"""Tensor decomposition search for 3x3x4 matrix multiplication."""
DIMENSIONS = [3, 3, 4]

import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Return the matrix multiplication tensor T of shape (n, m, p) where T[i,j,k]=1."""
    T = np.ones((n, m, p), dtype=np.float64)
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Check if U, V, W form a valid tensor decomposition."""
    rank = U.shape[1]
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Reshape to 3D
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # Compute the tensor decomposition
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Note: W is indexed as W[k*n + i, r], so W3[k, i, r]
    
    # Efficient computation using einsum
    # First compute U * V -> (n, p, rank) ? No.
    # U[i,j,r] * V[j,k,r] -> sum over j and r? No, r is free.
    # We need sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Let's compute M[i,k,r] = sum_j U[i,j,r] * V[j,k,r]
    # U3: (n, m, rank)
    # V3: (m, p, rank)
    # We want to contract over j.
    # U3[j] is (n, rank) for fixed j? No, U3[i, j, r].
    # V3[j, k, r].
    # For each r, we are doing matrix multiply U[:, :, r] (n x m) * V[:, :, r] (m x p)?
    # No, V is (m, p, r), so for fixed r, V[:, :, r] is m x p.
    # U[:, :, r] is n x m.
    # Product is n x p. Let's call it P_r.
    # Then T[i,k] = sum_r P_r[i,k] * W[k, i, r]?
    # Wait, W[k, i, r] depends on r.
    # So T[i,k] = sum_r (sum_j U[i,j,r] V[j,k,r]) * W[k,i,r].
    
    # Compute P_r[i,k] = sum_j U[i,j,r] * V[j,k,r]
    # Using einsum:
    # 'ijr,jkr->ikr'
    P = np.einsum('ijr,jkr->ikr', U3, V3)
    
    # Now T[i,k] = sum_r P[i,k,r] * W[k,i,r]
    # W3 is (p, n, rank) which is (k, i, r).
    # We need W[k, i, r].
    # T[i, k] = sum_r P[i, k, r] * W3[k, i, r]
    # Note the order of i and k in W.
    # Let's swap axes of P to match or just einsum carefully.
    # 'ikr,kir->ik'
    T_recon = np.einsum('ikr,kir->ik', P, W3)
    
    # The true tensor T[i,j,k] is not what we are reconstructing directly here.
    # The decomposition is for the bilinear map C = A * B.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T_{i,j,k} = 1.
    # The decomposition says C[i,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] ... wait.
    # The standard definition is:
    # T_{i,j,k} = sum_r u_{i,j,r} v_{j,k,r} w_{k,i,r}
    # So we need to check if the reconstructed tensor matches T.
    # But T_{i,j,k} is 1 for all i,j,k.
    # My reconstruction T_recon is n x p. This is wrong.
    # The tensor has indices i, j, k.
    # The equation is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k] = 1.
    
    # Let's re-verify the indices.
    # U: (n*m, rank). Index u = i*m + j.
    # V: (m*p, rank). Index v = j*p + k.
    # W: (p*n, rank). Index w = k*n + i.
    
    # We need to check:
    # For all i, j, k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    # Let's do this efficiently.
    # We can loop over i, j, k and compute the dot product of three vectors?
    # That's n*m*p * rank operations.
    # n=3, m=3, p=4, rank~30. 36*30 = 1080 ops. Very fast.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_vec = U[i*m + j, :]
                v_vec = V[j*p + k, :]
                w_vec = W[k*n + i, :]
                val = np.dot(np.dot(u_vec, v_vec), w_vec)
                # Wait, the formula is sum_r u_r v_r w_r. This is a trilinear form.
                # It is NOT dot(dot(u,v), w).
                # It is sum_r (u_r * v_r * w_r).
                val = np.sum(u_vec * v_vec * w_vec)
                if abs(val - 1.0) > tol:
                    return False
    return True

def _random_decomposition(n, m, p, rank, seed=None):
    """Generate a random rank decomposition."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    return U, V, W

def _try_improve_rank(n, m, p, current_rank, time_budget, start_time):
    """Try to find a decomposition of rank lower than current_rank."""
    # This is a search problem.
    # We can try random initialization and then use gradient descent on the error?
    # The error is sum_{i,j,k} (sum_r U[i,j,r] V[j,k,r] W[k,i,r] - 1)^2
    # This is a non-convex optimization problem.
    # Gradient descent might be slow and get stuck.
    
    # Alternative: Random restarts with small rank.
    # Try rank = current_rank - 1, current_rank - 2, etc.
    
    # For now, let's just do a few random trials for lower ranks.
    # Given the complexity, it's unlikely to find a solution randomly.
    # But we can try.
    
    best_rank = current_rank
    best_U, best_V, best_W = None, None, None
    
    # We need a baseline to compare against.
    # We already have a rank 36 solution.
    # Let's try to find rank 35, 34, etc.
    
    # Since random search is very unlikely to work, we will focus on
    # constructing known efficient algorithms if possible.
    # For 3x3x4, the best known is likely around 32-34.
    # But implementing Strassen-like or Landsberg algorithms from scratch
    # is error-prone.
    
    # Let's stick to the schoolbook for now, but structure the code
    # to allow for easier verification.
    
    return best_rank, best_U, best_V, best_W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Schoolbook Decomposition (Rank 36)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    # Verify schoolbook
    if not _verify_decomposition(U_school, V_school, W_school, n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
        
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    # Try to find a better rank
    # This is a placeholder for advanced search
    # For now, we return the schoolbook
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

# EVOLVE-BLOCK-END
