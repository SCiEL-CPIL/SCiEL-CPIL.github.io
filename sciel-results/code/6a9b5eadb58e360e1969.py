"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 9]
DEADLINE = 135.0  # safety margin before 150s kill

def _build_schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition as numpy arrays."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Check if the decomposition is exact: U[i,j,r]*V[j,k,r]*W[k,i,r] = delta pattern."""
    # The matrix multiplication tensor T[i,j,k] = 1 if the output C[i,k] = sum_j A[i,j]*B[j,k]
    # The decomposition requires: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    # where T[i,j,k] = 1 for all valid i,j,k (since C[i,k] = sum_j A[i,j] B[j,k])
    # Actually T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # So we need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k
    
    # Efficient check: for each r, the outer product structure
    # Let's just do a direct check for small ranks
    n, m, p = n, m, p
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    
    # Build the tensor T of shape (n, m, p)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We can compute this efficiently:
    # For each r: U_r (n*m,1) * V_r (m*p,1) * W_r (p*n,1)
    # This is a rank-1 tensor in the (i,j,k) space with specific index mappings.
    
    # Let's reshape:
    # U_r reshaped to (n, m), V_r to (m, p), W_r to (p, n)
    # The contribution to T[i,j,k] from rank r is: U_r[i,j] * V_r[j,k] * W_r[k,i]
    
    T = np.zeros((n, m, p))
    for r in range(rank):
        Ur = U[:, r].reshape(n, m)
        Vr = V[:, r].reshape(m, p)
        Wr = W[:, r].reshape(p, n)
        # T[i,j,k] += Ur[i,j] * Vr[j,k] * Wr[k,i]
        # This is: for each i, Ur[i,:] * (Vr * Wr[:,i])^T ... 
        # Let's do it directly:
        for i in range(n):
            # Ur[i, :] has shape (m,)
            # We need Vr[j,k] * Wr[k,i] for all j,k
            # Wr[:, i] has shape (p,)
            # So Vr * Wr[:, i][:, None] has shape (m, p)
            T[i, :, :] += Ur[i, :][:, None] * (Vr * Wr[:, i][None, :])
    
    # Check if T is all ones
    return np.allclose(T, np.ones((n, m, p)), atol=1e-9)

def _verify_fast(U, V, W, n, m, p):
    """Faster verification using matrix operations."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # For fixed i,k: sum_j T[i,j,k] should equal 1*m? No, T[i,j,k]=1 for each j.
    # Let's compute T directly.
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # = sum_r (U3[i,j,r] * W3[k,i,r]) * V3[j,k,r]
    
    # For each i, k:
    # Let A_ik[r] = U3[i,:,r] * W3[k,i,r]  -- no, U3[i,j,r] depends on j
    # T[i,j,k] = sum_r U3[i,j,r] * W3[k,i,r] * V3[j,k,r]
    
    # Let's compute: for each i, build matrix M_ik of shape (m, p, rank)
    # Actually, let's just do the full sum.
    # T has shape (n, m, p)
    T = np.zeros((n, m, p))
    for r in range(rank):
        Ur = U3[:, :, r]  # (n, m)
        Vr = V3[:, :, r]  # (m, p)
        Wr = W3[:, :, r]  # (p, n)
        # T[i,j,k] += Ur[i,j] * Vr[j,k] * Wr[k,i]
        # For each i: T[i,:, :] += Ur[i,:][:,None] * (Vr * Wr[:,i][None,:])
        for i in range(n):
            T[i, :, :] += Ur[i, :][:, None] * (Vr * Wr[:, i][None, :])
    
    return np.allclose(T, np.ones((n, m, p)), atol=1e-9)

def _try_decomposition(n, m, p, rank, max_iters=100, tol=1e-8):
    """Try to find a rank-rank decomposition using iterative refinement.
    
    We use the approach of starting from a random/schoolbook subset and refining.
    """
    if rank < n * m * p:
        # Try to find a valid decomposition
        # Use alternating minimization / gradient-free approach
        # Start from a random initialization or truncated schoolbook
        
        # For now, this is a placeholder - the main search will be in construct()
        pass
    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U0, V0, W0 = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
    
    # Try to find lower rank decompositions
    # Known bounds: for 2x3 x 3x9, the rank is at most 54 (schoolbook)
    # Lower bounds exist but let's try to search
    
    # Strategy: try ranks from a lower bound upward
    # A simple lower bound: max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    # But actually the tensor rank lower bound is more complex.
    
    # Let's try a simple approach: for each rank r from lower_bound to best_rank-1,
    # try random initializations and see if we can converge.
    
    lower_bound = max(n * m, m * p, p * n)
    
    for rank in range(lower_bound, best_rank):
        if time.time() - start_time > DEADLINE:
            break
        
        # Try multiple random initializations
        found = False
        for attempt in range(20):
            if time.time() - start_time > DEADLINE:
                break
            
            # Random initialization
            random.seed(hash((rank, attempt)) % (2**32))
            np.random.seed(hash((rank, attempt)) % (2**32))
            
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Iterative refinement
            converged = False
            for it in range(50):
                if time.time() - start_time > DEADLINE:
                    break
                
                # Compute current T
                U3 = U.reshape(n, m, rank)
                V3 = V.reshape(m, p, rank)
                W3 = W.reshape(p, n, rank)
                
                T = np.zeros((n, m, p))
                for r in range(rank):
                    Ur = U3[:, :, r]
                    Vr = V3[:, :, r]
                    Wr = W3[:, :, r]
                    for i in range(n):
                        T[i, :, :] += Ur[i, :][:, None] * (Vr * Wr[:, i][None, :])
                
                err = np.max(np.abs(T - 1.0))
                if err < tol:
                    converged = True
                    break
                
                # Gradient-free update: try small perturbations
                # This is very slow, so let's just check
                pass
            
            if converged:
                found = True
                break
        
        if found:
            best_rank = rank
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if time.time() - start_time > DEADLINE:
        return best_result
    
    return best_result
