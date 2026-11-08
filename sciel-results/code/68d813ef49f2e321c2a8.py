import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 7]
n, m, p = DIMENSIONS

def _verify(U, V, W, n, m, p):
    """Verify that the tensor decomposition is exact."""
    rank = len(U[0]) if U else 0
    # Build the multiplication tensor T[i,j,k] = 1 if we need (i,j)*(j,k) -> (i,k)
    # Actually, the tensor is T[i,j,k] such that C[i,k] = sum_j A[i,j]*B[j,k]
    # The decomposition requires: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # Let's verify by checking a few random matrices
    rng = np.random.RandomState(42)
    for _ in range(50):
        A = rng.randint(-3, 4, size=(n, m)).astype(float)
        B = rng.randint(-3, 4, size=(m, p)).astype(float)
        C = A @ B
        
        # Compute C' from decomposition
        C_prime = np.zeros((n, p))
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for c in range(rank):
            u_c = U_arr[:, c]  # shape (n*m,)
            v_c = V_arr[:, c]  # shape (m*p,)
            w_c = W_arr[:, c]  # shape (p*n,)
            
            # u_c reshaped to (n, m), v_c to (m, p), w_c to (p, n)
            u_reshaped = u_c.reshape(n, m)
            v_reshaped = v_c.reshape(m, p)
            w_reshaped = w_c.reshape(p, n)
            
            # The rank-1 contribution to C[i,k] is sum_j u[i,j] * v[j,k] * w[k,i]
            # Let's compute this properly
            # For each (i,k): sum_j u[i,j] * v[j,k] * w[k,i]
            # = w[k,i] * sum_j u[i,j] * v[j,k]
            # = w[k,i] * (u_reshaped @ v_reshaped)[i,k]
            
            contrib = u_reshaped @ v_reshaped  # (n, p)
            C_prime += (contrib * w_reshaped.T)  # w_reshaped is (p,n), so w_reshaped.T is (n,p)
        
        if not np.allclose(C, C_prime, atol=1e-10):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s out of 150s limit
    
    # Start with schoolbook
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_school
    
    # Try to find a lower-rank decomposition
    # The minimum rank for 2x4 x 4x7 is at least 2*4=8 (by submultiplicativity)
    # and at most 56 (schoolbook).
    # Known lower bounds: rank >= max(n*m, m*p, n*p) = max(8, 28, 14) = 28
    # Actually, the tensor rank for matrix multiplication [n,m,p] satisfies
    # R >= max(n*m, m*p, p*n) and R >= n*m*p - something.
    # For 2x4x4: rank is 9 (not 16 schoolbook).
    # For 2x2x7: rank is 13 (not 14 schoolbook).
    
    # Let's try a Kronecker-like decomposition.
    # 2x4 can be seen as 2 x (2x2). 
    # We can try to exploit the structure.
    
    # Strategy: Try random search for small perturbations of schoolbook
    # Or try known algebraic decompositions.
    
    # Actually, let's try a systematic approach:
    # For 2x4x7, we can think of it as multiplying 2x4 by 4x7.
    # The 4x7 part: we can use the fact that 4 = 2*2, 7 = 4+3.
    
    # Let's try a different approach: use the fact that matrix multiplication
    # is bilinear, and try to find a low-rank decomposition via
    # randomized SVD-like methods on the tensor.
    
    # But first, let's try a few known constructions:
    # 1. Treat 2x4 as block diagonal or something
    # 2. Use Strassen-like for 2x2 blocks
    
    # For a 2x4 matrix A and 4x7 matrix B:
    # A = [A1; A2] where A1, A2 are 1x4
    # B = [B1, B2, B3, B4] where Bj are 4x1
    # C[i,k] = A1[i]*B1[k] + A2[i]*B2[k] + A3[i]*B3[k] + A4[i]*B4[k]
    
    # This is just 4 separate 1x4 x 4x1 products, each of rank 4, total 16.
    # But that's worse than schoolbook (56/4=14 per... wait, schoolbook is 2*4*7=56).
    # Actually schoolbook for 2x4x7 is 2*4*7 = 56.
    
    # The lower bound is max(2*4, 4*7, 7*2) = max(8, 28, 14) = 28.
    
    # Let's try to achieve rank 28.
    # For 4x7: the rank of 4x7 multiplication is at least 28 (since 4*7=28).
    # Can we do better? For square matrices, Strassen gives 7 for 2x2.
    # For 4x4, it's 49 (Strassen^2) or lower.
    # For 4x7, we might use padding: 4x7 as 4x8 with one zero column.
    # 4x8 = 4x(4+4), use 4x4 Strassen twice: 49+49 = 98? No that's for the whole thing.
    
    # Actually, let me think differently. The tensor is T in R^{2x4 x 4x7 x 7x2}.
    # We want to find rank r such that T = sum_{c=1}^r u_c ⊗ v_c ⊗ w_c.
    
    # Let's try a numerical optimization approach:
    # Initialize with random factors and use gradient descent to minimize
    # ||T - sum_c u_c ⊗ v_c ⊗ w_c||_F^2.
    
    # But this is a hard non-convex problem. Let's try a simpler approach:
    # Use the fact that for small dimensions, we can try all "structured" decompositions.
    
    # Alternative: Use the known result that R(2,4,7) <= some value.
    # Let's try to construct a decomposition of rank 32 or so.
    
    # For now, let's try a random search with verification.
    rng = np.random.RandomState(12345)
    
    for target_rank in range(28, min(best_rank, 57)):
        if time.time() > deadline - 30:
            break
        
        # Try a few random initializations for this rank
        for attempt in range(3):
            if time.time() > deadline - 30:
                break
            
            U_rand = rng.randn(n * m, target_rank)
            V_rand = rng.randn(m * p, target_rank)
            W_rand = rng.randn(p * n, target_rank)
            
            # Scale to make entries O(1)
            U_rand /= np.sqrt(n * m)
            V_rand /= np.sqrt(m * p)
            W_rand /= np.sqrt(p * n)
            
            # Verify
            if _verify(U_rand.tolist(), V_rand.tolist(), W_rand.tolist(), n, m, p):
                best_rank = target_rank
                best_result = {
                    "U": U_rand.tolist(),
                    "V": V_rand.tolist(),
                    "W": W_rand.tolist()
                }
                break
        
        if best_rank == target_rank:
            break
    
    return best_result
