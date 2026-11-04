import time
import numpy as np
from fractions import Fraction
from itertools import combinations

DIMENSIONS = [2, 4, 8]
_START_TIME = time.time()
_DEADLINE = 120.0  # Safety margin before 150s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Build the rank-56 Strassen decomposition
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 rows
    V = [[0] * rank for _ in range(m * p)]  # 32 rows
    W = [[0] * rank for _ in range(p * n)]  # 16 rows
    
    def add_term(u_coeffs, v_coeffs, w_coeffs, col_idx):
        for (i, j), val in u_coeffs.items():
            if val != 0:
                U[i*m + j][col_idx] += val
        for (j, k), val in v_coeffs.items():
            if val != 0:
                V[j*p + k][col_idx] += val
        for (k, i), val in w_coeffs.items():
            if val != 0:
                W[k*n + i][col_idx] += val

    col = 0
    
    for j_block in [0, 2]:
        for k_block in [0, 2, 4, 6]:
            # Strassen 7 terms for 2x2 * 2x2
            # 1. (A00+A11)(B00+B11)
            add_term({(0, j_block): 1, (1, j_block+1): 1}, 
                     {(j_block, k_block): 1, (j_block+1, k_block+1): 1}, 
                     {(k_block, 0): 1, (k_block+1, 1): 1}, col); col += 1
            
            # 2. (A10+A11)(B00)
            add_term({(1, j_block): 1, (1, j_block+1): 1}, 
                     {(j_block, k_block): 1}, 
                     {(k_block, 1): 1, (k_block+1, 1): -1}, col); col += 1
            
            # 3. (A00)(B01-B11)
            add_term({(0, j_block): 1}, 
                     {(j_block, k_block+1): 1, (j_block+1, k_block+1): -1}, 
                     {(k_block+1, 0): 1, (k_block+1, 1): 1}, col); col += 1
            
            # 4. (A11)(B10-B00)
            add_term({(1, j_block+1): 1}, 
                     {(j_block+1, k_block): 1, (j_block, k_block): -1}, 
                     {(k_block, 0): 1, (k_block, 1): 1}, col); col += 1
            
            # 5. (A00+A01)(B11)
            add_term({(0, j_block): 1, (0, j_block+1): 1}, 
                     {(j_block+1, k_block+1): 1}, 
                     {(k_block, 0): -1, (k_block+1, 0): 1}, col); col += 1
            
            # 6. (A10-A00)(B00+B01)
            add_term({(1, j_block): 1, (0, j_block): -1}, 
                     {(j_block, k_block): 1, (j_block, k_block+1): 1}, 
                     {(k_block+1, 1): 1}, col); col += 1
            
            # 7. (A01-A11)(B10+B11)
            add_term({(0, j_block+1): 1, (1, j_block+1): -1}, 
                     {(j_block+1, k_block): 1, (j_block+1, k_block+1): 1}, 
                     {(k_block, 0): 1}, col); col += 1
    
    # Verify correctness first
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
    
    T_recon = np.einsum('ur,vr,wr->uvw', U_np, V_np, W_np)
    
    if not np.allclose(T_recon, T_target, atol=1e-9):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Now attempt rank reduction via exact linear algebra
    # The tensor T = sum_r u_r ⊗ v_r ⊗ w_r
    # Mode-1 flattening: for each r, the vector is v_r ⊗ w_r (flattened)
    # If the 56 vectors {v_r ⊗ w_r} are linearly dependent, we can reduce rank
    
    # Actually, a better approach: check if the columns of U, V, W have
    # linear dependencies that allow term elimination.
    
    # The rank of the tensor is at most the minimum of:
    # - rank of mode-1 unfolding (n*m) x (m*p*p*n)
    # - rank of mode-2 unfolding (m*p) x (n*m*p*n)  
    # - rank of mode-3 unfolding (p*n) x (n*m*m*p)
    
    # But more directly: if we can find a non-trivial linear combination
    # c_1*u_1⊗v_1⊗w_1 + ... + c_56*u_56⊗v_56⊗w_56 = 0,
    # we can eliminate one term.
    
    # Let's check the mode-1 flattening: T_(1) is (n*m) x (m*p * p*n)
    # Each column r of T_(1) corresponds to vec(v_r ⊗ w_r)
    
    # Build mode-1 flattening matrix
    # T_(1)[i, (j,k)] = T[i,j,k] for fixed i
    # In terms of factors: T_(1)[:, r] = sum over... 
    # Actually T_(1) = U * (V ⊙ W)^T where ⊙ is Khatri-Rao product
    
    # Khatri-Rao product of V (m*p x rank) and W (p*n x rank):
    # (V ⊙ W)[(j,k), r] = V[j*p+k, r] * W[k*n+i, r]... 
    # Wait, the Khatri-Rao for mode-1 flattening:
    # T_(1) is (n*m) x (m*p * p*n), but that's not right.
    
    # Let me think more carefully. The tensor T has indices (i,j,k) with
    # i in [0,n), j in [0,m), k in [0,p).
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Mode-1 flattening: reshape T to (n*m) x (m*p)
    # T_(1)[(i), (j,k)] = T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    # This isn't a simple matrix product because W depends on i and k.
    
    # Alternative: just check if any subset of the 56 rank-1 terms is linearly
    # dependent by checking the rank of the set of vectors {u_r ⊗ v_r ⊗ w_r}
    # in the full tensor space (n*m*p dimensional).
    
    # Flatten each rank-1 term into a vector of length n*m*p
    # term_r[i*m*p + j*p + k] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    dim = n * m * p  # 64
    terms = []
    for r in range(rank):
        vec = np.zeros(dim)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[i*m*p + j*p + k] = U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
        terms.append(vec)
    
    terms_np = np.array(terms, dtype=float)  # shape (56, 64)
    
    # Check the rank of these 56 vectors
    # If rank < 56, there are linear dependencies
    U_svd, S, Vt = np.linalg.svd(terms_np, full_matrices=False)
    
    # Count significant singular values
    tol = 1e-10
    num_independent = np.sum(S > tol)
    
    if num_independent < rank:
        # There are linear dependencies. Try to find a minimal subset.
        # This is complex; for now, just note the rank.
        # The actual tensor rank might be lower, but we need an explicit decomposition.
        # For this iteration, we'll keep the 56-term decomposition as it's verified.
        pass
    
    # Return the verified rank-56 decomposition
    return {
        "U": U_np.tolist(),
        "V": V_np.tolist(),
        "W": W_np.tolist()
    }

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result['U'][0]))
