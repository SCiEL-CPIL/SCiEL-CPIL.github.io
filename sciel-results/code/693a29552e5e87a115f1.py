import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 8]
_START_TIME = time.time()
_DEADLINE = 120.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    n, m, p = DIMENSIONS
    
    # Build the rank-56 Strassen decomposition
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
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
            add_term({(0, j_block): 1, (1, j_block+1): 1}, 
                     {(j_block, k_block): 1, (j_block+1, k_block+1): 1}, 
                     {(k_block, 0): 1, (k_block+1, 1): 1}, col); col += 1
            add_term({(1, j_block): 1, (1, j_block+1): 1}, 
                     {(j_block, k_block): 1}, 
                     {(k_block, 1): 1, (k_block+1, 1): -1}, col); col += 1
            add_term({(0, j_block): 1}, 
                     {(j_block, k_block+1): 1, (j_block+1, k_block+1): -1}, 
                     {(k_block+1, 0): 1, (k_block+1, 1): 1}, col); col += 1
            add_term({(1, j_block+1): 1}, 
                     {(j_block+1, k_block): 1, (j_block, k_block): -1}, 
                     {(k_block, 0): 1, (k_block, 1): 1}, col); col += 1
            add_term({(0, j_block): 1, (0, j_block+1): 1}, 
                     {(j_block+1, k_block+1): 1}, 
                     {(k_block, 0): -1, (k_block+1, 0): 1}, col); col += 1
            add_term({(1, j_block): 1, (0, j_block): -1}, 
                     {(j_block, k_block): 1, (j_block, k_block+1): 1}, 
                     {(k_block+1, 1): 1}, col); col += 1
            add_term({(0, j_block+1): 1, (1, j_block+1): -1}, 
                     {(j_block+1, k_block): 1, (j_block+1, k_block+1): 1}, 
                     {(k_block, 0): 1}, col); col += 1

    # Verify correctness
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
    
    # Attempt rank reduction via linear dependencies
    dim = n * m * p
    terms = []
    for r in range(rank):
        vec = np.zeros(dim)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[i*m*p + j*p + k] = U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
        terms.append(vec)
    
    terms_np = np.array(terms, dtype=float)
    
    # Find the rank of the terms
    U_svd, S, Vt = np.linalg.svd(terms_np, full_matrices=False)
    tol = 1e-10
    num_independent = np.sum(S > tol)
    
    if num_independent < rank:
        # We can eliminate (rank - num_independent) terms.
        # To do this, we need to find the null space of the term matrix.
        # Let c be a vector in the null space.
        # Then sum c_r term_r = 0.
        # We can set one of the terms to be a combination of others?
        # Actually, if sum c_r term_r = 0, we can remove any term r where c_r != 0
        # and replace it with the others? No, we just drop the term if we can
        # express it as a combination of the others?
        # The decomposition is T = sum term_r.
        # If sum c_r term_r = 0, then T = sum (1 + c_r) term_r? No.
        # T = sum term_r.
        # If term_k = - sum_{r!=k} (c_r/c_k) term_r, then we can substitute term_k.
        # But term_k is a rank-1 tensor. The sum of other rank-1 tensors is not necessarily rank-1.
        # So we cannot simply replace term_k with a sum of others.
        # We can only reduce the rank if we can find a subset of terms that sums to 0.
        # If term_1 + ... + term_k = 0, then we can drop all k terms?
        # No, T = sum term_r. If sum_{i=1}^k term_i = 0, then T = sum_{k+1}^R term_r.
        # So we can drop k terms.
        
        # We need to find a linear combination of terms that sums to 0.
        # This is equivalent to finding a vector c in the null space of terms_np.
        # If c is in the null space, sum c_r term_r = 0.
        # We can drop the terms corresponding to the non-zero components of c?
        # No, we can only drop them if we can remove them from the sum.
        # T = sum term_r.
        # If c is in null space, sum c_r term_r = 0.
        # This doesn't mean we can drop terms.
        # It means the terms are linearly dependent.
        # The rank of the tensor is the minimal number of terms.
        # If the terms are linearly dependent, the rank is less than the number of terms.
        # But we need to find the explicit minimal decomposition.
        # This is a hard problem.
        
        # For now, we will just return the 56-term decomposition.
        pass
    
    return {
        "U": U_np.tolist(),
        "V": V_np.tolist(),
        "W": W_np.tolist()
    }

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result['U'][0]))
