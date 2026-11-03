import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x5 matrix.
    
    Uses a combinatorial approach with structured sparse patterns.
    Falls back to schoolbook if no better solution is found within the time budget.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Build the target tensor for verification
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    def verify(U, V, W, tol=1e-9):
        """Verify decomposition using numpy."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Compute the tensor decomposition
        # T_computed[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        T_computed = np.einsum('ar,br,cr->abc', U, V, W)
        
        return np.allclose(T_computed, T_target, atol=tol)
    
    def clean_matrix(M, tol=1e-7):
        """Clean a matrix by snapping near-zero and near-half-integer values."""
        M = np.array(M, dtype=float)
        M_clean = np.zeros_like(M)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                val = M[i, j]
                if abs(val) < tol:
                    M_clean[i, j] = 0.0
                else:
                    val2 = round(val * 2)
                    if abs(val * 2 - val2) < tol * 10:
                        M_clean[i, j] = val2 / 2.0
                    else:
                        M_clean[i, j] = val
        return M_clean
    
    # Build schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Strategy: Try structured sparse decompositions
    # Each rank term r defines a pattern of which (i,j,k) it covers
    # For a term to contribute to (i,j,k): U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] != 0
    
    # Try ranks from 50 down to 40 (unlikely to beat schoolbook significantly)
    for rank in range(50, min(60, rank_schoolbook)):
        if time.time() > deadline:
            break
        
        # Try several structured patterns
        for seed in range(10):
            if time.time() > deadline:
                break
            
            np.random.seed(seed * 1000 + rank)
            
            # Create sparse factor matrices with random signs
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            # Assign each (i,j,k) to a rank term
            # Use a structured assignment
            assignments = {}
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Assign to rank term based on a hash-like function
                        r = (i * 7 + j * 13 + k * 17 + seed) % rank
                        if r not in assignments:
                            assignments[r] = []
                        assignments[r].append((i, j, k))
            
            # For each rank term, set the factors to cover its assigned triples
            for r, triples in assignments.items():
                if not triples:
                    continue
                
                # Collect unique indices
                i_set = set(i for i, j, k in triples)
                j_set = set(j for i, j, k in triples)
                k_set = set(k for i, j, k in triples)
                
                # Set U values for rows i*m+j
                for i in i_set:
                    for j in j_set:
                        U[i * m + j, r] = 1.0
                
                # Set V values for rows j*p+k
                for j in j_set:
                    for k in k_set:
                        V[j * p + k, r] = 1.0
                
                # Set W values for rows k*n+i
                for k in k_set:
                    for i in i_set:
                        W[k * n + i, r] = 1.0
            
            # Verify
            if verify(U, V, W):
                best_rank = rank
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                break
        
        if best_rank < rank_schoolbook:
            break
    
    # If we found a better rank, return it
    if best_rank < rank_schoolbook:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Otherwise, return the schoolbook decomposition
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
