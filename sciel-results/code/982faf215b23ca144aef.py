"""
Program for matrix multiplication 2x5x5 rank minimization.
Uses Strassen on 2x2 blocks and attempts to reduce rank by finding linear dependencies.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START

def _build_strassen_decomposition():
    """Build the rank-46 Strassen-based decomposition for 2x5x5."""
    n, m, p = 2, 5, 5
    terms = []
    
    def add_strassen(i_off, j_off, k_off):
        """Adds 7 terms for a 2x2x2 Strassen block at given offsets."""
        # M1 = (A00+A11)(B00+B11)
        terms.append((
            {(i_off, j_off): 1, (i_off+1, j_off+1): 1},
            {(j_off, k_off): 1, (j_off+1, k_off+1): 1},
            {(k_off, i_off): 1, (k_off+1, i_off+1): 1}
        ))
        # M2 = (A10+A11)B00
        terms.append((
            {(i_off+1, j_off): 1, (i_off+1, j_off+1): 1},
            {(j_off, k_off): 1},
            {(k_off, i_off+1): 1, (k_off+1, i_off+1): -1}
        ))
        # M3 = A00(B01-B11)
        terms.append((
            {(i_off, j_off): 1},
            {(j_off, k_off+1): 1, (j_off+1, k_off+1): -1},
            {(k_off+1, i_off): 1, (k_off+1, i_off+1): 1}
        ))
        # M4 = A11(B10-B00)
        terms.append((
            {(i_off+1, j_off+1): 1},
            {(j_off+1, k_off): 1, (j_off, k_off): -1},
            {(k_off, i_off): 1, (k_off, i_off+1): 1}
        ))
        # M5 = (A00+A01)B11
        terms.append((
            {(i_off, j_off): 1, (i_off, j_off+1): 1},
            {(j_off+1, k_off+1): 1},
            {(k_off, i_off): -1, (k_off+1, i_off): 1}
        ))
        # M6 = (A10-A00)(B00+B01)
        terms.append((
            {(i_off+1, j_off): 1, (i_off, j_off): -1},
            {(j_off, k_off): 1, (j_off, k_off+1): 1},
            {(k_off+1, i_off+1): 1}
        ))
        # M7 = (A01-A11)(B10+B11)
        terms.append((
            {(i_off, j_off+1): 1, (i_off+1, j_off+1): -1},
            {(j_off+1, k_off): 1, (j_off+1, k_off+1): 1},
            {(k_off, i_off): 1}
        ))
    
    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        """Adds schoolbook terms for the given ranges."""
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    terms.append((
                        {(i, j): 1},
                        {(j, k): 1},
                        {(k, i): 1}
                    ))
    
    # Partition inner dim 5 = 2+2+1, output dim 5 = 2+2+1
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_schoolbook(0, 2, 0, 2, 4, 5)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    add_schoolbook(0, 2, 2, 4, 4, 5)
    add_schoolbook(0, 2, 4, 5, 0, 5)
    
    return terms

def _try_reduce_rank(terms, max_reductions=5, deadline=None):
    """Try to reduce rank by finding linear dependencies among terms."""
    n, m, p = 2, 5, 5
    dim_u = n * m  # 10
    dim_v = m * p  # 25
    dim_w = p * n  # 10
    total_dim = dim_u + dim_v + dim_w  # 45
    
    def terms_to_matrix(terms):
        """Convert terms to a matrix where each row is a flattened term."""
        mat = []
        for u_dict, v_dict, w_dict in terms:
            row = [0] * total_dim
            for (i, j), val in u_dict.items():
                row[i * m + j] = val
            for (j, k), val in v_dict.items():
                row[dim_u + j * p + k] = val
            for (k, i), val in w_dict.items():
                row[dim_u + dim_v + k * n + i] = val
            mat.append(row)
        return np.array(mat, dtype=float)
    
    current_terms = list(terms)
    for _ in range(max_reductions):
        if deadline and time.time() > deadline - 10:
            break
        
        mat = terms_to_matrix(current_terms)
        n_terms = len(current_terms)
        
        # For each term, check if it can be expressed as a linear combination of others
        found_reduction = False
        for r in range(n_terms):
            if deadline and time.time() > deadline - 10:
                break
            
            # Get the r-th term and all others
            target = mat[r]
            others = np.delete(mat, r, axis=0)
            
            # Solve: target = others @ alpha
            # This is a system of total_dim equations in (n_terms-1) unknowns
            if n_terms - 1 < total_dim:
                continue
            
            # Use least squares to check if target is in the span of others
            alpha, residuals, rank, sv = np.linalg.lstsq(others, target, rcond=None)
            
            # Check if the fit is good
            if residuals.size > 0 and residuals[0] < 1e-10:
                # Term r is in the span of others, remove it
                current_terms.pop(r)
                found_reduction = True
                break
        
        if not found_reduction:
            break
    
    return current_terms

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2 by 5 matrix by a 5 by 5 matrix.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 5, 5
    
    # Build initial rank-46 decomposition
    terms = _build_strassen_decomposition()
    rank = len(terms)
    
    # Try to reduce rank
    if time.time() < deadline - 30:
        terms = _try_reduce_rank(terms, max_reductions=5, deadline=deadline)
        rank = len(terms)
    
    # Build output matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        u_dict, v_dict, w_dict = terms[c]
        for (i, j), val in u_dict.items():
            U[i * m + j][c] = val
        for (j, k), val in v_dict.items():
            V[j * p + k][c] = val
        for (k, i), val in w_dict.items():
            W[k * n + i][c] = val
    
    # Verify correctness
    if time.time() < deadline - 5.0:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        C_decomp = np.zeros((n, p))
        for r in range(rank):
            for i in range(n):
                for k in range(p):
                    s_u = 0.0
                    s_v = 0.0
                    for j in range(m):
                        s_u += U_arr[i*m+j, r] * A_test[i, j]
                        s_v += V_arr[j*p+k, r] * B_test[j, k]
                    C_decomp[i, k] += s_u * s_v * W_arr[k*n+i, r]
        
        if not np.allclose(C_decomp, C_expected, atol=1e-10):
            # Fallback to schoolbook if verification fails
            terms = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        terms.append((
                            {(i, j): 1},
                            {(j, k): 1},
                            {(k, i): 1}
                        ))
            rank = len(terms)
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            for c in range(rank):
                u_dict, v_dict, w_dict = terms[c]
                for (i, j), val in u_dict.items():
                    U[i * m + j][c] = val
                for (j, k), val in v_dict.items():
                    V[j * p + k][c] = val
                for (k, i), val in w_dict.items():
                    W[k * n + i][c] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
