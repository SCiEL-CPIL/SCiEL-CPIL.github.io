"""Matrix multiplication 2,5,6 rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Build rank-54 Strassen decomposition
    rank_base = 54
    U = [[0] * rank_base for _ in range(n * m)]
    V = [[0] * rank_base for _ in range(m * p)]
    W = [[0] * rank_base for _ in range(p * n)]
    col = 0
    
    def add_strassen_block(i_base, j_base, k_base):
        nonlocal col
        # M0: (A00+A11)(B00+B11)
        U[i_base*m+j_base][col] += 1
        U[(i_base+1)*m+(j_base+1)][col] += 1
        V[j_base*p+k_base][col] += 1
        V[(j_base+1)*p+(k_base+1)][col] += 1
        W[k_base*n+i_base][col] += 1
        W[(k_base+1)*n+(i_base+1)][col] += 1
        col += 1
        # M1: (A10+A11)B00
        U[(i_base+1)*m+j_base][col] += 1
        U[(i_base+1)*m+(j_base+1)][col] += 1
        V[j_base*p+k_base][col] += 1
        W[k_base*n+(i_base+1)][col] += 1
        W[(k_base+1)*n+(i_base+1)][col] -= 1
        col += 1
        # M2: A00(B01-B11)
        U[i_base*m+j_base][col] += 1
        V[j_base*p+(k_base+1)][col] += 1
        V[(j_base+1)*p+(k_base+1)][col] -= 1
        W[(k_base+1)*n+i_base][col] += 1
        W[(k_base+1)*n+(i_base+1)][col] += 1
        col += 1
        # M3: A11(B10-B00)
        U[(i_base+1)*m+(j_base+1)][col] += 1
        V[(j_base+1)*p+k_base][col] += 1
        V[j_base*p+k_base][col] -= 1
        W[k_base*n+i_base][col] += 1
        W[k_base*n+(i_base+1)][col] += 1
        col += 1
        # M4: (A00+A01)B11
        U[i_base*m+j_base][col] += 1
        U[i_base*m+(j_base+1)][col] += 1
        V[(j_base+1)*p+(k_base+1)][col] += 1
        W[k_base*n+i_base][col] -= 1
        W[(k_base+1)*n+i_base][col] += 1
        col += 1
        # M5: (A10-A00)(B00+B01)
        U[(i_base+1)*m+j_base][col] += 1
        U[i_base*m+j_base][col] -= 1
        V[j_base*p+k_base][col] += 1
        V[j_base*p+(k_base+1)][col] += 1
        W[(k_base+1)*n+(i_base+1)][col] += 1
        col += 1
        # M6: (A01-A11)(B10+B11)
        U[i_base*m+(j_base+1)][col] += 1
        U[(i_base+1)*m+(j_base+1)][col] -= 1
        V[(j_base+1)*p+k_base][col] += 1
        V[(j_base+1)*p+(k_base+1)][col] += 1
        W[k_base*n+i_base][col] += 1
        col += 1

    def add_naive_block(i_base, j_base, k_base):
        nonlocal col
        U[i_base*m+j_base][col] += 1
        V[j_base*p+k_base][col] += 1
        W[k_base*n+i_base][col] += 1
        col += 1
        U[i_base*m+j_base][col] += 1
        V[j_base*p+(k_base+1)][col] += 1
        W[(k_base+1)*n+i_base][col] += 1
        col += 1
        U[(i_base+1)*m+j_base][col] += 1
        V[j_base*p+k_base][col] += 1
        W[k_base*n+(i_base+1)][col] += 1
        col += 1
        U[(i_base+1)*m+j_base][col] += 1
        V[j_base*p+(k_base+1)][col] += 1
        W[(k_base+1)*n+(i_base+1)][col] += 1
        col += 1

    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 2, 2)
    add_strassen_block(0, 2, 4)
    add_naive_block(0, 4, 0)
    add_naive_block(0, 4, 2)
    add_naive_block(0, 4, 4)
    
    if col != rank_base:
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][i*m*p + j*p + k] = 1
                    V[j*p+k][i*m*p + j*p + k] = 1
                    W[k*n+i][i*m*p + j*p + k] = 1
        return {"U": U, "V": V, "W": W}
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Verify
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    T_dec = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    if not np.allclose(T_dec, T, atol=1e-10):
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i*m*p + j*p + k
                    U[i*m+j][idx] = 1
                    V[j*p+k][idx] = 1
                    W[k*n+i][idx] = 1
        return {"U": U, "V": V, "W": W}
    
    # Rank-1 term elimination: find linear dependencies among the 54 rank-1 terms
    # Each rank-1 term r has vectors u_r, v_r, w_r such that T = sum_r u_r ⊗ v_r ⊗ w_r
    # A linear dependency means sum_r c_r * (u_r ⊗ v_r ⊗ w_r) = 0
    # This is a bilinear constraint, not linear. But we can try to find terms
    # that can be expressed as linear combinations of others.
    
    # Approach: For each pair of terms, check if one can be written as a linear
    # combination of the others. This is expensive, so we use a heuristic.
    
    # Alternative: Use the fact that the tensor has a specific structure.
    # The 54 terms come from 6 Strassen blocks (7 each) and 3 naive blocks (4 each).
    # Within each Strassen block, the 7 terms are linearly independent (they form
    # a basis for the 2x2x2 tensor space).
    
    # Try to find cross-block dependencies by checking if any term can be
    # expressed as a linear combination of terms from other blocks.
    
    # For each term r, try to express it as a linear combination of all other terms.
    # This requires solving a system of size (n*m * m*p * p*n) = 60 equations
    # with 53 unknowns (excluding term r).
    
    # Build the tensor of each rank-1 term
    term_tensors = []
    for r in range(rank_base):
        tt = np.outer(np.outer(U_arr[:, r], V_arr[:, r]), W_arr[:, r])
        tt = tt.reshape(n*m, m*p, p*n)
        term_tensors.append(tt.flatten())
    
    # Try to find if any term is a linear combination of the others
    for r in range(rank_base):
        if time.time() > deadline - 5:
            break
        
        # Build matrix of all other terms
        other_indices = [i for i in range(rank_base) if i != r]
        A = np.column_stack([term_tensors[i] for i in other_indices])
        b = term_tensors[r]
        
        # Solve A * c = b in least squares
        try:
            c, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
            # Check if b is in the span of the other terms
            if len(residuals) > 0 and residuals[0] < 1e-10:
                # Term r is a linear combination of the others
                # We can remove it
                pass  # In a full implementation, we would remove this term
        except:
            pass
    
    # For now, return the rank-54 decomposition
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
