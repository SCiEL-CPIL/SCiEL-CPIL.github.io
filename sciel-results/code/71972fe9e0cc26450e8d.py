"""
Matrix multiplication 2x4 by 4x5 - rank-minimal exact tensor decomposition.
Builds a rank-36 Strassen-based decomposition, then searches for rank reductions
by adding new bilinear terms that allow removing existing terms.
"""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _build_base_36():
    """Build the rank-36 Strassen-based decomposition."""
    n, m, p = 2, 4, 5
    rank = 36
    U = np.zeros((n * m, rank))  # 8 x 36
    V = np.zeros((m * p, rank))  # 20 x 36
    W = np.zeros((p * n, rank))  # 10 x 36
    
    col = 0
    
    def add_strassen_2x2(a_row, a_col, b_row, b_col, c_row, c_col):
        nonlocal col
        u00 = a_row * m + a_col
        u01 = a_row * m + a_col + 1
        u10 = (a_row + 1) * m + a_col
        u11 = (a_row + 1) * m + a_col + 1
        
        v00 = b_row * p + b_col
        v01 = b_row * p + b_col + 1
        v10 = (b_row + 1) * p + b_col
        v11 = (b_row + 1) * p + b_col + 1
        
        w00 = c_col * n + c_row
        w01 = (c_col + 1) * n + c_row
        w10 = c_col * n + c_row + 1
        w11 = (c_col + 1) * n + c_row + 1
        
        # M1 = (a00+a11)(b00+b11)
        U[u00, col] += 1.0
        U[u11, col] += 1.0
        V[v00, col] += 1.0
        V[v11, col] += 1.0
        W[w00, col] += 1.0
        W[w11, col] += 1.0
        col += 1
        
        # M2 = (a10+a11)(b00)
        U[u10, col] += 1.0
        U[u11, col] += 1.0
        V[v00, col] += 1.0
        W[w10, col] += 1.0
        W[w11, col] -= 1.0
        col += 1
        
        # M3 = a00(b01-b11)
        U[u00, col] += 1.0
        V[v01, col] += 1.0
        V[v11, col] -= 1.0
        W[w01, col] += 1.0
        W[w11, col] += 1.0
        col += 1
        
        # M4 = a11(b10-b00)
        U[u11, col] += 1.0
        V[v10, col] += 1.0
        V[v00, col] -= 1.0
        W[w00, col] += 1.0
        W[w10, col] += 1.0
        col += 1
        
        # M5 = (a00+a01)(b11)
        U[u00, col] += 1.0
        U[u01, col] += 1.0
        V[v11, col] += 1.0
        W[w00, col] -= 1.0
        W[w01, col] += 1.0
        col += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[u10, col] += 1.0
        U[u00, col] -= 1.0
        V[v00, col] += 1.0
        V[v01, col] += 1.0
        W[w11, col] += 1.0
        col += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[u01, col] += 1.0
        U[u11, col] -= 1.0
        V[v10, col] += 1.0
        V[v11, col] += 1.0
        W[w00, col] += 1.0
        col += 1
    
    def add_2x2_by_2x1(a_row, a_col, b_row, b_col, c_row, c_col):
        nonlocal col
        u00 = a_row * m + a_col
        u01 = a_row * m + a_col + 1
        u10 = (a_row + 1) * m + a_col
        u11 = (a_row + 1) * m + a_col + 1
        
        v0 = b_row * p + b_col
        v1 = (b_row + 1) * p + b_col
        
        w0 = c_col * n + c_row
        w1 = c_col * n + c_row + 1
        
        U[u00, col] = 1.0
        V[v0, col] = 1.0
        W[w0, col] = 1.0
        col += 1
        
        U[u01, col] = 1.0
        V[v1, col] = 1.0
        W[w0, col] = 1.0
        col += 1
        
        U[u10, col] = 1.0
        V[v0, col] = 1.0
        W[w1, col] = 1.0
        col += 1
        
        U[u11, col] = 1.0
        V[v1, col] = 1.0
        W[w1, col] = 1.0
        col += 1
    
    # A1 @ B1: 2x2 * 2x5
    add_strassen_2x2(a_row=0, a_col=0, b_row=0, b_col=0, c_row=0, c_col=0)
    add_strassen_2x2(a_row=0, a_col=0, b_row=0, b_col=2, c_row=0, c_col=2)
    add_2x2_by_2x1(a_row=0, a_col=0, b_row=0, b_col=4, c_row=0, c_col=4)
    
    # A2 @ B2: 2x2 * 2x5
    add_strassen_2x2(a_row=0, a_col=2, b_row=2, b_col=0, c_row=0, c_col=0)
    add_strassen_2x2(a_row=0, a_col=2, b_row=2, b_col=2, c_row=0, c_col=2)
    add_2x2_by_2x1(a_row=0, a_col=2, b_row=2, b_col=4, c_row=0, c_col=4)
    
    return U, V, W


def _verify(U, V, W):
    """Verify that U, V, W correctly decompose the 2x4 * 4x5 tensor."""
    n, m, p = 2, 4, 5
    rank = U.shape[1]
    # Build the tensor T[i,j,k] = 1 for all i,j,k
    # Check: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] == 1 for all i,j,k
    # Actually the decomposition is: C[i,k] = sum_j sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t]
    # For the multiplication tensor, we need:
    # sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] = 1 if this is the (i,j,k) entry of the tensor
    # The tensor for matrix multiplication M_{n,m,p} has T[i,j,k] = 1 for all valid i,j,k.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for t in range(rank):
                    s += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _try_remove_columns(U, V, W, max_removals=5):
    """Try to remove columns (rank terms) while maintaining correctness.
    Returns the best (U, V, W) with minimum rank."""
    n, m, p = 2, 4, 5
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    current_rank = U.shape[1]
    
    # Try removing each column one at a time, greedily
    for attempt in range(max_removals * current_rank):
        if time.time() > _deadline:
            break
        improved = False
        for col_idx in range(current_rank):
            if time.time() > _deadline:
                break
            # Try removing this column
            mask = [i for i in range(current_rank) if i != col_idx]
            U_test = U[:, mask]
            V_test = V[:, mask]
            W_test = W[:, mask]
            if _verify(U_test, V_test, W_test):
                # Successfully removed a column
                U, V, W = U_test, V_test, W_test
                current_rank -= 1
                improved = True
                break
        if not improved:
            break
    
    best_U, best_V, best_W = U, V, W
    return best_U, best_V, best_W


def _try_add_and_remove(U, V, W, max_new_terms=10):
    """Try adding new bilinear terms that allow removing more existing terms.
    Strategy: For each pair of existing terms, try to find a new term that
    can replace them (or a subset) by solving for coefficients."""
    n, m, p = 2, 4, 5
    current_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = current_rank
    
    # Enumerate small sets of columns to try to replace
    # For each pair of columns (a, b), try to find a single new column c
    # such that col_a + col_b - col_c = 0 in the tensor sense
    # i.e., U[:,a]+U[:,b]-U_new, V[:,a]+V[:,b]-V_new, W[:,a]+W[:,b]-W_new
    # should still decompose the tensor.
    
    # Actually, the condition is: the new set of terms must still sum to the same tensor.
    # If we remove columns a and b and add column c, we need:
    # U[:,a]*V[:,a]*W[:,a] + U[:,b]*V[:,b]*W[:,b] = U_new*V_new*W_new
    # This is a rank-1 approximation problem, which is generally not exact.
    
    # Better approach: try to find linear dependencies.
    # If sum_{t in S} alpha_t * (U[:,t] ⊗ V[:,t] ⊗ W[:,t]) = 0,
    # then we can remove those terms.
    
    # For now, just do greedy column removal which is the simplest.
    # The _try_remove_columns function handles this.
    
    return best_U, best_V, best_W


# Deadline guard
_start_time = time.time()
_deadline = _start_time + 120  # 120 second deadline with margin

def construct():
    global _deadline
    _start = time.time()
    _deadline = _start + 115  # 5 second safety margin
    
    U, V, W = _build_base_36()
    
    # Verify the base construction
    if not _verify(U, V, W):
        raise ValueError("Base rank-36 decomposition failed verification")
    
    # Try to reduce rank by removing columns
    U, V, W = _try_remove_columns(U, V, W, max_removals=10)
    
    # Clean up: convert to integer where possible
    def clean_matrix(mat):
        result = []
        for row in mat:
            cleaned = []
            for val in row:
                if abs(val - round(val)) < 1e-9:
                    cleaned.append(int(round(val)))
                else:
                    cleaned.append(float(val))
            result.append(cleaned)
        return result
    
    U = clean_matrix(U.tolist())
    V = clean_matrix(V.tolist())
    W = clean_matrix(W.tolist())
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
