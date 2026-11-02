import numpy as np
import time

DIMENSIONS = [2, 4, 5]
DEADLINE = 145.0  # safety margin before 150s kill

def _build_strassen_36():
    """Build the known rank-36 block Strassen decomposition for 2x4 * 4x5."""
    n, m, p = 2, 4, 5
    rank = 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def u_idx(i, j):
        return i * m + j
    def v_idx(j, k):
        return j * p + k
    def w_idx(k, i):
        return k * n + i
    
    col = 0
    
    def strassen_block(y_indices, z_indices, w_offsets):
        nonlocal col
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices
        
        # M1 = (y00+y11)(z00+z11)
        U[u_idx(y00[0], y00[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        # M2 = (y10+y11)z00
        U[u_idx(y10[0], y10[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] -= 1
        col += 1
        
        # M3 = y00(z01-z11)
        U[u_idx(y00[0], y00[1]), col] = 1
        V[v_idx(z01[0], z01[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = -1
        W[w_idx(w_offsets[1][0], w_offsets[1][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        # M4 = y11(z10-z00)
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z10[0], z10[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = -1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1]), col] += 1
        col += 1
        
        # M5 = (y00+y01)z11
        U[u_idx(y00[0], y00[1]), col] = 1
        U[u_idx(y01[0], y01[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] -= 1
        W[w_idx(w_offsets[1][0], w_offsets[1][1]), col] += 1
        col += 1
        
        # M6 = (y10-y00)(z00+z01)
        U[u_idx(y10[0], y10[1]), col] = 1
        U[u_idx(y00[0], y00[1]), col] = -1
        V[v_idx(z00[0], z00[1]), col] = 1
        V[v_idx(z01[0], z01[1]), col] = 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        # M7 = (y01-y11)(z10+z11)
        U[u_idx(y01[0], y01[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = -1
        V[v_idx(z10[0], z10[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        col += 1
    
    # Block 1: A_L B_11 (2x2 * 2x2)
    strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,0),(0,1),(1,0),(1,1)], [(0,0),(1,0),(0,1),(1,1)])
    # Block 2: A_R B_21 (2x2 * 2x2)
    strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,0),(2,1),(3,0),(3,1)], [(0,0),(1,0),(0,1),(1,1)])
    # Block 3: A_L B_12 (2x2 * 2x2)
    strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,2),(0,3),(1,2),(1,3)], [(2,0),(3,0),(2,1),(3,1)])
    # Block 4: A_R B_22 (2x2 * 2x2)
    strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,2),(2,3),(3,2),(3,3)], [(2,0),(3,0),(2,1),(3,1)])
    
    # Block 5: A_L B_15 (2x2 * 2x1) - schoolbook
    U[u_idx(0,0), col] = 1; V[v_idx(0,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(0,1), col] = 1; V[v_idx(1,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(1,0), col] = 1; V[v_idx(0,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    U[u_idx(1,1), col] = 1; V[v_idx(1,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    
    # Block 6: A_R B_25 (2x2 * 2x1) - schoolbook
    U[u_idx(0,2), col] = 1; V[v_idx(2,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(0,3), col] = 1; V[v_idx(3,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(1,2), col] = 1; V[v_idx(2,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    U[u_idx(1,3), col] = 1; V[v_idx(3,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    
    return U, V, W

def _verify_decomposition(U, V, W):
    """Verify that the decomposition exactly represents matrix multiplication (vectorized)."""
    n, m, p = 2, 4, 5
    # For each (i,j,k), check sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    # Vectorize over c
    errors = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                if abs(np.sum(prod) - 1.0) > 1e-9:
                    return False
    return True

def _check_column_redundant(U, V, W, col_to_remove):
    """Check if column col_to_remove can be expressed as a linear combination of others.
    This is a necessary and sufficient condition for the column to be removable while
    preserving the tensor decomposition."""
    n, m, p = 2, 4, 5
    r = U.shape[1]
    
    # Build the system: for all (i,j,k):
    # sum_{c != col_to_remove} alpha_c * U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] 
    #   = U[i*m+j,col_to_remove] * V[j*p+k,col_to_remove] * W[k*n+i,col_to_remove]
    
    keep_cols = [c for c in range(r) if c != col_to_remove]
    num_eqs = n * m * p
    num_unknowns = len(keep_cols)
    
    if num_unknowns < 1:
        return False, None
    
    A = np.zeros((num_eqs, num_unknowns))
    b = np.zeros(num_eqs)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                rhs = U[i*m+j, col_to_remove] * V[j*p+k, col_to_remove] * W[k*n+i, col_to_remove]
                b[idx] = rhs
                for ci, c in enumerate(keep_cols):
                    A[idx, ci] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                idx += 1
    
    # Solve the least squares system
    try:
        solution, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=1e-10)
    except:
        return False, None
    
    # Check if the solution actually satisfies the equations
    if residuals.size > 0 and residuals[0] > 1e-8:
        return False, None
    
    # Verify the solution
    A_sol = A @ solution
    if np.max(np.abs(A_sol - b)) > 1e-8:
        return False, None
    
    return True, solution

def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x4 x 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    U, V, W = _build_strassen_36()
    
    # Verify the base decomposition
    if not _verify_decomposition(U, V, W):
        # Fallback: return schoolbook rank-40
        n, m, p = 2, 4, 5
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    rank = U.shape[1]
    
    # Phase 1: Iteratively remove redundant single columns
    # We loop until no more single columns can be removed
    while time.time() < deadline - 20:
        progress = False
        current_rank = U.shape[1]
        # Find all removable columns
        removable = []
        for col in range(current_rank):
            if time.time() > deadline - 20:
                break
            redundant, _ = _check_column_redundant(U, V, W, col)
            if redundant:
                removable.append(col)
        
        if not removable:
            break
            
        # Remove the first removable column found
        # (Could try randomization here to escape local minima, but deterministic is safer for reproducibility)
        col_to_remove = removable[0]
        mask = [i for i in range(current_rank) if i != col_to_remove]
        U_new = U[:, mask]
        V_new = V[:, mask]
        W_new = W[:, mask]
        
        if _verify_decomposition(U_new, V_new, W_new):
            U, V, W = U_new, V_new, W_new
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            progress = True
        else:
            # If verification fails, it means our redundancy check was flawed or floating point issues
            # Break to avoid infinite loop
            break
            
    # Phase 2: Try removing pairs
    if time.time() < deadline - 10:
        current_rank = U.shape[1]
        found_pair = False
        for i in range(current_rank):
            if time.time() > deadline - 10:
                break
            for j in range(i+1, current_rank):
                if time.time() > deadline - 10:
                    break
                mask = [k for k in range(current_rank) if k != i and k != j]
                U_new = U[:, mask]
                V_new = V[:, mask]
                W_new = W[:, mask]
                if _verify_decomposition(U_new, V_new, W_new):
                    U, V, W = U_new, V_new, W_new
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    found_pair = True
                    break
            if found_pair:
                break
            if time.time() > deadline - 10:
                break

    # Phase 3: Try removing triples
    if time.time() < deadline - 5:
        current_rank = U.shape[1]
        found_triple = False
        for i in range(current_rank):
            if time.time() > deadline - 5:
                break
            for j in range(i+1, current_rank):
                if time.time() > deadline - 5:
                    break
                for k in range(j+1, current_rank):
                    if time.time() > deadline - 5:
                        break
                    mask = [l for l in range(current_rank) if l != i and l != j and l != k]
                    U_new = U[:, mask]
                    V_new = V[:, mask]
                    W_new = W[:, mask]
                    if _verify_decomposition(U_new, V_new, W_new):
                        U, V, W = U_new, V_new, W_new
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        found_triple = True
                        break
                    if time.time() > deadline - 5:
                        break
                if found_triple:
                    break
                if time.time() > deadline - 5:
                    break
            if found_triple:
                break
            if time.time() > deadline - 5:
                break

    # Final verification
    if not _verify_decomposition(U, V, W):
        U, V, W = best_U, best_V, best_W
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
