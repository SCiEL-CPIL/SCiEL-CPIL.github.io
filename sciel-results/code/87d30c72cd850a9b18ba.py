import time
import sys
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Build the rank-50 decomposition
    rank_init = 50
    U = np.zeros((n * m, rank_init), dtype=object)
    V = np.zeros((m * p, rank_init), dtype=object)
    W = np.zeros((p * n, rank_init), dtype=object)
    
    col = 0
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1: (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
        U[i0*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # M2: (A21+A22)B11 -> C21 += M2, C22 -= M2
        U[(i0+1)*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        W[k0*n + (i0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] -= 1
        col += 1
        
        # M3: A11(B12-B22) -> C12 += M3, C22 += M3
        U[i0*m + j0, col] += 1
        V[j0*p + (k0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] -= 1
        W[(k0+1)*n + i0, col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # M4: A22(B21-B11) -> C11 += M4, C21 += M4
        U[(i0+1)*m + (j0+1), col] += 1
        V[(j0+1)*p + k0, col] += 1
        V[j0*p + k0, col] -= 1
        W[k0*n + i0, col] += 1
        W[k0*n + (i0+1), col] += 1
        col += 1
        
        # M5: (A11+A12)B22 -> C11 -= M5, C12 += M5
        U[i0*m + j0, col] += 1
        U[i0*m + (j0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] -= 1
        W[(k0+1)*n + i0, col] += 1
        col += 1
        
        # M6: (A21-A11)(B11+B12) -> C22 += M6
        U[(i0+1)*m + j0, col] += 1
        U[i0*m + j0, col] -= 1
        V[j0*p + k0, col] += 1
        V[j0*p + (k0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # M7: (A12-A22)(B21+B22) -> C11 += M7
        U[i0*m + (j0+1), col] += 1
        U[(i0+1)*m + (j0+1), col] -= 1
        V[(j0+1)*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        col += 1

    # 6 Strassen blocks for the 2x4x6 part (k=0..5)
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    add_strassen(0, 0, 4)
    add_strassen(0, 2, 4)
    
    # 8 rank-1 terms for k=6
    for i in range(2):
        for j in range(4):
            U[i*m + j, col] = 1
            V[j*p + 6, col] = 1
            W[6*n + i, col] = 1
            col += 1
    
    assert col == rank_init, f"Expected rank {rank_init}, got {col}"
    
    def verify_decomposition(U_m, V_m, W_m, n, m, p):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1
                    actual = sum(U_m[i*m+j, c] * V_m[j*p+k, c] * W_m[k*n+i, c] for c in range(W_m.shape[1]))
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    # Verify initial decomposition
    Uf = U.astype(float)
    Vf = V.astype(float)
    Wf = W.astype(float)
    
    if not verify_decomposition(Uf, Vf, Wf, n, m, p):
        # Return the rank-50 solution as fallback
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Now try to reduce rank
    # Strategy: find columns c where W[:,c] is in the span of other columns,
    # and the corresponding U[:,c], V[:,c] can be absorbed
    
    best_U, best_V, best_W = Uf.copy(), Vf.copy(), Wf.copy()
    best_rank = rank_init
    
    def try_eliminate_column(U_arr, V_arr, W_arr, target_col, other_cols):
        """Try to eliminate target_col by expressing it in terms of other_cols."""
        n_rows = U_arr.shape[0]
        m_rows = V_arr.shape[0]
        p_rows = W_arr.shape[0]
        rank = U_arr.shape[1]
        
        # For each (i,j,k) where target has non-zero contribution,
        # we need the sum over other_cols to match
        # This is a linear system
        
        # Collect constraints: for each (i,j,k), 
        # U[i*m+j, target] * V[j*p+k, target] * W[k*n+i, target] 
        # must be matched by sum over other_cols
        
        # Build linear system
        constraints = []
        rhs = []
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target_val = U_arr[i*m+j, target_col] * V_arr[j*p+k, target_col] * W_arr[k*n+i, target_col]
                    if abs(target_val) > 1e-9:
                        # Need sum over other_cols of U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = target_val
                        # This is nonlinear in the coefficients... 
                        # Actually, we can't just scale columns independently.
                        # The rank-1 term structure means we need to find 
                        # coefficients alpha_c such that:
                        # For all (i,j,k): sum_c alpha_c * U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] 
                        #   = U[i*m+j,target] * V[j*p+k,target] * W[k*n+i,target]
                        # But this is still nonlinear because U,V,W entries are fixed.
                        # 
                        # Alternative: we can try to find a linear combination of the 
                        # other rank-1 terms that equals the target term.
                        # This requires: for all (i,j,k),
                        # sum_c alpha_c * U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] 
                        #   = U[i*m+j,target] * V[j*p+k,target] * W[k*n+i,target]
                        #
                        # This is a system of equations in alpha_c.
                        # But it's nonlinear because each term is a product of three entries.
                        #
                        # However, if we fix which columns participate, we can set up
                        # a linear system where the unknowns are the alpha_c values.
                        # The "coefficients" for each alpha_c in the linear system are 
                        # U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] for each constraint (i,j,k).
                        
                        row = []
                        for c in other_cols:
                            row.append(U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c])
                        constraints.append(row)
                        rhs.append(target_val)
        
        if not constraints:
            return None
        
        A_mat = np.array(constraints, dtype=float)
        b_vec = np.array(rhs, dtype=float)
        
        # Solve least squares
        try:
            alpha, residuals, rank_ls, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        except:
            return None
        
        # Check if solution is good
        if len(residuals) > 0 and residuals[0] > 1e-6:
            return None
        
        # Check if alpha is "simple" (small integer/half-integer values)
        # Round to nearest 0.5 and check
        alpha_rounded = np.round(alpha * 2) / 2
        
        # Verify the rounded solution
        max_err = 0
        for idx, (i, j, k) in enumerate([(i,j,k) for i in range(n) for j in range(m) for k in range(p)]):
            if abs(U_arr[i*m+j, target_col] * V_arr[j*p+k, target_col] * W_arr[k*n+i, target_col]) > 1e-9:
                pred = sum(alpha_rounded[c] * U_arr[i*m+j, other_cols[c]] * V_arr[j*p+k, other_cols[c]] * W_arr[k*n+i, other_cols[c]] for c in range(len(other_cols)))
                actual = U_arr[i*m+j, target_col] * V_arr[j*p+k, target_col] * W_arr[k*n+i, target_col]
                max_err = max(max_err, abs(pred - actual))
        
        if max_err > 1e-6:
            return None
        
        return alpha_rounded, other_cols
    
    # Try to eliminate each column
    # Start with columns that have the smallest support (fewest non-zero entries)
    col_supports = []
    for c in range(rank_init):
        support = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(Uf[i*m+j, c] * Vf[j*p+k, c] * Wf[k*n+i, c]) > 1e-9:
                        support += 1
                        break
        col_supports.append(support)
    
    # Sort columns by support (try to eliminate low-support columns first)
    sorted_cols = sorted(range(rank_init), key=lambda c: col_supports[c])
    
    # Try to eliminate columns one by one
    U_work = Uf.copy()
    V_work = Vf.copy()
    W_work = Wf.copy()
    active_cols = list(range(rank_init))
    
    for target in sorted_cols:
        if time.time() > deadline - 10:
            break
        if target not in active_cols:
            continue
        
        other_cols = [c for c in active_cols if c != target]
        
        # Try subsets of other_cols (start with small subsets)
        eliminated = False
        for subset_size in [1, 2, 3, 4, 5]:
            if time.time() > deadline - 10:
                break
            for subset in combinations(other_cols, subset_size):
                result = try_eliminate_column(U_work, V_work, W_work, target, list(subset))
                if result is not None:
                    alpha, cols_used = result
                    # Remove the target column
                    new_active = [c for c in active_cols if c != target]
                    active_cols = new_active
                    eliminated = True
                    break
            if eliminated:
                break
        
        if not eliminated:
            # Try with all other active columns
            result = try_eliminate_column(U_work, V_work, W_work, target, other_cols)
            if result is not None:
                active_cols = [c for c in active_cols if c != target]
    
    # Extract the reduced decomposition
    if len(active_cols) < best_rank:
        U_reduced = Uf[:, active_cols]
        V_reduced = Vf[:, active_cols]
        W_reduced = Wf[:, active_cols]
        
        # Verify
        if verify_decomposition(U_reduced, V_reduced, W_reduced, n, m, p):
            best_U = U_reduced
            best_V = V_reduced
            best_W = W_reduced
            best_rank = len(active_cols)
    
    # Convert to lists with integer/half-integer entries
    def to_list(arr):
        result = []
        for row in arr:
            r = []
            for val in row:
                # Round to nearest 0.5
                rounded = round(val * 2) / 2
                if abs(rounded - int(rounded)) < 1e-9:
                    r.append(int(rounded))
                else:
                    r.append(rounded)
            result.append(r)
        return result
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}

if __name__ == "__main__":
    construct()
