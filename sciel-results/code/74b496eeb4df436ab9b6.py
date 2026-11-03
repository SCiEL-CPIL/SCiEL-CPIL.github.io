import time
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s limit to be safe (150s wall-clock)
    
    n, m, p = DIMENSIONS
    
    # Helper to add a Strassen 2x2x2 block
    def add_strassen(U, V, W, i0, j0, k0, col):
        # M1
        U[i0 * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M2
        U[(i0 + 1) * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] -= 1
        col += 1
        
        # M3
        U[i0 * m + j0][col] += 1
        V[j0 * p + k0 + 1][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] -= 1
        W[(k0 + 1) * n + i0][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M4
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0][col] += 1
        V[j0 * p + k0][col] -= 1
        W[k0 * n + i0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        col += 1
        
        # M5
        U[i0 * m + j0][col] += 1
        U[i0 * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[(k0 + 1) * n + i0][col] += 1
        W[k0 * n + i0][col] -= 1
        col += 1
        
        # M6
        U[(i0 + 1) * m + j0][col] += 1
        U[i0 * m + j0][col] -= 1
        V[j0 * p + k0][col] += 1
        V[j0 * p + k0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M7
        U[i0 * m + j0 + 1][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] -= 1
        V[(j0 + 1) * p + k0][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        col += 1
        
        return col

    # Helper to add schoolbook terms for a sub-block
    def add_schoolbook(U, V, W, i_range, j_range, k_range, col):
        for i in i_range:
            for j in j_range:
                for k in k_range:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return col

    # Build baseline with Strassen for 2x2x2 blocks and schoolbook for rest
    # i in {0,1} is size 2. i in {2} is size 1.
    # j in {0,1}, {2,3} are size 2. j in {4} is size 1.
    # k in {0,1}, {2,3}, {4,5} are size 2. k in {6} is size 1.
    
    # We can form Strassen blocks for:
    # i in {0,1}, j in {0,1}, k in {0,1}
    # i in {0,1}, j in {0,1}, k in {2,3}
    # i in {0,1}, j in {0,1}, k in {4,5}
    # i in {0,1}, j in {2,3}, k in {0,1}
    # i in {0,1}, j in {2,3}, k in {2,3}
    # i in {0,1}, j in {2,3}, k in {4,5}
    
    # Total 6 Strassen blocks.
    
    rank_baseline = 105
    U_base = [[0] * rank_baseline for _ in range(n * m)]
    V_base = [[0] * rank_baseline for _ in range(m * p)]
    W_base = [[0] * rank_baseline for _ in range(p * n)]
    
    col = 0
    
    # Add Strassen blocks
    for j0 in [0, 2]:
        for k0 in [0, 2, 4]:
            if time.time() > deadline:
                break
            col = add_strassen(U_base, V_base, W_base, 0, j0, k0, col)
    
    # Add schoolbook for the rest
    # i in {0,1}, j in {0,1}, k in {6}
    col = add_schoolbook(U_base, V_base, W_base, [0, 1], [0, 1], [6], col)
    # i in {0,1}, j in {2,3}, k in {6}
    col = add_schoolbook(U_base, V_base, W_base, [0, 1], [2, 3], [6], col)
    # i in {0,1}, j in {4}, k in range(7)
    col = add_schoolbook(U_base, V_base, W_base, [0, 1], [4], range(7), col)
    # i in {2}, j in range(5), k in range(7)
    col = add_schoolbook(U_base, V_base, W_base, [2], range(5), range(7), col)
    
    # col should be 99
    # 6 * 7 + 2 * 2 + 2 * 2 + 2 * 7 + 1 * 5 * 7 = 42 + 4 + 4 + 14 + 35 = 99
    
    def verify(U, V, W, n, m, p):
        """Verify that the decomposition is exact."""
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r_idx in range(r):
                        total += U[i * m + j][r_idx] * V[j * p + k][r_idx] * W[k * n + i][r_idx]
                    if total != 1:
                        return False
        return True
    
    # Check if the baseline is valid
    if not verify(U_base, V_base, W_base, n, m, p):
        # Fallback to schoolbook
        U_school = [[0] * 105 for _ in range(n * m)]
        V_school = [[0] * 105 for _ in range(m * p)]
        W_school = [[0] * 105 for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j][col] = 1
                    V_school[j * p + k][col] = 1
                    W_school[k * n + i][col] = 1
                    col += 1
        return {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to reduce rank via Gaussian elimination
    def find_redundant_columns(U, V, W, n, m, p):
        r = len(U[0])
        num_entries = n * m * p
        
        # Build the constraint matrix: each row is a (i,j,k) entry, each col is a rank term
        # M[idx, r_idx] = U[i*m+j][r_idx] * V[j*p+k][r_idx] * W[k*n+i][r_idx]
        # We want to find which columns are linearly dependent
        
        # Use Gaussian elimination with partial pivoting
        # We'll work with a copy of the matrix
        A = [[0] * r for _ in range(num_entries)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r_idx in range(r):
                        A[idx][r_idx] = U[i * m + j][r_idx] * V[j * p + k][r_idx] * W[k * n + i][r_idx]
                    idx += 1
        
        pivot_cols = []
        current_row = 0
        
        for col_idx in range(r):
            if time.time() > deadline:
                break
            if current_row >= num_entries:
                continue
            
            # Find pivot
            pivot_row = -1
            for row_idx in range(current_row, num_entries):
                if A[row_idx][col_idx] != 0:
                    pivot_row = row_idx
                    break
            
            if pivot_row == -1:
                continue
            
            # Swap rows
            A[current_row], A[pivot_row] = A[pivot_row], A[current_row]
            
            # Normalize pivot row
            pivot_val = A[current_row][col_idx]
            for c in range(col_idx, r):
                A[current_row][c] /= pivot_val
            
            # Eliminate below
            for row_idx in range(current_row + 1, num_entries):
                if A[row_idx][col_idx] != 0:
                    factor = A[row_idx][col_idx]
                    for c in range(col_idx, r):
                        A[row_idx][c] -= factor * A[current_row][c]
            
            pivot_cols.append(col_idx)
            current_row += 1
        
        # Columns not in pivot_cols are redundant
        pivot_set = set(pivot_cols)
        redundant = set(range(r)) - pivot_set
        return redundant
    
    if time.time() < deadline - 20:  # Need at least 20s for verification
        redundant_cols = find_redundant_columns(U_base, V_base, W_base, n, m, p)
        
        if redundant_cols:
            keep_cols = [c for c in range(len(U_base[0])) if c not in redundant_cols]
            new_rank = len(keep_cols)
            
            if new_rank < len(U_base[0]):
                U_new = [[U_base[i][c] for c in keep_cols] for i in range(n * m)]
                V_new = [[V_base[i][c] for c in keep_cols] for i in range(m * p)]
                W_new = [[W_base[i][c] for c in keep_cols] for i in range(p * n)]
                
                if verify(U_new, V_new, W_new, n, m, p):
                    return {"U": U_new, "V": V_new, "W": W_new}
    
    return {"U": U_base, "V": V_base, "W": W_base}
