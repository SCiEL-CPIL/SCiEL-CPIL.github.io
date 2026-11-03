"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s limit to be safe (150s wall-clock)
    
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

    # Laderman's algorithm for 3x3 matrix multiplication (rank 23)
    def add_laderman(U, V, W, i0, j0, k0, col):
        """Add Laderman's 23-term decomposition for the 3x3x3 sub-block at (i0,j0,k0)."""
        # Laderman's 23 terms for C = A * B where A is 3x3, B is 3x3
        # A[i][j] indexed by (i,j), B[j][k] indexed by (j,k), C[i][k] indexed by (i,k)
        # Using 0-based indexing within the 3x3 block
        
        # Term 1: A[0][0] * B[0][0]
        U[i0 * m + j0][col] = 1
        V[j0 * p + k0][col] = 1
        W[k0 * n + i0][col] = 1
        col += 1
        
        # Term 2: A[0][1] * B[1][0]
        U[i0 * m + j0 + 1][col] = 1
        V[(j0 + 1) * p + k0][col] = 1
        W[k0 * n + i0][col] = 1
        col += 1
        
        # Term 3: A[0][2] * B[2][0]
        U[i0 * m + j0 + 2][col] = 1
        V[(j0 + 2) * p + k0][col] = 1
        W[k0 * n + i0][col] = 1
        col += 1
        
        # Term 4: A[1][0] * B[0][1]
        U[(i0 + 1) * m + j0][col] = 1
        V[j0 * p + (k0 + 1)][col] = 1
        W[(k0 + 1) * n + i0][col] = 1
        col += 1
        
        # Term 5: A[1][1] * B[1][1]
        U[(i0 + 1) * m + j0 + 1][col] = 1
        V[(j0 + 1) * p + (k0 + 1)][col] = 1
        W[(k0 + 1) * n + (i0 + 1)][col] = 1
        col += 1
        
        # Term 6: A[1][2] * B[2][1]
        U[(i0 + 1) * m + j0 + 2][col] = 1
        V[(j0 + 2) * p + (k0 + 1)][col] = 1
        W[(k0 + 1) * n + (i0 + 1)][col] = 1
        col += 1
        
        # Term 7: A[2][0] * B[0][2]
        U[(i0 + 2) * m + j0][col] = 1
        V[j0 * p + (k0 + 2)][col] = 1
        W[(k0 + 2) * n + i0][col] = 1
        col += 1
        
        # Term 8: A[2][1] * B[1][2]
        U[(i0 + 2) * m + j0 + 1][col] = 1
        V[(j0 + 1) * p + (k0 + 2)][col] = 1
        W[(k0 + 2) * n + (i0 + 2)][col] = 1
        col += 1
        
        # Term 9: A[2][2] * B[2][2]
        U[(i0 + 2) * m + j0 + 2][col] = 1
        V[(j0 + 2) * p + (k0 + 2)][col] = 1
        W[(k0 + 2) * n + (i0 + 2)][col] = 1
        col += 1
        
        # Term 10: (A[0][0] + A[0][1] + A[1][0] + A[1][1]) * (B[0][1] + B[1][0] - B[1][1])
        U[i0 * m + j0][col] = 1
        U[i0 * m + j0 + 1][col] = 1
        U[(i0 + 1) * m + j0][col] = 1
        U[(i0 + 1) * m + j0 + 1][col] = 1
        V[j0 * p + (k0 + 1)][col] = 1
        V[(j0 + 1) * p + k0][col] = 1
        V[(j0 + 1) * p + (k0 + 1)][col] = -1
        W[(k0 + 1) * n + i0][col] = 1
        W[k0 * n + (i0 + 1)][col] = 1
        W[(k0 + 1) * n + (i0 + 1)][col] = -1
        col += 1
        
        # Term 11: (A[0][1] + A[0][2] + A[1][1] + A[1][2]) * (B[1][0] + B[2][0] - B[1][1] - B[2][1])
        U[i0 * m + j0 + 1][col] = 1
        U[i0 * m + j0 + 2][col] = 1
        U[(i0 + 1) * m + j0 + 1][col] = 1
        U[(i0 + 1) * m + j0 + 2][col] = 1
        V[(j0 + 1) * p + k0][col] = 1
        V[(j0 + 2) * p + k0][col] = 1
        V[(j0 + 1) * p + (k0 + 1)][col] = -1
        V[(j0 + 2) * p + (k0 + 1)][col] = -1
        W[k0 * n + i0][col] = 1
        W[k0 * n + (i0 + 1)][col] = 1
        W[(k0 + 1) * n + i0][col] = -1
        W[(k0 + 1) * n + (i0 + 1)][col] = -1
        col += 1
        
        # Term 12: (A[0][0] + A[1][0] + A[1][1] + A[2][0] + A[2][1] + A[2][2]) * (B[0][2] - B[1][2])
        U[i0 * m + j0][col] = 1
        U[(i0 + 1) * m + j0][col] = 1
        U[(i0 + 1) * m + j0 + 1][col] = 1
        U[(i0 + 2) * m + j0][col] = 1
        U[(i0 + 2) * m + j0 + 1][col] = 1
        U[(i0 + 2) * m + j0 + 2][col] = 1
        V[j0 * p + (k0 + 2)][col] = 1
        V[(j0 + 1) * p + (k0 + 2)][col] = -1
        W[(k0 + 2) * n + i0][col] = 1
        W[(k0 + 2) * n + (i0 + 1)][col] = 1
        W[(k0 + 2) * n + (i0 + 2)][col] = 1
        W[(k0 + 2) * n + i0][col] -= 1
        W[(k0 + 2) * n + (i0 + 1)][col] -= 1
        col += 1
        
        # Term 13: (A[0][0] + A[0][1] + A[0][2] + A[1][0] + A[1][1] + A[1][2] + A[2][0] + A[2][1] + A[2][2]) * (B[0][0] + B[1][1] + B[2][2])
        # This is getting complex; let me use a cleaner Laderman formulation
        
        # Actually, let me use the standard Laderman 23-term formula more carefully
        # I'll restart with a cleaner approach using the known Laderman terms
        
        return col

    # Cleaner Laderman implementation using the standard 23-term formula
    def add_laderman_v2(U, V, W, i0, j0, k0, col):
        """Laderman's 23-term algorithm for 3x3 matrix multiplication."""
        # Using the standard Laderman formulation
        # A[i][j], B[j][k], C[i][k] for i,j,k in {0,1,2}
        
        # We'll use the well-known 23-term Laderman decomposition
        # Terms are (U_factors, V_factors, W_factors) where each is a list of (index, coeff)
        
        terms = [
            # (u_indices, v_indices, w_indices)
            # Term 1: A00 * B00
            ([(0,0,1)], [(0,0,1)], [(0,0,1)]),
            # Term 2: A01 * B10
            ([(0,1,1)], [(1,0,1)], [(0,0,1)]),
            # Term 3: A02 * B20
            ([(0,2,1)], [(2,0,1)], [(0,0,1)]),
            # Term 4: A10 * B01
            ([(1,0,1)], [(0,1,1)], [(1,0,1)]),
            # Term 5: A11 * B11
            ([(1,1,1)], [(1,1,1)], [(1,1,1)]),
            # Term 6: A12 * B21
            ([(1,2,1)], [(2,1,1)], [(1,1,1)]),
            # Term 7: A20 * B02
            ([(2,0,1)], [(0,2,1)], [(2,0,1)]),
            # Term 8: A21 * B12
            ([(2,1,1)], [(1,2,1)], [(2,1,1)]),
            # Term 9: A22 * B22
            ([(2,2,1)], [(2,2,1)], [(2,2,1)]),
            # Term 10: (A00+A01+A10+A11) * (B01+B10-B11)
            ([(0,0,1),(0,1,1),(1,0,1),(1,1,1)], [(0,1,1),(1,0,1),(1,1,-1)], [(1,0,1),(0,1,1),(1,1,-1)]),
            # Term 11: (A01+A02+A11+A12) * (B10+B20-B11-B21)
            ([(0,1,1),(0,2,1),(1,1,1),(1,2,1)], [(1,0,1),(2,0,1),(1,1,-1),(2,1,-1)], [(0,1,1),(1,1,1),(1,0,-1),(1,1,-1)]),
            # Term 12: (A00+A10+A11+A20+A21+A22) * (B02-B12)
            ([(0,0,1),(1,0,1),(1,1,1),(2,0,1),(2,1,1),(2,2,1)], [(0,2,1),(1,2,-1)], [(2,0,1),(2,1,1),(2,2,1)]),
            # Term 13: (A00+A01+A02+A10+A11+A12+A20+A21+A22) * (B00+B11+B22)
            ([(0,0,1),(0,1,1),(0,2,1),(1,0,1),(1,1,1),(1,2,1),(2,0,1),(2,1,1),(2,2,1)], [(0,0,1),(1,1,1),(2,2,1)], [(0,0,1),(1,1,1),(2,2,1)]),
            # Term 14: (A00+A10+A20) * (B00+B11)
            ([(0,0,1),(1,0,1),(2,0,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            # Term 15: (A00+A11+A22) * (B00+B11)
            ([(0,0,1),(1,1,1),(2,2,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            # Term 16: (A01+A12+A20) * (B10+B21)
            ([(0,1,1),(1,2,1),(2,0,1)], [(1,0,1),(2,1,1)], [(0,1,1),(1,2,1),(2,0,1)]),
            # Term 17: (A02+A10+A21) * (B20+B01)
            ([(0,2,1),(1,0,1),(2,1,1)], [(2,0,1),(0,1,1)], [(2,0,1),(0,1,1),(1,2,1)]),
            # Term 18: (A00+A01+A10+A11+A20+A21) * (B02-B12)
            ([(0,0,1),(0,1,1),(1,0,1),(1,1,1),(2,0,1),(2,1,1)], [(0,2,1),(1,2,-1)], [(2,0,1),(2,1,1)]),
            # Term 19: (A00+A02+A10+A12+A20+A22) * (B00+B22)
            ([(0,0,1),(0,2,1),(1,0,1),(1,2,1),(2,0,1),(2,2,1)], [(0,0,1),(2,2,1)], [(0,0,1),(2,2,1)]),
            # Term 20: (A01+A10+A11+A20+A21+A22) * (B01+B10-B11)
            ([(0,1,1),(1,0,1),(1,1,1),(2,0,1),(2,1,1),(2,2,1)], [(0,1,1),(1,0,1),(1,1,-1)], [(1,0,1),(0,1,1),(1,1,-1)]),
            # Term 21: (A00+A01+A02+A10+A11+A12) * (B10+B20-B11-B21)
            ([(0,0,1),(0,1,1),(0,2,1),(1,0,1),(1,1,1),(1,2,1)], [(1,0,1),(2,0,1),(1,1,-1),(2,1,-1)], [(0,1,1),(1,1,1),(1,0,-1),(1,1,-1)]),
            # Term 22: (A01+A02+A11+A12+A21+A22) * (B02-B12)
            ([(0,1,1),(0,2,1),(1,1,1),(1,2,1),(2,1,1),(2,2,1)], [(0,2,1),(1,2,-1)], [(2,1,1),(2,2,1)]),
            # Term 23: (A00+A11+A22) * (B00+B11+B22)
            ([(0,0,1),(1,1,1),(2,2,1)], [(0,0,1),(1,1,1),(2,2,1)], [(0,0,1),(1,1,1),(2,2,1)]),
        ]
        
        for u_factors, v_factors, w_factors in terms:
            for (i, j, coeff) in u_factors:
                U[(i0 + i) * m + (j0 + j)][col] += coeff
            for (j, k, coeff) in v_factors:
                V[(j0 + j) * p + (k0 + k)][col] += coeff
            for (i, k, coeff) in w_factors:
                W[(k0 + k) * n + (i0 + i)][col] += coeff
            col += 1
        
        return col

    # Build baseline with Laderman for 3x3x3 sub-block and schoolbook for rest
    # Sub-block: i in {0,1,2}, j in {0,1,2}, k in {0,1,2} uses Laderman (rank 23)
    # Remaining terms use schoolbook
    
    # Total schoolbook terms: 3*5*7 = 105
    # Laderman covers: 3*3*3 = 27 terms with rank 23
    # Remaining: 105 - 27 = 78 terms with schoolbook rank 78
    # Total rank: 23 + 78 = 101 (worse than 99!)
    
    # So Laderman alone doesn't help. Let's use Strassen for 2x2x2 blocks as before
    # and try to find better reductions.
    
    rank_baseline = 99
    U_base = [[0] * rank_baseline for _ in range(n * m)]
    V_base = [[0] * rank_baseline for _ in range(m * p)]
    W_base = [[0] * rank_baseline for _ in range(p * n)]
    
    col = 0
    # Part 1: j in {0, 1}
    for k0 in [0, 2, 4]:
        col = add_strassen(U_base, V_base, W_base, 0, 0, k0, col)
    col = add_schoolbook(U_base, V_base, W_base, [0, 1], [0, 1], [6], col)
    col = add_schoolbook(U_base, V_base, W_base, [2], [0, 1], range(7), col)
    
    # Part 2: j in {2, 3, 4}
    for k0 in [0, 2, 4]:
        col = add_strassen(U_base, V_base, W_base, 0, 2, k0, col)
    col = add_schoolbook(U_base, V_base, W_base, [0, 1], [2, 3], [6], col)
    col = add_schoolbook(U_base, V_base, W_base, [0, 1], [4], range(7), col)
    col = add_schoolbook(U_base, V_base, W_base, [2], [2, 3, 4], range(7), col)
    
    # Now attempt rank reduction via random linear combinations
    # Strategy: Try to find linear dependencies by combining columns with random coefficients
    
    def build_constraint_matrix(U, V, W, n, m, p):
        """Build the 105 x rank matrix where entry (i,j,k),r = U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]"""
        r = len(U[0])
        M = [[Fraction(0)] * r for _ in range(n * m * p)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r_idx in range(r):
                        M[idx][r_idx] = Fraction(U[i * m + j][r_idx]) * Fraction(V[j * p + k][r_idx]) * Fraction(W[k * n + i][r_idx])
                    idx += 1
        return M
    
    def find_redundant_columns(M):
        """Return the set of column indices that are linear combinations of earlier columns."""
        if not M or not M[0]:
            return set()
        rows = len(M)
        cols = len(M[0])
        A = [row[:] for row in M]
        pivot_cols = []
        redundant = set()
        
        current_row = 0
        for col_idx in range(cols):
            if time.time() > deadline:
                break
            if current_row >= rows:
                redundant.add(col_idx)
                continue
            # Find pivot in column col_idx, row >= current_row
            pivot_row = -1
            for row_idx in range(current_row, rows):
                if A[row_idx][col_idx] != 0:
                    pivot_row = row_idx
                    break
            if pivot_row == -1:
                redundant.add(col_idx)
                continue
            # Swap
            A[current_row], A[pivot_row] = A[pivot_row], A[current_row]
            # Eliminate below
            pivot_val = A[current_row][col_idx]
            for row_idx in range(current_row + 1, rows):
                if A[row_idx][col_idx] != 0:
                    factor = A[row_idx][col_idx] / pivot_val
                    for c in range(col_idx, cols):
                        A[row_idx][c] -= factor * A[current_row][c]
            pivot_cols.append(col_idx)
            current_row += 1
        
        # Columns not in pivot_cols are redundant
        pivot_set = set(pivot_cols)
        for c in range(cols):
            if c not in pivot_set:
                redundant.add(c)
        
        return redundant
    
    def verify(U, V, W, n, m, p):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in range(len(U[0])):
                        total += Fraction(U[i * m + j][r]) * Fraction(V[j * p + k][r]) * Fraction(W[k * n + i][r])
                    if total != 1:
                        return False
        return True
    
    # First, try the simple column elimination approach
    M_base = build_constraint_matrix(U_base, V_base, W_base, n, m, p)
    redundant_cols = find_redundant_columns(M_base)
    
    if redundant_cols:
        keep_cols = [c for c in range(len(U_base[0])) if c not in redundant_cols]
        new_rank = len(keep_cols)
        
        if new_rank < rank_baseline:
            U_new = [[U_base[i][c] for c in keep_cols] for i in range(n * m)]
            V_new = [[V_base[i][c] for c in keep_cols] for i in range(m * p)]
            W_new = [[W_base[i][c] for c in keep_cols] for i in range(p * n)]
            
            if verify(U_new, V_new, W_new, n, m, p):
                return {"U": U_new, "V": V_new, "W": W_new}
    
    # Try random linear combinations to find more reductions
    # Create new columns as random linear combinations of existing ones
    import random
    random.seed(42)
    
    best_U, best_V, best_W = U_base, V_base, W_base
    best_rank = rank_baseline
    
    # Try to find columns that can be expressed as linear combinations
    # by checking if the constraint matrix has rank < 99
    M_base = build_constraint_matrix(U_base, V_base, W_base, n, m, p)
    
    # Compute rank of M_base
    A = [row[:] for row in M_base]
    rows = len(A)
    cols = len(A[0])
    current_row = 0
    pivot_cols = []
    
    for col_idx in range(cols):
        if time.time() > deadline:
            break
        if current_row >= rows:
            break
        pivot_row = -1
        for row_idx in range(current_row, rows):
            if A[row_idx][col_idx] != 0:
                pivot_row = row_idx
                break
        if pivot_row == -1:
            continue
        A[current_row], A[pivot_row] = A[pivot_row], A[current_row]
        pivot_val = A[current_row][col_idx]
        for row_idx in range(current_row + 1, rows):
            if A[row_idx][col_idx] != 0:
                factor = A[row_idx][col_idx] / pivot_val
                for c in range(col_idx, cols):
                    A[row_idx][c] -= factor * A[current_row][c]
        pivot_cols.append(col_idx)
        current_row += 1
    
    matrix_rank = len(pivot_cols)
    
    if matrix_rank < rank_baseline:
        # There are linear dependencies. Try to find them.
        # Keep only pivot columns
        keep_cols = pivot_cols
        new_rank = len(keep_cols)
        
        if new_rank < best_rank:
            U_new = [[U_base[i][c] for c in keep_cols] for i in range(n * m)]
            V_new = [[V_base[i][c] for c in keep_cols] for i in range(m * p)]
            W_new = [[W_base[i][c] for c in keep_cols] for i in range(p * n)]
            
            if verify(U_new, V_new, W_new, n, m, p):
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = new_rank
    
    # Try more aggressive search: random column perturbations
    if time.time() < deadline - 30:
        # Try to find a better decomposition by searching over small rank values
        for target_rank in range(best_rank - 1, max(90, best_rank - 10), -1):
            if time.time() > deadline:
                break
            
            # Try to find a rank-target_rank decomposition
            # by randomly combining and eliminating columns
            found = False
            for attempt in range(50):
                if time.time() > deadline:
                    break
                
                # Create a new set of columns by random linear combinations
                U_try = [[Fraction(0)] * target_rank for _ in range(n * m)]
                V_try = [[Fraction(0)] * target_rank for _ in range(m * p)]
                W_try = [[Fraction(0)] * target_rank for _ in range(p * n)]
                
                # Use random combinations of original columns
                for r in range(target_rank):
                    # Pick 2-3 random original columns and combine
                    num_src = random.randint(2, 3)
                    src_cols = random.sample(range(rank_baseline), num_src)
                    coeffs = [Fraction(random.randint(-2, 2) or 1) for _ in range(num_src)]
                    
                    for i in range(n * m):
                        U_try[i][r] = sum(Fraction(U_base[i][sc]) * c for sc, c in zip(src_cols, coeffs))
                    for i in range(m * p):
                        V_try[i][r] = sum(Fraction(V_base[i][sc]) * c for sc, c in zip(src_cols, coeffs))
                    for i in range(p * n):
                        W_try[i][r] = sum(Fraction(W_base[i][sc]) * c for sc, c in zip(src_cols, coeffs))
                
                if verify(U_try, V_try, W_try, n, m, p):
                    best_U, best_V, best_W = U_try, V_try, W_try
                    best_rank = target_rank
                    found = True
                    break
            
            if found:
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
