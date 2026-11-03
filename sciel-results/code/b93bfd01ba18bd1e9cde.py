"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s limit to be safe (150s wall-clock)
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    nm = n * m  # 15
    mp = m * p  # 35
    pn = p * n  # 21
    
    def verify_solution(U, V, W, n, m, p):
        """Verify that the tensor decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in range(len(U[0])):
                        total += Fraction(U[i * m + j][r]) * Fraction(V[j * p + k][r]) * Fraction(W[k * n + i][r])
                    if total != 1:
                        return False
        return True
    
    # Build baseline: pure schoolbook rank = n*m*p = 105
    def build_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][idx] = 1
                    V[j * p + k][idx] = 1
                    W[k * n + i][idx] = 1
                    idx += 1
        return U, V, W
    
    # Build Strassen+Schoolbook baseline (rank 99)
    def build_strassen_schoolbook():
        def add_strassen(U, V, W, i0, j0, k0, col):
            # M1: (A00+A11)(B00+B11)
            U[i0 * m + j0][col] += 1
            U[(i0 + 1) * m + j0 + 1][col] += 1
            V[j0 * p + k0][col] += 1
            V[(j0 + 1) * p + k0 + 1][col] += 1
            W[k0 * n + i0][col] += 1
            W[(k0 + 1) * n + i0 + 1][col] += 1
            col += 1
            
            # M2: A11(B10-B00)
            U[(i0 + 1) * m + j0][col] += 1
            U[(i0 + 1) * m + j0 + 1][col] += 1
            V[j0 * p + k0][col] += 1
            W[k0 * n + i0 + 1][col] += 1
            W[(k0 + 1) * n + i0 + 1][col] -= 1
            col += 1
            
            # M3: (A00-A11)(B11+B22)
            U[i0 * m + j0][col] += 1
            V[j0 * p + k0 + 1][col] += 1
            V[(j0 + 1) * p + k0 + 1][col] -= 1
            W[(k0 + 1) * n + i0][col] += 1
            W[(k0 + 1) * n + i0 + 1][col] += 1
            col += 1
            
            # M4: A22(B22-B00) -- adjusted for 2x2 block
            U[(i0 + 1) * m + j0 + 1][col] += 1
            V[(j0 + 1) * p + k0][col] += 1
            V[j0 * p + k0][col] -= 1
            W[k0 * n + i0][col] += 1
            W[k0 * n + i0 + 1][col] += 1
            col += 1
            
            # M5: (A00+A10)(B11-B21)
            U[i0 * m + j0][col] += 1
            U[i0 * m + j0 + 1][col] += 1
            V[(j0 + 1) * p + k0 + 1][col] += 1
            W[(k0 + 1) * n + i0][col] += 1
            W[k0 * n + i0][col] -= 1
            col += 1
            
            # M6: (A10+A11)(B01+B11)
            U[(i0 + 1) * m + j0][col] += 1
            U[i0 * m + j0][col] -= 1
            V[j0 * p + k0][col] += 1
            V[j0 * p + k0 + 1][col] += 1
            W[(k0 + 1) * n + i0 + 1][col] += 1
            col += 1
            
            # M7: (A00+A01)(B20-B10)
            U[i0 * m + j0 + 1][col] += 1
            U[(i0 + 1) * m + j0 + 1][col] -= 1
            V[(j0 + 1) * p + k0][col] += 1
            V[(j0 + 1) * p + k0 + 1][col] += 1
            W[k0 * n + i0][col] += 1
            col += 1
            
            return col
        
        def add_schoolbook_block(U, V, W, i_range, j_range, k_range, col):
            for i in i_range:
                for j in j_range:
                    for k in k_range:
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
            return col
        
        rank_baseline = 99
        U_base = [[0] * rank_baseline for _ in range(nm)]
        V_base = [[0] * rank_baseline for _ in range(mp)]
        W_base = [[0] * rank_baseline for _ in range(pn)]
        
        col = 0
        # Part 1: j in {0, 1}, use Strassen for k-pairs (0,1), (2,3), (4,5), schoolbook for k=6
        for k0 in [0, 2, 4]:
            col = add_strassen(U_base, V_base, W_base, 0, 0, k0, col)
        col = add_schoolbook_block(U_base, V_base, W_base, [0, 1], [0, 1], [6], col)
        col = add_schoolbook_block(U_base, V_base, W_base, [2], [0, 1], range(7), col)
        
        # Part 2: j in {2, 3, 4}, use Strassen for k-pairs (0,1), (2,3), (4,5)
        for k0 in [0, 2, 4]:
            col = add_strassen(U_base, V_base, W_base, 0, 2, k0, col)
        col = add_schoolbook_block(U_base, V_base, W_base, [0, 1], [2, 3], [6], col)
        col = add_schoolbook_block(U_base, V_base, W_base, [0, 1], [4], range(7), col)
        col = add_schoolbook_block(U_base, V_base, W_base, [2], [2, 3, 4], range(7), col)
        
        return U_base, V_base, W_base
    
    # Build the full tensor representation (105 x rank matrix)
    def build_tensor_matrix(U, V, W, n, m, p):
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
    
    # Find redundant columns via Gaussian elimination
    def find_redundant_columns(M, deadline):
        if not M or not M[0]:
            return set()
        rows = len(M)
        cols = len(M[0])
        A = [row[:] for row in M]
        pivot_cols = []
        
        current_row = 0
        for col_idx in range(cols):
            if time.time() > deadline:
                break
            if current_row >= rows:
                continue
            # Find pivot
            pivot_row = -1
            for row_idx in range(current_row, rows):
                if A[row_idx][col_idx] != 0:
                    pivot_row = row_idx
                    break
            if pivot_row == -1:
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
        
        pivot_set = set(pivot_cols)
        redundant = set()
        for c in range(cols):
            if c not in pivot_set:
                redundant.add(c)
        return redundant
    
    # Try to reduce rank by removing redundant columns
    def reduce_rank(U, V, W, n, m, p, deadline):
        M = build_tensor_matrix(U, V, W, n, m, p)
        redundant = find_redundant_columns(M, deadline)
        
        if redundant:
            keep_cols = [c for c in range(len(U[0])) if c not in redundant]
            new_rank = len(keep_cols)
            
            if new_rank < len(U[0]):
                U_new = [[U[i][c] for c in keep_cols] for i in range(nm)]
                V_new = [[V[i][c] for c in keep_cols] for i in range(mp)]
                W_new = [[W[i][c] for c in keep_cols] for i in range(pn)]
                
                if verify_solution(U_new, V_new, W_new, n, m, p):
                    return U_new, V_new, W_new
        return U, V, W
    
    # Alternative: try different Strassen tilings to find better base
    def build_alt_strassen():
        """Try different ways to tile with Strassen 2x2x2 blocks."""
        def add_strassen_block(U, V, W, i0, j0, k0, col):
            # Standard Strassen 2x2x2
            U[i0 * m + j0][col] += 1
            U[(i0 + 1) * m + j0 + 1][col] += 1
            V[j0 * p + k0][col] += 1
            V[(j0 + 1) * p + k0 + 1][col] += 1
            W[k0 * n + i0][col] += 1
            W[(k0 + 1) * n + i0 + 1][col] += 1
            col += 1
            
            U[(i0 + 1) * m + j0][col] += 1
            U[(i0 + 1) * m + j0 + 1][col] += 1
            V[j0 * p + k0][col] += 1
            W[k0 * n + i0 + 1][col] += 1
            W[(k0 + 1) * n + i0 + 1][col] -= 1
            col += 1
            
            U[i0 * m + j0][col] += 1
            V[j0 * p + k0 + 1][col] += 1
            V[(j0 + 1) * p + k0 + 1][col] -= 1
            W[(k0 + 1) * n + i0][col] += 1
            W[(k0 + 1) * n + i0 + 1][col] += 1
            col += 1
            
            U[(i0 + 1) * m + j0 + 1][col] += 1
            V[(j0 + 1) * p + k0][col] += 1
            V[j0 * p + k0][col] -= 1
            W[k0 * n + i0][col] += 1
            W[k0 * n + i0 + 1][col] += 1
            col += 1
            
            U[i0 * m + j0][col] += 1
            U[i0 * m + j0 + 1][col] += 1
            V[(j0 + 1) * p + k0 + 1][col] += 1
            W[(k0 + 1) * n + i0][col] += 1
            W[k0 * n + i0][col] -= 1
            col += 1
            
            U[(i0 + 1) * m + j0][col] += 1
            U[i0 * m + j0][col] -= 1
            V[j0 * p + k0][col] += 1
            V[j0 * p + k0 + 1][col] += 1
            W[(k0 + 1) * n + i0 + 1][col] += 1
            col += 1
            
            U[i0 * m + j0 + 1][col] += 1
            U[(i0 + 1) * m + j0 + 1][col] -= 1
            V[(j0 + 1) * p + k0][col] += 1
            V[(j0 + 1) * p + k0 + 1][col] += 1
            W[k0 * n + i0][col] += 1
            col += 1
            
            return col
        
        # Try tiling i={0,1} with j={0,1}, k={0,1,2,3,4,5}
        U_alt = [[0] * 99 for _ in range(nm)]
        V_alt = [[0] * 99 for _ in range(mp)]
        W_alt = [[0] * 99 for _ in range(pn)]
        
        col = 0
        # Try: i={0,1}, j={0,1,2}, k={0,1} as one big 2x3x2 block
        # Split into two 2x2x2 Strassen blocks
        col = add_strassen_block(U_alt, V_alt, W_alt, 0, 0, 0, col)
        col = add_strassen_block(U_alt, V_alt, W_alt, 0, 2, 0, col)
        # i={0,1}, j={3,4}, k={0,1}
        col = add_strassen_block(U_alt, V_alt, W_alt, 0, 3, 0, col)
        # i={0,1}, j={0,1,2,3,4}, k={2,3}
        col = add_strassen_block(U_alt, V_alt, W_alt, 0, 0, 2, col)
        col = add_strassen_block(U_alt, V_alt, W_alt, 0, 2, 2, col)
        # i={0,1}, j={0,1,2,3,4}, k={4,5}
        col = add_strassen_block(U_alt, V_alt, W_alt, 0, 0, 4, col)
        col = add_strassen_block(U_alt, V_alt, W_alt, 0, 2, 4, col)
        
        # Remaining: i={0,1}, j=0..4, k=6 (2x5x1 = 10)
        # i=2, j=0..4, k=0..6 (1x5x7 = 35)
        for i in [0, 1]:
            for j in range(5):
                U_alt[i * m + j][col] = 1
                V_alt[j * p + 6][col] = 1
                W_alt[6 * n + i][col] = 1
                col += 1
        for i in [2]:
            for j in range(5):
                for k in range(7):
                    U_alt[i * m + j][col] = 1
                    V_alt[j * p + k][col] = 1
                    W_alt[k * n + i][col] = 1
                    col += 1
        
        if col != 99:
            return None
        return U_alt, V_alt, W_alt
    
    # Strategy: try multiple base constructions, reduce each, keep best
    best_rank = 99
    best_U, best_V, best_W = None, None, None
    
    # 1. Original Strassen+Schoolbook
    U1, V1, W1 = build_strassen_schoolbook()
    U1r, V1r, W1r = reduce_rank(U1, V1, W1, n, m, p, deadline)
    if U1r is not None and len(U1r[0]) < best_rank:
        best_rank = len(U1r[0])
        best_U, best_V, best_W = U1r, V1r, W1r
    
    if time.time() > deadline:
        return {"U": U1, "V": V1, "W": W1}
    
    # 2. Alternative Strassen tiling
    alt = build_alt_strassen()
    if alt is not None:
        U2, V2, W2 = alt
        U2r, V2r, W2r = reduce_rank(U2, V2, W2, n, m, p, deadline)
        if U2r is not None and len(U2r[0]) < best_rank:
            best_rank = len(U2r[0])
            best_U, best_V, best_W = U2r, V2r, W2r
    
    if time.time() > deadline:
        if best_U is not None:
            return {"U": best_U, "V": best_V, "W": best_W}
        return {"U": U1, "V": V1, "W": W1}
    
    # 3. Try schoolbook base with rank reduction (should find same or better)
    U3, V3, W3 = build_schoolbook()
    U3r, V3r, W3r = reduce_rank(U3, V3, W3, n, m, p, deadline)
    if U3r is not None and len(U3r[0]) < best_rank:
        best_rank = len(U3r[0])
        best_U, best_V, best_W = U3r, V3r, W3r
    
    if best_U is not None:
        return {"U": best_U, "V": best_V, "W": best_W}
    return {"U": U1, "V": V1, "W": W1}
# EVOLVE-BLOCK-END
