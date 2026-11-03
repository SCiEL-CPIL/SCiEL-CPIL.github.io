"""Improved program: block-Strassen + rational Gaussian elimination for 4x5x5 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [4, 5, 5]
_DEADLINE_SECONDS = 115  # safety margin before 120s kill

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """Strassen's 7-product algorithm for 2x2x2.
    Returns U (4x7), V (4x7), W (4x7) as lists of lists (integers).
    """
    U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, -1, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V = [
        [1, 1, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 1]
    ]
    W = [
        [1, 0, 0, -1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, -1, 0, 0, 1, 0]
    ]
    return U, V, W


def _schoolbook_block(n, m, p):
    """Schoolbook decomposition for n x m times m x p = n x p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _build_block_strassen_4x5x5():
    """Build the block-Strassen decomposition (rank 98) as starting point."""
    n, m, p = 4, 5, 5
    total_rank = 98
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    sU, sV, sW = _strassen_2x2()
    
    # Sub-part 1a: A_top_left (2x2) * B_top_left (2x2) - Strassen
    a_rows = [0, 1, 5, 6]
    b_rows = [0, 1, 5, 6]
    c_rows = [0, 1, 4, 5]
    for i in range(4):
        for c in range(7):
            U[a_rows[i]][col + c] = sU[i][c]
            V[b_rows[i]][col + c] = sV[i][c]
            W[c_rows[i]][col + c] = sW[i][c]
    col += 7
    
    # Sub-part 1b: A_top_left (2x2) * B_top_right (2x3) - schoolbook
    a_rows = [0, 1, 5, 6]
    b_rows = [2, 3, 4, 7, 8, 9]
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[a_rows[i * 2 + j]][col] = 1
                V[b_rows[j * 3 + k]][col] = 1
                W[(2 + k) * 4 + i][col] = 1
                col += 1
    
    # Sub-part 1c: A_bot_left (2x2) * B_top_left (2x2) - Strassen
    sU, sV, sW = _strassen_2x2()
    a_rows = [10, 11, 15, 16]
    b_rows = [0, 1, 5, 6]
    c_rows = [8, 10, 12, 14]
    for i in range(4):
        for c in range(7):
            U[a_rows[i]][col + c] = sU[i][c]
            V[b_rows[i]][col + c] = sV[i][c]
            W[c_rows[i]][col + c] = sW[i][c]
    col += 7
    
    # Sub-part 1d: A_bot_left (2x2) * B_top_right (2x3) - schoolbook
    a_rows = [10, 11, 15, 16]
    b_rows = [2, 3, 4, 7, 8, 9]
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[a_rows[i * 2 + j]][col] = 1
                V[b_rows[j * 3 + k]][col] = 1
                W[(2 + k) * 4 + (2 + i)][col] = 1
                col += 1
    
    # Part 2: A_right (4x3) * B_bot (3x5) - schoolbook
    for i in range(4):
        for j in range(3):
            for k in range(5):
                U[i * 5 + (2 + j)][col] = 1
                V[(2 + j) * 5 + k][col] = 1
                W[k * 4 + i][col] = 1
                col += 1
    
    assert col == total_rank
    return U, V, W


def _extract_tensor_coefficients(U, V, W, n, m, p):
    """Extract the tensor T[i,j,k] coefficients from the decomposition.
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    Returns a dict mapping (i,j,k) -> coefficient (should be 1 for all valid (i,j,k)).
    """
    rank = len(W[0])
    tensor = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                tensor[(i, j, k)] = val
    return tensor


def _build_linear_system(U, V, W, n, m, p):
    """Build the linear system for Gaussian elimination.
    Each rank-1 term r defines a tensor T_r[i,j,k] = U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r].
    We need sum_r T_r = T_matrix_mult where T_matrix_mult[i,j,k] = 1.
    
    The constraint is: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1.
    This is nonlinear in U,V,W. But if we fix the structure (which entries are nonzero),
    we can work with the linear dependencies.
    
    Alternative approach: treat each rank-1 term as a variable x_r, and the constraint is
    that the tensor sum equals the target. The rank-1 tensors form a set, and we want
    the minimum subset (with appropriate scaling) that sums to the target.
    
    For exact integer/half-integer entries, we work over Q.
    """
    rank = len(W[0])
    # The tensor has n*m*p = 100 entries, all should equal 1
    # Each rank-1 term contributes to all 100 entries
    # We want to find if we can remove some terms while maintaining the sum
    
    # Build matrix A where A[(i,j,k), r] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # Target b[(i,j,k)] = 1
    # We want to find x_r such that A @ x = b, minimizing number of nonzero x_r
    # with x_r in {0, 1/2, 1} (or similar small rationals)
    
    # This is a sparse linear system. Use rational Gaussian elimination.
    num_constraints = n * m * p  # 100
    A = [[Fraction(0) for _ in range(rank)] for _ in range(num_constraints)]
    b = [Fraction(1) for _ in range(num_constraints)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                for r in range(rank):
                    A[idx][r] = Fraction(U[i * m + j][r]) * Fraction(V[j * p + k][r]) * Fraction(W[k * n + i][r])
    
    return A, b


def _gaussian_elimination_rational(A, b, deadline):
    """Perform Gaussian elimination over rationals.
    Returns (rank, pivot_cols, solution) or None if timeout.
    """
    t0 = time.time()
    m = len(A)
    n = len(A[0]) if m > 0 else 0
    
    # Make copies
    M = [row[:] for row in A]
    rhs = b[:]
    
    pivot_cols = []
    rank = 0
    
    for col in range(n):
        if time.time() - t0 > deadline - 5:
            return None
        
        # Find pivot
        pivot_row = -1
        for row in range(rank, m):
            if M[row][col] != 0:
                pivot_row = row
                break
        
        if pivot_row == -1:
            continue
        
        # Swap rows
        if pivot_row != rank:
            M[rank], M[pivot_row] = M[pivot_row], M[rank]
            rhs[rank], rhs[pivot_row] = rhs[pivot_row], rhs[rank]
        
        pivot_cols.append(col)
        
        # Normalize pivot row
        pivot_val = M[rank][col]
        for j in range(col, n):
            M[rank][j] = M[rank][j] / pivot_val
        rhs[rank] = rhs[rank] / pivot_val
        
        # Eliminate other rows
        for row in range(m):
            if row == rank:
                continue
            factor = M[row][col]
            if factor != 0:
                for j in range(col, n):
                    M[row][j] = M[row][j] - factor * M[rank][j]
                rhs[row] = rhs[row] - factor * rhs[rank]
        
        rank += 1
    
    return rank, pivot_cols, M, rhs


def _find_minimal_basis(U, V, W, n, m, p, deadline):
    """Try to find a minimal basis using rational elimination.
    
    Strategy: 
    1. Start with the block-Strassen decomposition (rank 98)
    2. Build the linear system
    3. Perform Gaussian elimination to find the rank
    4. Extract a minimal basis from the RREF
    
    The key insight: if the rank of the system is less than the number of terms,
    some terms are linearly dependent and can be eliminated.
    """
    t0 = time.time()
    rank = len(W[0])
    
    # Build the linear system
    A, b = _build_linear_system(U, V, W, n, m, p)
    
    # Perform Gaussian elimination
    result = _gaussian_elimination_rational(A, b, deadline)
    if result is None:
        return None
    
    sys_rank, pivot_cols, M_rref, rhs_rref = result
    
    # The system has sys_rank independent constraints out of 100
    # We have 98 variables (rank-1 terms)
    # If sys_rank < 98, there are free variables, meaning we can potentially reduce
    
    # For a minimal solution, set free variables to 0 and solve for pivots
    num_vars = rank
    free_vars = [c for c in range(num_vars) if c not in pivot_cols]
    pivot_set = set(pivot_cols)
    
    # Build minimal solution: free vars = 0, solve for pivot vars
    x_minimal = [Fraction(0) for _ in range(num_vars)]
    for i in range(sys_rank):
        col = pivot_cols[i]
        x_minimal[col] = rhs_rref[i]
    
    # Check which terms are nonzero in the minimal solution
    nonzero_terms = [r for r in range(num_vars) if x_minimal[r] != 0]
    
    if len(nonzero_terms) < rank:
        # Build reduced decomposition
        new_rank = len(nonzero_terms)
        term_map = {r: idx for idx, r in enumerate(nonzero_terms)}
        
        new_U = [[0] * new_rank for _ in range(len(U))]
        new_V = [[0] * new_rank for _ in range(len(V))]
        new_W = [[0] * new_rank for _ in range(len(W))]
        
        for r in nonzero_terms:
            new_idx = term_map[r]
            scale = x_minimal[r]
            for i in range(len(U)):
                new_U[i][new_idx] = U[i][r] * scale
            for j in range(len(V)):
                new_V[j][new_idx] = V[j][r] * scale
            for k in range(len(W)):
                new_W[k][new_idx] = W[k][r] * scale
        
        return new_U, new_V, new_W
    
    return None


def _verify_decomposition(U, V, W, n, m, p, num_trials=3):
    """Verify that the decomposition correctly computes matrix multiplication."""
    np.random.seed(42)
    rank = len(W[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for trial in range(num_trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for r in range(rank):
                        C_computed[i, k] += U_np[i * m + j, r] * V_np[j * p + k, r] * W_np[k * n + i, r]
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True


def _try_direct_minimal_construction():
    """Try to directly construct a lower-rank decomposition using known results.
    
    For 4x5x5 matrix multiplication, the known upper bounds:
    - Schoolbook: 100
    - Block Strassen: 98 (our current)
    - The tensor rank is at most 98, possibly lower.
    
    Let's try a more aggressive block decomposition:
    Split 4x5x5 into blocks and apply Strassen where possible.
    
    Alternative: Use the fact that 5 = 2+3 and try different splits.
    """
    n, m, p = 4, 5, 5
    
    # Try: split m=5 into 2+3
    # C = A[:, :2] @ B[:2, :] + A[:, 2:] @ B[2:, :]
    # Part 1: 4x2 * 2x5 = 4x5
    # Part 2: 4x3 * 3x5 = 4x5
    
    # Part 1: 4x2 * 2x5
    # Split 4 rows into 2+2:
    #   2x2 * 2x5 (rank 19 with Strassen on 2x2 part)
    #   2x2 * 2x5 (rank 19)
    # Total Part 1: 38
    
    # Part 2: 4x3 * 3x5
    # Split 4 rows into 2+2:
    #   2x3 * 3x5 (schoolbook rank 30)
    #   2x3 * 3x5 (schoolbook rank 30)
    # Total Part 2: 60
    
    # Grand total: 98. Same as before.
    
    # Try different split: m=5 into 1+4
    # Part 1: 4x1 * 1x5 = 4x5, rank 20
    # Part 2: 4x4 * 4x5 = 4x5
    #   Split 4x4 into 2x2 blocks:
    #   2x2*2x5: rank 19 (Strassen on 2x2)
    #   2x2*2x5: rank 19
    #   2x2*2x5: rank 19
    #   2x2*2x5: rank 19
    #   But wait, 4x4 * 4x5: split A into 2x2 blocks, B into 2x2 and 2x3
    #   A[0:2,0:2]*B[0:2,0:2]: 2x2*2x2, Strassen 7
    #   A[0:2,0:2]*B[0:2,2:5]: 2x2*2x3, schoolbook 12
    #   A[0:2,2:4]*B[2:4,0:2]: 2x2*2x2, Strassen 7
    #   A[0:2,2:4]*B[2:4,2:5]: 2x2*2x3, schoolbook 12
    #   A[2:4,0:2]*B[0:2,0:2]: 2x2*2x2, Strassen 7
    #   A[2:4,0:2]*B[0:2,2:5]: 2x2*2x3, schoolbook 12
    #   A[2:4,2:4]*B[2:4,0:2]: 2x2*2x2, Strassen 7
    #   A[2:4,2:4]*B[2:4,2:5]: 2x2*2x3, schoolbook 12
    #   Total: 4*7 + 4*12 = 28 + 48 = 76
    # Grand total: 20 + 76 = 96!
    
    # Let's build this!
    total_rank = 96
    U = [[0] * total_rank for _ in range(n * m)]  # 20 x 96
    V = [[0] * total_rank for _ in range(m * p)]  # 25 x 96
    W = [[0] * total_rank for _ in range(p * n)]  # 20 x 96
    
    col = 0
    
    # Part 1: 4x1 * 1x5 (A[:,0] * B[0,:])
    # A[i,0] for i=0..3 -> U rows i*5+0
    # B[0,k] for k=0..4 -> V rows 0*5+k = k
    # C[i,k] for i=0..3, k=0..4 -> W rows k*4+i
    for i in range(4):
        for k in range(5):
            U[i * 5 + 0][col] = 1
            V[0 * 5 + k][col] = 1
            W[k * 4 + i][col] = 1
            col += 1
    # col = 20
    
    # Part 2: 4x4 * 4x5 (A[:,1:5] * B[1:5,:])
    # A[i,j] for i=0..3, j=1..4 -> U rows i*5+j
    # B[j,k] for j=1..4, k=0..4 -> V rows j*5+k
    # C[i,k] for i=0..3, k=0..4 -> W rows k*4+i
    
    # Split A[:,1:5] into 2x2 blocks:
    # A_top_left: A[0:2, 1:3] -> A[0,1],A[0,2],A[1,1],A[1,2] -> U rows 1,2,6,7
    # A_top_right: A[0:2, 3:5] -> A[0,3],A[0,4],A[1,3],A[1,4] -> U rows 3,4,8,9
    # A_bot_left: A[2:4, 1:3] -> A[2,1],A[2,2],A[3,1],A[3,2] -> U rows 11,12,16,17
    # A_bot_right: A[2:4, 3:5] -> A[2,3],A[2,4],A[3,3],A[3,4] -> U rows 13,14,18,19
    
    # B[1:5,:] split cols into 2+3:
    # B_top: B[1:3, :] -> B[1,0..4], B[2,0..4] -> V rows 5..9, 10..14
    # B_bot: B[3:5, :] -> B[3,0..4], B[4,0..4] -> V rows 15..19, 20..24
    # B_top_left: B[1:3, 0:2] -> V rows 5,6,10,11
    # B_top_right: B[1:3, 2:5] -> V rows 7,8,9,12,13,14
    # B_bot_left: B[3:5, 0:2] -> V rows 15,16,20,21
    # B_bot_right: B[3:5, 2:5] -> V rows 17,18,19,22,23,24
    
    sU, sV, sW = _strassen_2x2()
    
    # A_top_left (2x2) * B_top_left (2x2) - Strassen
    a_rows = [1, 2, 6, 7]
    b_rows = [5, 6, 10, 11]
    c_rows = [0, 1, 4, 5]  # C[0,0],C[0,1],C[1,0],C[1,1]
    for i in range(4):
        for c in range(7):
            U[a_rows[i]][col + c] = sU[i][c]
            V[b_rows[i]][col + c] = sV[i][c]
            W[c_rows[i]][col + c] = sW[i][c]
    col += 7
    
    # A_top_left (2x2) * B_top_right (2x3) - schoolbook
    a_rows = [1, 2, 6, 7]
    b_rows = [7, 8, 9, 12, 13, 14]
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[a_rows[i * 2 + j]][col] = 1
                V[b_rows[j * 3 + k]][col] = 1
                W[(2 + k) * 4 + i][col] = 1
                col += 1
    
    # A_top_right (2x2) * B_top_left (2x2) - Strassen
    a_rows = [3, 4, 8, 9]
    b_rows = [5, 6, 10, 11]
    c_rows = [0, 1, 4, 5]
    for i in range(4):
        for c in range(7):
            U[a_rows[i]][col + c] = sU[i][c]
            V[b_rows[i]][col + c] = sV[i][c]
            W[c_rows[i]][col + c] = sW[i][c]
    col += 7
    
    # A_top_right (2x2) * B_top_right (2x3) - schoolbook
    a_rows = [3, 4, 8, 9]
    b_rows = [7, 8, 9, 12, 13, 14]
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[a_rows[i * 2 + j]][col] = 1
                V[b_rows[j * 3 + k]][col] = 1
                W[(2 + k) * 4 + i][col] = 1
                col += 1
    
    # A_bot_left (2x2) * B_bot_left (2x2) - Strassen
    a_rows = [11, 12, 16, 17]
    b_rows = [15, 16, 20, 21]
    c_rows = [8, 9, 12, 13]  # C[2,0],C[2,1],C[3,0],C[3,1]
    for i in range(4):
        for c in range(7):
            U[a_rows[i]][col + c] = sU[i][c]
            V[b_rows[i]][col + c] = sV[i][c]
            W[c_rows[i]][col + c] = sW[i][c]
    col += 7
    
    # A_bot_left (2x2) * B_bot_right (2x3) - schoolbook
    a_rows = [11, 12, 16, 17]
    b_rows = [17, 18, 19, 22, 23, 24]
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[a_rows[i * 2 + j]][col] = 1
                V[b_rows[j * 3 + k]][col] = 1
                W[(2 + k) * 4 + (2 + i)][col] = 1
                col += 1
    
    # A_bot_right (2x2) * B_bot_left (2x2) - Strassen
    a_rows = [13, 14, 18, 19]
    b_rows = [15, 16, 20, 21]
    c_rows = [8, 9, 12, 13]
    for i in range(4):
        for c in range(7):
            U[a_rows[i]][col + c] = sU[i][c]
            V[b_rows[i]][col + c] = sV[i][c]
            W[c_rows[i]][col + c] = sW[i][c]
    col += 7
    
    # A_bot_right (2x2) * B_bot_right (2x3) - schoolbook
    a_rows = [13, 14, 18, 19]
    b_rows = [17, 18, 19, 22, 23, 24]
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[a_rows[i * 2 + j]][col] = 1
                V[b_rows[j * 3 + k]][col] = 1
                W[(2 + k) * 4 + (2 + i)][col] = 1
                col += 1
    
    assert col == total_rank, f"Expected {total_rank}, got {col}"
    return U, V, W


def construct():
    """Construct the rank-minimal exact tensor decomposition for 4x5x5 matrix multiplication."""
    t0 = time.time()
    
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Try the improved 96-rank construction first
    U, V, W = _try_direct_minimal_construction()
    rank = len(W[0])
    
    # Verify correctness
    if time.time() - t0 < _DEADLINE_SECONDS - 10:
        if not _verify_decomposition(U, V, W, n, m, p, num_trials=3):
            # Fallback to block-Strassen 98
            U, V, W = _build_block_strassen_4x5x5()
            rank = len(W[0])
            if time.time() - t0 < _DEADLINE_SECONDS - 5:
                if not _verify_decomposition(U, V, W, n, m, p, num_trials=3):
                    # Ultimate fallback: schoolbook
                    U, V, W = _schoolbook_block(n, m, p)
    
    # Check deadline
    if time.time() - t0 > _DEADLINE_SECONDS:
        U, V, W = _schoolbook_block(n, m, p)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
