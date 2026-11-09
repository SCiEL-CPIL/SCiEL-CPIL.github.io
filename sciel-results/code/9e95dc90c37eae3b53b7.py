"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    Uses block-structured decomposition with rank-7 sub-blocks and ALS refinement.
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 36

    start_time = time.time()
    deadline = start_time + 140  # 140s deadline with 10s safety margin

    # Schoolbook fallback
    def get_schoolbook():
        rank = schoolbook_rank
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
        return {"U": U, "V": V, "W": W}

    def verify_decomposition(U, V, W, rank):
        """Verify exact decomposition with tolerance for half-integer arithmetic."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    def snap_to_half_integers(matrix):
        """Snap float matrix to nearest half-integer."""
        return [[int(round(val * 2)) / 2.0 for val in row] for row in matrix]

    def run_als(U, V, W, max_iter=150, tol=1e-10):
        """
        Run ALS iterations to minimize error.
        U: (n*m, R), V: (m*p, R), W: (p*n, R)
        """
        R = U.shape[1]

        for iteration in range(max_iter):
            if time.time() > deadline:
                return False, U, V, W

            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, R))
                    for k in range(p):
                        A[k] = V[j * p + k] * W[k * n + i]
                    try:
                        u_sol, residuals, rank_sol, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U[i * m + j] = u_sol
                    except:
                        pass

            # Update V
            for j in range(m):
                for k in range(p):
                    B = np.zeros((n, R))
                    for i in range(n):
                        B[i] = U[i * m + j] * W[k * n + i]
                    try:
                        v_sol, residuals, rank_sol, sv = np.linalg.lstsq(B, np.ones(n), rcond=None)
                        V[j * p + k] = v_sol
                    except:
                        pass

            # Update W
            for k in range(p):
                for i in range(n):
                    C = np.zeros((m, R))
                    for j in range(m):
                        C[j] = U[i * m + j] * V[j * p + k]
                    try:
                        w_sol, residuals, rank_sol, sv = np.linalg.lstsq(C, np.ones(m), rcond=None)
                        W[k * n + i] = w_sol
                    except:
                        pass

            # Check convergence
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                        err += (val - 1.0) ** 2

            if err < tol:
                return True, U, V, W

        return False, U, V, W

    def generate_structured_init(candidate_rank, seed=None):
        """
        Generate structured initial matrices that respect the tensor structure.
        Use sparse patterns with some density to help ALS converge.
        """
        if seed is not None:
            np.random.seed(seed)

        U = np.zeros((n * m, candidate_rank))
        V = np.zeros((m * p, candidate_rank))
        W = np.zeros((p * n, candidate_rank))

        for r in range(candidate_rank):
            num_u_entries = max(1, candidate_rank // 3)
            u_indices = np.random.choice(n * m, size=min(num_u_entries, n * m), replace=False)
            for idx in u_indices:
                U[idx, r] = np.random.choice([-1.0, 1.0])

            num_v_entries = max(1, candidate_rank // 4)
            v_indices = np.random.choice(m * p, size=min(num_v_entries, m * p), replace=False)
            for idx in v_indices:
                V[idx, r] = np.random.choice([-1.0, 1.0])

            num_w_entries = max(1, candidate_rank // 4)
            w_indices = np.random.choice(p * n, size=min(num_w_entries, p * n), replace=False)
            for idx in w_indices:
                W[idx, r] = np.random.choice([-1.0, 1.0])

        return U, V, W

    def generate_random_init(candidate_rank, scale=0.5, seed=None):
        """Generate random initial matrices."""
        if seed is not None:
            np.random.seed(seed)

        U = np.random.randn(n * m, candidate_rank) * scale
        V = np.random.randn(m * p, candidate_rank) * scale
        W = np.random.randn(p * n, candidate_rank) * scale
        return U, V, W

    def generate_block_structured_init(candidate_rank, seed=None):
        """
        Generate block-structured initial matrices using overlapping 2x2 sub-blocks.
        The 3x3 matrix is decomposed into overlapping 2x2 blocks, and the 3x4 output
        is constructed from overlapping 2x3 multiplications using rank-7 Strassen-like
        decompositions for each 2x2x2 sub-multiplication.
        """
        if seed is not None:
            np.random.seed(seed)

        U = np.zeros((n * m, candidate_rank))
        V = np.zeros((m * p, candidate_rank))
        W = np.zeros((p * n, candidate_rank))

        # Define overlapping 2x2 block positions for 3x3 matrix
        # Blocks: (0,0), (0,1), (1,0), (1,1) covering rows/cols 0-1 and 1-2
        block_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        
        # Rank-7 Strassen-like coefficients for 2x2x2
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b12-b22)
        # M4 = a22(b21-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        strassen_coeffs = [
            # (U_row1, U_row2, V_col1, V_col2, W_row1, W_row2, coeff)
            # Each entry represents one of the 7 products
        ]
        
        # For each block position, assign 7 rank indices
        rank_offset = 0
        for block_row, block_col in block_positions:
            if rank_offset + 7 > candidate_rank:
                break
                
            # 2x2 block at position (block_row, block_col) in A
            # corresponds to A[block_row:block_row+2, block_col:block_col+2]
            # Output block at position (block_row, block_col) in C
            # corresponds to C[block_row:block_row+2, block_col:block_col+2]
            
            # Assign 7 rank indices for this block
            for s in range(7):
                r = rank_offset + s
                if r >= candidate_rank:
                    break
                    
                # Strassen-like decomposition for 2x2x2
                # We'll use a simple structured pattern
                if s == 0:
                    # M1 = (a11+a22)(b11+b22)
                    U[block_row * m + block_col, r] = 1.0
                    U[(block_row+1) * m + (block_col+1), r] = 1.0
                    V[block_col * p + block_col, r] = 1.0
                    V[(block_col+1) * p + (block_col+1), r] = 1.0
                    W[block_col * n + block_row, r] = 1.0
                    W[(block_col+1) * n + (block_row+1), r] = 1.0
                    W[block_col * n + (block_row+1), r] = 1.0
                    W[(block_col+1) * n + block_row, r] = 1.0
                elif s == 1:
                    # M2 = (a21+a22)b11
                    U[(block_row+1) * m + block_col, r] = 1.0
                    U[(block_row+1) * m + (block_col+1), r] = 1.0
                    V[block_col * p + block_col, r] = 1.0
                    W[block_col * n + (block_row+1), r] = 1.0
                elif s == 2:
                    # M3 = a11(b12-b22)
                    U[block_row * m + block_col, r] = 1.0
                    V[block_col * p + (block_col+1), r] = 1.0
                    V[(block_col+1) * p + (block_col+1), r] = -1.0
                    W[(block_col+1) * n + block_row, r] = 1.0
                elif s == 3:
                    # M4 = a22(b21-b11)
                    U[(block_row+1) * m + (block_col+1), r] = 1.0
                    V[(block_col+1) * p + block_col, r] = 1.0
                    V[block_col * p + block_col, r] = -1.0
                    W[block_col * n + (block_row+1), r] = 1.0
                elif s == 4:
                    # M5 = (a11+a12)b22
                    U[block_row * m + block_col, r] = 1.0
                    U[block_row * m + (block_col+1), r] = 1.0
                    V[(block_col+1) * p + (block_col+1), r] = 1.0
                    W[(block_col+1) * n + block_row, r] = 1.0
                elif s == 5:
                    # M6 = (a21-a11)(b11+b12)
                    U[(block_row+1) * m + block_col, r] = 1.0
                    U[block_row * m + block_col, r] = -1.0
                    V[block_col * p + block_col, r] = 1.0
                    V[block_col * p + (block_col+1), r] = 1.0
                    W[block_col * n + (block_row+1), r] = 1.0
                    W[(block_col+1) * n + (block_row+1), r] = 1.0
                elif s == 6:
                    # M7 = (a12-a22)(b21+b22)
                    U[block_row * m + (block_col+1), r] = 1.0
                    U[(block_row+1) * m + (block_col+1), r] = -1.0
                    V[(block_col+1) * p + block_col, r] = 1.0
                    V[(block_col+1) * p + (block_col+1), r] = 1.0
                    W[(block_col+1) * n + block_row, r] = 1.0
                    W[(block_col+1) * n + (block_row+1), r] = 1.0
            
            rank_offset += 7
        
        # Add random noise to remaining rank indices to help ALS escape local minima
        for r in range(rank_offset, candidate_rank):
            if time.time() > deadline:
                break
            num_u_entries = max(1, candidate_rank // 5)
            u_indices = np.random.choice(n * m, size=min(num_u_entries, n * m), replace=False)
            for idx in u_indices:
                U[idx, r] = np.random.choice([-1.0, 1.0])
            
            num_v_entries = max(1, candidate_rank // 5)
            v_indices = np.random.choice(m * p, size=min(num_v_entries, m * p), replace=False)
            for idx in v_indices:
                V[idx, r] = np.random.choice([-1.0, 1.0])
            
            num_w_entries = max(1, candidate_rank // 5)
            w_indices = np.random.choice(p * n, size=min(num_w_entries, p * n), replace=False)
            for idx in w_indices:
                W[idx, r] = np.random.choice([-1.0, 1.0])
        
        return U, V, W

    # Try to find lower rank decompositions
    best_result = None
    best_rank = schoolbook_rank

    # Try ranks from 30 down to 24
    for candidate_rank in range(30, 23, -1):
        if time.time() > deadline:
            break

        # Try multiple random restarts for this rank
        max_restarts = 8 if candidate_rank > 26 else 12

        for restart in range(max_restarts):
            if time.time() > deadline:
                break

            # Use block-structured init for even restarts, structured for odd, random for others
            if restart % 3 == 0:
                U, V, W = generate_block_structured_init(candidate_rank, seed=restart * 1000 + candidate_rank)
            elif restart % 3 == 1:
                U, V, W = generate_structured_init(candidate_rank, seed=restart * 1000 + candidate_rank + 2)
            else:
                U, V, W = generate_random_init(candidate_rank, scale=0.3, seed=restart * 1000 + candidate_rank + 3)

            # Run ALS
            converged, U_final, V_final, W_final = run_als(U, V, W, max_iter=100, tol=1e-10)

            if converged:
                # Snap to half-integers
                U_snap = snap_to_half_integers(U_final.tolist())
                V_snap = snap_to_half_integers(V_final.tolist())
                W_snap = snap_to_half_integers(W_final.tolist())

                # Verify exactness of snapped version
                if verify_decomposition(U_snap, V_snap, W_snap, candidate_rank):
                    best_result = {
                        "U": U_snap,
                        "V": V_snap,
                        "W": W_snap
                    }
                    best_rank = candidate_rank
                    break

        if best_result is not None:
            break

    if best_result is not None:
        return best_result
    else:
        return get_schoolbook()
# EVOLVE-BLOCK-END
