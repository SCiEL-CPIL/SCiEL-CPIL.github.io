"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(i, k):
        return k * n + i

    def build_decomposition(rank, U, V, W):
        """Build a decomposition of given rank using block structure."""
        col = 0
        
        def add_term(u_vals, v_vals, w_vals):
            nonlocal col
            if col >= rank:
                return False
            for idx, val in u_vals:
                U[idx][col] += val
            for idx, val in v_vals:
                V[idx][col] += val
            for idx, val in w_vals:
                W[idx][col] += val
            col += 1
            return True

        # Strassen 2x2x2 helper
        def strassen_2x2(u_base, v_base, w_base):
            nonlocal col
            if col + 7 > rank:
                return False
            
            # M1: (A00+A11)(B00+B11)
            add_term(
                [(u_base(0,0), 1), (u_base(1,1), 1)],
                [(v_base(0,0), 1), (v_base(1,1), 1)],
                [(w_base(0,0), 1), (w_base(1,1), 1)]
            )
            # M2: (A10+A11)B00
            add_term(
                [(u_base(1,0), 1), (u_base(1,1), 1)],
                [(v_base(0,0), 1)],
                [(w_base(1,0), 1), (w_base(1,1), -1)]
            )
            # M3: A00(B01-B11)
            add_term(
                [(u_base(0,0), 1)],
                [(v_base(0,1), 1), (v_base(1,1), -1)],
                [(w_base(0,1), 1), (w_base(1,1), 1)]
            )
            # M4: A11(B10-B00)
            add_term(
                [(u_base(1,1), 1)],
                [(v_base(1,0), 1), (v_base(0,0), -1)],
                [(w_base(0,0), 1), (w_base(1,0), 1)]
            )
            # M5: (A00+A01)B11
            add_term(
                [(u_base(0,0), 1), (u_base(0,1), 1)],
                [(v_base(1,1), 1)],
                [(w_base(0,0), -1), (w_base(0,1), 1)]
            )
            # M6: (A10-A00)(B00+B01)
            add_term(
                [(u_base(1,0), 1), (u_base(0,0), -1)],
                [(v_base(0,0), 1), (v_base(0,1), 1)],
                [(w_base(1,1), 1)]
            )
            # M7: (A01-A11)(B10+B11)
            add_term(
                [(u_base(0,1), 1), (u_base(1,1), -1)],
                [(v_base(1,0), 1), (v_base(1,1), 1)],
                [(w_base(0,0), 1)]
            )
            return True

        # Standard multiplication for small blocks
        def standard_block(a_rows, b_cols, c_rows, c_cols):
            nonlocal col
            for i in range(a_rows):
                for j in range(m):
                    for k in range(b_cols):
                        if col + 1 > rank:
                            return False
                        add_term([(u_idx(i, j), 1)], [(v_idx(j, k), 1)], [(w_idx(c_rows + i, c_cols + k), 1)])
            return True

        # Try different block decompositions
        # Decomposition 1: A rows 2+1, B cols 2+3
        # Blocks: (2x2x2), (2x2x2), (2x2x3), (2x2x3), (1x2x2), (1x2x2), (1x2x3), (1x2x3)
        # Ranks: 7, 7, 11, 11, 4, 4, 6, 6 = 56
        
        # Decomposition 2: A rows 1+2, B cols 3+2
        # Blocks: (1x3x3), (1x3x2), (2x3x3), (2x3x2)
        # For (1x3x3): rank 9 (standard)
        # For (1x3x2): rank 6 (standard)
        # For (2x3x3): split into 2x1x3 + 2x2x3 = 6 + 11 = 17? Or use Strassen-like
        # For (2x3x2): split into 2x1x2 + 2x2x2 = 4 + 7 = 11? 
        # Total: 9 + 6 + 17 + 11 = 43? Let me think more carefully.
        
        # Actually, let me try a different approach. Use Strassen for 2x2x2 blocks
        # and try to optimize the 2x2x3 and 1x2x3 blocks.
        
        # For 2x2x3: C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, k in {0,1,2}
        # Standard rank: 12
        # Can we do better? Let's try to use Strassen on the 2x2 part and handle the third column separately.
        # That gives 7 + 4 = 11 (as in current code).
        # Can we do 10? Let me think...
        
        # For 1x2x3: C[k] = A[0]*B[0,k] + A[1]*B[1,k] for k in {0,1,2}
        # Standard rank: 6
        # This is essentially computing 3 dot products of length 2.
        # The tensor rank is exactly 6 (I believe), so we can't improve.
        
        # So the main optimization opportunity is in the 2x2x3 blocks.
        # Let me try to find a rank-10 decomposition for 2x2x3.
        
        # Actually, let me just verify the current decomposition works and return it.
        # The current code uses rank 56, which is a valid improvement over 60.
        
        # Let me build the standard rank-56 decomposition.
        # 1. A[0:2, 0:2] * B[0:2, 0:2] (2x2x2) -> Strassen rank 7
        strassen_2x2(
            lambda i, j: u_idx(i, j),
            lambda j, k: v_idx(j, k),
            lambda i, k: w_idx(i, k)
        )
        
        # 2. A[0:2, 2:4] * B[2:4, 0:2] (2x2x2) -> Strassen rank 7
        strassen_2x2(
            lambda i, j: u_idx(i, j + 2),
            lambda j, k: v_idx(j + 2, k),
            lambda i, k: w_idx(i, k)
        )
        
        # 3. A[0:2, 0:2] * B[0:2, 2:5] (2x2x3) -> Strassen on first 2 cols + standard on last col = 7 + 4 = 11
        strassen_2x2(
            lambda i, j: u_idx(i, j),
            lambda j, k: v_idx(j, k + 2),
            lambda i, k: w_idx(i, k + 2)
        )
        for i in range(2):
            for j in range(2):
                add_term([(u_idx(i, j), 1)], [(v_idx(j, 4), 1)], [(w_idx(i, 4), 1)])
        
        # 4. A[0:2, 2:4] * B[2:4, 2:5] (2x2x3) -> Strassen on first 2 cols + standard on last col = 7 + 4 = 11
        strassen_2x2(
            lambda i, j: u_idx(i, j + 2),
            lambda j, k: v_idx(j + 2, k + 2),
            lambda i, k: w_idx(i, k + 2)
        )
        for i in range(2):
            for j in range(2):
                add_term([(u_idx(i, j + 2), 1)], [(v_idx(j + 2, 4), 1)], [(w_idx(i, 4), 1)])
        
        # 5. A[2, 0:2] * B[0:2, 0:2] (1x2x2) -> standard rank 4
        for j in range(2):
            for k in range(2):
                add_term([(u_idx(2, j), 1)], [(v_idx(j, k), 1)], [(w_idx(2, k), 1)])
        
        # 6. A[2, 2:4] * B[2:4, 0:2] (1x2x2) -> standard rank 4
        for j in range(2):
            for k in range(2):
                add_term([(u_idx(2, j + 2), 1)], [(v_idx(j + 2, k), 1)], [(w_idx(2, k), 1)])
        
        # 7. A[2, 0:2] * B[0:2, 2:5] (1x2x3) -> standard rank 6
        for j in range(2):
            for k in range(3):
                add_term([(u_idx(2, j), 1)], [(v_idx(j, k + 2), 1)], [(w_idx(2, k + 2), 1)])
        
        # 8. A[2, 2:4] * B[2:4, 2:5] (1x2x3) -> standard rank 6
        for j in range(2):
            for k in range(3):
                add_term([(u_idx(2, j + 2), 1)], [(v_idx(j + 2, k + 2), 1)], [(w_idx(2, k + 2), 1)])
        
        return col

    # Try rank 56 first (known to work)
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    used_cols = build_decomposition(rank, U, V, W)
    
    if used_cols != rank:
        raise ValueError(f"Expected {rank} columns, got {used_cols}")
    
    # Verify the decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check: for all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1 if i,j,k are valid indices, 0 otherwise
    # Actually, the tensor decomposition means:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T[i,j,k] = 1 if i < n, j < m, k < p, else 0
    
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                error += abs(actual - expected)
    
    if error > 1e-9:
        raise ValueError(f"Decomposition error: {error}")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
