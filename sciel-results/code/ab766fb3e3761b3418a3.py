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

        # Rank-10 decomposition for 2x2x3: C[i,k] = sum_j A[i,j]*B[j,k]
        # Uses 5 Strassen-like products:
        # M1: (A00+A11)(B00+B11) -> contributes to C00,C02,C10,C12
        # M2: (A10+A11)B00 -> contributes to C10,C12
        # M3: A00(B01-B11) -> contributes to C01,C02
        # M4: A11(B10-B00) -> contributes to C00,C10
        # M5: (A00+A01)B11 -> contributes to C00,C02
        # Plus 5 standard terms for remaining contributions
        def strassen_2x2x3(u_base, v_base, w_base):
            nonlocal col
            if col + 10 > rank:
                return False
            
            # M1: (A00+A11)(B00+B11)
            add_term(
                [(u_base(0,0), 1), (u_base(1,1), 1)],
                [(v_base(0,0), 1), (v_base(1,1), 1)],
                [(w_base(0,0), 1), (w_base(0,2), 1), (w_base(1,0), 1), (w_base(1,2), 1)]
            )
            # M2: (A10+A11)B00
            add_term(
                [(u_base(1,0), 1), (u_base(1,1), 1)],
                [(v_base(0,0), 1)],
                [(w_base(1,0), 1), (w_base(1,2), -1)]
            )
            # M3: A00(B01-B11)
            add_term(
                [(u_base(0,0), 1)],
                [(v_base(0,1), 1), (v_base(1,1), -1)],
                [(w_base(0,1), 1), (w_base(0,2), 1)]
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
                [(w_base(0,0), -1), (w_base(0,2), 1)]
            )
            # Standard terms for remaining contributions:
            # C01 needs A01*B01 (not covered by M3 which gives A00*B01 - A00*B11)
            add_term([(u_base(0,1), 1)], [(v_base(0,1), 1)], [(w_base(0,1), 1)])
            # C11 needs A10*B01 + A11*B11
            add_term([(u_base(1,0), 1)], [(v_base(0,1), 1)], [(w_base(1,1), 1)])
            add_term([(u_base(1,1), 1)], [(v_base(1,1), 1)], [(w_base(1,1), 1)])
            # C02 needs A01*B02 (M3 gives A00*B02 - A00*B12, M1 gives A00*B02+A00*B12+A11*B02+A11*B12, M5 gives A00*B12+A01*B12)
            # Let me just add the missing standard terms
            add_term([(u_base(0,1), 1)], [(v_base(0,2), 1)], [(w_base(0,2), 1)])
            # C12 needs A10*B02 + A11*B12 (M1 gives A11*B02+A11*B12, M2 gives A10*B00+A11*B00, but we need B02 and B12)
            add_term([(u_base(1,0), 1)], [(v_base(0,2), 1)], [(w_base(1,2), 1)])
            # Wait, this is getting complicated. Let me just use standard terms for all 12 and see if I can reduce.
            # Actually, let me just return False and use the standard 11-term approach.
            # For now, let's keep it simple and use the known working approach.
            return False
        
        # Use the proven rank-56 decomposition with Strassen for 2x2x2 blocks
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
