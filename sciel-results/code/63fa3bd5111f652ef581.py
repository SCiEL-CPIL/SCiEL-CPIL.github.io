import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def _time_ok(deadline):
    return time.time() < deadline

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0
    
    # Strassen components for 2x2x2
    # local_a: 0=A11, 1=A12, 2=A21, 3=A22
    # local_b: 0=B11, 1=B12, 2=B21, 3=B22
    # local_w: 0=C11, 1=C12, 2=C21, 3=C22
    strassen = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),   # M1
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),           # M2
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),           # M3
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),           # M4
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),           # M5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),           # M6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),           # M7
    ]
    
    # Blocks: (a_rows, a_cols, b_rows, b_cols)
    # We split B into three 4x2 blocks.
    # A is 2x4.
    # Each block is a 2x4x2 multiplication.
    # Rank of 2x4x2 is 14 (two 2x2x2).
    # Total rank 42.
    
    blocks = [
        ([0, 1], [0, 1, 2, 3], [0, 1], [0, 1]), # B cols 0,1
        ([0, 1], [0, 1, 2, 3], [2, 3], [0, 1]), # B cols 2,3
        ([0, 1], [0, 1, 2, 3], [4, 5], [0, 1]), # B cols 4,5
    ]
    
    # Wait, the previous code used 6 blocks of 2x2x2.
    # My new analysis says 3 blocks of 2x4x2.
    # 2x4x2 is two 2x2x2 multiplications.
    # So for each 2x4x2 block, we need 14 terms.
    # Total 42 terms.
    
    # Let's stick to the 6 blocks of 2x2x2 to be safe and consistent with previous logic,
    # but verify the mapping.
    
    # Previous blocks:
    # ([0, 1], [0, 1], [0, 1], [0, 1]) -> A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # ([0, 1], [0, 1], [0, 1], [2, 3]) -> A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # ...
    # This covers all 2x2 sub-matrices.
    # There are 2 (A rows) * 2 (A cols) * 3 (B col pairs) = 12? No.
    # A is 2x4. B is 4x6.
    # A rows: 0,1. A cols: 0,1,2,3.
    # B rows: 0,1,2,3. B cols: 0,1,2,3,4,5.
    # C rows: 0,1. C cols: 0,1,2,3,4,5.
    
    # The multiplication is C = A B.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # We can decompose this into sums of 2x2x2 multiplications.
    # Let's split A cols into pairs: (0,1) and (2,3).
    # Let's split B rows into pairs: (0,1) and (2,3).
    # Let's split B cols into pairs: (0,1), (2,3), (4,5).
    
    # Term (a_col_pair, b_row_pair, b_col_pair)
    # A block: 2x2 (rows 0,1; cols a_cp)
    # B block: 2x2 (rows b_rp; cols b_cp)
    # C block: 2x2 (rows 0,1; cols b_cp)
    
    # There are 2 (a_cp) * 2 (b_rp) * 3 (b_cp) = 12 such blocks.
    # Each block is a 2x2x2 multiplication, rank 7.
    # Total rank 12 * 7 = 84?
    # No, the previous code had 6 blocks.
    # Why 6?
    # Blocks in previous code:
    # 1. A[0:2, 0:2] * B[0:2, 0:2]
    # 2. A[0:2, 0:2] * B[0:2, 2:4]
    # 3. A[0:2, 0:2] * B[0:2, 4:6]
    # 4. A[0:2, 2:4] * B[2:4, 0:2]
    # 5. A[0:2, 2:4] * B[2:4, 2:4]
    # 6. A[0:2, 2:4] * B[2:4, 4:6]
    
    # This misses A[0:2, 0:2] * B[2:4, ...] and A[0:2, 2:4] * B[0:2, ...]
    # Wait, B rows must match A cols.
    # A cols 0,1 match B rows 0,1.
    # A cols 2,3 match B rows 2,3.
    # So only 6 blocks are non-zero!
    # Yes, because A[i, j] is zero if j doesn't match the block? No, A is dense.
    # But B[j, k] is zero if j doesn't match? No, B is dense.
    # However, the summation is over j.
    # C[i, k] = A[i, 0]B[0, k] + A[i, 1]B[1, k] + A[i, 2]B[2, k] + A[i, 3]B[3, k].
    # We can group terms:
    # (A[i, 0]B[0, k] + A[i, 1]B[1, k]) + (A[i, 2]B[2, k] + A[i, 3]B[3, k]).
    # First part: A[0:2, 0:2] * B[0:2, k].
    # Second part: A[0:2, 2:4] * B[2:4, k].
    # So for each k (or pair of k), we have two 2x2x2 multiplications.
    # 3 pairs of k -> 3 * 2 = 6 blocks.
    # Each block rank 7.
    # Total rank 42.
    
    # So the previous construction is correct and minimal for this decomposition.
    
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for a_rows, a_cols, b_rows, b_cols in blocks:
        if not _time_ok(deadline):
            break
        
        a_map = {
            0: a_rows[0] * m + a_cols[0],
            1: a_rows[0] * m + a_cols[1],
            2: a_rows[1] * m + a_cols[0],
            3: a_rows[1] * m + a_cols[1]
        }
        b_map = {
            0: b_rows[0] * p + b_cols[0],
            1: b_rows[0] * p + b_cols[1],
            2: b_rows[1] * p + b_cols[0],
            3: b_rows[1] * p + b_cols[1]
        }
        w_map = {
            0: b_cols[0] * n + a_rows[0],
            1: b_cols[1] * n + a_rows[0],
            2: b_cols[0] * n + a_rows[1],
            3: b_cols[1] * n + a_rows[1]
        }
        
        for u_coeffs, v_coeffs, w_coeffs in strassen:
            if not _time_ok(deadline):
                break
            for local_idx, coeff in u_coeffs:
                U[a_map[local_idx], col] += coeff
            for local_idx, coeff in v_coeffs:
                V[b_map[local_idx], col] += coeff
            for local_idx, coeff in w_coeffs:
                W[w_map[local_idx], col] += coeff
            col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
