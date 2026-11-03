import time

DIMENSIONS = [2, 4, 6]

def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0])
    for u_idx in range(n * m):
        for v_idx in range(m * p):
            for w_idx in range(p * n):
                val = sum(U[u_idx][r] * V[v_idx][r] * W[w_idx][r] for r in range(rank))
                i = u_idx // m
                j = u_idx % m
                j2 = v_idx // p
                k = v_idx % p
                k2 = w_idx // n
                i2 = w_idx % n
                if i == i2 and j == j2 and k == k2:
                    if val != 1:
                        return False
                else:
                    if val != 0:
                        return False
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block definitions: (a_rows, a_cols, b_rows, b_cols)
    # A is 2x4. B is 4x6.
    # We split A into left (cols 0-1) and right (cols 2-3)
    # We split B into top (rows 0-1) and bottom (rows 2-3)
    # And B columns into 3 blocks of 2
    
    blocks = [
        # Block 0: A_left * B_top, cols 0-1
        ([0, 1], [0, 1], [0, 1], [0, 1]),
        # Block 1: A_left * B_top, cols 2-3
        ([0, 1], [0, 1], [0, 1], [2, 3]),
        # Block 2: A_left * B_top, cols 4-5
        ([0, 1], [0, 1], [0, 1], [4, 5]),
        # Block 3: A_right * B_bottom, cols 0-1
        ([0, 1], [2, 3], [2, 3], [0, 1]),
        # Block 4: A_right * B_bottom, cols 2-3
        ([0, 1], [2, 3], [2, 3], [2, 3]),
        # Block 5: A_right * B_bottom, cols 4-5
        ([0, 1], [2, 3], [2, 3], [4, 5]),
    ]
    
    # Strassen 7 multiplications for 2x2x2
    # local_a: 0=A11, 1=A12, 2=A21, 3=A22
    # local_b: 0=B11, 1=B12, 2=B21, 3=B22
    # local_z: 0=Z11, 1=Z12, 2=Z21, 3=Z22
    strassen = [
        # M1 = (A11+A22)(B11+B22) -> Z11 += M1, Z22 += M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2 = (A21+A22)B11 -> Z21 += M2, Z22 -= M2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3 = A11(B22-B11) -> Z12 += M3, Z22 += M3
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4 = A22(B11-B21) -> Z11 += M4, Z21 += M4
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5 = (A11+A12)B22 -> Z11 -= M5, Z12 += M5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6 = (A21-A11)(B11+B12) -> Z21 += M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7 = (A12-A22)(B21+B22) -> Z11 += M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    col = 0
    for block in blocks:
        if time.time() > deadline:
            break
        
        a_rows, a_cols, b_rows, b_cols = block
        
        # Map local indices to global indices
        a_map = {
            0: a_rows[0] * m + a_cols[0],
            1: a_rows[0] * m + a_cols[1],
            2: a_rows[1] * m + a_cols[0],
            3: a_rows[1] * m + a_cols[1],
        }
        
        b_map = {
            0: b_rows[0] * p + b_cols[0],
            1: b_rows[0] * p + b_cols[1],
            2: b_rows[1] * p + b_cols[0],
            3: b_rows[1] * p + b_cols[1],
        }
        
        # W index: k * n + i where k is column of B, i is row of A
        w_map = {
            0: b_cols[0] * n + a_rows[0],
            1: b_cols[1] * n + a_rows[0],
            2: b_cols[0] * n + a_rows[1],
            3: b_cols[1] * n + a_rows[1],
        }
        
        for u_coeffs, v_coeffs, w_coeffs in strassen:
            for local_idx, coeff in u_coeffs:
                global_idx = a_map[local_idx]
                U[global_idx][col] += coeff
            
            for local_idx, coeff in v_coeffs:
                global_idx = b_map[local_idx]
                V[global_idx][col] += coeff
            
            for local_idx, coeff in w_coeffs:
                global_idx = w_map[local_idx]
                W[global_idx][col] += coeff
            
            col += 1
    
    # Verify correctness
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Invalid decomposition")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
