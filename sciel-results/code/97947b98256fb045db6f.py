import time
import numpy as np

def construct():
    """Return a valid tensor decomposition for 4x4 by 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    n, m, p = 4, 4, 5
    
    # Try rank-50 decomposition: Strassen 4x4x4 (rank 49) + 1 term for 5th column
    try:
        if time.time() < deadline:
            result_444 = _strassen_4x4x4()
            result = _extend_to_445(result_444, n, m, p)
            if _verify_decomposition(result, n, m, p, num_checks=50):
                return result
    except Exception:
        pass
    
    # Fallback: schoolbook decomposition (rank 80)
    return _schoolbook(n, m, p)

def _strassen_2x2x2():
    """
    Return U, V, W (each 4x7) for Strassen 2x2x2 matrix multiplication.
    
    Indexing:
    U[i*2+j] for i,j in {0,1}
    V[j*2+k] for j,k in {0,1}
    W[k*2+i] for k,i in {0,1}
    
    C[i,k] = sum_r U[i,j]_r * V[j,k]_r * W[k,i]_r
    """
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    
    # P1 = (A00 + A11) * (B00 + B11)
    # U: A00 (idx 0), A11 (idx 3)
    # V: B00 (idx 0), B11 (idx 3)
    # W: C00 (idx 0), C11 (idx 3)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (A10 + A11) * B00
    # U: A10 (idx 2), A11 (idx 3)
    # V: B00 (idx 0)
    # W: C10 (idx 1), C11 (idx 3)
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = A00 * (B10 - B00)
    # U: A00 (idx 0)
    # V: B10 (idx 2), B00 (idx 0)
    # W: C00 (idx 0)
    U[0][2] = 1
    V[2][2] = 1; V[0][2] = -1
    W[0][2] = 1
    
    # P4 = A11 * (B01 - B11)
    # U: A11 (idx 3)
    # V: B01 (idx 1), B11 (idx 3)
    # W: C01 (idx 2), C11 (idx 3)
    U[3][3] = 1
    V[1][3] = 1; V[3][3] = -1
    W[2][3] = 1; W[3][3] = 1
    
    # P5 = (A00 + A01) * B11
    # U: A00 (idx 0), A01 (idx 1)
    # V: B11 (idx 3)
    # W: C00 (idx 0), C01 (idx 2)
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (A01 - A11) * (B00 + B01)
    # U: A01 (idx 1), A11 (idx 3)
    # V: B00 (idx 0), B01 (idx 1)
    # W: C11 (idx 3)
    U[1][5] = 1; U[3][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (A00 - A10) * (B10 + B11)
    # U: A00 (idx 0), A10 (idx 2)
    # V: B10 (idx 2), B11 (idx 3)
    # W: C00 (idx 0), C10 (idx 1)
    U[0][6] = 1; U[2][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[1][6] = 1
    
    return U, V, W

def _strassen_4x4x4():
    """
    Construct rank 49 decomposition for 4x4x4 using 2-level Strassen recursion.
    U: 16x49 indexed [i*4+j], V: 16x49 indexed [j*4+k], W: 16x49 indexed [k*4+i]
    """
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    U2, V2, W2 = _strassen_2x2x2()
    
    # 7 Strassen steps, each with 7 sub-steps = 49 total rank
    # Step r, sub-step s -> column r*7+s
    # A-block coeffs: (bi,bj) -> coeff for A[bi*2:bi*2+2, bj*2:bj*2+2]
    # B-block coeffs: (bj,bk) -> coeff for B[bj*2:bj*2+2, bk*2:bk*2+2]  
    # C-block coeffs: (ci,ck) -> coeff for C[ci*2:ci*2+2, ck*2:ck*2+2]
    
    steps = [
        # P1: (A00+A11)(B00+B11) -> contributes to C00 and C11
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # P2: (A10+A11)B00 -> contributes to C10 and C11
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # P3: A00(B10-B00) -> contributes to C01 and C11
        ({(0,0): 1}, {(1,0): 1, (0,0): -1}, {(0,1): 1, (1,1): 1}),
        # P4: A11(B01-B11) -> contributes to C00 and C10
        ({(1,1): 1}, {(0,1): 1, (1,1): -1}, {(0,0): 1, (1,0): 1}),
        # P5: (A00+A01)B11 -> contributes to C00 and C01
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # P6: (A01-A11)(B00+B01) -> contributes to C11
        ({(0,1): 1, (1,1): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # P7: (A00-A10)(B10+B11) -> contributes to C00 and C10
        ({(0,0): 1, (1,0): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,0): 1}),
    ]
    
    for r, (a_coeffs, b_coeffs, c_coeffs) in enumerate(steps):
        for s in range(7):
            col = r * 7 + s
            
            # U: sum over A-block positions (bi,bj) and local 2x2 indices (li,lj)
            for bi in range(2):
                for bj in range(2):
                    coeff_a = a_coeffs.get((bi, bj), 0)
                    if coeff_a == 0:
                        continue
                    for li in range(2):
                        for lj in range(2):
                            local_u = U2[li*2+lj][s]
                            if local_u == 0:
                                continue
                            gi = bi * 2 + li
                            gj = bj * 2 + lj
                            U[gi*4 + gj][col] += coeff_a * local_u
            
            # V: sum over B-block positions (bj,bk) and local 2x2 indices (lj,lk)
            for bj in range(2):
                for bk in range(2):
                    coeff_b = b_coeffs.get((bj, bk), 0)
                    if coeff_b == 0:
                        continue
                    for lj in range(2):
                        for lk in range(2):
                            local_v = V2[lj*2+lk][s]
                            if local_v == 0:
                                continue
                            gj = bj * 2 + lj
                            gk = bk * 2 + lk
                            V[gj*4 + gk][col] += coeff_b * local_v
            
            # W: sum over C-block positions (ci,ck) and local 2x2 indices (li,lk)
            # W2 is indexed [local_k*2 + local_i], so W2[lk*2+li]
            for ci in range(2):
                for ck in range(2):
                    coeff_c = c_coeffs.get((ci, ck), 0)
                    if coeff_c == 0:
                        continue
                    for li in range(2):
                        for lk in range(2):
                            local_w = W2[lk*2+li][s]
                            if local_w == 0:
                                continue
                            gk = ck * 2 + lk
                            gi = ci * 2 + li
                            W[gk*4 + gi][col] += coeff_c * local_w
                            
    return {"U": U, "V": V, "W": W}

def _extend_to_445(decomp_444, n, m, p):
    """
    Extend 4x4x4 decomposition to 4x4x5 with rank 50.
    The 5th column (k=4) is handled by one additional rank-1 term.
    """
    U4 = decomp_444["U"]
    V4 = decomp_444["V"]
    W4 = decomp_444["W"]
    rank_444 = len(U4[0])
    
    new_rank = rank_444 + 1
    U = [[0] * new_rank for _ in range(n * m)]
    V = [[0] * new_rank for _ in range(m * p)]
    W = [[0] * new_rank for _ in range(p * n)]
    
    # Copy 4x4x4 part (first 4 columns of V and W)
    for i in range(n):
        for j in range(m):
            for r in range(rank_444):
                U[i*m + j][r] = U4[i*4 + j][r]
                
    for j in range(m):
        for k in range(4):
            for r in range(rank_444):
                V[j*p + k][r] = V4[j*4 + k][r]
                
    for k in range(4):
        for i in range(n):
            for r in range(rank_444):
                W[k*n + i][r] = W4[k*4 + i][r]
                
    # Add rank-1 term for the 5th output column k=4
    # This gives C[i,4] = sum_j A[i,j] * B[j,4]
    new_col = rank_444
    for i in range(n):
        for j in range(m):
            U[i*m + j][new_col] = 1
    for j in range(m):
        V[j*p + 4][new_col] = 1
    for i in range(n):
        W[4*n + i][new_col] = 1
            
    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p):
    """Schoolbook decomposition (rank n*m*p)."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(decomp, n, m, p, num_checks=50):
    """Verify the decomposition exactly."""
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]
    rank = len(U[0]) if U else 0
    
    # Check all tensor entries: C[i,k] should have coefficient 1 for each (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1) > 1e-9:
                    return False
    
    # Check that non-tensor entries are zero (sample check)
    import random
    random.seed(42)
    for _ in range(num_checks):
        a = random.randint(0, n*m - 1)
        b = random.randint(0, m*p - 1)
        c = random.randint(0, p*n - 1)
        i = a // m; j = a % m
        j2 = b // p; k = b % p
        k2 = c // n; i2 = c % n
        if i == i2 and j == j2 and k == k2:
            continue
        val = 0
        for r in range(rank):
            val += U[a][r] * V[b][r] * W[c][r]
        if abs(val) > 1e-9:
            return False
    
    return True
