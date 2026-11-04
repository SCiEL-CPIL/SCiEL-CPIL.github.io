import time
import numpy as np

DIMENSIONS = [4, 4, 5]
DEADLINE = 130.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Return U, V, W (each 4x7) for Strassen 2x2x2 matrix multiplication."""
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    
    # Indices: A[i,j] -> U[i*2+j], B[j,k] -> V[j*2+k], C[i,k] -> W[k*2+i]
    # C[0,0] -> W[0], C[0,1] -> W[2], C[1,0] -> W[1], C[1,1] -> W[3]
    
    # P1 = (A00 + A11) * (B00 + B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (A10 + A11) * B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = A00 * (B10 - B00)
    U[0][2] = 1
    V[2][2] = 1; V[0][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = A11 * (B01 - B11)
    U[3][3] = 1
    V[1][3] = 1; V[3][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (A00 + A01) * B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[2][4] = 1; W[0][4] = -1
    
    # P6 = (A01 - A11) * (B00 + B01)
    U[1][5] = 1; U[3][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (A00 - A10) * (B10 + B11)
    U[0][6] = 1; U[2][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _strassen_4x4x4():
    """Construct rank 49 decomposition for 4x4x4."""
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    U2, V2, W2 = _strassen_2x2x2()
    
    steps = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(1,0): 1, (0,0): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(0,1): 1, (1,1): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        ({(0,1): 1, (1,1): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,0): 1, (1,0): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    for r, (a_coeffs, b_coeffs, c_coeffs) in enumerate(steps):
        for s in range(7):
            col = r * 7 + s
            
            for bi in range(2):
                for bj in range(2):
                    coeff_a = a_coeffs.get((bi, bj), 0)
                    if coeff_a == 0: continue
                    for li in range(2):
                        for lj in range(2):
                            local_u = U2[li*2+lj][s]
                            if local_u == 0: continue
                            global_i = bi*2 + li
                            global_j = bj*2 + lj
                            U[global_i*4 + global_j][col] += coeff_a * local_u
            
            for bj in range(2):
                for bk in range(2):
                    coeff_b = b_coeffs.get((bj, bk), 0)
                    if coeff_b == 0: continue
                    for lj in range(2):
                        for lk in range(2):
                            local_v = V2[lj*2+lk][s]
                            if local_v == 0: continue
                            global_j = bj*2 + lj
                            global_k = bk*2 + lk
                            V[global_j*4 + global_k][col] += coeff_b * local_v
            
            for ci in range(2):
                for ck in range(2):
                    coeff_c = c_coeffs.get((ci, ck), 0)
                    if coeff_c == 0: continue
                    for li in range(2):
                        for lk in range(2):
                            local_w = W2[lk*2+li][s]
                            if local_w == 0: continue
                            global_k = ck*2 + lk
                            global_i = ci*2 + li
                            W[global_k*4 + global_i][col] += coeff_c * local_w
                            
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(decomp, n, m, p, num_checks=50):
    """Verify the decomposition."""
    import random
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]
    rank = len(U[0]) if U else 0
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if abs(val - 1) > 1e-9:
                    return False
    
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
        val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
        if abs(val) > 1e-9:
            return False
    
    return True

def _try_rank_reduction(decomp, n, m, p, max_attempts=2000, timeout=60.0):
    """Try to find a lower-rank decomposition by column elimination."""
    start_time = time.time()
    U = [row[:] for row in decomp["U"]]
    V = [row[:] for row in decomp["V"]]
    W = [row[:] for row in decomp["W"]]
    rank = len(U[0])
    
    # Precompute the target tensor values
    # For each (a, b) pair where a=i*m+j, b=j*p+k, the valid c values are k*n+i
    # The tensor T[a, b, c] = 1 if c = k*n+i where a=i*m+j and b=j*p+k, else 0
    
    # For each pair (a, b), determine the valid c indices
    valid_c_for_ab = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_c_for_ab[(a, b)] = c
    
    # For each pair (a, b), the invalid c indices
    all_c_indices = set(range(p * n))
    invalid_c_for_ab = {}
    for (a, b) in valid_c_for_ab.keys():
        valid_c = valid_c_for_ab[(a, b)]
        invalid_c_for_ab[(a, b)] = [c for c in all_c_indices if c != valid_c]
    
    # Precompute the contribution of each rank term to each (a, b) pair
    # contrib[r][a][b] = U[a][r] * V[b][r]
    # This is O(rank * n*m * m*p)
    
    # Try to eliminate columns
    for attempt in range(max_attempts):
        if time.time() - start_time > timeout:
            break
        
        # Randomly pick a column to try to eliminate
        col_to_remove = attempt % rank
        if col_to_remove >= rank:
            col_to_remove = rank - 1
            
        # Try to express this column as a combination of other columns
        # We need to find coefficients alpha_r such that for all valid (a,b,c):
        #   sum_r alpha_r * U[a][r] * V[b][r] * W[c][r] = U[a][col_to_remove] * V[b][col_to_remove] * W[c][col_to_remove]
        # And for all invalid (a,b,c):
        #   sum_r alpha_r * U[a][r] * V[b][r] * W[c][r] = 0
        
        # This is a system of linear equations in the alpha_r variables
        # Let's set up the system
        
        other_cols = [r for r in range(rank) if r != col_to_remove]
        
        # Build the linear system
        # For each valid (a,b) pair and its valid c:
        #   sum_r alpha_r * U[a][r] * V[b][r] * W[c][r] = U[a][col_to_remove] * V[b][col_to_remove] * W[c][col_to_remove]
        
        # For each invalid (a,b) pair and each invalid c:
        #   sum_r alpha_r * U[a][r] * V[b][r] * W[c][r] = 0
        
        # This system is large, so let's use a simpler approach:
        # Try to find a column that is redundant (can be expressed as a linear combination)
        
        # Check if the column is all zeros
        if all(U[a][col_to_remove] == 0 for a in range(n*m)) and \
           all(V[b][col_to_remove] == 0 for b in range(m*p)) and \
           all(W[c][col_to_remove] == 0 for c in range(p*n)):
            # Column is zero, remove it
            for a in range(n*m):
                U[a].pop(col_to_remove)
            for b in range(m*p):
                V[b].pop(col_to_remove)
            for c in range(p*n):
                W[c].pop(col_to_remove)
            rank -= 1
            continue
        
        # Try a simple elimination: check if two columns are proportional
        found_elimination = False
        for other_col in other_cols:
            if other_col >= rank:
                continue
                
            # Check if column col_to_remove is a scalar multiple of other_col
            # Find a non-zero entry to determine the ratio
            ratio = None
            valid = True
            for a in range(n*m):
                if U[other_col][a] != 0:
                    if ratio is None:
                        ratio = U[a][col_to_remove] / U[a][other_col]
                    else:
                        if abs(U[a][col_to_remove] - ratio * U[a][other_col]) > 1e-9:
                            valid = False
                            break
                    break
            
            if not valid:
                continue
            
            # Check V and W
            for b in range(m*p):
                if V[other_col][b] != 0:
                    expected = ratio * V[other_col][b]
                    if abs(V[b][col_to_remove] - expected) > 1e-9:
                        valid = False
                        break
            
            if not valid:
                continue
            
            for c in range(p*n):
                if W[other_col][c] != 0:
                    expected = ratio * W[other_col][c]
                    if abs(W[c][col_to_remove] - expected) > 1e-9:
                        valid = False
                        break
            
            if valid:
                # Column is proportional, remove it
                for a in range(n*m):
                    U[a].pop(col_to_remove)
                for b in range(m*p):
                    V[b].pop(col_to_remove)
                for c in range(p*n):
                    W[c].pop(col_to_remove)
                rank -= 1
                found_elimination = True
                break
        
        if found_elimination:
            # Verify and continue
            if _verify_decomposition({"U": U, "V": V, "W": W}, n, m, p):
                continue
            else:
                # Shouldn't happen, but restore if it does
                break
    
    if rank < len(decomp["U"][0]):
        return {"U": U, "V": V, "W": W}
    
    return decomp

def construct():
    """Return a valid tensor decomposition for 4x4 by 4x5 matrix multiplication."""
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    try:
        result_444 = _strassen_4x4x4()
        
        # Extend to 4x4x5
        U4 = result_444["U"]
        V4 = result_444["V"]
        W4 = result_444["W"]
        rank_444 = len(U4[0])
        
        new_rank = rank_444 + 1
        U = [[0] * new_rank for _ in range(n * m)]
        V = [[0] * new_rank for _ in range(m * p)]
        W = [[0] * new_rank for _ in range(p * n)]
        
        for i in range(16):
            for r in range(rank_444):
                U[i][r] = U4[i][r]
                
        for j in range(4):
            for k in range(4):
                for r in range(rank_444):
                    V[j*5 + k][r] = V4[j*4 + k][r]
                    
        for k in range(4):
            for i in range(4):
                for r in range(rank_444):
                    W[k*4 + i][r] = W4[k*4 + i][r]
                    
        new_col = rank_444
        for i in range(4):
            for j in range(4):
                U[i*4 + j][new_col] = 1
            V[j*5 + 4][new_col] = 1
            W[4*4 + i][new_col] = 1
        
        result = {"U": U, "V": V, "W": W}
        
        if _verify_decomposition(result, n, m, p):
            # Try to reduce rank
            remaining_time = DEADLINE - (time.time() - start_time)
            if remaining_time > 5:
                result = _try_rank_reduction(result, n, m, p, timeout=min(remaining_time - 5, 30))
            
            if _verify_decomposition(result, n, m, p):
                return result
    
    except Exception as e:
        pass
    
    # Fallback: schoolbook decomposition (rank 80)
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

# EVOLVE-BLOCK-END
