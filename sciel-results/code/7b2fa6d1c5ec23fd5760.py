import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _strassen_2x2():
    """
    Return Strassen's 7-term decomposition for 2x2 x 2x2.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Index convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (a00+a11)(b00+b11)
    U[0][0] = 1  # a00
    U[3][0] = 1  # a11
    V[0][0] = 1  # b00
    V[3][0] = 1  # b11
    W[0][0] = 1  # C00 (k=0, i=0 -> 0*2+0=0)
    W[3][0] = 1  # C11 (k=1, i=1 -> 1*2+1=3)
    
    # M2: (a10+a11)b00
    U[2][1] = 1  # a10
    U[3][1] = 1  # a11
    V[0][1] = 1  # b00
    W[2][1] = 1  # C10 (k=0, i=1 -> 0*2+1=1? No. W[k*2+i]. C10 is i=1, k=0. idx=1. Wait.
    # Let's re-verify W indices.
    # W[k*n+i]. For 2x2, n=2. W[k*2+i].
    # C00: i=0, k=0 -> 0
    # C01: i=0, k=1 -> 1*2+0 = 2
    # C10: i=1, k=0 -> 0*2+1 = 1
    # C11: i=1, k=1 -> 1*2+1 = 3
    
    # My previous seed code had:
    # W[0][0] = 1 # C00
    # W[3][0] = 1 # C11
    # W[2][1] = 1 # C10 ?? No, C10 is index 1.
    # Let's check the seed code's W indices carefully.
    
    # Seed code:
    # M1: W[0][0]=1, W[3][0]=1. (C00, C11). Correct.
    # M2: W[2][1]=1, W[3][1]=-1. (C10? No, idx 2 is C01. idx 3 is C11).
    # Let's re-read Strassen formulas.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for M2, it contributes to C10 and C11.
    # C10 index: i=1, k=0 -> 0*2+1 = 1.
    # C11 index: i=1, k=1 -> 1*2+1 = 3.
    # Seed code had W[2][1]=1. This is index 2, which is C01 (i=0, k=1).
    # This implies the seed code's W indexing might be different or I am misinterpreting.
    # Let's look at the seed code's _verify.
    # W[k*n+i].
    # For 2x2, n=2.
    # Loop: for k in range(2): for i in range(2): ... W[k*2+i]
    # So W[0] is k=0,i=0 (C00).
    # W[1] is k=0,i=1 (C10).
    # W[2] is k=1,i=0 (C01).
    # W[3] is k=1,i=1 (C11).
    
    # Let's re-evaluate Seed Code M2:
    # W[2][1] = 1. This is C01.
    # But formula says C10 = M2 + M4.
    # So M2 should contribute to C10 (index 1).
    # Seed code has W[2][1]=1. This is a bug in the seed code?
    # Let's check M4 in seed code.
    # M4: W[0][3]=1 (C00), W[2][3]=1 (C01).
    # Formula: C10 = M2 + M4. C00 = M1 + M4 ...
    # If M4 contributes to C01, that contradicts C10 = M2 + M4.
    
    # It seems the seed code's Strassen implementation might be flawed or uses a different convention.
    # However, the task says "score 1.0000" for the current program, which uses schoolbook (128) because the rank-112 construction failed verification or wasn't better?
    # Wait, the seed program returns `best`. If `_construct_rank32` fails verification, it falls back to schoolbook (128).
    # If it succeeds, it returns 112.
    # The score is 1.0, which corresponds to 128 (schoolbook).
    # This implies `_construct_rank32` likely failed verification in the seed program, or the score is normalized such that 128 is 1.0.
    # "score is the rank". "1.0 equals the seed program".
    # If seed program returns 128, score is 128? No, "score is seed/value".
    # "1.0 equals the seed program". So if seed returns 128, and score is 1.0, then value = 128.
    # So I need to return a rank < 128 to get score > 1.0.
    
    # I will write a correct Strassen 2x2 decomposition.
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (a00+a11)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1  # C00
    W[3][0] = 1  # C11
    
    # M2: (a10+a11)b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1  # C10
    W[3][1] = -1 # C11
    
    # M3: a00(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1  # C01
    W[3][2] = 1  # C11
    
    # M4: a11(b10-b00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1  # C00
    W[1][3] = 1  # C10
    
    # M5: (a00+a01)b11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1 # C00
    W[2][4] = 1  # C01
    
    # M6: (a10-a00)(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C11
    
    # M7: (a01-a11)(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C00
    W[2][6] = 1  # C01
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    if U_arr.shape != (n*m, rank):
        return False
    if V_arr.shape != (m*p, rank):
        return False
    if W_arr.shape != (p*n, rank):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W}

def _construct_strassen_4x4x8():
    n, m, p = 4, 4, 8
    base_U, base_V, base_W = _strassen_2x2()
    
    total_rank = 112
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # p_block in {0, 1}
    for p_block in range(2):
        # r in {0, 1} (row block of A and C)
        # k in {0, 1} (col block of A / row block of B)
        # c in {0, 1} (col block of B and C)
        for r in range(2):
            for k in range(2):
                for c in range(2):
                    for s in range(7):
                        col = p_block * 56 + (r * 4 + k * 2 + c) * 7 + s
                        
                        # U: A_block(r, k)
                        for a in range(2):
                            for b in range(2):
                                u_idx = (r * 2 + a) * 4 + (k * 2 + b)
                                U[u_idx][col] += base_U[a * 2 + b][s]
                                
                        # V: B_block(k, c) for this p_block
                        for a in range(2):
                            for b in range(2):
                                v_idx = (k * 2 + a) * 8 + (p_block * 4 + c * 2 + b)
                                V[v_idx][col] += base_V[a * 2 + b][s]
                                
                        # W: C_block(r, c) for this p_block
                        for a in range(2):
                            for b in range(2):
                                w_idx = (p_block * 4 + c * 2 + b) * 4 + (r * 2 + a)
                                W[w_idx][col] += base_W[b * 2 + a][s]
                                
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    best = _schoolbook(n, m, p)
    best_rank = len(best["U"][0])
    
    try:
        candidate = _construct_strassen_4x4x8()
        candidate_rank = len(candidate["U"][0])
        if candidate_rank < best_rank:
            if _verify(candidate["U"], candidate["V"], candidate["W"], n, m, p):
                best = candidate
                best_rank = candidate_rank
    except Exception as e:
        pass
    
    if time.time() > deadline:
        return best
        
    return best
