"""
Strategy: Use Strassen 2x2x2 decomposition for 2x2x8 part (rank 28),
then absorb the 2x1x8 part into the existing terms using M1 of each block.
This should yield rank 28 if the absorption works.
"""
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

    # Schoolbook fallback
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2 decomposition
    # U indices: (i,a) -> i*2+a for i,a in {0,1}
    # V indices: (a,k) -> a*2+k for a,k in {0,1}
    # W indices: (k,i) -> k*2+i for k,i in {0,1}
    
    S_U = [
        [1, 0, 0, 1],   # M1: A00 + A11
        [0, 0, 1, 1],   # M2: A10 + A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00 + A01
        [-1, 0, 1, 0],  # M6: A10 - A00
        [0, 1, 0, -1]   # M7: A01 - A11
    ]
    S_V = [
        [1, 0, 0, 1],   # M1: B00 + B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01 - B11
        [-1, 0, 1, 0],  # M4: B10 - B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00 + B01
        [0, 0, 1, 1]    # M7: B10 + B11
    ]
    S_W = [
        [1, 0, 0, 1],   # M1: C00 + C01 + C10 + C11... 
        # Wait, let me re-derive Strassen properly
    ]
    
    # Let me use the standard Strassen formulas:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Where:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B01-B11)
    # M4 = (A11)(B10-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U vectors (for A coefficients):
    S_U = [
        [1, 0, 0, 1],   # M1: A00 + A11
        [0, 0, 1, 1],   # M2: A10 + A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00 + A01
        [-1, 0, 1, 0],  # M6: A10 - A00
        [0, 1, 0, -1]   # M7: A01 - A11
    ]
    # V vectors (for B coefficients):
    S_V = [
        [1, 0, 0, 1],   # M1: B00 + B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01 - B11
        [-1, 0, 1, 0],  # M4: B10 - B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00 + B01
        [0, 0, 1, 1]    # M7: B10 + B11
    ]
    # W vectors (for C coefficients):
    # C00 = M1 + M4 - M5 - M7 -> W for C00: [1, 0, 0, 1, -1, 0, -1]
    # C01 = M3 + M5 -> W for C01: [0, 0, 1, 0, 1, 0, 0]
    # C10 = M2 + M4 -> W for C10: [0, 1, 0, 1, 0, 0, 0]
    # C11 = M1 - M2 + M3 + M6 -> W for C11: [1, -1, 1, 0, 0, 1, 0]
    S_W = [
        [1, 0, 0, 1],   # M1
        [0, 1, 0, -1],  # M2
        [0, 0, 1, 1],   # M3
        [1, 1, 0, 0],   # M4
        [-1, 0, 1, 0],  # M5
        [0, 0, 0, 1],   # M6
        [1, 0, 0, 0]    # M7
    ]
    
    # Verify Strassen works for 2x2x2
    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][i * 2 + a]
                        v_val = S_V[r][a * 2 + k]
                        w_val = S_W[r][k * 2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # Build 28-term decomposition for 2x2x8
    rank_s = 28
    U_s = [[0] * rank_s for _ in range(n * m)]  # 6 rows (i*3+a), but we only fill a=0,1
    V_s = [[0] * rank_s for _ in range(m * p)]  # 24 rows (a*8+k), but we only fill a=0,1
    W_s = [[0] * rank_s for _ in range(p * n)]  # 16 rows (k*2+i)

    col = 0
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            # Fill U: i in {0,1}, a in {0,1}
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        U_s[li * m + la][col] = u_val
            # Fill V: a in {0,1}, k in {k0, k0+1}
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        V_s[la * p + (k0 + lk)][col] = v_val
            # Fill W: k in {k0, k0+1}, i in {0,1}
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        W_s[(k0 + lk) * n + li][col] = w_val
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Now try to absorb the 2x1x8 part into the 28 terms
    # We need: for each (i,k) with i in {0,1}, k in {0..7}:
    #   sum_r U[i*3+2, r] * V[2*8+k, r] * W[k*2+i, r] = 1
    # where W[k*2+i, r] are the existing values from the Strassen part
    
    # For each block b (k0=2b, k1=2b+1), the M1 term is at r=b*7
    # W_s[k, i, b*7] = S_W[0][k_local*2 + i] where k_local = k - k0
    # S_W[0] = [1, 0, 0, 1] which means W[0,0]=1, W[0,1]=0, W[1,0]=0, W[1,1]=1
    # So W_s[k0, 0, b*7] = 1, W_s[k0, 1, b*7] = 0, W_s[k1, 0, b*7] = 0, W_s[k1, 1, b*7] = 1
    
    # This doesn't give uniform 1s. Let me check other terms.
    # Actually, let me just try to solve this systematically.
    
    # For rank 28, we need to find U[i,2,r] and V[2,k,r] such that the constraint holds.
    # Let's try a simple approach: for each block, use multiple terms.
    
    U28 = [[0] * rank_s for _ in range(n * m)]
    V28 = [[0] * rank_s for _ in range(m * p)]
    W28 = [[0] * rank_s for _ in range(p * n)]
    
    # Copy Strassen part
    for r in range(rank_s):
        for i in range(n):
            for a in range(2):
                U28[i * m + a][r] = U_s[i * m + a][r]
        for a in range(2):
            for k in range(p):
                V28[a * p + k][r] = V_s[a * p + k][r]
        for k in range(p):
            for i in range(n):
                W28[k * n + i][r] = W_s[k * n + i][r]
    
    # Try to absorb: for each block, use the 7 terms to satisfy 4 equations
    # This is a bilinear system. Let's try setting values and checking.
    
    # Simple attempt: for each block b, set U[i,2,b*7+r] and V[2,k,b*7+r] 
    # to solve the 4x4 system using the 7 available terms.
    
    # Actually, let me just try the 44-term version first, then see if we can reduce.
    
    # Build 44-term decomposition
    rank_44 = 44
    U44 = [[0] * rank_44 for _ in range(n * m)]
    V44 = [[0] * rank_44 for _ in range(m * p)]
    W44 = [[0] * rank_44 for _ in range(p * n)]
    
    for r in range(rank_s):
        for i in range(n):
            for a in range(2):
                U44[i * m + a][r] = U_s[i * m + a][r]
        for a in range(2):
            for k in range(p):
                V44[a * p + k][r] = V_s[a * p + k][r]
        for k in range(p):
            for i in range(n):
                W44[k * n + i][r] = W_s[k * n + i][r]
    
    col = rank_s
    for i in range(n):
        for k in range(p):
            U44[i * m + 2][col] = 1
            V44[2 * p + k][col] = 1
            W44[k * n + i][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()
    
    candidate44 = {"U": U44, "V": V44, "W": W44}
    if _validate(candidate44):
        # Try to reduce to rank 28
        if _elapsed_ok():
            # For each block, we have 7 terms and need to satisfy 4 equations
            # Let's try to find U[i,2,r] and V[2,k,r] for r in the block's 7 terms
            # such that sum_r U[i,2,r] * V[2,k,r] * W[k,i,r] = 1
            
            # This is a bilinear system. Let's try a simple ansatz:
            # For each block b, use only 2 of the 7 terms.
            
            # Actually, let me try a different approach: 
            # The W values for the Strassen part are fixed. We need to find
            # U[i,2,r] and V[2,k,r] that work.
            
            # Let's try: for each (i,k), find a term r where W[k,i,r] != 0,
            # and set U[i,2,r] * V[2,k,r] = 1/W[k,i,r].
            
            # But this must be consistent across all (i,k).
            
            # Let me just try the rank 28 with the M1 term approach and validate.
            
            # For M1 term of block b (r = b*7):
            # W_s[k, i, b*7] = S_W[0][(k-k0)*2 + i]
            # S_W[0] = [1, 0, 0, 1]
            # So W_s[k0, 0, b*7] = 1, W_s[k0, 1, b*7] = 0
            #    W_s[k1, 0, b*7] = 0, W_s[k1, 1, b*7] = 1
            
            # This doesn't give a simple solution. Let me try all 7 terms per block.
            
            # For block b, the 7 terms have W values:
            # r=b*7+0: S_W[0] = [1,0,0,1] -> W[k0,0]=1, W[k0,1]=0, W[k1,0]=0, W[k1,1]=1
            # r=b*7+1: S_W[1] = [0,1,0,-1] -> W[k0,0]=0, W[k0,1]=1, W[k1,0]=0, W[k1,1]=-1
            # r=b*7+2: S_W[2] = [0,0,1,1] -> W[k0,0]=0, W[k0,1]=0, W[k1,0]=1, W[k1,1]=1
            # r=b*7+3: S_W[3] = [1,1,0,0] -> W[k0,0]=1, W[k0,1]=1, W[k1,0]=0, W[k1,1]=0
            # r=b*7+4: S_W[4] = [-1,0,1,0] -> W[k0,0]=-1, W[k0,1]=0, W[k1,0]=1, W[k1,1]=0
            # r=b*7+5: S_W[5] = [0,0,0,1] -> W[k0,0]=0, W[k0,1]=0, W[k1,0]=0, W[k1,1]=1
            # r=b*7+6: S_W[6] = [1,0,0,0] -> W[k0,0]=1, W[k0,1]=0, W[k1,0]=0, W[k1,1]=0
            
            # We need to find U[i,2,r] for i in {0,1} and V[2,k,r] for k in {k0,k1}
            # such that for each (i,k): sum_{r=0}^{6} U[i,2,b*7+r] * V[2,k,b*7+r] * W[k,i,b*7+r] = 1
            
            # This is 4 equations with 14 unknowns (2*7 for U, 2*7 for V).
            # Let's try to solve this by setting most to zero and finding a sparse solution.
            
            # Try: use only r=0 (M1) and r=3 (M4)
            # r=0: W[k0,0]=1, W[k0,1]=0, W[k1,0]=0, W[k1,1]=1
            # r=3: W[k0,0]=1, W[k0,1]=1, W[k1,0]=0, W[k1,1]=0
            
            # Let U[0,2,r0]=a, U[1,2,r0]=b, U[0,2,r3]=c, U[1,2,r3]=d
            # Let V[2,k0,r0]=e, V[2,k1,r0]=f, V[2,k0,r3]=g, V[2,k1,r3]=h
            
            # Equations:
            # (i=0,k=k0): a*e*1 + c*g*1 = 1
            # (i=1,k=k0): b*e*0 + d*g*1 = 1 -> d*g = 1
            # (i=0,k=k1): a*f*0 + c*h*0 = 1 -> 0 = 1 (IMPOSSIBLE)
            
            # So this doesn't work. Let me try a different combination.
            
            # Actually, let me just return the 44-term solution for now.
            return candidate44
    
    return _schoolbook()

# EVOLVE-BLOCK-END
