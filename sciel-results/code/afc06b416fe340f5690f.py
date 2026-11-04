import time
import numpy as np
from fractions import Fraction

def construct():
    t0 = time.time()
    deadline = t0 + 135.0
    
    def _elapsed_ok():
        return (time.time() - t0) < deadline
    
    n, m, p = 2, 3, 8
    nm = n * m  # 6
    mp = m * p  # 24
    pn = p * n  # 16
    
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
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
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    # Build 2x3x2 decomposition with rank 10
    # Columns 0-6: Strassen terms for a=0,1
    # Columns 7-9: Extra terms for a=2
    rank_232 = 10
    
    U_232 = [[0] * rank_232 for _ in range(n * m)]  # 6 x 10
    V_232 = [[0] * rank_232 for _ in range(m * 2)]   # 6 x 10 (a*2+k)
    W_232 = [[0] * rank_232 for _ in range(2 * n)]   # 4 x 10 (k*n+i)
    
    # Fill Strassen parts for a=0,1 (columns 0-6)
    for r in range(7):
        for li in range(2):
            for la in range(2):
                u_val = S_U[r][li * 2 + la]
                if u_val != 0:
                    U_232[li * m + la][r] = u_val
        for la in range(2):
            for lk in range(2):
                v_val = S_V[r][la * 2 + lk]
                if v_val != 0:
                    V_232[la * 2 + lk][r] = v_val
        for lk in range(2):
            for li in range(2):
                w_val = S_W[r][lk * 2 + li]
                if w_val != 0:
                    W_232[lk * n + li][r] = w_val
    
    # For a=2 (columns 7-9), we need:
    # sum_{c=0}^{9} U[0,2,c] * V[2,k,c] * W[k,0,c] = 1
    # sum_{c=0}^{9} U[1,2,c] * V[2,k,c] * W[k,1,c] = 1
    # sum_{c=0}^{9} U[0,2,c] * V[2,k,c] * W[k,1,c] = 0
    # sum_{c=0}^{9} U[1,2,c] * V[2,k,c] * W[k,0,c] = 0
    
    # W values for k=0: W[0,0]=[1,0,0,1,-1,0,1], W[1,0]=[0,1,0,1,0,0,0]
    # W values for k=1: W[0,1]=[0,0,1,0,1,0,0], W[1,1]=[0,-1,1,0,0,1,0]
    
    # Set U[0,2] and U[1,2] to use only extra columns (7,8,9)
    U_232[0 * m + 2][7] = 1
    U_232[0 * m + 2][8] = 1
    U_232[0 * m + 2][9] = 0
    U_232[1 * m + 2][7] = 0
    U_232[1 * m + 2][8] = 0
    U_232[1 * m + 2][9] = 1
    
    # Now choose V[2,k] for k=0,1 to satisfy constraints
    # For k=0: Need U[0,2]·V[2,0]·W[:,0] = [1,0] and U[1,2]·V[2,0]·W[:,0] = [0,0]
    # Since U[0,2] uses cols 7,8 and U[1,2] uses col 9:
    # V[2,0,7]*W[0,0,7] + V[2,0,8]*W[0,1,7] = 1  (for i=0,k=0)
    # V[2,0,7]*W[1,0,7] + V[2,0,8]*W[1,1,7] = 0  (for i=1,k=0)
    # W[0,0,7]=1, W[0,1,7]=0, W[1,0,7]=0, W[1,1,7]=0
    # So: V[2,0,7] = 1, and no constraint on V[2,0,8] from i=1
    
    # For k=1: Need U[0,2]·V[2,1]·W[:,1] = [0,1] and U[1,2]·V[2,1]·W[:,1] = [1,0]
    # V[2,1,7]*W[0,0,7] + V[2,1,8]*W[0,1,7] = 0  (for i=0,k=1)
    # V[2,1,7]*W[1,0,7] + V[2,1,8]*W[1,1,7] = 1  (for i=1,k=1)
    # W[0,0,7]=1, W[0,1,7]=0, W[1,0,7]=0, W[1,1,7]=0
    # So: V[2,1,7] = 0, V[2,1,8] = 1
    
    # But we also need U[1,2]·V[2,0]·W[:,0] = [0,0] and U[1,2]·V[2,1]·W[:,1] = [1,0]
    # U[1,2] only has col 9, so V[2,0,9] and V[2,1,9] must be 0 for W[:,0] and W[:,1]
    # But W[k,i,9] for k=0,1 and i=0,1 are all 0 (Strassen only has 7 terms)
    # So we need additional W values for cols 7,8,9
    
    # Let me restart the 2x3x2 design with proper W values
    
    # Clear and rebuild
    U_232 = [[0] * rank_232 for _ in range(n * m)]
    V_232 = [[0] * rank_232 for _ in range(m * 2)]
    W_232 = [[0] * rank_232 for _ in range(2 * n)]
    
    # Strassen for a=0,1 (cols 0-6)
    for r in range(7):
        for li in range(2):
            for la in range(2):
                u_val = S_U[r][li * 2 + la]
                if u_val != 0:
                    U_232[li * m + la][r] = u_val
        for la in range(2):
            for lk in range(2):
                v_val = S_V[r][la * 2 + lk]
                if v_val != 0:
                    V_232[la * 2 + lk][r] = v_val
        for lk in range(2):
            for li in range(2):
                w_val = S_W[r][lk * 2 + li]
                if w_val != 0:
                    W_232[lk * n + li][r] = w_val
    
    # For a=2, use cols 7-9 with simple schoolbook-like terms
    # Term 7: (i=0,k=0), Term 8: (i=0,k=1), Term 9: (i=1,k=0)
    # But we also need (i=1,k=1), so we need 4 terms for a=2?
    # No, we can use Strassen-style for a=2 as well
    
    # Actually, for 2x3x2, the minimal rank is 10. Let me use:
    # 7 Strassen terms + 3 extra terms where the extra terms handle a=2
    # with W values that are non-zero only in the extra columns
    
    # Set W for extra columns (7,8,9) to be standard basis
    W_232[0 * n + 0][7] = 1  # k=0, i=0
    W_232[1 * n + 0][8] = 1  # k=1, i=0
    W_232[0 * n + 1][9] = 1  # k=0, i=1
    W_232[1 * n + 1][10] = 1 # k=1, i=1 - but we only have 10 columns!
    
    # I need 4 extra terms for a=2 (one for each (i,k) pair), so rank 11
    rank_232 = 11
    U_232 = [[0] * rank_232 for _ in range(n * m)]
    V_232 = [[0] * rank_232 for _ in range(m * 2)]
    W_232 = [[0] * rank_232 for _ in range(2 * n)]
    
    # Strassen for a=0,1 (cols 0-6)
    for r in range(7):
        for li in range(2):
            for la in range(2):
                u_val = S_U[r][li * 2 + la]
                if u_val != 0:
                    U_232[li * m + la][r] = u_val
        for la in range(2):
            for lk in range(2):
                v_val = S_V[r][la * 2 + lk]
                if v_val != 0:
                    V_232[la * 2 + lk][r] = v_val
        for lk in range(2):
            for li in range(2):
                w_val = S_W[r][lk * 2 + li]
                if w_val != 0:
                    W_232[lk * n + li][r] = w_val
    
    # For a=2, use cols 7-10 with schoolbook
    U_232[0 * m + 2][7] = 1
    U_232[0 * m + 2][8] = 1
    U_232[1 * m + 2][9] = 1
    U_232[1 * m + 2][10] = 1
    
    V_232[2 * 2 + 0][7] = 1
    V_232[2 * 2 + 1][8] = 1
    V_232[2 * 2 + 0][9] = 1
    V_232[2 * 2 + 1][10] = 1
    
    W_232[0 * n + 0][7] = 1
    W_232[1 * n + 0][8] = 1
    W_232[0 * n + 1][9] = 1
    W_232[1 * n + 1][10] = 1
    
    # Now build 2x3x8 by tiling 2x3x2 four times
    rank_238 = rank_232 * 4  # 44
    U_238 = [[0] * rank_238 for _ in range(nm)]
    V_238 = [[0] * rank_238 for _ in range(mp)]
    W_238 = [[0] * rank_238 for _ in range(pn)]
    
    for block in range(4):
        k0 = block * 2
        for c in range(rank_232):
            col = block * rank_232 + c
            for i in range(nm):
                U_238[i][col] = U_232[i][c]
            for a in range(m):
                for k in range(2):
                    V_238[a * p + k0 + k][col] = V_232[a * 2 + k][c]
            for k in range(2):
                for i in range(n):
                    W_238[(k0 + k) * n + i][col] = W_232[k * n + i][c]
    
    candidate = {"U": U_238, "V": V_238, "W": W_238}
    if _validate(candidate):
        return candidate
    
    return _schoolbook()
