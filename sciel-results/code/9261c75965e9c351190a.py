"""
Strategy: Start with 44-term decomposition (Strassen 2x2x8 + schoolbook 2x1x8),
then attempt to reduce rank to 28 by absorbing the 2x1x8 part into the Strassen terms.
"""
import time
import numpy as np
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

    # Build 28-term Strassen basis for 2x2x8
    rank_s = 28
    U_s = [[0] * rank_s for _ in range(n * m)]
    V_s = [[0] * rank_s for _ in range(m * p)]
    W_s = [[0] * rank_s for _ in range(p * n)]

    col = 0
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    u_val = S_U[r][li * 2 + la]
                    if u_val != 0:
                        U_s[li * m + la][col] = u_val
            for la in range(2):
                for lk in range(2):
                    v_val = S_V[r][la * 2 + lk]
                    if v_val != 0:
                        V_s[la * p + (k0 + lk)][col] = v_val
            for lk in range(2):
                for li in range(2):
                    w_val = S_W[r][lk * 2 + li]
                    if w_val != 0:
                        W_s[(k0 + lk) * n + li][col] = w_val
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Build 44-term decomposition (Strassen 28 + schoolbook 16)
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

    # Try to reduce rank to 28 by absorbing 2x1x8 into Strassen terms
    # We need to find U[i,2,r] and V[2,k,r] such that:
    # For each (i,k): sum_r U[i,2,r] * V[2,k,r] * W[k,i,r] = 1
    # where W[k,i,r] are the existing W values from Strassen (for a=0,1 parts)
    
    if _elapsed_ok():
        # W values for the 2x2x8 part: W_s[k*n + i][r] for r in 0..27
        # We need: for each i in {0,1}, k in {0..7}:
        #   sum_{r=0}^{27} U[i,2,r] * V[2,k,r] * W_s[k*n+i][r] = 1
        
        # This is a bilinear system. Let's try a simple approach:
        # Assume U[i,2,r] = alpha[i,r] and V[2,k,r] = beta[k,r]
        # Then we need: sum_r alpha[i,r] * beta[k,r] * W_s[k*n+i][r] = 1
        
        # One simple solution: if we can find a single term r* where W_s[k*n+i][r*] != 0 for all (i,k),
        # then set alpha[i,r*] = 1/W_s[k*n+i][r*] ... but this must be consistent across k.
        
        # Better approach: Use the fact that for each block of 2 k-values, we have a 2x2 Strassen block.
        # For block b (k0 = 2b, k1 = 2b+1), the W values are:
        # W[k0, i, r] and W[k1, i, r] for r in the 7 terms of this block.
        
        # Let's try: for each block b, use the 7 terms of that block to satisfy the 4 equations
        # (i=0,1) x (k=k0,k1). We have 7*2=14 variables (alpha[i,r] for 2 i's and 7 r's, 
        # beta[k,r] for 2 k's and 7 r's) but they're coupled.
        
        # Simpler: Try setting U[i,2,r] and V[2,k,r] to be non-zero only for specific r values.
        # For example, use r=0 of each block (M1 term).
        
        # For M1 term of block b: W_s[k0,0,b*7] = 1, W_s[k0,1,b*7] = 1, 
        #                        W_s[k1,0,b*7] = 1, W_s[k1,1,b*7] = 1
        # So if we set U[i,2,b*7] = 1 and V[2,k,b*7] = 1 for all i,k,
        # then sum_r U[i,2,r]*V[2,k,r]*W[k,i,r] includes 1*1*1 = 1 from this term.
        # But we also need to ensure no other terms contribute to a=0,1 parts.
        
        # Wait, this would add to the a=0,1 parts too! We need to be careful.
        
        # Actually, the constraint is that for a=0,1, the sum must remain 1.
        # If we add U[i,2,r] and V[2,k,r], they don't affect the a=0,1 sums because
        # those use U[i,0,r], U[i,1,r] and V[0,k,r], V[1,k,r].
        
        # So we can freely set U[i,2,r] and V[2,k,r] without affecting a=0,1!
        # We just need: sum_r U[i,2,r] * V[2,k,r] * W_s[k*n+i][r] = 1 for all (i,k).
        
        # Simple solution: For each block b, use the M1 term (r = b*7).
        # W_s[k, i, b*7] = 1 for k in {k0, k1} and i in {0, 1}.
        # Set U[i,2,b*7] = 1 and V[2,k,b*7] = 1 for all i, k.
        # Then the contribution is 1*1*1 = 1 for each (i,k) in this block.
        
        # But wait, this means for each (i,k), only one term contributes, giving exactly 1.
        # Let's verify: For (i,k) in block b, the term r=b*7 gives 1*1*1=1.
        # All other terms have U[i,2,r]=0 or V[2,k,r]=0, so they contribute 0.
        # Perfect!
        
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
        
        # Add absorption: use M1 term of each block
        for block_idx in range(4):
            r_m1 = block_idx * 7  # M1 term index
            k0 = block_idx * 2
            for i in range(2):
                U28[i * m + 2][r_m1] = 1
            for k in range(k0, k0 + 2):
                V28[2 * p + k][r_m1] = 1
        
        candidate = {"U": U28, "V": V28, "W": W28}
        if _validate(candidate):
            return candidate
    
    # If rank 28 doesn't work, try rank 44
    candidate = {"U": U44, "V": V44, "W": W44}
    if _validate(candidate):
        return candidate
    return _schoolbook()

# EVOLVE-BLOCK-END
