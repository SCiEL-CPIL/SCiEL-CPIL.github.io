import numpy as np
import time

DIMENSIONS = [4, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 145.0

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    R = 44  # 28 (for 4x4x4) + 16 (for 4x4x2)
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    # Strassen 2x2x2 coefficients
    # A blocks: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 0, 1: 0, 2: 1, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1}
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 1, 3: 1}
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    col = 0
    # Part 1: 4x4x4 using 4 Strassen 2x2x2 decompositions (rank 28)
    # A is 4x4, split into 4 2x2 blocks: A[ar,br] where ar,br in {0,1}
    # B first 4 cols: 4x4, split into 4 2x2 blocks: B[ac,bc] where ac,bc in {0,1}, bc in {0,1} for cols 0-3
    # C first 4 cols: 4x4, split into 4 2x2 blocks: C[a,b] where a,b in {0,1}
    
    for ar in range(2):
        for ac in range(2):
            for s in range(7):
                # U: A block (ar, *)
                for a in range(4):
                    ca = alpha[s][a]
                    if ca == 0:
                        continue
                    x_row, x_col = a // 2, a % 2
                    # A[ar*2+x_row, x_col] within the 2x2 block starting at row ar*2
                    # Full A index: (ar*2 + x_row, x_col) for the 2x2 sub-block
                    # But we need A[ar, br] as 2x2 blocks
                    # A full index: row = ar*2 + x_row, col = x_col (for the left 2x2) or br*2+x_col
                    # Wait, A is 4x4. Block (ar, br) is A[ar*2:ar*2+2, br*2:br*2+2]
                    # For Strassen, we combine blocks. alpha[s] gives coefficients for A blocks.
                    # A block index a: a//2 = row block, a%2 = col block
                    ab_r, ab_c = a // 2, a % 2
                    # U index: A[ar*2+ab_r, ab_c*2+x_col]... 
                    # Actually U[i*m+k, t] where i is output row, k is inner dim
                    # U row index = i*m + k = i*4 + k
                    # For A, the "row" in U corresponds to output row i, "col" to inner k
                    # A[i, k] where i in 0..3, k in 0..3
                    # Block (ab_r, ab_c) of A: A[ab_r*2+u, ab_c*2+v]
                    # So U index = (ab_r*2 + x_row) * 4 + (ab_c*2 + x_col)
                    u_idx = (ab_r * 2 + x_row) * 4 + (ab_c * 2 + x_col)
                    U[u_idx, col] += ca
                
                # V: B block (*, ac) for cols 0-3
                for b in range(4):
                    cb = beta[s][b]
                    if cb == 0:
                        continue
                    y_row, y_col = b // 2, b % 2
                    # B block (ac, y_col): B[ac*2+v, y_col*2+w]
                    # V index = k*p + r = k*6 + r
                    # B[k, r] where k in 0..3, r in 0..5
                    # Block (ac, bc) of B: B[ac*2+v, bc*2+w]
                    # For first 4 cols, bc in {0,1}
                    v_idx = (ac * 2 + y_row) * 6 + (y_col * 2 + y_col)  # BUG: should be (bc*2 + w)
                    # Fix: v_idx = (ac*2 + y_row) * 6 + (y_col*2 + 0) for w=0, but we need all w
                    # Actually for the 2x2 block, we need all 4 entries
                    # V[k*p+r] = V[k*6+r], B[k,r]
                    # Block (ac, bc): rows ac*2..ac*2+1, cols bc*2..bc*2+1
                    # So for entry (y_row, y_col) in the block: k = ac*2+y_row, r = bc*2+y_col
                    # But bc = y_col here since b%2 = y_col is the col block index
                    # Wait, b is the B block index: b//2 = row block, b%2 = col block
                    # So bc = b%2 = y_col
                    v_idx = (ac * 2 + y_row) * 6 + (y_col * 2 + 0)
                    # This is wrong - we need to sum over all w in the 2x2 block
                    # Let me redo: V[k*p+r, t] where k is inner, r is output col
                    # B[k, r]. Block (ac, bc): k in {ac*2, ac*2+1}, r in {bc*2, bc*2+1}
                    # For the Strassen combination, beta[s][b] multiplies B block b
                    # B block b has 4 entries. We need V for each entry.
                    # So for each (v_row, v_col) in 2x2 block:
                    #   k = ac*2 + v_row, r = bc*2 + v_col
                    #   V[k*6+r, col] += cb
                    bc = y_col  # col block index of B
                    for v_row in range(2):
                        for v_col in range(2):
                            v_idx = (ac * 2 + v_row) * 6 + (bc * 2 + v_col)
                            V[v_idx, col] += cb
                    # But we also need the y_row, y_col to index into the Strassen combination
                    # Actually no - the Strassen combination is on the BLOCK level
                    # Each block is a 2x2 matrix. The Strassen formula combines blocks.
                    # So for B, the "inner" dimension is the row block (ac fixed), and 
                    # the "outer" is... hmm, I'm confusing myself.
                    
                    # Let me restart the V part more carefully.
                    # V[k*p+r, t]: k is the shared inner dimension (0..3), r is output col (0..5)
                    # B[k, r]
                    # For the 4x4x4 part, r in 0..3
                    # B block (ac, bc): B[ac*2+i, bc*2+j] for i,j in {0,1}
                    # In Strassen, beta[s][b] where b indexes B blocks
                    # b//2 = row block of B, b%2 = col block of B
                    # So for B block b: rows (b//2)*2 + i, cols (b%2)*2 + j
                    # V[(b//2)*2+i]*6 + (b%2)*2+j, col] += beta[s][b]
                    # But ac is fixed (we're computing C[ar, ac] block)
                    # So b//2 should equal ac? No, b//2 ranges over all row blocks of B
                    # The Strassen formula for C[ar,ac] uses all B row blocks
                    # So we don't fix b//2 = ac. Let me remove the ac constraint on b.
                    
                    # Actually I think my loop structure is wrong. Let me redo.
                    pass
                col += 1
    
    # This is getting complex. Let me use a cleaner approach.
    # Reset and try a direct construction.
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    col = 0
    
    # 4x4x4 part: rank 28
    # Split into 4 independent 2x2x2 Strassen problems
    # C[i,j] for i in 0..3, j in 0..3
    # Block (a,b) of C: C[a*2: a*2+2, b*2: b*2+2] = sum_k A[a*2:a*2+2, k] * B[k, b*2:b*2+2]
    # This is a 2x2 x 2x2 matrix multiplication with inner dim 4
    # Not directly 2x2x2. Need to split inner dim too.
    
    # Alternative: just use schoolbook for 4x4x4 (rank 64) and 4x4x2 (rank 16) = 80. Worse.
    
    # Let me try: 4x4x6 = sum of 4x4x1 six times, each rank 16, total 96. Worse.
    
    # Best known: use Strassen on 4x4x4 (rank 28) + schoolbook 4x4x2 (rank 16) = 44
    # For 4x4x4 with Strassen: split A into 2x2 blocks (4 blocks), B into 2x2 blocks (4 blocks)
    # C = sum of 16 block products, but Strassen reduces to 7 per "level"
    # Actually 4x4x4: split into 2x2 sub-blocks. A has 4 2x2 blocks, B has 4 2x2 blocks.
    # C has 4 2x2 blocks. Each C block = sum of 4 products of A blocks and B blocks.
    # Total 16 block multiplications, each 2x2x2 with rank 7, total 112. That's worse!
    
    # I was wrong. 4x4x4 via Strassen means treating it as (2x2 block matrix) * (2x2 block matrix)
    # where each block is 2x2. The Strassen algorithm for 2x2 block matrices uses 7 block
    # multiplications, each of which is a 2x2x2 matrix multiplication (rank 7).
    # Total rank = 7 * 7 = 49. Not 28.
    
    # Hmm, so 4x4x4 rank via Strassen is 49, not 28.
    # 4x4x4 schoolbook: 64. Strassen: 49. 
    # 4x4x2 schoolbook: 32.
    # Total: 49 + 32 = 81. Worse than 77!
    
    # Let me reconsider. The current program gets rank 77.
    # 4x4x6 schoolbook: 96.
    # Current: 77 via two-level Strassen.
    
    # What about: split 4x4x6 into two 4x4x3 problems.
    # 4x4x3: can we do better than schoolbook 48?
    # 4x4x3 schoolbook: 48. 
    # Two of them: 96. Worse.
    
    # Or: 2x4x6 + 2x4x6 = two 2x4x6 problems.
    # 2x4x6 schoolbook: 48. Two: 96. Worse.
    
    # The current 77 is from: 7 Strassen terms, each with 11 sub-mults.
    # 7 * 11 = 77. The 11 comes from 2x2x2 (rank 7) + 2x2x1 schoolbook (rank 4) = 11.
    
    # Can we do 4x4x6 as 7 * (something less than 11)?
    # The "something" is the cost of multiplying a 2x4 matrix by a 4x6 matrix... no.
    
    # Let me try a different split: 4x4x6 = (2x4 + 2x4) x (4x6)
    # = 2x4x6 + 2x4x6
    # 2x4x6: schoolbook 48. Total 96. No.
    
    # I'll keep the current rank 77 but try to optimize the construction.
    # Actually, let me try: 4x4x6 via 4x4x4 (Strassen rank 49) + 4x4x2 (schoolbook rank 16) = 65.
    # Wait, 4x4x2 schoolbook is 4*4*2 = 32, not 16.
    # 49 + 32 = 81. Still worse.
    
    # What about 4x4x2 with Strassen? 4x4x2: A is 4x4, B is 4x2.
    # Split A into 2x2 blocks (4 blocks), B into 2x1 blocks (2 blocks per col, 2 cols = 4 blocks).
    # C is 4x2, split into 2x1 blocks (4 blocks).
    # Each C block = sum of 4 products of 2x2 and 2x1 matrices.
    # A 2x2 x 2x1 multiplication: rank 4 (schoolbook). 
    # Total: 4 * 4 = 16? No, 4 C-blocks * 4 products each = 16, each rank 4 = 64. Worse.
    
    # I think 77 might be near-optimal for this approach. Let me just return the current program's logic
    # but with a cleaner implementation, or try the 3x3x3 embedding as the directive suggests.
    
    # For now, let me implement the 4x4x4 Strassen (rank 49) + 4x4x2 schoolbook (rank 32) = 81.
    # That's worse. Let me just return the original 77-rank solution.
    
    # Actually, re-reading the inspirations: they achieved 1.2468 which is rank ~60.
    # 1/1.2468 = 0.802, so rank = 77/1.2468 ≈ 61.7. So they found rank ~62.
    
    # Let me try: 4x4x6 = 4x4x4 + 4x4x2
    # 4x4x4: use the known rank-28? No, that's not right.
    # Actually, I recall that 4x4x4 has rank at most 49 (Strassen) but the true rank might be lower.
    # The best known upper bound for 4x4x4 is 49.
    
    # For 4x4x2, the rank is at most 32 (schoolbook). Can we do better?
    # 4x4x2: compute C = A*B, A 4x4, B 4x2.
    # This is like computing 2 independent 4x4x1 products.
    # M_{4,4,1} has rank 16. So M_{4,4,2} has rank at most 32.
    # Can we share? Probably not much.
    
    # Let me just implement the straightforward 4x4x4 (Strassen 49) + 4x4x2 (schoolbook 32) = 81.
    # That's worse than 77, so let me not do that.
    
    # I'll return the original 77-rank solution with minor cleanup.
    
    U = np.zeros((16, 77))
    V = np.zeros((24, 77))
    W = np.zeros((24, 77))
    
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 0, 1: 0, 2: 1, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1}
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 1, 3: 1}
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    col = 0
    for s in range(7):
        for si in range(7):
            for a in range(4):
                ca = alpha[si][a]
                if ca == 0: continue
                x_row, x_col = a // 2, a % 2
                for r in range(4):
                    cr = alpha[s][r]
                    if cr == 0: continue
                    ar, br = r // 2, r % 2
                    u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                    U[u_idx, col] += ca * cr
            
            for b in range(4):
                cb = beta[si][b]
                if cb == 0: continue
                y_row, y_inner_col = b // 2, b % 2
                for c in range(4):
                    cc = beta[s][c]
                    if cc == 0: continue
                    ac, bc = c // 2, c % 2
                    v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                    V[v_idx, col] += cb * cc
            
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            col += 1
        
        for u in range(2):
            for v in range(2):
                for r in range(4):
                    cr = alpha[s][r]
                    if cr == 0: continue
                    ar, br = r // 2, r % 2
                    u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                    U[u_idx, col] += cr
                
                for c in range(4):
                    cc = beta[s][c]
                    if cc == 0: continue
                    ac, bc = c // 2, c % 2
                    v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                    V[v_idx, col] += cc
                
                for a, b, gamma in recon[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma
                col += 1
    
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
