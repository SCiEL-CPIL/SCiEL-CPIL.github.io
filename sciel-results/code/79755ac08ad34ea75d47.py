import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 100.0

    n, m, p = 2, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 8, 32, 16

    # Block-structured Strassen: treat A as 2x(2x2) blocks and B as (2x2)x4 blocks.
    # Partition A into 2 row-blocks (each 1x4) and B into 4 col-blocks (each 4x2).
    # Each sub-problem is 1x4 * 4x2 = 1x2, which is a "row times column" product.
    # But that's just standard. Let's use a different partitioning.
    
    # Actually, let's try: A is 2x4, B is 4x8.
    # Split A into 2 blocks of 2x2 (along columns): A = [A_L | A_R] where A_L, A_R are 2x2
    # Split B into 4 blocks of 2x2 (along rows AND columns): B = [[B_00, B_01], [B_10, B_11]] (4x8)
    # C = A * B = [A_L * B_top + A_R * B_bottom] where B_top is top 4 rows (4x8), B_bottom is bottom 4 rows (4x8)
    # This doesn't decompose nicely.
    
    # Better approach: Use the fact that 2x4 * 4x8 = 2x8.
    # View A as 2 row-vectors of length 4, B as 4 column-vectors of length 4 giving 8 output columns.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Let's use a tensor decomposition approach with a different block structure.
    # Split the 4 middle dimension into 2 groups of 2: j = 2*q + r where q in {0,1}, r in {0,1}
    # A[i, 2q+r], B[2q+r, k]
    # C[i,k] = sum_{q,r} A[i, 2q+r] * B[2q+r, k]
    
    # For each (i, k) pair, we need sum over q,r. That's 8 products per (i,k) pair, total 2*8*4 = 64.
    # Strassen on the 2x2 (q,r) structure for each (i,k) would give 7 per (i,k) pair = 56.
    # But that's what we already have.
    
    # Let me think differently. Can we share multiplications across different (i,k) pairs?
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0] + A[0,2]B[2,0] + A[0,3]B[3,0]
    # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1] + A[0,2]B[2,1] + A[0,3]B[3,1]
    # ...
    
    # Group by j: For each j, we have A[:,j] (2-vector) * B[j,:] (8-vector) = 2x8 outer product contribution.
    # So C = sum_j A[:,j] * B[j,:]
    # This is a sum of 4 rank-1 (in the 2x8 space) outer products, each of "rank" 16 in tensor sense.
    # Total rank = 4 * 2 * 8 = 64 for the naive approach.
    
    # The Strassen approach groups 4 of these into a 2x2x2 structure and saves 1 multiplication.
    # 8 such groups * 7 = 56.
    
    # Can we do better? The tensor is <2,4,8>. Known bounds:
    # R(<2,4,8>) <= 56 (via Strassen)
    # Lower bound: max(2*4+4*8+2*8 - 4 - 8 + 1) = max(8+32+16-12) = 44 (Bayer-Strassen)
    # Actually the lower bound is n*m + m*p + p*n - m - p + 1 = 8 + 32 + 16 - 4 - 8 + 1 = 45
    
    # Let me try to find a rank-50 or lower decomposition.
    # One idea: use a different blocking that shares more structure.
    
    # Alternative: Use Winograd's algorithm ideas or a custom decomposition.
    
    # Let me try a different block structure:
    # Split A into 4 blocks of 2x1 (each column is a block)
    # Split B into 8 blocks of 1x1... no, B is 4x8, so B[j,k] is scalar.
    
    # Let me just verify the rank-56 decomposition is correct and see if there's a way to reduce.
    
    # Actually, let me try a completely different approach:
    # Use the fact that we can choose arbitrary (not just ±1) coefficients.
    # Try to find a decomposition of rank < 56 using optimization.
    
    # But first, let me make sure the rank-56 is correct.
    
    rank = 56
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)
    
    # The 8 blocks: for each (a, b) where a in 0..1, b in 0..3
    # This treats A as having 2 column-pairs and B as having 4 column-pairs
    # Within each, it's a 2x2 * 2x2 -> 2x2 Strassen
    
    for a in range(2):
        for b in range(4):
            base = 7 * (4 * a + b)
            
            # A indices: A[i, 2a], A[i, 2a+1] for i=0,1
            a00 = 0 * m + 2 * a
            a01 = 0 * m + 2 * a + 1
            a10 = 1 * m + 2 * a
            a11 = 1 * m + 2 * a + 1
            
            # B indices: B[2a+j, 2b+k]
            b00 = (2 * a) * p + 2 * b
            b01 = (2 * a) * p + 2 * b + 1
            b10 = (2 * a + 1) * p + 2 * b
            b11 = (2 * a + 1) * p + 2 * b + 1
            
            # W indices: W[k*n + i] for C[i, k]
            c00 = (2 * b) * n + 0
            c01 = (2 * b + 1) * n + 0
            c10 = (2 * b) * n + 1
            c11 = (2 * b + 1) * n + 1
            
            # s=0: M1 = (A00+A11)(B00+B11)
            U[a00, base] += 1; U[a11, base] += 1
            V[b00, base] += 1; V[b11, base] += 1
            W[c00, base] += 1; W[c11, base] += 1
            
            # s=1: M2 = (A10+A11)B00
            U[a10, base+1] += 1; U[a11, base+1] += 1
            V[b00, base+1] += 1
            W[c10, base+1] += 1; W[c11, base+1] -= 1
            
            # s=2: M3 = A00(B01-B11)
            U[a00, base+2] += 1
            V[b01, base+2] += 1; V[b11, base+2] -= 1
            W[c01, base+2] += 1; W[c11, base+2] += 1
            
            # s=3: M4 = A11(B10-B00)
            U[a11, base+3] += 1
            V[b10, base+3] += 1; V[b00, base+3] -= 1
            W[c00, base+3] += 1; W[c10, base+3] += 1
            
            # s=4: M5 = (A00+A01)B11
            U[a00, base+4] += 1; U[a01, base+4] += 1
            V[b11, base+4] += 1
            W[c00, base+4] -= 1; W[c01, base+4] += 1
            
            # s=5: M6 = (A10-A00)(B00+B01)
            U[a10, base+5] += 1; U[a00, base+5] -= 1
            V[b00, base+5] += 1; V[b01, base+5] += 1
            W[c11, base+5] += 1
            
            # s=6: M7 = (A01-A11)(B10+B11)
            U[a01, base+6] += 1; U[a11, base+6] -= 1
            V[b10, base+6] += 1; V[b11, base+6] += 1
            W[c00, base+6] += 1
    
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = float(np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :]))
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook
        rank = 64
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
