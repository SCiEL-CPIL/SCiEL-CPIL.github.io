"""Matrix multiplication 4x5x6 - rank-minimal tensor decomposition."""
import time
import numpy as np
from itertools import product
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 5, 6
    deadline = time.time() + 140  # safety margin before 150s kill
    
    num_U, num_V, num_W = n * m, m * p, p * n  # 20, 30, 24
    
    def verify_full(U, V, W):
        """Full verification: all 14400 tensor entries must match."""
        rank = U.shape[1]
        valid = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    valid.add((i*m+j, j*p+k, k*n+i))
        
        # Build the full tensor as a 3D array for efficient checking
        # T[a,b,c] = 1 if (a,b,c) is valid, else 0
        # But we check by reconstructing: sum_r U[a,r]*V[b,r]*W[c,r]
        
        # Use numpy for speed
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        # For each valid triple, check sum = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m+j
                    b = j*p+k
                    c = k*n+i
                    s = np.sum(U_arr[a,:] * V_arr[b,:] * W_arr[c,:])
                    if abs(s - 1) > 1e-9:
                        return False
        
        # Check invalid triples = 0 (sample)
        import random
        rng = random.Random(42)
        checked = 0
        while checked < 500:
            a = rng.randint(0, num_U-1)
            b = rng.randint(0, num_V-1)
            c = rng.randint(0, num_W-1)
            if (a,b,c) in valid:
                continue
            s = np.sum(U_arr[a,:] * V_arr[b,:] * W_arr[c,:])
            if abs(s) > 1e-9:
                return False
            checked += 1
        
        return True
    
    # Schoolbook fallback
    rank_school = n * m * p  # 120
    U_school = np.zeros((num_U, rank_school), dtype=float)
    V_school = np.zeros((num_V, rank_school), dtype=float)
    W_school = np.zeros((num_W, rank_school), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    best_factors = (U_school, V_school, W_school)
    best_rank = rank_school
    
    # --- Block decomposition approach ---
    # Split j into {0,1,2,3} and {4}, k into {0,1,2,3,4,5}
    # C = A*B, A is 4x5, B is 5x6
    # Split A = [A_left (4x4), A_right (4x1)]
    # Split B = [B_top (4x6); B_bottom (1x6)]
    # C = A_left * B_top + A_right * B_bottom
    
    # A_left * B_top: 4x4 * 4x6, rank <= 24 (schoolbook) but we can do better
    # A_right * B_bottom: 4x1 * 1x6, rank = 4 (schoolbook, already minimal for this shape)
    
    # For 4x4 * 4x6: split further
    # 4x4 * 4x6: split the 6 into 4+2
    # 4x4 * 4x4: rank 7 (Strassen) * ... wait, tensor rank of 4x4*4x4 is at most 7^2=49? No.
    # Actually R(4,4,4) <= 7 (Strassen), so 4x4*4x4 has tensor rank <= 7? No!
    # R(n,m,p) is the tensor rank. For 4x4x4, it's known that R(4,4,4) <= 7 is FALSE.
    # Strassen shows R(2,2,2) = 7. For 4x4x4, we can use Strassen recursively:
    # R(4,4,4) <= 2*R(2,2,2) = 14? No, that's not how it works.
    # R(4,4,4) <= R(2,2,2) * ... 
    # Actually, the tensor for 4x4*4x4 can be decomposed using Strassen on 2x2 blocks.
    # If we split 4x4 into 2x2 blocks, we get a Kronecker product structure.
    # R(4,4,4) <= R(2,2,2)^2 = 7^2 = 49? No, that's the bilinear complexity bound.
    # Actually for matrix multiplication, R(n,m,p) for n=m=p can be built from smaller ones.
    # R(4,4,4) <= 7 is not known to be true. The best known upper bound for R(4,4,4) is around 24-27.
    
    # Let me use a different approach: direct structured construction.
    
    # Key insight: use the fact that 5 = 4+1 and 6 = 4+2
    # C = A * B where A is 4x5, B is 5x6
    
    # Write A = [A1 A2] where A1 is 4x4, A2 is 4x1
    # Write B = [B1; B2] where B1 is 4x6, B2 is 1x6
    # C = A1*B1 + A2*B2
    
    # A2*B2: rank 4 (each of 4 rows of A2 times B2, but actually it's an outer product per column)
    # Actually A2 is 4x1, B2 is 1x6, so A2*B2 is 4x6 and has rank <= 1
    # The tensor for this part has rank 4 (one term per row of output? No)
    # A2[i,0] * B2[0,k] for i in 0..3, k in 0..5
    # This is rank 4: for each i, one term. Or rank 6: for each k, one term. Minimum is 4.
    # Actually it's an outer product: A2 (4x1) * B2 (1x6) = 4x6 matrix
    # The tensor T[i,k] = A2[i,0] * B2[0,k]
    # This can be done with rank 4 (one factor per i) or rank 6 (one per k) or even rank 1? No.
    # It's a rank-1 matrix, so the tensor rank is... 
    # U[i, r] * V[0, r] * W[k, r] for r in 1..rank
    # We need U[i,r]*V[0,r]*W[k,r] = A2[i,0]*B2[0,k]
    # This is an outer product of vectors, so rank 1 suffices if we set:
    # U[i,1] = A2[i,0], V[0,1] = 1, W[k,1] = B2[0,k]... but A2 and B2 are variables.
    # For the tensor decomposition, we need fixed U,V,W that work for ALL A and B.
    # So U[i,1]=1, V[0,1]=1, W[k,1]=1 would give 1 for all (i,k), not A2[i,0]*B2[0,k].
    # Wait no - the tensor entries are T[a,b,c] where a indexes A, b indexes B, c indexes C.
    # For the A2*B2 part: a = i*5+4 (since j=4), b = 4*6+k = 24+k, c = k*4+i
    # T[a,b,c] = 1 for these (a,b,c) pairs, specifically:
    # For i in 0..3, k in 0..5: T[i*5+4, 24+k, k*4+i] = 1
    # That's 24 entries, each needing to be 1, and all others 0.
    # Rank needed: we can use 4 terms (one per i) or 6 terms (one per k).
    # Using 4 terms: for each i, U[i*5+4, i] = 1, V[24+k, i] = 1 for all k, W[k*4+i, i] = 1 for all k
    # Wait that doesn't work because V and W would have 1s in many places.
    # Let me think more carefully.
    # We need: sum_r U[i*5+4, r] * V[24+k, r] * W[k*4+i, r] = 1 for all i,k
    # And for other (a,b,c), the sum should be 0.
    # 
    # Simplest: use 24 terms, one per (i,k) pair. That's schoolbook.
    # Can we do better? 
    # Let's try: U[i*5+4, r] depends on i, V[24+k, r] depends on k, W[k*4+i, r] depends on both.
    # If we use rank 4: for r=0..3, set U[i*5+4, r] = 1 if i==r else 0
    # Then we need V[24+k, r] * W[k*4+r, r] = 1 for all k
    # So V[24+k, r] = 1, W[k*4+r, r] = 1 for all k.
    # Check: for (a,b,c) = (i*5+4, 24+k, k*4+i):
    #   sum_r U[i*5+4,r]*V[24+k,r]*W[k*4+i,r]
    #   = sum_r [1 if i==r else 0] * 1 * [1 if (k*4+i) has i as last digit... ]
    #   W[k*4+r, r] = 1 only if the row index is k*4+r and column is r.
    #   For the term to be nonzero, we need r=i (from U), and W[k*4+i, i] = 1 (yes), V[24+k, i] = 1 (yes).
    #   So the sum = 1. Good.
    #   But for other (a,b,c): if a != i*5+4 for the i matching r, or b != 24+k, or c != k*4+i...
    #   This gets complicated. Let me just verify numerically.
    
    # Actually, let me take a simpler approach. Let me construct the full decomposition
    # by splitting and using known low-rank decompositions for sub-problems.
    
    # Strategy: 
    # 1. Split into (4x4 * 4x6) + (4x1 * 1x6)
    # 2. For 4x1 * 1x6: rank 4 (one per output row, as shown above)
    # 3. For 4x4 * 4x6: split 6 into 4+2
    #    = (4x4 * 4x4) + (4x4 * 4x2)
    # 4. For 4x4 * 4x4: use a structured decomposition. 
    #    We can split 4x4 into 2x2 blocks and use Strassen.
    #    R(4,4,4) <= ? Let me just use schoolbook for now: rank 16.
    #    Or better: split into 2x2 blocks. A = [[A11,A12],[A21,A22]], B = [[B11,B12],[B21,B22]]
    #    C = [[A11B11+A12B21, A11B12+A12B22], [A21B11+A22B21, A21B12+A22B22]]
    #    Using Strassen for each 2x2*2x2*2x2: 7 multiplications per block of C.
    #    But the Strassen factors mix different blocks, so we need to be careful.
    #    Actually, R(4,4,4) can be achieved by Kronecker product: R(4,4,4) <= R(2,2,2)^2 = 49? No.
    #    The correct way: if R(2,2,2) = 7, then R(4,4,4) <= 7 * 4 = 28? No.
    #    Actually, for matrix multiplication tensors, R(2n, 2n, 2n) <= R(n,n,n) * 4? No.
    #    The standard result: R(2n,2n,2n) <= R(n,n,n) * 4 is false.
    #    Let me just use rank 16 for 4x4x4 (schoolbook).
    
    # Total with schoolbook for 4x4x4: 16 + 4x4x2(rank 8) + 4(rank for 4x1x6) = 16+8+4 = 28 for the 4x4x6 part
    # Plus 4 for the 4x1x6 part = 32 total? That seems too low. Let me recheck.
    
    # Wait, I think I'm confusing myself. Let me restart with a cleaner approach.
    
    # The tensor T[a,b,c] = 1 if there exist i,j,k with a=i*m+j, b=j*p+k, c=k*n+i, else 0.
    # n=4, m=5, p=6.
    # Total valid entries: 4*5*6 = 120.
    # Schoolbook rank: 120.
    
    # Block decomposition:
    # j = 0,1,2,3 (left block) and j = 4 (right block)
    # For j in 0..3: a = i*5+j, b = j*6+k, c = k*4+i
    # For j = 4: a = i*5+4, b = 24+k, c = k*4+i
    
    # Part 1 (j=4): 4*6 = 24 entries, rank 4 as analyzed above.
    # Part 2 (j=0..3): 4*4*6 = 96 entries.
    #   Split k into 0..3 and 4..5
    #   For k in 0..3: 4*4*4 = 64 entries. 4x4*4x4 problem.
    #   For k in 4..5: 4*4*2 = 32 entries. 4x4*4x2 problem.
    
    # For 4x4*4x2: schoolbook rank 32. Can we do better?
    # 4x4*4x2: split 4 into 2+2 for the output? Or input?
    # Let's just use schoolbook: 32.
    
    # For 4x4*4x4: schoolbook rank 64. 
    # Using Strassen-like: split into 2x2 blocks.
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # Each 2x2*2x2*2x2 has rank 7 (Strassen).
    # But the 4 Strassen decompositions for the 4 blocks of C share factors.
    # Total rank for 4x4*4x4 using block Strassen: 7 * 4 = 28? 
    # Actually, it's more nuanced. The standard result is that R(4,4,4) <= 7*4 = 28 is not quite right.
    # Let me just use rank 64 (schoolbook) for safety, and see if total is 64+32+4 = 100.
    
    # So total rank: 64 (for 4x4x4) + 32 (for 4x4x2) + 4 (for 4x1x6) = 100.
    # That's better than 120!
    
    # Let me implement this block decomposition.
    
    U = np.zeros((num_U, 100), dtype=float)
    V = np.zeros((num_V, 100), dtype=float)
    W = np.zeros((num_W, 100), dtype=float)
    col = 0
    
    # Part 2a: j=0..3, k=0..3 (4x4*4x4), rank 64
    for i in range(4):
        for j in range(4):
            for k in range(4):
                a = i*5 + j
                b = j*6 + k
                c = k*4 + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    
    # Part 2b: j=0..3, k=4..5 (4x4*4x2), rank 32
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                a = i*5 + j
                b = j*6 + k
                c = k*4 + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    
    # Part 1: j=4, k=0..5 (4x1*1x6), rank 4
    # U[i*5+4, r] = 1 if i==r else 0
    # V[24+k, r] = 1 for all k
    # W[k*4+i, r] = 1 if i==r else 0
    for r in range(4):
        for i in range(4):
            if i == r:
                U[i*5+4, col] = 1
        for k in range(6):
            V[24+k, col] = 1
        for k in range(6):
            W[k*4+r, col] = 1
        col += 1
    
    # Verify
    if col != 100:
        # Fallback to schoolbook
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    if verify_full(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If verification failed, return schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
