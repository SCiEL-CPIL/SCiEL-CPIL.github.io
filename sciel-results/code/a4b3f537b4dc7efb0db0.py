import numpy as np
import time

def construct():
    n, m, p = 2, 3, 10
    # Start with schoolbook rank 60, then try to improve
    # Using block decomposition: split p=10 into blocks, and m=3 into 2+1
    
    # Strategy: 
    # 1. Use Strassen on 2x2 blocks of the middle dimension where possible
    # 2. The known result: R(2,3,2) = 10 (from literature or we can try to find)
    #    If R(2,3,2)=10, then R(2,3,10) <= 10*5 = 50 by splitting p into 5 blocks of 2
    #    But actually splitting p means we need R(2,3,2) for each block independently
    #    So R(2,3,10) <= 5 * R(2,3,2)
    
    # Let's try to construct a rank-50 decomposition using 5 blocks of size 2 in p
    # For each block of p (2 columns), we need to compute 2x3 * 3x2 = 2x2
    # The rank of M_{2,3,2} - let's try to find a rank-10 decomposition
    
    # Actually, let me think more carefully.
    # M_{2,3,10}: A is 2x3, B is 3x10, C is 2x10
    # C[i,k] = sum_{j=0}^{2} A[i,j] B[j,k]
    
    # Lower bound: max(n*m, m*p, p*n) = max(6, 30, 20) = 30
    # So rank >= 30
    
    # Known: R(2,2,2) = 7 (Strassen)
    # Can we use Strassen-like tricks?
    
    # Let me try a direct approach: enumerate all possible rank-r decompositions
    # for small r and see if we can find one that works.
    
    # Actually, let me try a smarter structural approach.
    # 
    # Consider splitting the 3 middle dimension: A = [A_left (2x2), A_right (2x1)]
    # B = [B_top (2x10); B_bottom (1x10)]
    # C = A_left * B_top + A_right * B_bottom
    # 
    # A_left * B_top is 2x2 * 2x10 -> can use Strassen if we further split
    # A_right * B_bottom is 2x1 * 1x10 -> this is just 2*10 = 20 multiplications
    # 
    # For A_left * B_top (2x2 * 2x10):
    #   Split B_top into 5 blocks of 2: B_top = [B_t0, B_t1, ..., B_t4] each 2x2
    #   For each block, use Strassen: rank 7
    #   Total for this part: 5 * 7 = 35
    #   Plus the A_right * B_bottom part: 20
    #   Total: 35 + 20 = 55
    #
    # That's worse than 60? No, 55 < 60. Good improvement!
    #
    # But wait, can we do better?
    #
    # Alternative: Don't split m. Use the full 2x3 * 3x10 structure.
    # 
    # Let me think about R(2,3,2). 
    # A is 2x3, B is 3x2, C is 2x2.
    # C[i,k] = sum_j A[i,j] B[j,k] for i=0,1; k=0,1; j=0,1,2
    # That's 4 output entries, each a sum of 3 products = 12 terms total.
    # Lower bound: max(6, 6, 4) = 6.
    # 
    # Can we find a rank-6 decomposition? Probably not easily.
    # What about rank-8 or rank-9?
    #
    # Actually, let me just go with the 55-rank decomposition I described.
    # It's a clear improvement over 60.
    
    # But first, let me verify the structure:
    # A (2x3) = [a00 a01 a02; a10 a11 a12]
    # B (3x10) = rows b0 (10), b1 (10), b2 (10)
    # C (2x10) = A*B
    
    # Split: A_left = [a00 a01; a10 a11] (2x2), A_right = [a02; a12] (2x1)
    # B_top = [b0; b1] (2x10), B_bottom = b2 (1x10)
    # C = A_left * B_top + A_right * B_bottom
    #
    # A_right * B_bottom: C[i,k] gets contribution A_right[i] * B_bottom[k]
    #   This is rank 2*10 = 20 (each product is separate)
    #
    # A_left * B_top: 2x2 * 2x10
    #   Split B_top into 5 blocks of 2 columns: B_top = [B0, B1, B2, B3, B4]
    #   Each Bt is 2x2
    #   For each block, compute A_left * Bt (2x2 * 2x2 = 2x2) using Strassen
    #   Strassen rank for 2x2x2 is 7
    #   So 5 blocks * 7 = 35
    #
    # Total rank: 35 + 20 = 55
    
    # Now let me construct this explicitly.
    # 
    # Strassen for 2x2 * 2x2:
    # A = [a b; c d], B = [e f; g h], C = AB
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C = [M1+M4-M5+M7,  M3+M5;     M2+M4,      M1-M2+M3+M6]
    
    # In our tensor decomposition, each "M" is a rank-1 term.
    # For A_left * B_top,block_t:
    # A_left = [a00 a01; a10 a11]
    # Bt = [b0,2t b0,2t+1; b1,2t b1,2t+1]
    
    # Let me define the indices:
    # A indices: row i (0,1), col j (0,1,2) -> index i*3+j
    # B indices: row j (0,1,2), col k (0,...,9) -> index j*10+k
    # C indices: row i (0,1), col k (0,...,9) -> index k*2+i (W[k*n+i])
    
    # For the A_right * B_bottom part (rank 20):
    # A_right[i] = A[i,2], B_bottom[k] = B[2,k]
    # C[i,k] += A[i,2] * B[2,k]
    # U[i*3+2, r] = 1, V[2*10+k, r] = 1, W[k*2+i, r] = 1
    # for i in 0,1 and k in 0..9: that's 20 terms
    
    # For the A_left * B_top part, block t (t=0..4):
    # A_left = [A[0,0] A[0,1]; A[1,0] A[1,1]]
    # Bt = [B[0,2t] B[0,2t+1]; B[1,2t] B[1,2t+1]]
    # Ct = [C[0,2t] C[0,2t+1]; C[1,2t] C[1,2t+1]]
    
    # Strassen:
    # a = A[0,0], b = A[0,1], c = A[1,0], d = A[1,1]
    # e = B[0,2t], f = B[0,2t+1], g = B[1,2t], h = B[1,2t+1]
    
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C[0,2t]   = M1 + M4 - M5 + M7
    # C[0,2t+1] = M3 + M5
    # C[1,2t]   = M2 + M4
    # C[1,2t+1] = M1 - M2 + M3 + M6
    
    # For each M, we need to express it as a product of linear forms in A and B.
    # M1 = (a+d)(e+h) = (A[0,0]+A[1,1])(B[0,2t]+B[1,2t+1])
    #   U: A[0,0] and A[1,1] get +1
    #   V: B[0,2t] and B[1,2t+1] get +1
    #   W: contributes to C[0,2t] (+1) and C[1,2t+1] (+1)
    
    rank = 55
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    # Part 1: A_right * B_bottom (20 terms)
    for i in range(2):
        for k in range(10):
            U[i*3+2, col] = 1
            V[2*10+k, col] = 1
            W[k*2+i, col] = 1
            col += 1
    
    # Part 2: A_left * B_top, 5 Strassen blocks (35 terms)
    for t in range(5):
        k0 = 2*t
        k1 = 2*t+1
        
        # M1 = (A[0,0]+A[1,1])(B[0,k0]+B[1,k1])
        # Contributes to C[0,k0] and C[1,k1]
        U[0*3+0, col] = 1; U[1*3+1, col] = 1
        V[0*10+k0, col] = 1; V[1*10+k1, col] = 1
        W[k0*2+0, col] = 1; W[k1*2+1, col] = 1
        col += 1
        
        # M2 = (A[1,0]+A[1,1]) * B[0,k0]
        # Contributes to C[1,k0]
        U[1*3+0, col] = 1; U[1*3+1, col] = 1
        V[0*10+k0, col] = 1
        W[k0*2+1, col] = 1
        col += 1
        
        # M3 = A[0,0] * (B[0,k1]-B[1,k1])
        # Contributes to C[0,k1]
        U[0*3+0, col] = 1
        V[0*10+k1, col] = 1; V[1*10+k1, col] = -1
        W[k1*2+0, col] = 1
        col += 1
        
        # M4 = A[1,1] * (B[1,k0]-B[0,k0])
        # Contributes to C[0,k0] and C[1,k0]
        U[1*3+1, col] = 1
        V[1*10+k0, col] = 1; V[0*10+k0, col] = -1
        W[k0*2+0, col] = 1; W[k0*2+1, col] = 1
        col += 1
        
        # M5 = (A[0,0]+A[0,1]) * B[1,k1]
        # Contributes to C[0,k0] (-1) and C[0,k1] (+1)
        U[0*3+0, col] = 1; U[0*3+1, col] = 1
        V[1*10+k1, col] = 1
        W[k0*2+0, col] = -1; W[k1*2+0, col] = 1
        col += 1
        
        # M6 = (A[1,0]-A[0,0]) * (B[0,k0]+B[0,k1])
        # Contributes to C[1,k1]
        U[1*3+0, col] = 1; U[0*3+0, col] = -1
        V[0*10+k0, col] = 1; V[0*10+k1, col] = 1
        W[k1*2+1, col] = 1
        col += 1
        
        # M7 = (A[0,1]-A[1,1]) * (B[1,k0]+B[1,k1])
        # Contributes to C[0,k0]
        U[0*3+1, col] = 1; U[1*3+1, col] = -1
        V[1*10+k0, col] = 1; V[1*10+k1, col] = 1
        W[k0*2+0, col] = 1
        col += 1
    
    assert col == 55, f"Expected 55 terms, got {col}"
    
    # Verify
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # Quick verification
    for i in range(2):
        for k in range(10):
            s = 0.0
            for j in range(3):
                # C[i,k] should equal sum_j A[i,j]*B[j,k]
                # In our decomposition: sum_r U[i*3+j, r] * V[j*10+k, r] * W[k*2+i, r]
                pass
    
    # More careful check
    for i in range(2):
        for k in range(10):
            s = 0.0
            for j in range(3):
                for r in range(55):
                    s += U[i*3+j, r] * V[j*10+k, r] * W[k*2+i, r]
            expected = 1.0  # because T[i,j,k] = 1 for the matrix mult tensor
            # Wait, the tensor T[i,j,k] = 1 if and only if this is the (i,j,k) entry
            # of the matrix multiplication tensor. For M_{2,3,10}, T[i,j,k]=1 for all valid i,j,k.
            # So sum_r U[i*3+j, r] * V[j*10+k, r] * W[k*2+i, r] should equal 1 for each (i,j,k)
            # But that's not right either. Let me re-read the problem.
            #
            # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # And T[i,j,k] should be 1 for all i,j,k (it's the matrix multiplication tensor)
            # Wait no. The matrix multiplication tensor M_{n,m,p} has entries:
            # M_{n,m,p}[i,j,k] = 1 if... actually it's a 3-tensor where
            # (M_{n,m,p})_{ijk} = 1 for all i in [n], j in [m], k in [p]
            # and the decomposition satisfies:
            # delta_{ij} ... no.
            #
            # Let me re-read: "The tensor T[i, j, k] for M_{n,m,p} is 1."
            # And T[i, j, k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
            # So for EVERY (i,j,k) triple, the sum should equal 1.
            # But that can't be right because that would mean the tensor is all ones,
            # which is rank 1.
            #
            # I think I'm misunderstanding. Let me re-read the task.
            # "exactly decompose the matrix multiplication tensor"
            # The matrix multiplication tensor M_{n,m,p} is defined such that
            # for matrices A (n x m) and B (m x p), the product C = A*B is:
            # C_{ik} = sum_j A_{ij} B_{jk}
            # This can be written as:
            # C_{ik} = sum_{j} (A_{ij} B_{jk} * 1)
            # The tensor is T with T_{ijk} = 1 for all i,j,k? No...
            #
            # Actually, the standard definition: the tensor T for matrix multiplication
            # M_{n,m,p} is a 3-tensor of size n x m x p where T[i,j,k] = 1 for all entries.
            # And the decomposition means:
            # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]  (with appropriate indexing)
            #
            # But if T[i,j,k] = 1 for all i,j,k, then we need:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
            #
            # That's what the verification function checks. Let me verify with schoolbook:
            # In schoolbook, for each (i,j,k), there's exactly one r where all three are 1,
            # and for all other r, at least one is 0. So the sum is 1. Correct.
            #
            # So my verification should check that for all (i,j,k), the sum equals 1.
            
            # Let me redo the check properly
            pass
    
    # Proper verification
    ok = True
    for i in range(2):
        for j in range(3):
            for k in range(10):
                s = 0.0
                for r in range(55):
                    s += U[i*3+j, r] * V[j*10+k, r] * W[k*2+i, r]
                if abs(s - 1.0) > 1e-9:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    
    if not ok:
        # Fall back to schoolbook
        rank = 60
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(10):
                    U[i*3+j, col] = 1
                    V[j*10+k, col] = 1
                    W[k*2+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
