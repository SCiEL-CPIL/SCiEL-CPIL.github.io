"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    def verify(U, V, W, tol=1e-6):
        """Check if decomposition is exact with half-integer entries."""
        # Check entries are half-integers
        for arr in [U, V, W]:
            for val in arr:
                if abs(val * 2 - round(val * 2)) > tol:
                    return False
        # Verify decomposition
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(len(U[0])):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def round_half(arr):
        return np.round(arr * 2) / 2.0
    
    def schoolbook_block(nB, mB, pB):
        """Return U (nB*mB x nB*mB*pB), V (mB*pB x nB*mB*pB), W (pB*nB x nB*mB*pB) for schoolbook."""
        rank = nB * mB * pB
        U = np.zeros((nB*mB, rank))
        V = np.zeros((mB*pB, rank))
        W = np.zeros((pB*nB, rank))
        col = 0
        for i in range(nB):
            for j in range(mB):
                for k in range(pB):
                    U[i*mB+j, col] = 1
                    V[j*pB+k, col] = 1
                    W[k*nB+i, col] = 1
                    col += 1
        return U, V, W
    
    def strassen_2x2x2():
        """Return U(4x7), V(4x7), W(4x7) for Strassen 2x2x2."""
        # Strassen formulas for 2x2:
        # M1 = (A11+A22)*(B11+B22)
        # M2 = (A21+A22)*B11
        # M3 = A11*(B11-B21)
        # M4 = A22*(B22-B12)
        # M5 = (A11+A12)*B22
        # M6 = (A21-A11)*(B11+B12)
        # M7 = (A12-A22)*(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        U = np.array([
            [1, 0, 1, 0, 1, 0, 0],  # A11
            [0, 0, 0, 0, 0, -1, 0], # A12
            [0, 1, 0, 0, 0, 1, 0],  # A21
            [1, 1, 0, 1, 0, 0, -1], # A22
        ])
        V = np.array([
            [1, 1, 1, 0, 0, 1, 0],  # B11
            [-1, 0, -1, -1, 0, 0, 0], # B12
            [1, 0, 0, 0, 1, 0, 1],  # B21
            [1, 0, 0, 1, 1, 0, 1],  # B22
        ])
        W = np.array([
            [1, 0, 0, 1, -1, 0, 1],  # C11 (i=0,k=0)
            [0, 0, 1, 0, 1, 0, 0],   # C12 (i=0,k=1)
            [0, 1, 0, 1, 0, 0, 0],   # C21 (i=1,k=0)
            [1, -1, 1, 0, 0, 1, 0],  # C22 (i=1,k=1)
        ])
        return U, V, W
    
    def embed_2x2x2_to_4x4x4(U2, V2, W2):
        """Embed 2x2x2 decomposition into 4x4x4 by applying to each 2x2 block of a 4x4 multiplication.
        Actually, for 4x4x4, we can think of it as 4 independent 2x2 multiplications? No, that's not right.
        A 4x4 times 4x4 gives 4x4. We can partition into 2x2 blocks:
        A = [A11 A12; A21 A22] where each Ai is 2x2
        B = [B11 B12; B21 B22]
        C = [C11 C12; C21 C22]
        Each Ci = sum of 2 products of 2x2 matrices.
        This doesn't directly give us a 4x4x4 decomposition from 2x2x2.
        
        Actually, the standard way: 4x4x4 can be done with rank 4*7=28 by applying Strassen to each of the 4 "rows" of 2x2 blocks?
        No. Let me think differently.
        
        For 4x4x4: We can view it as multiplying a (2x2) x (2x2) block matrix.
        A[i,j] for i=0..3, j=0..3. B[j,k] for j=0..3, k=0..3. C[i,k] for i=0..3, k=0..3.
        
        Partition into 2x2 super-matrices:
        A = [[A_00, A_01], [A_10, A_11]] where A_ij is 2x2
        B = [[B_00, B_01], [B_10, B_11]]
        C = [[C_00, C_01], [C_10, C_11]]
        
        C_00 = A_00*B_00 + A_01*B_10
        C_01 = A_00*B_01 + A_01*B_11
        C_10 = A_10*B_00 + A_11*B_10
        C_11 = A_10*B_01 + A_11*B_11
        
        Each C_ij is a 2x2 matrix = sum of two 2x2 matrix products.
        Using Strassen for each pair of products... this gets complicated.
        
        Alternative: Just use rank 49 for 4x4x4 (known optimal) or even schoolbook 64.
        """
        pass
    
    # Strategy: Build decomposition for 4x4x7 as combination of:
    # - 4x4x4 part (first 4 columns of output): rank r1
    # - 4x4x3 part (last 3 columns of output): rank r2
    # Total rank = r1 + r2
    
    # For 4x4x4: Use Strassen-based approach
    # 4x4x4 can be decomposed using 4 copies of 2x2x2 Strassen applied appropriately
    # Actually, the standard result: m(4,4,4) = 49 (proven optimal)
    # But let's try to get rank 28 or so using structured approaches
    
    # Let me try: for 4x4x4, partition into 2x2 blocks and use Strassen
    # This gives rank = 4 * 7 = 28? Let me verify.
    
    # Actually, here's a cleaner approach:
    # 4x4x4: View A as 2x2 block matrix of 2x2 blocks, same for B.
    # C_ij = A_i0*B_0j + A_i1*B_1j for i,j in {0,1}
    # For each fixed i, j: C_ij is a 2x2 matrix product sum.
    # Using Strassen to compute each sum of two 2x2 products: 7 mults per (i,j)
    # Total: 4 * 7 = 28 mults, but we also need the 2x2 block multiplications...
    
    # Let me just use known decompositions:
    # 4x4x4: rank 49 is optimal, but let's use rank 28 if possible or fall back to 49/64
    # 4x4x3: rank 48 schoolbook, or try to improve
    
    # For now, let's construct:
    # 4x4x4 with rank 28 using block Strassen
    # 4x4x3 with schoolbook rank 48
    # Total: 76
    
    # Actually, let me try a different structural approach:
    # Use the fact that 4x4x7 = 4x4x4 + 4x4x3
    # For 4x4x4: rank 28 (if we can construct it) or 49 or 64
    # For 4x4x3: try rank 36 (4*3*3) or better
    
    # Let me construct 4x4x4 with rank 28:
    # Partition indices: i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2 where i1,j1,k1 in {0,1}, i2,j2,k2 in {0,1}
    # Then A[2*i1+i2, 2*j1+j2] * B[2*j1+j2, 2*k1+k2] summed over j1,j2
    # = sum_{j1,j2} A[i1,i2;j1,j2] * B[j1,j2;k1,k2]
    
    # Group by j1: sum_{j1} sum_{j2} A[i1,i2;j1,j2] * B[j1,j2;k1,k2]
    # For each j1, the inner sum over j2 is a 2x2 matrix product:
    # (A[i1,:;j1,:] @ B[j1,:;k1,:])_{i2,k2}
    # So C[i1,i2;k1,k2] = sum_{j1=0,1} (A[i1,:;j1,:] @ B[j1,:;k1,:])_{i2,k2}
    
    # This is: C_super[i1,k1] = A_super[i1,0] @ B_super[0,k1] + A_super[i1,1] @ B_super[1,k1]
    # where A_super[i1,j1] is the 2x2 block, etc.
    
    # So we need to compute, for each (i1,k1) in {0,1}^2:
    #   C_super[i1,k1] = A_super[i1,0]*B_super[0,k1] + A_super[i1,1]*B_super[1,k1]
    # This is a sum of two 2x2 matrix products.
    # Using Strassen-like approach for sum of two products:
    # Let X = A_super[i1,0], Y = A_super[i1,1], Z = B_super[0,k1], W = B_super[1,k1]
    # We want X*Z + Y*W
    # This can be done in 7 multiplications (Strassen applied to [[X,Y],[0,0]] * [[Z,0],[W,0]]? No...)
    # Actually: XZ + YW. Let's use:
    # M1 = (X+Y)*Z = XZ + YZ
    # M2 = Y*(W-Z) = YW - YZ
    # So XZ + YW = M1 + M2
    # That's only 2 multiplications! But wait, (X+Y)*Z requires computing X+Y first (free addition).
    # So rank per (i1,k1) is 2? That gives total rank 4*2 = 8? That can't be right.
    
    # Oh wait, the issue is that X, Y, Z, W are 2x2 matrices, and their "multiplication" is not scalar.
    # The product (X+Y)*Z means matrix multiplication of (X+Y) (a 2x2 matrix) by Z (a 2x2 matrix).
    # So we're back to needing to multiply 2x2 matrices, which costs 7 with Strassen.
    
    # So for each (i1,k1), we need:
    #   C_super[i1,k1] = (X+Y)*Z + Y*(W-Z)
    # where each "*" is a 2x2 matrix product.
    # Each 2x2 product costs 7 rank units.
    # So 2 products * 7 = 14 rank units per (i1,k1).
    # Total: 4 * 14 = 56. That's worse than schoolbook 64? No, 56 < 64. OK.
    
    # But wait, the X, Y, Z, W for different (i1,k1) are different, so we can't share.
    # Total rank for 4x4x4 = 56 with this approach.
    
    # Hmm, let me just use schoolbook for 4x4x4 (rank 64) and schoolbook for 4x4x3 (rank 48).
    # Total: 112. That's the same as the original.
    
    # Let me try: 4x4x4 with rank 49 (optimal but hard to construct) -> use 64 schoolbook
    # 4x4x3: try rank 36 = 4*3*3 (schoolbook) -> same as 48? No, 4*4*3=48.
    
    # OK let me just go with a clean structural decomposition:
    # 4x4x4: schoolbook rank 64
    # 4x4x3: schoolbook rank 48
    # Total: 112
    
    # But the inspirations suggest rank 93 is achievable (49+44).
    # Let me try to construct 4x4x4 with rank 49 and 4x4x3 with rank 44.
    
    # For 4x4x4 rank 49: This is the known optimal. Hard to construct explicitly.
    # Let me use rank 28 approach if I can make it work, or just use 64.
    
    # For simplicity and correctness, let me construct:
    # 4x4x4: rank 64 (schoolbook)
    # 4x4x3: rank 48 (schoolbook)
    # Total: 112
    
    # Then try to improve 4x4x3 part.
    
    U_total = np.zeros((n*m, n*m*p))
    V_total = np.zeros((m*p, n*m*p))
    W_total = np.zeros((p*n, n*m*p))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_total[i*m+j, col] = 1
                V_total[j*p+k, col] = 1
                W_total[k*n+i, col] = 1
                col += 1
    
    best_U = U_total.copy()
    best_V = V_total.copy()
    best_W = W_total.copy()
    best_rank = n*m*p  # 112
    
    # Now try to improve by decomposing into 4x4x4 and 4x4x3 parts
    # and using Strassen for the 2x2 sub-blocks
    
    def build_4x4x4_strassen():
        """Build 4x4x4 decomposition using block Strassen. Target rank ~56."""
        # As derived: for each (i1,k1), compute (X+Y)*Z + Y*(W-Z) where X,Y,Z,W are 2x2 blocks
        # Each 2x2 product uses Strassen (rank 7)
        # Total rank = 4 * 2 * 7 = 56
        
        U = np.zeros((16, 56))
        V = np.zeros((16, 56))
        W = np.zeros((16, 56))
        
        U2, V2, W2 = strassen_2x2x2()
        
        col = 0
        for i1 in range(2):
            for k1 in range(2):
                # X = A[i1,:; 0,:] = rows 2*i1, 2*i1+1; cols 0,1
                # Y = A[i1,:; 1,:] = rows 2*i1, 2*i1+1; cols 2,3
                # Z = B[0,:; k1,:] = rows 0,1; cols 2*k1, 2*k1+1
                # Wmat = B[1,:; k1,:] = rows 2,3; cols 2*k1, 2*k1+1
                
                # First product: (X+Y)*Z
                # X+Y has entries: A[2*i1+i2, 2*j1+j2] for j1=0,1
                # i.e., A[2*i1+i2, j2] + A[2*i1+i2, 2+j2] for i2,j2 in {0,1}
                
                # Apply Strassen to compute (X+Y)*Z:
                # U for (X+Y): for each (i2, j2) in 2x2, U_val = A[2*i1+i2, j2] + A[2*i1+i2, 2+j2]
                # This maps to index (2*i1+i2)*4 + j2 and (2*i1+i2)*4 + 2+j2 in U
                # V for Z: B[j2, 2*k1+k2] for j2 in {0,1} (since Z is rows 0,1 of B)
                #   maps to j2*4 + 2*k1+k2
                # W for result: C[2*i1+i2, 2*k1+k2] maps to (2*k1+k2)*4 + (2*i1+i2)
                
                for c in range(7):
                    # U contribution from (X+Y)
                    for i2 in range(2):
                        for j2 in range(2):
                            idx_a1 = (2*i1+i2)*4 + j2      # A[2*i1+i2, j2]
                            idx_a2 = (2*i1+i2)*4 + 2 + j2   # A[2*i1+i2, 2+j2]
                            u_val = U2[i2*2+j2, c]
                            U[idx_a1, col] += u_val
                            U[idx_a2, col] += u_val
                    # V contribution from Z
                    for j2 in range(2):
                        for k2 in range(2):
                            idx_b = j2*4 + 2*k1+k2  # B[j2, 2*k1+k2]
                            v_val = V2[j2*2+k2, c]
                            V[idx_b, col] += v_val
                    # W contribution
                    for i2 in range(2):
                        for k2 in range(2):
                            idx_c = (2*k1+k2)*4 + (2*i1+i2)
                            w_val = W2[k2*2+i2, c]
                            W[idx_c, col] += w_val
                    col += 1
                
                # Second product: Y*(W-Z)
                # Y = A[i1,:; 1,:] = rows 2*i1, 2*i1+1; cols 2,3
                # W-Z = B[1,:; k1,:] - B[0,:; k1,:]
                #   = rows 2,3 minus rows 0,1; cols 2*k1, 2*k1+1
                #   entry (j2', k2) = B[2+j2', 2*k1+k2] - B[j2', 2*k1+k2]
                
                for c in range(7):
                    # U contribution from Y
                    for i2 in range(2):
                        for j2 in range(2):
                            idx_a = (2*i1+i2)*4 + 2 + j2  # A[2*i1+i2, 2+j2]
                            u_val = U2[i2*2+j2, c]
                            U[idx_a, col] += u_val
                    # V contribution from (W-Z)
                    for j2 in range(2):
                        for k2 in range(2):
                            idx_b_w = (2+j2)*4 + 2*k1+k2  # B[2+j2, 2*k1+k2]
                            idx_b_z = j2*4 + 2*k1+k2      # B[j2, 2*k1+k2]
                            v_val = V2[j2*2+k2, c]
                            V[idx_b_w, col] += v_val
                            V[idx_b_z, col] -= v_val
                    # W contribution
                    for i2 in range(2):
                        for k2 in range(2):
                            idx_c = (2*k1+k2)*4 + (2*i1+i2)
                            w_val = W2[k2*2+i2, c]
                            W[idx_c, col] += w_val
                    col += 1
        
        return U, V, W, 56
    
    def build_4x4x3_schoolbook():
        """Build 4x4x3 schoolbook decomposition. Rank 48."""
        U = np.zeros((16, 48))
        V = np.zeros((12, 48))
        W = np.zeros((12, 48))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    U[i*4+j, col] = 1
                    V[j*3+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
        return U, V, W, 48
    
    # Try building the combined decomposition
    try:
        U1, V1, W1, r1 = build_4x4x4_strassen()
        U2, V2, W2, r2 = build_4x4x3_schoolbook()
        
        # Combine: U_total (16 x (r1+r2)), V_total (16 x (r1+r2) for first part, 12 x (r1+r2) for second)
        # Wait, V has shape (m*p, rank) = (4*7, rank) = (28, rank)
        # For the 4x4x4 part, V is 16 x r1 (since m*p = 4*4 = 16 for that part)
        # For the 4x4x3 part, V is 12 x r2 (since m*p = 4*3 = 12 for that part)
        # But the full V is 28 x total_rank
        # The 4x4x4 part uses rows 0..15 of V (j*p+k for j in 0..3, k in 0..3)
        # The 4x4x3 part uses rows 16..27 of V (j*p+k for j in 0..3, k in 4..6)
        
        total_rank = r1 + r2
        U_comb = np.zeros((16, total_rank))
        V_comb = np.zeros((28, total_rank))
        W_comb = np.zeros((28, total_rank))  # p*n = 7*4 = 28
        
        # Fill 4x4x4 part (columns 0..r1-1)
        U_comb[:, :r1] = U1
        # V1 is 16 x r1, goes into rows 0..15 of V_comb
        V_comb[:16, :r1] = V1
        # W1 is 16 x r1 (p*n for 4x4x4 is 4*4=16), goes into rows 0..15 of W_comb
        W_comb[:16, :r1] = W1
        
        # Fill 4x4x3 part (columns r1..total_rank-1)
        U_comb[:, r1:] = U2
        # V2 is 12 x r2, goes into rows 16..27 of V_comb
        V_comb[16:, r1:] = V2
        # W2 is 12 x r2 (p*n for 4x4x3 is 3*4=12), goes into rows 16..27 of W_comb
        W_comb[16:, r1:] = W2
        
        # Verify
        if verify(U_comb, V_comb, W_comb, tol=1e-6):
            best_U = U_comb
            best_V = V_comb
            best_W = W_comb
            best_rank = total_rank
            print(f"Found decomposition with rank {total_rank}")
    except Exception as e:
        print(f"Error in Strassen construction: {e}")
    
    # Also try pure schoolbook as fallback (already have it)
    
    # Try to improve 4x4x3 part using Strassen-like methods
    # 4x4x3: can we do better than 48?
    # Partition 4x4x3 into two 4x2x3 and one 4x0x3? Or 2x4x3 + 2x4x3?
    # Let's try: split the 4 rows of A into two groups of 2
    # Group 1: rows 0,1 of A -> 2x4 x 4x3 = 2x4x3
    # Group 2: rows 2,3 of A -> 2x4 x 4x3 = 2x4x3
    # Each 2x4x3: schoolbook rank 24. Total 48. Same.
    
    # Try splitting columns of B: 4x4x3 -> 4x2x3 + 4x2x3 (split B's 4 cols into 2+2, but B is 4x3)
    # Actually B is 4x3, so we can't split it into two 4x1.5.
    # Split B into rows: 4x4x3 -> (2x4 x 4x3) + (2x4 x 4x3) = 24 + 24 = 48. Same.
    
    # What if we use Strassen for 2x2x2 blocks within 4x4x3?
    # This is getting complex. Let's just go with what we have.
    
    # If the Strassen 4x4x4 didn't work, we still have schoolbook 112.
    # If it did, we have rank 56+48=104.
    
    # Let's also try: 4x4x4 with rank 49 (hard) or just use 64 schoolbook
    # and see if 104 is achievable.
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
