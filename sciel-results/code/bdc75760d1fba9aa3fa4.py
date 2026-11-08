import numpy as np
import time

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 8
    
    # Build a rank-98 decomposition: two independent 4x4x4 Strassen-recursive decompositions
    # 4x4x4 via recursive Strassen: split into 2x2 blocks, use 7 multiplications of 2x2x2
    # Each 2x2x2 uses Strassen (rank 7)
    # Total: 7 * 7 = 49 per 4x4x4, so 98 for 4x4x8
    
    # Strassen 2x2x2: U_2, V_2, W_2 each 4x7
    # Local indices: A[r,s] -> r*2+s, B[r,s] -> r*2+s, C[r,s] -> r*2+s
    def strassen_2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # P0 = (A00+A11)*(B00+B11) -> C00+=1, C11+=1
        U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
        # P1 = (A10+A11)*B00 -> C10+=1, C11-=1
        U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
        # P2 = A00*(B01-B11) -> C01+=1, C11+=1
        U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
        # P3 = A11*(B10-B00) -> C00+=1, C10+=1
        U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
        # P4 = (A00+A01)*B11 -> C00-=1, C01+=1
        U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
        # P5 = (A10-A00)*(B00+B01) -> C11+=1
        U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
        # P6 = (A01-A11)*(B10+B11) -> C00+=1
        U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
        return U, V, W
    
    # 4x4x4 via recursive Strassen: 7 multiplications of 2x2x2
    # A = [[A00,A01],[A10,A11]], B = [[B00,B01],[B10,B11]], C = [[C00,C01],[C10,C11]]
    # Each 2x2x2 has 4 elements. Global index for A block (a,c), local (r,s): (a*2+r)*4 + (c*2+s)
    # Global index for B block (c,b), local (r,s): (c*2+r)*4 + (b*2+s)
    # Global index for C block (a,b), local (r,s): (b*2+s)*4 + (a*2+r)  [W convention: j*4+i]
    
    def build_4x4x4():
        """Build rank-49 decomposition of 4x4x4 multiplication."""
        U2, V2, W2 = strassen_2x2()
        # 7 sub-multiplications, each 2x2x2 (rank 7)
        # Sub-mult 0: M0 = (A00+A11)*(B00+B11) -> contributes to C00, C11
        # Sub-mult 1: M1 = (A10+A11)*B00 -> contributes to C10, C11
        # Sub-mult 2: M2 = A00*(B01-B11) -> contributes to C01, C11
        # Sub-mult 3: M3 = A11*(B10-B00) -> contributes to C00, C10
        # Sub-mult 4: M4 = (A00+A01)*B11 -> contributes to C00, C01
        # Sub-mult 5: M5 = (A10-A00)*(B00+B01) -> contributes to C11
        # Sub-mult 6: M6 = (A01-A11)*(B10+B11) -> contributes to C00
        
        # For each sub-mult s, we need to map the 2x2 A-input, 2x2 B-input, 2x2 C-output
        # to global 4x4 indices.
        
        # Sub-mult 0: A-input = A00+A11, B-input = B00+B11, C-output = C00+C11
        # A00: block (0,0), A11: block (1,1)
        # B00: block (0,0), B11: block (1,1)
        # C00: block (0,0), C11: block (1,1)
        
        # We need: for each of the 7 Strassen terms within sub-mult 0,
        # the U vector has entries in A00 and A11 positions,
        # the V vector has entries in B00 and B11 positions,
        # the W vector has entries in C00 and C11 positions.
        
        rank_4x4 = 49
        U = np.zeros((16, rank_4x4))
        V = np.zeros((16, rank_4x4))
        W = np.zeros((16, rank_4x4))
        
        # For each of the 7 sub-multiplications, define which A-blocks, B-blocks, C-blocks are involved
        # and the coefficients
        # sub_mult s: A_blocks = [(block_idx, coeff), ...], B_blocks = [...], C_blocks = [...]
        
        sub_mults = [
            # s=0: A00+A11, B00+B11, C00+C11
            ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
            # s=1: A10+A11, B00, C10+C11
            ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),
            # s=2: A00, B01-B11, C01+C11
            ([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)]),
            # s=3: A11, B10-B00, C00+C10
            ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),
            # s=4: A00+A01, B11, C00+C01
            ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),
            # s=5: A10-A00, B00+B01, C11
            ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
            # s=6: A01-A11, B10+B11, C00
            ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
        ]
        
        col = 0
        for s in range(7):
            a_blocks, b_blocks, c_blocks = sub_mults[s]
            for t in range(7):
                # Strassen term t within sub-mult s
                u_pat = U2[:, t]
                v_pat = V2[:, t]
                w_pat = W2[:, t]
                
                # Fill U: for each local A index (r*2+s_local), add coeff * u_pat[local]
                # a_blocks gives which global 2x2 blocks and their coefficients
                for (block_idx, coeff) in a_blocks:
                    br, bc = divmod(block_idx, 2)  # block row, block col
                    for local_idx in range(4):
                        val = u_pat[local_idx]
                        if val != 0:
                            lr, ls = divmod(local_idx, 2)
                            global_idx = (br*2 + lr)*4 + (bc*2 + ls)
                            U[global_idx, col] += coeff * val
                
                # Fill V: similar for B
                for (block_idx, coeff) in b_blocks:
                    br, bc = divmod(block_idx, 2)
                    for local_idx in range(4):
                        val = v_pat[local_idx]
                        if val != 0:
                            lr, ls = divmod(local_idx, 2)
                            global_idx = (br*2 + lr)*4 + (bc*2 + ls)
                            V[global_idx, col] += coeff * val
                
                # Fill W: similar for C (W index = j*4 + i)
                for (block_idx, coeff) in c_blocks:
                    br, bc = divmod(block_idx, 2)  # C block row=a, col=b
                    for local_idx in range(4):
                        val = w_pat[local_idx]
                        if val != 0:
                            lr, ls = divmod(local_idx, 2)
                            global_i = br*2 + lr
                            global_j = bc*2 + ls
                            global_W_idx = global_j*4 + global_i
                            W[global_W_idx, col] += coeff * val
                
                col += 1
        
        return U, V, W
    
    # Build two 4x4x4 decompositions: one for B columns 0-3, one for B columns 4-7
    U4, V4, W4 = build_4x4x4()
    
    # For the first half (B cols 0-3): U is A, V is B[:, 0:4], W is C[:, 0:4]
    # For the second half (B cols 4-7): U is A, V is B[:, 4:8], W is C[:, 4:8]
    
    # Full U: 16 x 98
    # Full V: 32 x 98 (B has 4*8=32 entries)
    # Full W: 32 x 98 (C has 8*4=32 entries, W index = j*4+i)
    
    rank_total = 98
    U_full = np.zeros((16, rank_total))
    V_full = np.zeros((32, rank_total))
    W_full = np.zeros((32, rank_total))
    
    # First half: columns 0-47
    U_full[:, 0:49] = U4
    # V for first half: B indices 0-15 (rows 0-3, cols 0-3)
    V_full[0:16, 0:49] = V4
    # W for first half: C indices for cols 0-3, W index = j*4+i where j in 0-3
    # W4 has entries at positions j*4+i where j is local col (0-3), i is local row (0-3)
    # Global W index for first half: j_global*4 + i_global where j_global = j_local (0-3)
    # So W_full[0:16, 0:49] = W4
    W_full[0:16, 0:49] = W4
    
    # Second half: columns 49-97
    U_full[:, 49:98] = U4
    # V for second half: B indices 16-31 (rows 0-3, cols 4-7)
    V_full[16:32, 49:98] = V4
    # W for second half: C indices for cols 4-7, W index = j*4+i where j in 4-7
    # Global W index: j_global*4 + i_global where j_global = 4 + j_local
    # So W index = (4+j_local)*4 + i_local = 16 + j_local*4 + i_local
    # W4 has entries at j_local*4 + i_local, so we need to shift by 16
    W_full[16:32, 49:98] = W4
    
    # Verify: check the tensor identity on a few entries
    # M[A_idx, B_idx, W_idx] should be 1 iff A_idx=i*4+k, B_idx=k*8+j, W_idx=j*4+i
    # Let's verify by checking that the decomposition reconstructs the tensor
    
    # Quick verification: for a random (i,k,j), check that the contribution is 1
    # This is expensive, so let's just do a spot check
    
    # Actually, let's verify the 4x4x4 part first
    # For 4x4x4: M[A_idx, B_idx, W_idx] = 1 iff A_idx=i*4+k, B_idx=k*4+j, W_idx=j*4+i
    
    # Let's build the target tensor and compare
    # Target: T[A_idx, B_idx, W_idx] where A_idx = i*4+k, B_idx = k*4+j, W_idx = j*4+i
    
    # Reconstruct from decomposition: T_recon = sum_r U[:,r] * V[:,r] * W[:,r]
    # This is a 16x16x16 tensor, too large to build explicitly.
    # Instead, check specific entries.
    
    # For 4x4x4, check: for i=0,k=0,j=0: A_idx=0, B_idx=0, W_idx=0
    # T[0,0,0] should be 1
    # T_recon[0,0,0] = sum_r U[0,r]*V[0,r]*W[0,r]
    
    # Let's do a numerical check on a subset
    # Build target for 4x4x4
    target_4 = np.zeros((16, 16, 16))
    for i in range(4):
        for k in range(4):
            for j in range(4):
                A_idx = i*4 + k
                B_idx = k*4 + j
                W_idx = j*4 + i
                target_4[A_idx, B_idx, W_idx] = 1
    
    # Reconstruct from U4, V4, W4
    recon_4 = np.zeros((16, 16, 16))
    for r in range(49):
        recon_4 += np.einsum('i,j,k->ijk', U4[:,r], V4[:,r], W4[:,r])
    
    # Check
    max_err = np.max(np.abs(recon_4 - target_4))
    if max_err > 1e-10:
        # Fallback: use naive rank-64 decomposition for 4x4x4
        U_naive = np.zeros((16, 64))
        V_naive = np.zeros((16, 64))
        W_naive = np.zeros((16, 64))
        col = 0
        for i in range(4):
            for k in range(4):
                for j in range(4):
                    U_naive[i*4+k, col] = 1
                    V_naive[k*4+j, col] = 1
                    W_naive[j*4+i, col] = 1
                    col += 1
        U4, V4, W4 = U_naive, V_naive, W_naive
        # Rebuild full
        rank_total = 128
        U_full = np.zeros((16, rank_total))
        V_full = np.zeros((32, rank_total))
        W_full = np.zeros((32, rank_total))
        U_full[:, 0:64] = U4
        V_full[0:16, 0:64] = V4
        W_full[0:16, 0:64] = W4
        U_full[:, 64:128] = U4
        V_full[16:32, 64:128] = V4
        W_full[16:32, 64:128] = W4
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Verify full 4x4x8
    # Target: T[A_idx, B_idx, W_idx] = 1 iff A_idx=i*4+k, B_idx=k*8+j, W_idx=j*4+i
    # Spot check a few entries
    # For i=0,k=0,j=0: A=0, B=0, W=0
    # T_recon[0,0,0] = sum_r U_full[0,r]*V_full[0,r]*W_full[0,r]
    
    # Let's do a more thorough check on a random subset
    import random
    random.seed(42)
    errors = []
    for _ in range(100):
        i = random.randint(0, 3)
        k = random.randint(0, 3)
        j = random.randint(0, 7)
        A_idx = i*4 + k
        B_idx = k*8 + j
        W_idx = j*4 + i
        # Expected: 1
        recon_val = 0.0
        for r in range(rank_total):
            recon_val += U_full[A_idx, r] * V_full[B_idx, r] * W_full[W_idx, r]
        if abs(recon_val - 1.0) > 1e-10:
            errors.append((i,k,j, recon_val))
    
    # Also check that non-support entries are 0
    for _ in range(100):
        A_idx = random.randint(0, 15)
        B_idx = random.randint(0, 31)
        W_idx = random.randint(0, 31)
        # Check if this is a support entry
        i = A_idx // 4
        k = A_idx % 4
        k2 = B_idx // 8
        j = B_idx % 8
        j2 = W_idx // 4
        i2 = W_idx % 4
        if not (k == k2 and j == j2 and i == i2):
            # Should be 0
            recon_val = 0.0
            for r in range(rank_total):
                recon_val += U_full[A_idx, r] * V_full[B_idx, r] * W_full[W_idx, r]
            if abs(recon_val) > 1e-10:
                errors.append((A_idx, B_idx, W_idx, recon_val))
    
    if errors:
        # Fallback to naive
        U_naive = np.zeros((16, 128))
        V_naive = np.zeros((32, 128))
        W_naive = np.zeros((32, 128))
        col = 0
        for i in range(4):
            for k in range(4):
                for j in range(8):
                    U_naive[i*4+k, col] = 1
                    V_naive[k*8+j, col] = 1
                    W_naive[j*4+i, col] = 1
                    col += 1
        return {"U": U_naive.tolist(), "V": V_naive.tolist(), "W": W_naive.tolist()}
    
    elapsed = time.time() - start_time
    if elapsed > 140:
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
