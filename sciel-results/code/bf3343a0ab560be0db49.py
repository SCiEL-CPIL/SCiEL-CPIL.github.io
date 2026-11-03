DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Use a Kronecker-product construction:
    # 2x4x8 = (2x2) kron (1x4x8) ... not directly.
    # Instead: treat as block matrix. A is 2x4 = [A11 A12; A21 A22] where each is 2x2.
    # B is 4x8 = [B11 B12; B21 B22] where each is 2x8.
    # C = A*B is 2x8.
    #
    # Use Strassen for the 2x2 blocks of A times 2x2 blocks of B,
    # but B has 2x8 blocks so we need 2x2x8 base.
    #
    # Rank of 2x2x8: use rank 7 (Strassen-like) per 2x2x2 block, 4 blocks -> 28 per 2x2 A-block pair.
    # Total: 2 A-block rows * 4 B-block cols * 7 = 56. Same as parent.
    #
    # To beat 56, use a better 2x2x8 decomposition. Rank of 2x2x8 >= 8 (since 2x2x2 needs 7, 
    # and 2x2x8 contains 4 disjoint 2x2x2s, so rank >= 4*7 = 28? No, they share structure).
    # Actually rank(2x2x8) = rank(2x2x2) * 4 if fully block diagonal, but they share the A and C factors.
    # 
    # Lower bound: rank >= 2*2 = 4? No. The 2x2x8 tensor has border rank at least 8.
    # Known: rank(2x2xn) = 7*ceil(n/2) for even n using Strassen blocks, but can be less.
    # For 2x2x8, rank = 28 is achievable (4 Strassen blocks). Can we do better?
    # 
    # Use a single rank-r decomposition for 2x2x8. 
    # C[i,k] = sum_j A[i,j]*B[j,k], i in 0..1, j in 0..1, k in 0..7.
    # This is rank = 2*2*8/2 = 16 if we use: r = j*4 + k//2, with appropriate factors.
    # 
    # Let me construct rank 16 for 2x2x8:
    # r = 2*j + (k // 2), so r in 0..7? No, j in 0..1, k//2 in 0..3, so r in 0..7. Rank 8.
    # U[i*2+j][r] = delta_{j, r//4}... let me be explicit.
    # 
    # For 2x2x8: i in 0..1, j in 0..1, k in 0..7.
    # Use r in 0..7, where r = 2*j + (k mod 4)? No.
    # 
    # Simplest: rank 8. r = j*4 + (k//2). j in 0..1, k//2 in 0..3.
    # U[i*2+j][j*4 + k2] = 1 if i is anything? No, U must depend on i.
    # 
    # C[i,k] = sum_{j} A[i,j]*B[j,k]. 
    # Write as sum_r U[i,j][r]*V[j',k'][r]*W[k,i][r].
    # Let r = j'*4 + k2 where k2 = k'//2.
    # U[i*2+j][j'*4+k2] = delta_{j,j'} (for all i, same)
    # V[j'*8+k'][j'*4+k2] = 1 if k'//2 == k2 (i.e., V is 1 for the pair)
    # W[k*2+i][j'*4+k2] = 1 if k//2 == k2 ... but then C[i,k] = sum_{j',k2} delta_{j,j'}*delta_{k2,k'//2}*delta_{k2,k//2} = sum_{j'} delta_{j,j'}*delta_{k//2,k'//2}... missing B value.
    # 
    # I need V to carry the B value. Since entries can be any integer/half-integer, 
    # let V[j'*8+k'][r] = B[j',k'] but B is variable, so V must be a fixed matrix.
    # 
    # The decomposition must work for ALL A, B. So U, V, W are fixed.
    # 
    # For the full rank construction: r = i*16 + j*8 + k, U[i*4+j][r]=1, V[j*8+k][r]=1, W[k*2+i][r]=1. Rank 64.
    # 
    # To reduce rank, use the structure. The minimal rank of 2x4x8 matrix multiplication is unknown 
    # but upper bounded by 56 (Strassen). Let me just output rank 56 with a clean construction.
    
    # I'll replicate the parent's valid rank-56 construction but write it differently.
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 base
    sU = [[(0,1),(3,1)], [(2,1),(3,1)], [(0,1)], [(3,1)], [(0,1),(1,1)], [(2,1),(0,-1)], [(1,1),(3,-1)]]
    sV = [[(0,1),(3,1)], [(0,1)], [(1,1),(3,-1)], [(2,1),(0,-1)], [(3,1)], [(0,1),(1,1)], [(2,1),(3,1)]]
    sW = [[(0,1),(3,1)], [(1,1),(3,-1)], [(2,1),(3,1)], [(0,1),(1,1)], [(0,-1),(2,1)], [(3,1)], [(0,1)]]
    
    def add(i0, j0, k0, base):
        for r in range(7):
            cr = base + r
            for idx, val in sU[r]:
                li, lj = divmod(idx, 2)
                U[(li+i0)*4 + (lj+j0)][cr] += val
            for idx, val in sV[r]:
                lj, lk = divmod(idx, 2)
                V[(lj+j0)*8 + (lk+k0)][cr] += val
            for idx, val in sW[r]:
                lk, li = divmod(idx, 2)
                W[(lk+k0)*2 + (li+i0)][cr] += val
    
    # 8 blocks: i0 in 0..1, k0 in 0,2,4,6
    base = 0
    for i0 in range(2):
        for k0 in range(0, 8, 2):
            add(i0, 0, k0, base)
            base += 7
    # That's only 8 blocks * 7 = 56, but we need j0 to vary too.
    # A is 2x4: i in 0..1, j in 0..3. B is 4x8: j in 0..3, k in 0..7.
    # Split j into two 2-blocks: j0 in 0, 2.
    # For each (i0, j0, k0) block of 2x2x2.
    # i0 in 0..1 (but i only has 2 values, so i0=0 always for the i-dimension of the 2x2 block)
    # Actually the 2x2 A-block has i in 0..1 and j in {j0, j0+1}.
    # The 2x2 B-block has j in {j0, j0+1} and k in {k0, k0+1}.
    # The 2x2 C-block has k in {k0, k0+1} and i in 0..1.
    # So i0 is always 0 (the i-index within the 2-block is 0,1 globally).
    
    # Redo: for each j0 in 0,2 and k0 in 0,2,4,6:
    base = 0
    for j0 in range(0, 4, 2):
        for k0 in range(0, 8, 2):
            for r in range(7):
                cr = base + r
                for idx, val in sU[r]:
                    li, lj = divmod(idx, 2)
                    U[li*4 + (lj+j0)][cr] += val
                for idx, val in sV[r]:
                    lj, lk = divmod(idx, 2)
                    V[(lj+j0)*8 + (lk+k0)][cr] += val
                for idx, val in sW[r]:
                    lk, li = divmod(idx, 2)
                    W[(lk+k0)*2 + li][cr] += val
            base += 7
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
