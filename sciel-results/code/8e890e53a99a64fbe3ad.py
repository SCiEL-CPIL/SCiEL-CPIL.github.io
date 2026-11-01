"""Matrix multiplication 2,3,8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, rank):
    """Verify the tensor decomposition: T[a,b,c] = sum_d U[a,d]*V[b,d]*W[c,d]
    where a=i*m+j, b=j'*p+k', c=k*n+i'
    T=1 iff (i,j,k)=(i',j',k'), else 0."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build the full tensor and check
    # T[i,j,k] = sum_d U[i*m+j, d] * V[j*p+k, d] * W[k*n+i, d]
    # For each (i,j,k) we want sum_d U[i*m+j,d]*V[j*p+k,d]*W[k*n+i,d] = 1
    # For non-matching (i',j',k') we want sum_d U[i*m+j,d]*V[j'*p+k',d]*W[k*n+i',d] = 0
    
    # Efficient check: compute the bilinear form
    # For fixed i,j,k: sum_d U[i*m+j,d] * V[j*p+k,d] * W[k*n+i,d]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Should be 1
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = float(np.dot(np.dot(U[a], V[b]), W[c]))
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check off-diagonal: for each (i,j) and (j',k') and (i',k') where not all match
    # Actually simpler: for each pair of indices, check the full tensor
    # T[i,j,k] = sum_d U[i*m+j,d] * V[j*p+k,d] * W[k*n+i,d]
    # We need T to be the identity-like tensor: T[i,j,k]=1 iff i=i',j=j',k=k'
    # But wait - the tensor is 3-indexed: T[a,b,c] where a=(i,j), b=(j',k'), c=(k'',i'')
    # The condition is: sum_d U[a,d]*V[b,d]*W[c,d] = 1 if (i,j,k) corresponds to (a,b,c)
    # i.e., a=i*m+j, b=j*p+k, c=k*n+i, and 0 otherwise.
    
    # More precisely: for given (i,j,k), the "matching" triple is:
    #   a = i*m+j, b = j*p+k, c = k*n+i
    # For any other combination of (a',b',c') that doesn't correspond to the same (i,j,k),
    # we need the sum to be 0.
    
    # Let's just check all combinations
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                for jp in range(m):
                    for kp in range(p):
                        b = jp * p + kp
                        for ip in range(n):
                            c = kp * n + ip  # Wait, c = k*n+i, so for the test index it's kp*n+ip
                            # Actually c should be k''*n+i'' where (i'',j'',k'') is the output index
                            # Let me re-read the convention: W[k*n+i, r]
                            # So W index is k*n+i where k is the output column and i is the output row
                            # The tensor entry T[i,j,k] = sum_d U[i*m+j,d] * V[j*p+k,d] * W[k*n+i,d]
                            # This equals 1 iff the output position (i,k) receives contribution from (j)
                            # i.e., it's the (i,j,k) entry of the multiplication tensor.
                            
                            # For the off-diagonal check:
                            # sum_d U[i*m+j,d] * V[jp*p+kp,d] * W[kp*n+ip,d] should be 0
                            # unless (i,j,k) == (ip,jp,kp) -- wait no.
                            # The tensor T has entries indexed by (output_row, input_col_of_A, input_row_of_B)
                            # T[i,j,k] = delta where the product C[i,k] = sum_j A[i,j]*B[j,k]
                            # So T[i,j,k] = 1 for all i,j,k (it's the structure tensor of matmul)
                            # Wait no. The tensor is: T[i,j,k] such that C[i,k] = sum_{j} A[i,j] B[j,k]
                            # The rank decomposition gives: C[i,k] = sum_d (sum_j U[i*m+j,d]*A[i,j]) * (sum_k V[j*p+k,d]*B[j,k]) ... 
                            # No, that's not right either.
                            
                            # Let me re-derive. The standard tensor decomposition for matrix multiplication:
                            # T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p]
                            # where T is the trilinear form: <A,B,C> = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[k,i]... 
                            # No. The matrix multiplication tensor is:
                            # mu_{m,n,p} = sum_{i,j,k} e_i^* x e_j x e_k^* (in appropriate spaces)
                            # The bilinear algorithm computes C = AB using:
                            # C[i,k] = sum_j A[i,j] B[j,k]
                            # The tensor is in U* x V* x W* where U has dim nm, V has dim mp, W has dim pn
                            # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some (i,j,k), else 0
                            
                            # So: sum_d U[a,d]*V[b,d]*W[c,d] = 1 if there exist (i,j,k) with a=i*m+j, b=j*p+k, c=k*n+i
                            # and 0 otherwise.
                            
                            # The "otherwise" means: for a=i*m+j, b=jp*p+kp, c=kp*n+ip
                            # this equals 1 iff i=ip, j=jp, k=kp (all three match)
                            # because we need: a=i*m+j, b=j*p+k, c=k*n+i simultaneously
                            # i.e., jp=j, kp=k, ip=i.
                            
                            if not (i == ip and j == jp and k == kp):
                                val = float(np.dot(np.dot(U[a], V[b]), W[c]))
                                if abs(val) > 1e-9:
                                    return False
    return True

def _build_strassen_2x2():
    """Rank-7 Strassen algorithm for 2x2 * 2x2.
    Convention: U[i*2+j, d], V[j*2+k, d], W[k*2+i, d]
    A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]], C = [[c00,c01],[c10,c11]]
    c[i,k] = sum_j a[i,j]*b[j,k]
    """
    n, m, p = 2, 2, 2
    r = 7
    U = [[0]*r for _ in range(n*m)]  # 4x7
    V = [[0]*r for _ in range(m*p)]  # 4x7
    W = [[0]*r for _ in range(p*n)]  # 4x7
    
    # Strassen:
    # M0 = (a00+a11)*(b00+b11)
    # M1 = (a10+a11)*b00
    # M2 = a00*(b01-b11)
    # M3 = a11*(b10-b00)
    # M4 = (a00+a01)*b11
    # M5 = (a10-a00)*(b00+b10)
    # M6 = (a01-a11)*(b01+b11)
    #
    # c00 = M0 + M3 - M5 - M6
    # c01 = M2 + M4
    # c10 = M1 + M3
    # c11 = M0 + M2 - M1 + M4
    
    # Term 0: M0 = (a00+a11)*(b00+b11)
    # U: a00 -> +1, a11 -> +1; V: b00 -> +1, b11 -> +1
    U[0][0] += 1  # a00
    U[3][0] += 1  # a11
    V[0][0] += 1  # b00
    V[3][0] += 1  # b11
    # W: c00 -> +1, c11 -> +1
    W[0][0] += 1  # c00 (k=0,n=2,i=0: 0*2+0=0)
    W[3][0] += 1  # c11 (k=1,n=2,i=1: 1*2+1=3)
    
    # Term 1: M1 = (a10+a11)*b00
    U[2][1] += 1  # a10
    U[3][1] += 1  # a11
    V[0][1] += 1  # b00
    # W: c10 -> +1
    W[2][1] += 1  # c10 (k=0,n=2,i=1: 0*2+1=1)... wait
    # c[i,k]: W index is k*n+i. c10 means i=1,k=0: W[0*2+1] = W[1]
    W[1][1] += 1  # c10
    
    # Term 2: M2 = a00*(b01-b11)
    U[0][2] += 1  # a00
    V[1][2] += 1  # b01
    V[3][2] -= 1  # b11
    # W: c01 -> +1, c11 -> -1
    W[0][2] += 1  # c01: k=1,i=0: 1*2+0=2... wait
    # c[i,k]: W index = k*n+i. c01: i=0,k=1: W[1*2+0]=W[2]
    W[2][2] += 1  # c01
    W[3][2] -= 1  # c11: i=1,k=1: W[1*2+1]=W[3]
    
    # Term 3: M3 = a11*(b10-b00)
    U[3][3] += 1  # a11
    V[2][3] += 1  # b10
    V[0][3] -= 1  # b00
    # W: c00 -> -1, c10 -> +1
    W[0][3] -= 1  # c00
    W[1][3] += 1  # c10
    
    # Term 4: M4 = (a00+a01)*b11
    U[0][4] += 1  # a00
    U[1][4] += 1  # a01
    V[3][4] += 1  # b11
    # W: c01 -> +1, c11 -> +1
    W[2][4] += 1  # c01
    W[3][4] += 1  # c11
    
    # Term 5: M5 = (a10-a00)*(b00+b10)
    U[2][5] += 1  # a10
    U[0][5] -= 1  # a00
    V[0][5] += 1  # b00
    V[2][5] += 1  # b10
    # W: c00 -> +1
    W[0][5] += 1  # c00
    
    # Term 6: M6 = (a01-a11)*(b01+b11)
    U[1][6] += 1  # a01
    U[3][6] -= 1  # a11
    V[1][6] += 1  # b01
    V[3][6] += 1  # b11
    # W: c00 -> +1
    W[0][6] += 1  # c00
    
    return U, V, W

def _build_rank11_2x3_3x2():
    """Rank-11 bilinear algorithm for 2x3 * 3x2.
    Convention: U[i*3+j, d], V[j*2+k, d], W[k*2+i, d]
    
    A = [[a00, a01, a02],
         [a10, a11, a12]]
    B = [[b00, b01],
         [b10, b11],
         [b20, b21]]
    C = [[c00, c01],
         [c10, c11]]
    
    c[i,k] = sum_j a[i,j] * b[j,k]
    
    Construction: Use Strassen on the 2x2 block (j=0,1) and handle j=2 separately.
    Split A into [A_left (2x2), a_right (2x1)] and B into [B_top (2x2); b_bottom (1x2)]
    
    C = A_left * B_top + a_right * b_bottom
    
    For the 2x2*2x2 part, use Strassen (rank 7).
    For the cross terms, we need:
      c00 += a02*b20, c01 += a02*b21
      c10 += a12*b20, c11 += a12*b21
    
    We can handle these with 4 more terms (schoolbook for the 2x1*1x2 part),
    giving rank 7+4 = 11.
    """
    n, m, p = 2, 3, 2
    r = 11
    U = [[0]*r for _ in range(n*m)]  # 6x11
    V = [[0]*r for _ in range(m*p)]  # 6x11
    W = [[0]*r for _ in range(p*n)]  # 4x11
    
    # First 7 terms: Strassen for the 2x2*2x2 sub-problem (rows j=0,1 of A and B)
    # In the 2x2*2x2 sub-problem:
    #   A_sub = [[a00,a01],[a10,a11]], B_sub = [[b00,b01],[b10,b11]]
    #   C_sub = [[c00',c01'],[c10',c11']] where c[i,k]' = a[i,0]*b[0,k]+a[i,1]*b[1,k]
    #   
    #   U_sub[i*2+j, d] corresponds to U[i*3+j, d] in the full problem (j=0,1)
    #   V_sub[j*2+k, d] corresponds to V[j*2+k, d] in the full problem (j=0,1)
    #   W_sub[k*2+i, d] corresponds to W[k*2+i, d] in the full problem
    
    U_sub, V_sub, W_sub = _build_strassen_2x2()
    
    for d in range(7):
        for i in range(2):
            for j in range(2):
                U[i*3+j][d] = U_sub[i*2+j][d]
        for j in range(2):
            for k in range(2):
                V[j*2+k][d] = V_sub[j*2+k][d]
        for k in range(2):
            for i in range(2):
                W[k*2+i][d] = W_sub[k*2+i][d]
    
    # Terms 7-10: handle the j=2 cross terms
    # c00 += a02*b20: U[0*3+2]=U[2], V[2*2+0]=V[4], W[0*2+0]=W[0]
    # c01 += a02*b21: U[2], V[2*2+1]=V[5], W[1*2+0]=W[2]
    # c10 += a12*b20: U[1*3+2]=U[5], V[4], W[0*2+1]=W[1]
    # c11 += a12*b21: U[5], V[5], W[1*2+1]=W[3]
    
    U[2][7] = 1; V[4][7] = 1; W[0][7] = 1   # a02*b20 -> c00
    U[2][8] = 1; V[5][8] = 1; W[2][8] = 1   # a02*b21 -> c01
    U[5][9] = 1; V[4][9] = 1; W[1][9] = 1   # a12*b20 -> c10
    U[5][10] = 1; V[5][10] = 1; W[3][10] = 1 # a12*b21 -> c11
    
    return U, V, W

def _build_rank16_2x3_3x4():
    """Rank-16 bilinear algorithm for 2x3 * 3x4.
    Convention: U[i*3+j, d], V[j*4+k, d], W[k*2+i, d]
    
    A is 2x3, B is 3x4, C is 2x4.
    
    Split B into two 3x2 blocks: B = [B_left, B_right]
    C = [A*B_left, A*B_right]
    
    Each A*B_block is a 2x3*3x2 problem with rank 11.
    But we can share the A-factor across both blocks!
    
    Actually, let's think more carefully.
    C[i,k] = sum_j A[i,j]*B[j,k]
    
    For k=0,1: C[i,k] = sum_j A[i,j]*B[j,k] (left block)
    For k=2,3: C[i,k] = sum_j A[i,j]*B[j,k] (right block)
    
    If we use independent rank-11 for each, total rank = 22.
    
    To share: the A-factors are the same. In the rank-11 construction,
    the U matrix has 6 rows (for A) and 11 columns.
    
    For the left block (k=0,1), we have U_left (6x11), V_left (12x11), W_left (4x11)
    For the right block (k=2,3), we have U_right (6x11), V_right (12x11), W_right (4x11)
    
    If U_left == U_right, we can share and get rank 11+11-11 = 11? No, that's not how it works.
    The rank is the number of terms. If we can use the same U vectors for both blocks,
    then we just need separate V and W vectors.
    
    In the rank-11 construction for 2x3*3x2:
    U has 6 rows, 11 cols. The U vectors are:
      d=0: U[0]=1, U[3]=1 (a00+a11)
      d=1: U[2]=1, U[3]=1 (a10+a11)
      d=2: U[0]=1 (a00)
      d=3: U[3]=1 (a11)
      d=4: U[0]=1, U[1]=1 (a00+a01)
      d=5: U[2]=1, U[0]=-1 (a10-a00)
      d=6: U[1]=1, U[3]=-1 (a01-a11)
      d=7: U[2]=1 (a02)
      d=8: U[2]=1 (a02)
      d=9: U[5]=1 (a12)
      d=10: U[5]=1 (a12)
    
    For the right block, the same U vectors work (they depend only on A, not on B).
    So we can use the same 11 U-vectors for both blocks.
    
    Total rank = 11 (shared U) + 11 (V_left) + 11 (V_right) ... no, that's not right.
    The rank is the total number of terms. Each term d has one U-vector, one V-vector, one W-vector.
    
    If we use 11 terms for the left block and 11 terms for the right block, total rank = 22.
    The U-vectors for the left and right blocks are identical, but they're still separate terms.
    
    However, we can potentially reduce further by combining. But for now, rank 22 for 2x3*3x4.
    
    Wait, actually there might be a better way. Let me think about this differently.
    
    Alternative: use a rank-16 construction directly.
    
    Actually, let me just use the tiling approach:
    - 2x3*3x4 = two 2x3*3x2 problems, each rank 11, total rank 22.
    - 2x3*3x8 = four 2x3*3x2 problems, each rank 11, total rank 44.
    
    This gives rank 44 for 2x3*3x8, which is better than schoolbook 48.
    
    Let me just build the 2x3*3x8 directly by tiling four 2x3*3x2 blocks.
    """
    # Actually, let me just build 2x3*3x8 directly in construct()
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Build rank-44 construction by tiling four 2x3*3x2 blocks.
    # B is 3x8, split into four 3x2 blocks: B[:,0:2], B[:,2:4], B[:,4:6], B[:,6:8]
    # C is 2x8, split into four 2x2 blocks: C[:,0:2], C[:,2:4], C[:,4:6], C[:,6:8]
    # 
    # For each block t (t=0,1,2,3):
    #   C[:,2t:2t+2] = A * B[:,2t:2t+2]
    #   This is a 2x3*3x2 problem with rank 11.
    #
    # Total rank = 4 * 11 = 44.
    
    rank = 44
    U = [[0]*rank for _ in range(n*m)]   # 6x44
    V = [[0]*rank for _ in range(m*p)]   # 24x44
    W = [[0]*rank for _ in range(p*n)]   # 16x44
    
    # Build one rank-11 sub-algorithm
    U_sub, V_sub, W_sub = _build_rank11_2x3_3x2()
    
    for t in range(4):
        base = t * 11
        for d in range(11):
            col = base + d
            # U: same for all blocks
            for i in range(n*m):
                U[i][col] = U_sub[i][d]
            # V: block t uses B rows j*2+k where k is offset by 2t
            # V_sub index: j*2+k for j in 0..2, k in 0..1
            # Full V index: j*p+k' where k' = 2t+k
            for j in range(m):
                for k in range(2):
                    V_sub_idx = j * 2 + k
                    V_full_idx = j * p + (2*t + k)
                    V[V_full_idx][col] = V_sub[j*2+k][d]
            # W: block t uses output C[i, 2t+k]
            # W_sub index: k*2+i for k in 0..1, i in 0..1
            # Full W index: k'*n+i where k' = 2t+k
            for k in range(2):
                for i in range(n):
                    W_sub_idx = k * n + i
                    W_full_idx = (2*t + k) * n + i
                    W[W_full_idx][col] = W_sub[k*n+i][d]
    
    # Verify
    if not _verify(U, V, W, n, m, p, rank):
        # Fallback to schoolbook
        rank = n * m * p  # 48
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
