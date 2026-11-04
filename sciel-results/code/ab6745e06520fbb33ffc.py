"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """Return rank-7 decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W as lists of lists (dense matrices).
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Indexing: U[i*2+j, t], V[j*2+k, t], W[k*2+i, t]
    """
    # Strassen's 7 products for 2x2 matrices
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = A*B = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-d)(f+h)
    # M7 = (b-c)(e+h)
    # C11 = M1 - M4 + M5 + M6
    # C12 = M3 + M4
    # C21 = M2 + M5
    # C22 = M1 - M2 + M3 + M7
    
    # U[i*2+j, t] = coefficient of A[i,j] in product t
    # V[j*2+k, t] = coefficient of B[j,k] in product t
    # W[k*2+i, t] = coefficient of product t in C[i,k]
    
    # Products:
    # t=0: M1 = (a+d)(e+h) = a*e + a*h + d*e + d*h
    # t=1: M2 = (c+d)*e = c*e + d*e
    # t=2: M3 = a*(f-h) = a*f - a*h
    # t=3: M4 = d*(g-e) = d*g - d*e
    # t=4: M5 = (a+b)*h = a*h + b*h
    # t=5: M6 = (c-d)*(f+h) = c*f + c*h - d*f - d*h
    # t=6: M7 = (b-c)*(e+h) = b*e + b*h - c*e - c*h
    
    U = [[0]*7 for _ in range(4)]  # rows: i*2+j, cols: t
    V = [[0]*7 for _ in range(4)]  # rows: j*2+k, cols: t
    W = [[0]*7 for _ in range(4)]  # rows: k*2+i, cols: t
    
    # t=0: M1 = (a+d)(e+h)
    # U: a=U[0,0]=1, d=U[3,0]=1
    # V: e=V[0,0]=1, h=V[3,0]=1
    # W: C11 += M1 => W[0*2+0,0]=W[0,0]=1
    #      C22 += M1 => W[1*2+1,0]=W[3,0]=1
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # t=1: M2 = (c+d)*e
    # U: c=U[2,1]=1, d=U[3,1]=1
    # V: e=V[0,1]=1
    # W: C21 += M2 => W[0*2+1,1]=W[1,1]=1
    #      C22 -= M2 => W[1*2+1,1]=W[3,1]=-1
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[1,1] = 1; W[3,1] = -1
    
    # t=2: M3 = a*(f-h)
    # U: a=U[0,2]=1
    # V: f=V[1,2]=1, h=V[3,2]=-1
    # W: C12 += M3 => W[1*2+0,2]=W[2,2]=1
    #      C22 += M3 => W[1*2+1,2]=W[3,2]=1
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[2,2] = 1; W[3,2] = 1
    
    # t=3: M4 = d*(g-e)
    # U: d=U[3,3]=1
    # V: g=V[2,3]=1, e=V[0,3]=-1
    # W: C11 -= M4 => W[0*2+0,3]=W[0,3]=-1
    #      C12 += M4 => W[1*2+0,3]=W[2,3]=1
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[0,3] = -1; W[2,3] = 1
    
    # t=4: M5 = (a+b)*h
    # U: a=U[0,4]=1, b=U[1,4]=1
    # V: h=V[3,4]=1
    # W: C11 += M5 => W[0*2+0,4]=W[0,4]=1
    #      C21 += M5 => W[0*2+1,4]=W[1,4]=1
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = 1; W[1,4] = 1
    
    # t=5: M6 = (c-d)*(f+h)
    # U: c=U[2,5]=1, d=U[3,5]=-1
    # V: f=V[1,5]=1, h=V[3,5]=1
    # W: C11 += M6 => W[0*2+0,5]=W[0,5]=1
    U[2,5] = 1; U[3,5] = -1
    V[1,5] = 1; V[3,5] = 1
    W[0,5] = 1
    
    # t=6: M7 = (b-c)*(e+h)
    # U: b=U[1,6]=1, c=U[2,6]=-1
    # V: e=V[0,6]=1, h=V[3,6]=1
    # W: C22 += M7 => W[1*2+1,6]=W[3,6]=1
    U[1,6] = 1; U[2,6] = -1
    V[0,6] = 1; V[3,6] = 1
    W[3,6] = 1
    
    return U, V, W

def _strassen_4x4():
    """Return rank-49 decomposition for 4x4x4 matrix multiplication using Strassen recursively.
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    """
    U2, V2, W2 = _strassen_2x2()
    
    # For 4x4: split into 2x2 blocks
    # A is 4x4, split into 4 blocks of 2x2
    # B is 4x4, split into 4 blocks of 2x2
    # C = A*B, each C block = sum of products of A blocks * B blocks
    # Using Strassen: 7 "multiplications" of 2x2 blocks, each done via Strassen (rank 7)
    # Total rank = 7 * 7 = 49
    
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    # Block indexing: A block (bi,bj) has rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # For 2x2 Strassen at the block level:
    # The 7 block products correspond to linear combinations of 2x2 block multiplications
    # Each block multiplication is a 2x2x2 problem solved by Strassen (rank 7)
    
    # Let's denote the 7 Strassen products at block level as M_t for t in 0..6
    # Each M_t involves specific 2x2 block products
    # At the element level, each 2x2 block product has 7 Strassen sub-products
    
    # Block-level Strassen (same coefficients as 2x2):
    # Block A = [[A00, A01],[A10, A11]], Block B = [[B00, B01],[B10, B11]]
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A11)(B01+B11)
    # M7 = (A01-A10)(B00+B11)
    
    # For each block-level product t (0..6), we need to compute the 2x2 block product
    # using Strassen at the element level (7 sub-products)
    
    # The block-level U, V, W matrices (2x2 block structure):
    # U_block[i*2+j, t] where i,j in {0,1} are block indices
    # This is exactly the same as the 2x2 Strassen U, V, W
    
    # For each block-level product t, and each element-level Strassen product s (0..6):
    # The combined index is t*7 + s
    
    # Element-level: U2[i*2+j, s], V2[j*2+k, s], W2[k*2+i, s]
    # where i,j,k are within the 2x2 block
    
    # Block-level: U_block[bi*2+bj, t], V_block[bj*2+bk, t], W_block[bk*2+bi, t]
    
    # Full element index for A: (bi*2+i)*4 + (bj*2+j) = bi*8 + bj*4 + i*2 + j
    # Wait, A is 4x4, row = bi*2+i, col = bj*2+j
    # So U full index = (bi*2+i)*4 + (bj*2+j)
    
    # Let me be more careful:
    # A is 4x4. Row r = bi*2 + i, Col c = bj*2 + j
    # U index = r*4 + c = (bi*2+i)*4 + (bj*2+j) = bi*8 + i*4 + bj*4 + j
    
    # B is 4x4. Row r = bj*2 + j, Col c = bk*2 + k
    # V index = r*4 + c = (bj*2+j)*4 + (bk*2+k) = bj*8 + j*4 + bk*4 + k
    
    # C is 4x4. Row r = bi*2 + i, Col c = bk*2 + k
    # W index = r*4 + c = (bi*2+i)*4 + (bk*2+k) = bi*8 + i*4 + bk*4 + k
    
    # But the task uses W[k*n+i, r] convention... let me re-read.
    # "The output factor follows the official trace convention W[k*n+i,r]"
    # So W has shape (p*n, rank) where p=4, n=4, so 16 rows.
    # W index = k*4 + i where k is column of C, i is row of C
    # So W[k*4+i, t]
    
    # For A (n x m = 4x4): U[i*4+j, t] where i is row, j is col
    # For B (m x p = 4x4): V[j*4+k, t] where j is row, k is col
    # For C (p x n = 4x4): W[k*4+i, t] where k is col, i is row
    
    # Now combining:
    # A element at (bi*2+i, bj*2+j): U index = (bi*2+i)*4 + (bj*2+j)
    # B element at (bj*2+j, bk*2+k): V index = (bj*2+j)*4 + (bk*2+k)
    # C element at (bi*2+i, bk*2+k): W index = (bk*2+k)*4 + (bi*2+i)
    
    for t in range(7):
        for s in range(7):
            combined_t = t * 7 + s
            # Block-level coefficients
            ub = U2  # U2[i*2+j, t] for 2x2
            vb = V2
            wb = W2
            
            # For this block-level product t, the involved block pairs:
            # We need to iterate over all (bi,bj,bk) combinations
            # and multiply block-level coeff * element-level coeff
            
            for bi in range(2):
                for bj in range(2):
                    for bk in range(2):
                        # Block-level U coefficient for A block (bi,bj) in product t
                        # U_block[bi*2+bj, t]
                        ub_coeff = ub[bi*2+bj, t]
                        if ub_coeff == 0:
                            continue
                        
                        # Block-level V coefficient for B block (bj,bk) in product t
                        vb_coeff = vb[bj*2+bk, t]
                        if vb_coeff == 0:
                            continue
                        
                        # Block-level W coefficient for C block (bi,bk) in product t
                        wb_coeff = wb[bk*2+bi, t]
                        if wb_coeff == 0:
                            continue
                        
                        # Now for element-level product s within this 2x2 block product:
                        # Element-level U: U2[i*2+j, s] for i,j in 0..1
                        # Element-level V: V2[j*2+k, s] for j,k in 0..1
                        # Element-level W: W2[k*2+i, s] for k,i in 0..1
                        
                        for i in range(2):
                            for j in range(2):
                                ue = ub[i*2+j, s]
                                if ue == 0:
                                    continue
                                U_full_idx = (bi*2+i)*4 + (bj*2+j)
                                U[U_full_idx][combined_t] += ub_coeff * ue * vb_coeff * wb_coeff / (vb_coeff * wb_coeff) if vb_coeff != 0 and wb_coeff != 0 else 0
                                # Wait, this is getting confused. Let me restart the logic.
    
    # Let me redo this more carefully.
    # The Kronecker structure: 
    # Full U[row, combined_t] = sum over (block_row, elem_row) of U_block[block_row, t] * U_elem[elem_row, s]
    # where combined_t = t*7 + s
    # But actually, the rank-1 tensor for (t,s) is:
    # U_full[a_idx, t*7+s] = U_block[a_block_idx, t] * U_elem[a_elem_idx, s]
    # where a_idx = a_block_idx * 4 + a_elem_idx (for 4x4, block size 2x2)
    # a_block_idx = bi*2+bj, a_elem_idx = i*2+j
    # a_idx = (bi*2+bj)*4 + (i*2+j)
    
    # Similarly for V and W.
    
    # So for each (t, s):
    # For each (bi, bj, i, j):
    #   U_full[(bi*2+bj)*4 + (i*2+j), t*7+s] += U2[bi*2+bj, t] * U2[i*2+j, s]
    # For each (bj, bk, j, k):
    #   V_full[(bj*2+bk)*4 + (j*2+k), t*7+s] += V2[bj*2+bk, t] * V2[j*2+k, s]
    # For each (bi, bk, i, k):
    #   W_full[(bk*2+i)*4 + (bi*2+k), t*7+s] += W2[bk*2+bi, t] * W2[i*2+k, s]
    
    # Wait, W convention: W[k*n+i, r] where n=4. So W index = k*4+i.
    # In block terms: k = bk*2+k_elem, i = bi*2+i_elem
    # W index = (bk*2+k_elem)*4 + (bi*2+i_elem)
    # W_block index for 2x2: bk*2+bi
    # W_elem index for 2x2: k_elem*2+i_elem
    
    # Let me just build it directly.
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    for t in range(7):
        for s in range(7):
            ct = t * 7 + s
            for bi in range(2):
                for bj in range(2):
                    ub = U2[bi*2+bj, t]
                    for i in range(2):
                        for j in range(2):
                            ue = U2[i*2+j, s]
                            if ub != 0 and ue != 0:
                                U[(bi*2+i)*4 + (bj*2+j)][ct] += ub * ue
            
            for bj in range(2):
                for bk in range(2):
                    vb = V2[bj*2+bk, t]
                    for j in range(2):
                        for k in range(2):
                            ve = V2[j*2+k, s]
                            if vb != 0 and ve != 0:
                                V[(bj*2+j)*4 + (bk*2+k)][ct] += vb * ve
            
            for bi in range(2):
                for bk in range(2):
                    wb = W2[bk*2+bi, t]
                    for i in range(2):
                        for k in range(2):
                            we = W2[k*2+i, s]
                            if wb != 0 and we != 0:
                                W[(bk*2+k)*4 + (bi*2+i)][ct] += wb * we
    
    return U, V, W

def _strassen_4x8():
    """Return rank decomposition for 4x4x8 (4x4 matrix times 4x8 matrix).
    Pad to 4x4x8: A is 4x4, B is 4x8, C is 4x8.
    Split B into two 4x4 blocks: B = [B1 | B2]
    C = A*B1 + A*B2
    Each A*B_i is 4x4x4 with rank 49.
    Total rank = 49 + 49 = 98.
    
    But we can do better: use Strassen on the 4x4x4 part.
    Actually, 4x4x8: A(4x4) * B(4x8) = C(4x8)
    This is like multiplying 4x4 by two 4x4 matrices.
    Rank = 2 * M(4,4,4) = 2 * 49 = 98.
    
    Hmm, that's not great. Let me think differently.
    
    Alternative: view 4x4x8 as 4x4x4 x 2 (in the output dimension).
    Actually, the tensor is T[i,j,k] for i in 0..3, j in 0..3, k in 0..7.
    T[i,j,k] = sum over l of A[i,l] * B[l,k]
    
    For k in 0..3: T[i,j,k] involves B[l,k] for l in 0..3
    For k in 4..7: T[i,j,k] involves B[l,k] for l in 0..3
    
    So we have two independent 4x4x4 problems (one for k=0..3, one for k=4..7).
    Total rank = 2 * 49 = 98.
    """
    U4, V4, W4 = _strassen_4x4()
    
    U = [[0]*98 for _ in range(16)]  # 4*4 = 16
    V = [[0]*98 for _ in range(32)]  # 4*8 = 32
    W = [[0]*98 for _ in range(32)]  # 8*4 = 32
    
    # First half: k=0..3
    for i in range(16):
        for t in range(49):
            U[i][t] = U4[i][t]
            V[i][t] = V4[i][t]  # V index: j*4+k where j in 0..3, k in 0..3
            W[i][t] = W4[i][t]  # W index: k*4+i where k in 0..3, i in 0..3
    
    # Second half: k=4..7
    for i in range(16):
        for t in range(49):
            U[i][t+49] = U4[i][t]
            # V index for second block: j*4+(k+4) = j*4+k+4
            # V4[i][t] has index j*4+k for k in 0..3
            # So V[j*4+k+4][t+49] = V4[j*4+k][t]
            for j in range(4):
                for k in range(4):
                    v4_idx = j*4+k
                    v_full_idx = j*4+k+4
                    V[v_full_idx][t+49] = V4[v4_idx][t]
            # W index for second block: (k+4)*4+i = k*4+i+16
            for k in range(4):
                for i in range(4):
                    w4_idx = k*4+i
                    w_full_idx = k*4+i+16
                    W[w_full_idx][t+49] = W4[w4_idx][t]
    
    return U, V, W

def _pad_and_project(U_full, V_full, W_full, n, m, p):
    """Project a decomposition for padded dimensions back to original.
    
    U_full: shape (n_pad * m_pad, rank)
    V_full: shape (m_pad * p_pad, rank)  
    W_full: shape (p_pad * n_pad, rank)
    
    We need:
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    
    The padding zeros out rows/cols that don't correspond to original dimensions.
    For U: row index i*m_pad + j. We keep rows where i < n and j < m.
    For V: row index j*p_pad + k. We keep rows where j < m and k < p.
    For W: row index k*n_pad + i. We keep rows where k < p and i < n.
    """
    n_pad = n + (4 - n % 4) % 4 if n % 4 != 0 else n
    m_pad = m + (4 - m % 4) % 4 if m % 4 != 0 else m
    p_pad = p + (4 - p % 4) % 4 if p % 4 != 0 else p
    
    # Actually, let's just use the exact padded dimensions
    # For 3x4x6 padded to 4x4x8:
    # n_pad=4, m_pad=4, p_pad=8
    
    rank = len(U_full[0])
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Map original indices to padded indices
    for i in range(n):
        for j in range(m):
            u_idx = i * m_pad + j
            for t in range(rank):
                U[i*m+j][t] = U_full[u_idx][t]
    
    for j in range(m):
        for k in range(p):
            v_idx = j * p_pad + k
            for t in range(rank):
                V[j*p+k][t] = V_full[v_idx][t]
    
    for k in range(p):
        for i in range(n):
            w_idx = k * n_pad + i
            for t in range(rank):
                W[k*n+i][t] = W_full[w_idx][t]
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin
    
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # Schoolbook decomposition: rank = n*m*p = 72
    # This is guaranteed correct and fast.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() - start_time > deadline - 5.0:
                    # Should not happen, but guard anyway
                    break
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
