"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 Strassen, rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen's 7 products for 2x2 matrix multiplication
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    # C11 = p1+p4-p5+p7
    # C12 = p3+p5
    # C21 = p2+p4
    # C22 = p1-p2+p3+p6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Row indices for U: i*2+j -> (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # Row indices for V: j*2+k -> (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # Row indices for W: k*2+i -> (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # p1 = (a+d)(e+h): U[0,0]=1, U[3,0]=1; V[0,0]=1, V[3,0]=1; W contributes to C11 and C22
    # W[k*2+i]: C11 is i=0,k=0 -> W[0,0]; C22 is i=1,k=1 -> W[3,0]
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C11 += 1, C22 += 1
    
    # p2 = (c+d)e: U[2,1]=1, U[3,1]=1; V[0,1]=1; W: C21 (i=1,k=0) -> W[2,1]=1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1  # C21 += 1
    
    # p3 = a(f-h): U[0,2]=1; V[1,2]=1, V[3,2]=-1; W: C12 (i=0,k=1) -> W[1,2]=1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1  # C12 += 1
    
    # p4 = d(g-e): U[3,3]=1; V[2,3]=1, V[0,3]=-1; W: C11 (i=0,k=0) -> W[0,3]=1, C21 (i=1,k=0) -> W[2,3]=1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1  # C11 += 1, C21 += 1
    
    # p5 = (a+b)h: U[0,4]=1, U[1,4]=1; V[3,4]=1; W: C11 (i=0,k=0) -> W[0,4]=-1, C12 (i=0,k=1) -> W[1,4]=1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1  # C11 -= 1, C12 += 1
    
    # p6 = (c-a)(e+f): U[2,5]=1, U[0,5]=-1; V[0,5]=1, V[1,5]=1; W: C22 (i=1,k=1) -> W[3,5]=1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C22 += 1
    
    # p7 = (b-d)(g+h): U[1,6]=1, U[3,6]=-1; V[2,6]=1, V[3,6]=1; W: C11 (i=0,k=0) -> W[0,6]=1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C11 += 1
    
    return U, V, W

def _schoolbook_block(nn, mm, pp):
    """Return (U, V, W) for nn x mm x pp schoolbook, rank nn*mm*pp."""
    rank = nn * mm * pp
    U = np.zeros((nn * mm, rank))
    V = np.zeros((mm * pp, rank))
    W = np.zeros((pp * nn, rank))
    col = 0
    for i in range(nn):
        for j in range(mm):
            for k in range(pp):
                U[i * mm + j, col] = 1
                V[j * pp + k, col] = 1
                W[k * nn + i, col] = 1
                col += 1
    return U, V, W

def _block_combine(blocks, n, m, p):
    """Combine list of (U_block, V_block, W_block) into full (U, V, W) matrices.
    Each block has shape (n*m, rank_b), (m*p, rank_b), (p*n, rank_b)
    """
    total_rank = sum(b[0].shape[1] for b in blocks)
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for Ub, Vb, Wb in blocks:
        rank_b = Ub.shape[1]
        U[:, col:col+rank_b] = Ub
        V[:, col:col+rank_b] = Vb
        W[:, col:col+rank_b] = Wb
        col += rank_b
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify that T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(U.shape[1]):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-9:
                    return False, (i, j, k, val)
    return True, None

def construct():
    n, m, p = 4, 5, 5
    deadline = time.time() + 110
    
    # Schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook_block(n, m, p)
    best = {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}
    best_rank = n * m * p
    
    # Try block decomposition: split i into 2+2, j into 2+3, k into 2+3
    # Blocks: (i_split, j_split, k_split)
    # (2,2,2): Strassen, rank 7
    # (2,2,3): 14
    # (2,3,2): 9
    # (2,3,3): 11
    # (2,2,2): 7
    # (2,2,3): 14
    # (2,3,2): 9
    # (2,3,3): 11
    # Total: 82
    
    # Let's build each block type
    
    # 2x2x2: Strassen
    U222, V222, W222 = _strassen_2x2x2()
    
    # 2x2x3: schoolbook would be 12, but we can do better
    # C(2x3) = A(2x2) * B(2x3)
    # Split B into B1(2x2) and B2(2x1)
    # A*B1: 2x2x2, rank 7
    # A*B2: 2x2x1, rank 2 (outer products)
    # Total: 9
    # Wait, 2x2x1: C[i,k] = sum_j A[i,j]*B[j,k]. With k only 0.
    # C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
    # This is a dot product, rank 2 (two rank-1 terms)
    
    U223 = np.zeros((4, 9))
    V223 = np.zeros((6, 9))
    W223 = np.zeros((6, 9))
    
    # First 7: Strassen for A(2x2)*B1(2x2) -> C1(2x2)
    U223[:, :7] = U222
    # V: B1 is j*2+k for j in 0,1, k in 0,1 -> rows 0,1,2,3
    V223[:4, :7] = V222
    # W: C1 is k*2+i for k in 0,1, i in 0,1 -> rows 0,1,2,3
    W223[:4, :7] = W222
    
    # Last 2: A(2x2)*B2(2x1) -> C2(2x1)
    # B2: j*3+k for j in 0,1, k=2 -> rows 2, 5
    # C2: k*2+i for k=2, i in 0,1 -> rows 4, 5
    # C2[i,0] = A[i,0]*B2[0,0] + A[i,1]*B2[1,0]
    # Term 1: A[i,0]*B2[0,0]: U[i*2+0, r]=1, V[2, r]=1, W[4+i, r]=1
    # Term 2: A[i,1]*B2[1,0]: U[i*2+1, r]=1, V[5, r]=1, W[4+i, r]=1
    U223[0, 7] = 1; U223[2, 7] = 1  # A[0,0], A[1,0]
    V223[2, 7] = 1  # B2[0,0] = B[0,2]
    W223[4, 7] = 1; W223[5, 7] = 1  # C[0,2], C[1,2]
    
    U223[1, 8] = 1; U223[3, 8] = 1  # A[0,1], A[1,1]
    V223[5, 8] = 1  # B2[1,0] = B[1,2]
    W223[4, 8] = 1; W223[5, 8] = 1  # C[0,2], C[1,2]
    
    # 2x3x2: C(2x2) = A(2x3) * B(3x2)
    # Split A into A1(2x2) and A2(2x1)
    # A1*B: 2x2x2, rank 7
    # A2*B: 2x1x2, rank 2
    # Total: 9
    
    U232 = np.zeros((6, 9))
    V232 = np.zeros((6, 9))
    W232 = np.zeros((4, 9))
    
    # First 7: A1(2x2)*B(3x2) -> C(2x2)
    # Wait, B is 3x2, not 2x2. Need to handle this.
    # A1(2x2) * B(3x2): C[i,k] = sum_{j=0}^{1} A1[i,j]*B[j,k]
    # This is 2x2x2 if we only use first 2 rows of B. But B has 3 rows.
    # Actually A1 only has 2 columns, so j goes 0 to 1, and B[j,k] for j=0,1.
    # So it's effectively 2x2x2 with B being the 2x2 submatrix B[0:2,:].
    # U: i*3+j for i in 0,1, j in 0,1 -> rows 0,1,3,4
    # V: j*2+k for j in 0,1, k in 0,1 -> rows 0,1,2,3
    # W: k*2+i for k in 0,1, i in 0,1 -> rows 0,1,2,3
    
    # Map Strassen U (4x7) to rows 0,1,3,4
    U232[0, :7] = U222[0, :]  # i=0,j=0
    U232[3, :7] = U222[1, :]  # i=0,j=1
    U232[1, :7] = U222[2, :]  # i=1,j=0
    U232[4, :7] = U222[3, :]  # i=1,j=1
    
    V232[:4, :7] = V222
    W232[:4, :7] = W222
    
    # Last 2: A2(2x1)*B(3x2) -> C(2x2)
    # A2: i*3+2 for i in 0,1 -> rows 2, 5
    # C[i,k] = A2[i,0]*B[2,k]
    # Term i=0: A[0,2]*B[2,k] for k=0,1: rank 2
    # Term i=1: A[1,2]*B[2,k] for k=0,1: rank 2
    # Total 4 terms, but we only have 2 slots. Let me reconsider.
    
    # Actually 2x1x2: C[i,k] = A[i,0]*B[0,k]. This is outer product, rank 2.
    # But here A2 is 2x1 and B is 3x2, but only B[2,:] matters.
    # C[i,k] = A2[i,0]*B[2,k]
    # = sum over... it's just outer product of A2[:,0] and B[2,:]
    # Rank 2: two rank-1 terms.
    # Term 1: A2[0,0]*B[2,0] -> U[2,7]=1, V[4,7]=1, W[0,7]=1
    # Term 2: A2[0,0]*B[2,1] -> U[2,8]=1, V[5,8]=1, W[1,8]=1
    # But we also need i=1 terms!
    # C[1,0] = A2[1,0]*B[2,0] -> U[5,7]=1, V[4,7]=1, W[2,7]=1
    # C[1,1] = A2[1,0]*B[2,1] -> U[5,8]=1, V[5,8]=1, W[3,8]=1
    
    # That's 4 terms, but we only allocated 2. Let me fix the rank.
    # 2x1x2 should have rank 4? No, outer product of two vectors of length 2 is rank 2.
    # C = A2 * B_row = [[a0*b0, a0*b1], [a1*b0, a1*b1]]
    # = a0*b0*[[1,0],[1,0]] + a0*b1*[[0,1],[0,1]] + a1*b0*[[0,0],[1,0]] + a1*b1*[[0,0],[0,1]]
    # That's 4 rank-1 terms. But we can do better:
    # = [[a0,a0],[a1,a1]] * [[b0,b1],[b0,b1]]... no that's not right.
    # Actually C = A2 @ B_row where A2 is 2x1 and B_row is 1x2.
    # C = outer(A2, B_row), which has rank 1 as a matrix, but as a tensor...
    # T[i,j,k] where j is fixed (j=2 in the full problem, but j=0 in the sub-block)
    # T[i,0,k] = A2[i,0]*B_row[0,k]
    # This is a rank-1 tensor! Because it factors as A2[i,0] * 1 * B_row[0,k]
    # So rank 1!
    
    # Wait, in the tensor decomposition: T[i,j,k] = sum_r U[i*mm+j,r]*V[j*pp+k,r]*W[k*nn+i,r]
    # For the sub-block 2x1x2: i in 0,1; j in 0; k in 0,1
    # T[i,0,k] = U[i*1+0, r]*V[0*2+k, r]*W[k*2+i, r]
    # = U[i, r]*V[k, r]*W[2k+i, r]
    # We want this to equal A2[i,0]*B[0,k] (using sub-block indices)
    # = A2[i,0] * B_row[k]
    # This factors as: U[i,r]=A2[i,0], V[k,r]=B_row[k], W[2k+i,r]=1
    # But W[2k+i,r] depends on both k and i, so it's not a simple product.
    # Actually for rank 1: U[i,0]=A2[i,0], V[k,0]=B_row[k], W[2k+i,0]=1
    # Then sum = A2[i,0]*B_row[k]*1 = A2[i,0]*B_row[k]. Yes!
    # So rank 1 suffices.
    
    U232[2, 7] = 1; U232[5, 7] = 1  # A2[0,0], A2[1,0]
    V232[4, 7] = 1; V232[5, 7] = 1  # B[2,0], B[2,1]
    W232[0, 7] = 1; W232[1, 7] = 1; W232[2, 7] = 1; W232[3, 7] = 1
    
    # So 2x3x2 has rank 8, not 9. Let me adjust.
    # Actually I allocated 9, let me just use 8 and set the 9th to 0.
    
    # Let me redo: 2x3x2 rank 8
    U232 = U232[:, :8]
    V232 = V232[:, :8]
    W232 = W232[:, :8]
    
    # 2x3x3: C(2x3) = A(2x3)*B(3x3)
    # Split A: A1(2x2), A2(2x1)
    # A1*B: 2x2x3, rank 9 (from above)
    # A2*B: 2x1x3, rank 1
    # Total: 10
    
    U233 = np.zeros((6, 10))
    V233 = np.zeros((9, 10))
    W233 = np.zeros((6, 10))
    
    # First 9: A1(2x2)*B(3x3) -> C1(2x3)
    # A1: i*3+j, i in 0,1, j in 0,1 -> rows 0,1,3,4
    # B: j*3+k, j in 0,1, k in 0,1,2 -> rows 0,1,2,3,4,5
    # C1: k*2+i, k in 0,1,2, i in 0,1 -> rows 0,1,2,3,4,5
    # This is 2x2x3, which we built as U223, V223, W223 but with different indexing.
    # U223: (4,9) with rows i*2+j
    # Need to map to i*3+j: row 0->0, 1->1, 2->3, 3->4
    U233[0, :9] = U223[0, :]
    U233[1, :9] = U223[1, :]
    U233[3, :9] = U223[2, :]
    U233[4, :9] = U223[3, :]
    
    # V223: (6,9) with rows j*2+k, j in 0,1, k in 0,1,2
    # Need j*3+k: row 0->0, 1->1, 2->3, 3->4, 4->5, 5->? 
    # j*2+k: (0,0)=0, (0,1)=1, (0,2)=2, (1,0)=3, (1,1)=4, (1,2)=5
    # j*3+k: (0,0)=0, (0,1)=1, (0,2)=2, (1,0)=3, (1,1)=4, (1,2)=5
    # Same! So V maps directly.
    V233[:6, :9] = V223
    
    # W223: (6,9) with rows k*2+i, k in 0,1,2, i in 0,1
    # k*2+i: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3, (2,0)=4, (2,1)=5
    # W233 needs k*2+i for k in 0,1,2, i in 0,1: same indexing!
    W233[:6, :9] = W223
    
    # Last 1: A2(2x1)*B(3x3) -> C2(2x3)
    # A2: i*3+2, i in 0,1 -> rows 2, 5
    # B: j*3+k, all 9 rows
    # C2: k*2+i, k in 0,1,2, i in 0,1 -> rows 0,1,2,3,4,5
    # C2[i,k] = A2[i,0]*B[2,k]
    # Rank 1: U[i*3+2, r]=A2[i,0]=1, V[2*3+k, r]=B[2,k], W[k*2+i, r]=1
    U233[2, 9] = 1; U233[5, 9] = 1
    V233[6, 9] = 1; V233[7, 9] = 1; V233[8, 9] = 1  # B[2,0], B[2,1], B[2,2]
    W233[0, 9] = 1; W233[1, 9] = 1; W233[2, 9] = 1; W233[3, 9] = 1; W233[4, 9] = 1; W233[5, 9] = 1
    
    # Now assemble the full 4x5x5 decomposition
    # i splits: [0,1] and [2,3] (offsets 0 and 2 in i)
    # j splits: [0,1] and [2,3,4] (offsets 0 and 2 in j)
    # k splits: [0,1] and [2,3,4] (offsets 0 and 2 in k)
    
    blocks = []
    
    # Block (i1,j1,k1): 2x2x2, offset i=0, j=0, k=0
    # U rows: i*5+j, i in 0,1, j in 0,1 -> 0,1,5,6
    # V rows: j*5+k, j in 0,1, k in 0,1 -> 0,1,5,6
    # W rows: k*4+i, k in 0,1, i in 0,1 -> 0,1,4,5
    # Need to map Strassen (4x7) to these rows
    U_b = np.zeros((n*m, 7))
    V_b = np.zeros((m*p, 7))
    W_b = np.zeros((p*n, 7))
    # Strassen U rows: i*2+j: 0,1,2,3 -> map to i*5+j: 0,1,5,6
    U_b[0, :] = U222[0, :]
    U_b[1, :] = U222[1, :]
    U_b[5, :] = U222[2, :]
    U_b[6, :] = U222[3, :]
    # Strassen V rows: j*2+k: 0,1,2,3 -> map to j*5+k: 0,1,5,6
    V_b[0, :] = V222[0, :]
    V_b[1, :] = V222[1, :]
    V_b[5, :] = V222[2, :]
    V_b[6, :] = V222[3, :]
    # Strassen W rows: k*2+i: 0,1,2,3 -> map to k*4+i: 0,1,4,5
    W_b[0, :] = W222[0, :]
    W_b[1, :] = W222[1, :]
    W_b[4, :] = W222[2, :]
    W_b[5, :] = W222[3, :]
    blocks.append((U_b, V_b, W_b))
    
    # Block (i1,j1,k2): 2x2x3, offset i=0, j=0, k=2
    # U rows: i*5+j, i in 0,1, j in 0,1 -> 0,1,5,6
    # V rows: j*5+k, j in 0,1, k in 2,3,4 -> 2,3,4,7,8,9
    # W rows: k*4+i, k in 2,3,4, i in 0,1 -> 8,9,12,13,16,17
    U_b = np.zeros((n*m, 9))
    V_b = np.zeros((m*p, 9))
    W_b = np.zeros((p*n, 9))
    U_b[0, :] = U223[0, :]
    U_b[1, :] = U223[1, :]
    U_b[5, :] = U223[2, :]
    U_b[6, :] = U223[3, :]
    # V223 rows: j*2+k: 0,1,2,3,4,5 -> map to j*5+k for j in 0,1, k in 2,3,4
    # j*2+k: (0,0)=0,(0,1)=1,(0,2)=2,(1,0)=3,(1,1)=4,(1,2)=5
    # j*5+k: (0,2)=2,(0,3)=3,(0,4)=4,(1,2)=7,(1,3)=8,(1,4)=9
    V_b[2, :] = V223[0, :]
    V_b[3, :] = V223[1, :]
    V_b[4, :] = V223[2, :]
    V_b[7, :] = V223[3, :]
    V_b[8, :] = V223[4, :]
    V_b[9, :] = V223[5, :]
    # W223 rows: k*2+i: 0,1,2,3,4,5 -> map to k*4+i for k in 2,3,4, i in 0,1
    # k*2+i: (0,0)=0,(0,1)=1,(1,0)=2,(1,1)=3,(2,0)=4,(2,1)=5
    # k*4+i: (2,0)=8,(2,1)=9,(3,0)=12,(3,1)=13,(4,0)=16,(4,1)=17
    W_b[8, :] = W223[0, :]
    W_b[9, :] = W223[1, :]
    W_b[12, :] = W223[2, :]
    W_b[13, :] = W223[3, :]
    W_b[16, :] = W223[4, :]
    W_b[17, :] = W223[5, :]
    blocks.append((U_b, V_b, W_b))
    
    # Block (i1,j2,k1): 2x3x2, offset i=0, j=2, k=0
    # U rows: i*5+j, i in 0,1, j in 2,3,4 -> 2,3,4,7,8,9
    # V rows: j*5+k, j in 2,3,4, k in 0,1 -> 10,11,15,16,20,21
    # W rows: k*4+i, k in 0,1, i in 0,1 -> 0,1,4,5
    U_b = np.zeros((n*m, 8))
    V_b = np.zeros((m*p, 8))
    W_b = np.zeros((p*n, 8))
    # U232 rows: i*3+j: 0,1,2,3,4,5 -> map to i*5+j for i in 0,1, j in 2,3,4
    # i*3+j: (0,0)=0,(0,1)=1,(0,2)=2,(1,0)=3,(1,1)=4,(1,2)=5
    # i*5+j: (0,2)=2,(0,3)=3,(0,4)=4,(1,2)=7,(1,3)=8,(1,4)=9
    U_b[2, :] = U232[0, :]
    U_b[3, :] = U232[1, :]
    U_b[4, :] = U232[2, :]
    U_b[7, :] = U232[3, :]
    U_b[8, :] = U232[4, :]
    U_b[9, :] = U232[5, :]
    # V232 rows: j*2+k: 0,1,2,3,4,5 -> map to j*5+k for j in 2,3,4, k in 0,1
    # j*2+k: (0,0)=0,(0,1)=1,(1,0)=2,(1,1)=3,(2,0)=4,(2,1)=5
    # j*5+k: (2,0)=10,(2,1)=11,(3,0)=15,(3,1)=16,(4,0)=20,(4,1)=21
    V_b[10, :] = V232[0, :]
    V_b[11, :] = V232[1, :]
    V_b[15, :] = V232[2, :]
    V_b[16, :] = V232[3, :]
    V_b[20, :] = V232[4, :]
    V_b[21, :] = V232[5, :]
    # W232 rows: k*2+i: 0,1,2,3 -> map to k*4+i for k in 0,1, i in 0,1
    # k*2+i: (0,0)=0,(0,1)=1,(1,0)=2,(1,1)=3
    # k*4+i: (0,0)=0,(0,1)=1,(1,0)=4,(1,1)=5
    W_b[0, :] = W232[0, :]
    W_b[1, :] = W232[1, :]
    W_b[4, :] = W232[2, :]
    W_b[5, :] = W232[3, :]
    blocks.append((U_b, V_b, W_b))
    
    # Block (i1,j2,k2): 2x3x3, offset i=0, j=2, k=2
    # U rows: i*5+j, i in 0,1, j in 2,3,4 -> 2,3,4,7,8,9
    # V rows: j*5+k, j in 2,3,4, k in 2,3,4 -> 12,13,14,17,18,19,22,23,24
    # W rows: k*4+i, k in 2,3,4, i in 0,1 -> 8,9,12,13,16,17
    U_b = np.zeros((n*m, 10))
    V_b = np.zeros((m*p, 10))
    W_b = np.zeros((p*n, 10))
    # U233 rows: i*3+j: 0,1,2,3,4,5 -> map to i*5+j for i in 0,1, j in 2,3,4
    U_b[2, :] = U233[0, :]
    U_b[3, :] = U233[1, :]
    U_b[4, :] = U233[2, :]
    U_b[7, :] = U233[3, :]
    U_b[8, :] = U233[4, :]
    U_b[9, :] = U233[5, :]
    # V233 rows: j*3+k: 0,1,2,3,4,5,6,7,8 -> map to j*5+k for j in 2,3,4, k in 2,3,4
    # j*3+k: (0,0)=0,...,(0,2)=2,(1,0)=3,...,(1,2)=5,(2,0)=6,...,(2,2)=8
    # j*5+k: (2,2)=12,(2,3)=13,(2,4)=14,(3,2)=17,(3,3)=18,(3,4)=19,(4,2)=22,(4,3)=23,(4,4)=24
    V_b[12, :] = V233[0, :]
    V_b[13, :] = V233[1, :]
    V_b[14, :] = V233[2, :]
    V_b[17, :] = V233[3, :]
    V_b[18, :] = V233[4, :]
    V_b[19, :] = V233[5, :]
    V_b[22, :] = V233[6, :]
    V_b[23, :] = V233[7, :]
    V_b[24, :] = V233[8, :]
    # W233 rows: k*2+i: 0,1,2,3,4,5 -> map to k*4+i for k in 2,3,4, i in 0,1
    # k*2+i: (0,0)=0,(0,1)=1,(1,0)=2,(1,1)=3,(2,0)=4,(2,1)=5
    # k*4+i: (2,0)=8,(2,1)=9,(3,0)=12,(3,1)=13,(4,0)=16,(4,1)=17
    W_b[8, :] = W233[0, :]
    W_b[9, :] = W233[1, :]
    W_b[12, :] = W233[2, :]
    W_b[13, :] = W233[3, :]
    W_b[16, :] = W233[4, :]
    W_b[17, :] = W233[5, :]
    blocks.append((U_b, V_b, W_b))
    
    # Block (i2,j1,k1): 2x2x2, offset i=2, j=0, k=0
    # U rows: i*5+j, i in 2,3, j in 0,1 -> 10,11,15,16
    # V rows: j*5+k, j in 0,1, k in 0,1 -> 0,1,5,6
    # W rows: k*4+i, k in 0,1, i in 2,3 -> 2,3,6,7
    U_b = np.zeros((n*m, 7))
    V_b = np.zeros((m*p, 7))
    W_b = np.zeros((p*n, 7))
    U_b[10, :] = U222[0, :]
    U_b[11, :] = U222[1, :]
    U_b[15, :] = U222[2, :]
    U_b[16, :] = U222[3, :]
    V_b[0, :] = V222[0, :]
    V_b[1, :] = V222[1, :]
    V_b[5, :] = V222[2, :]
    V_b[6, :] = V222[3, :]
    W_b[2, :] = W222[0, :]
    W_b[3, :] = W222[1, :]
    W_b[6, :] = W222[2, :]
    W_b[7, :] = W222[3, :]
    blocks.append((U_b, V_b, W_b))
    
    # Block (i2,j1,k2): 2x2x3, offset i=2, j=0, k=2
    # U rows: i*5+j, i in 2,3, j in 0,1 -> 10,11,15,16
    # V rows: j*5+k, j in 0,1, k in 2,3,4 -> 2,3,4,7,8,9
    # W rows: k*4+i, k in 2,3,4, i in 2,3 -> 10,11,14,15,18,19
    U_b = np.zeros((n*m, 9))
    V_b = np.zeros((m*p, 9))
    W_b = np.zeros((p*n, 9))
    U_b[10, :] = U223[0, :]
    U_b[11, :] = U223[1, :]
    U_b[15, :] = U223[2, :]
    U_b[16, :] = U223[3, :]
    V_b[2, :] = V223[0, :]
    V_b[3, :] = V223[1, :]
    V_b[4, :] = V223[2, :]
    V_b[7, :] = V223[3, :]
    V_b[8, :] = V223[4, :]
    V_b[9, :] = V223[5, :]
    W_b[10, :] = W223[0, :]
    W_b[11, :] = W223[1, :]
    W_b[14, :] = W223[2, :]
    W_b[15, :] = W223[3, :]
    W_b[18, :] = W223[4, :]
    W_b[19, :] = W223[5, :]
    blocks.append((U_b, V_b, W_b))
    
    # Block (i2,j2,k1): 2x3x2, offset i=2, j=2, k=0
    # U rows: i*5+j, i in 2,3, j in 2,3,4 -> 12,13,14,17,18,19
    # V rows: j*5+k, j in 2,3,4, k in 0,1 -> 10,11,15,16,20,21
    # W rows: k*4+i, k in 0,1, i in 2,3 -> 2,3,6,7
    U_b = np.zeros((n*m, 8))
    V_b = np.zeros((m*p, 8))
    W_b = np.zeros((p*n, 8))
    U_b[12, :] = U232[0, :]
    U_b[13, :] = U232[1, :]
    U_b[14, :] = U232[2, :]
    U_b[17, :] = U232[3, :]
    U_b[18, :] = U232[4, :]
    U_b[19, :] = U232[5, :]
    V_b[10, :] = V232[0, :]
    V_b[11, :] = V232[1, :]
    V_b[15, :] = V232[2, :]
    V_b[16, :] = V232[3, :]
    V_b[20, :] = V232[4, :]
    V_b[21, :] = V232[5, :]
    W_b[2, :] = W232[0, :]
    W_b[3, :] = W232[1, :]
    W_b[6, :] = W232[2, :]
    W_b[7, :] = W232[3, :]
    blocks.append((U_b, V_b, W_b))
    
    # Block (i2,j2,k2): 2x3x3, offset i=2, j=2, k=2
    # U rows: i*5+j, i in 2,3, j in 2,3,4 -> 12,13,14,17,18,19
    # V rows: j*5+k, j in 2,3,4, k in 2,3,4 -> 12,13,14,17,18,19,22,23,24
    # W rows: k*4+i, k in 2,3,4, i in 2,3 -> 10,11,14,15,18,19
    U_b = np.zeros((n*m, 10))
    V_b = np.zeros((m*p, 10))
    W_b = np.zeros((p*n, 10))
    U_b[12, :] = U233[0, :]
    U_b[13, :] = U233[1, :]
    U_b[14, :] = U233[2, :]
    U_b[17, :] = U233[3, :]
    U_b[18, :] = U233[4, :]
    U_b[19, :] = U233[5, :]
    V_b[12, :] = V233[0, :]
    V_b[13, :] = V233[1, :]
    V_b[14, :] = V233[2, :]
    V_b[17, :] = V233[3, :]
    V_b[18, :] = V233[4, :]
    V_b[19, :] = V233[5, :]
    V_b[22, :] = V233[6, :]
    V_b[23, :] = V233[7, :]
    V_b[24, :] = V233[8, :]
    W_b[10, :] = W233[0, :]
    W_b[11, :] = W233[1, :]
    W_b[14, :] = W233[2, :]
    W_b[15, :] = W233[3, :]
    W_b[18, :] = W233[4, :]
    W_b[19, :] = W233[5, :]
    blocks.append((U_b, V_b, W_b))
    
    # Assemble
    U_full, V_full, W_full = _block_combine(blocks, n, m, p)
    
    # Verify
    ok, err = _verify(U_full, V_full, W_full, n, m, p)
    if ok:
        best = {'U': U_full.tolist(), 'V': V_full.tolist(), 'W': W_full.tolist()}
        best_rank = U_full.shape[1]
    
    return best
# EVOLVE-BLOCK-END
