import time
import numpy as np
from collections import defaultdict

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank decomposition for 5x5 matrix multiplication using block partitioning.
    Partition 5 = 2 + 2 + 1. For each block (i,j,k), multiply n_i x m_j by m_j x p_k.
    Use Strassen (rank 7) for 2x2x2 blocks, schoolbook for all others.
    """
    start_time = time.time()
    deadline = 120.0
    safety_margin = 10.0
    
    n, m, p = DIMENSIONS
    partition = [2, 2, 1]
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # Each Strassen term is a rank-1 decomposition of the 2x2x2 tensor.
    # For term with coefficient c, A-indices (a1,a2), B-indices (b1,b2), C-indices (c1,c2):
    # U[a1*m+a2, r] += c_a
    # V[b1*p+b2, r] += c_b  
    # W[c2*n+c1, r] += c_c
    # where the product c_a * c_b * c_c = 1 for the delta condition
    
    # Actually, for Strassen, each intermediate M is a product of linear combinations.
    # M = (sum of A terms) * (sum of B terms)
    # This contributes to C as: C += M * (some pattern)
    
    # Let me think about this differently. The tensor T[i,j,k] = 1 if j==k (in 2x2x2).
    # Schoolbook: 8 terms, one for each (i,j,k).
    # Strassen: 7 terms.
    
    # For 2x2x2, the schoolbook terms are:
    # (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)
    # Each has U[i,j]=1, V[j,k]=1, W[k,i]=1
    
    # Strassen decomposes this into 7 rank-1 terms:
    # T1: A=(1,0)+(1,1), B=(1,0)+(1,1), C=(1,0) with coeff pattern
    # This is getting complex. Let me use the known Strassen decomposition.
    
    # Strassen for 2x2:
    # p1 = (a00+a11)(b00+b11)
    # p2 = (a10+a11)b00
    # p3 = a00(b01-b11)
    # p4 = a11(b10-b00)
    # p5 = (a00+a01)b11
    # p6 = (a10-a00)(b00+b01)
    # p7 = (a01-a11)(b10+b11)
    # c00 = p1+p4-p5+p7
    # c01 = p3+p5
    # c10 = p2+p4
    # c11 = p1-p2+p3+p6
    
    # Each p_i is a product of a linear form in A and a linear form in B.
    # For the tensor decomposition, each p_i contributes one rank-1 term.
    
    # p1: A_lin = A[0,0] + A[1,1], B_lin = B[0,0] + B[1,1]
    #     U[0,0]=1, U[1,1]=1, V[0,0]=1, V[1,1]=1, W[0,0]=1, W[1,1]=1
    #     But this gives c00 += 1, c01 += 1, c10 += 1, c11 += 1
    #     We need to combine with other terms to get the right C.
    
    # The issue is that each p_i contributes to multiple C entries.
    # The full Strassen decomposition has 7 rank-1 terms where:
    # Term 1: U has 1s at (0,0),(1,1); V has 1s at (0,0),(1,1); W has 1s at (0,0),(1,1)
    #   This contributes: C[0,0]+=1, C[0,1]+=1, C[1,0]+=1, C[1,1]+=1
    #   But we need C[0,0] to get +1 from p1, and C[0,1] to get 0 from p1 alone...
    
    # I think I'm overcomplicating this. Let me just directly construct the Strassen
    # rank-1 terms for the 2x2x2 tensor.
    
    # The 2x2x2 matmul tensor has entries T[i,j,k] = 1 if j==k else 0.
    # So T = T[0,0,0] + T[0,0,1] + T[0,1,0] + T[0,1,1] + T[1,0,0] + T[1,0,1] + T[1,1,0] + T[1,1,1]
    #      = (i=0,j=0,k=0) + (i=0,j=0,k=1) + (i=0,j=1,k=0) + (i=0,j=1,k=1) + ...
    
    # Wait, T[i,j,k] = 1 if j==k. So:
    # T[0,0,0]=1, T[0,0,1]=0, T[0,1,0]=0, T[0,1,1]=1
    # T[1,0,0]=1, T[1,0,1]=0, T[1,1,0]=0, T[1,1,1]=1
    
    # Hmm, that's only 4 non-zero entries. That can't be right for matrix multiplication.
    
    # Let me re-read the tensor definition:
    # T[a_row, a_col, b_row, b_col] = 1 if a_col == b_row else 0
    # For 2x2x2: a_row in {0,1}, a_col in {0,1}, b_row in {0,1}, b_col in {0,1}
    # T[0,0,0,0]=1, T[0,0,0,1]=1, T[0,0,1,0]=0, T[0,0,1,1]=0
    # T[0,1,0,0]=0, T[0,1,0,1]=0, T[0,1,1,0]=1, T[0,1,1,1]=1
    # T[1,0,0,0]=1, T[1,0,0,1]=1, T[1,0,1,0]=0, T[1,0,1,1]=0
    # T[1,1,0,0]=0, T[1,1,0,1]=0, T[1,1,1,0]=1, T[1,1,1,1]=1
    
    # So 8 non-zero entries. The schoolbook uses 8 rank-1 terms.
    # Strassen uses 7.
    
    # For Strassen, the 7 rank-1 terms for 2x2x2 are:
    # Each term r has U_r (in R^4), V_r (in R^4), W_r (in R^4)
    # such that sum_r U_r[i,j] * V_r[k,l] * W_r[l,i] = T[i,j,k,l]
    
    # Let me look up or derive the Strassen decomposition in tensor form.
    
    # Actually, let me just use a simpler approach: try the block decomposition and
    # see if the total rank is less than 125. If not, fall back to schoolbook.
    
    def strassen_2x2x2_terms():
        """
        Return the 7 rank-1 terms for 2x2x2 matrix multiplication.
        Each term is (U_vec, V_vec, W_vec) where:
        U_vec has 4 entries (for a_row*2+a_col)
        V_vec has 4 entries (for b_row*2+b_col)  
        W_vec has 4 entries (for b_col*2+a_row)
        """
        terms = []
        
        # p1 = (a00+a11)(b00+b11)
        # U: a00=1, a11=1 -> U[0]=1, U[3]=1
        # V: b00=1, b11=1 -> V[0]=1, V[3]=1
        # W: contributes to c00 and c11
        # c00 += p1, c11 += p1
        # W[0,0]=1, W[1,1]=1 -> W[0]=1, W[3]=1
        u1 = np.array([1,0,0,1])
        v1 = np.array([1,0,0,1])
        w1 = np.array([1,0,0,1])
        terms.append((u1, v1, w1))
        
        # p2 = (a10+a11)b00
        # U: a10=1, a11=1 -> U[2]=1, U[3]=1
        # V: b00=1 -> V[0]=1
        # c10 += p2
        # W[1,0]=1 -> W[2]=1
        u2 = np.array([0,0,1,1])
        v2 = np.array([1,0,0,0])
        w2 = np.array([0,0,1,0])
        terms.append((u2, v2, w2))
        
        # p3 = a00(b01-b11)
        # U: a00=1 -> U[0]=1
        # V: b01=1, b11=-1 -> V[1]=1, V[3]=-1
        # c01 += p3
        # W[0,1]=1 -> W[1]=1
        u3 = np.array([1,0,0,0])
        v3 = np.array([0,1,0,-1])
        w3 = np.array([0,1,0,0])
        terms.append((u3, v3, w3))
        
        # p4 = a11(b10-b00)
        # U: a11=1 -> U[3]=1
        # V: b10=1, b00=-1 -> V[2]=1, V[0]=-1
        # c10 += p4, c00 += p4
        # W[1,0]=1, W[0,0]=1 -> W[2]=1, W[0]=1
        u4 = np.array([0,0,0,1])
        v4 = np.array([-1,0,1,0])
        w4 = np.array([1,0,1,0])
        terms.append((u4, v4, w4))
        
        # p5 = (a00+a01)b11
        # U: a00=1, a01=1 -> U[0]=1, U[1]=1
        # V: b11=1 -> V[3]=1
        # c01 += p5, c00 -= p5
        # W[0,1]=1, W[0,0]=-1 -> W[1]=1, W[0]=-1
        u5 = np.array([1,1,0,0])
        v5 = np.array([0,0,0,1])
        w5 = np.array([-1,1,0,0])
        terms.append((u5, v5, w5))
        
        # p6 = (a10-a00)(b00+b01)
        # U: a10=1, a00=-1 -> U[2]=1, U[0]=-1
        # V: b00=1, b01=1 -> V[0]=1, V[1]=1
        # c11 += p6
        # W[1,1]=1 -> W[3]=1
        u6 = np.array([-1,0,1,0])
        v6 = np.array([1,1,0,0])
        w6 = np.array([0,0,0,1])
        terms.append((u6, v6, w6))
        
        # p7 = (a01-a11)(b10+b11)
        # U: a01=1, a11=-1 -> U[1]=1, U[3]=-1
        # V: b10=1, b11=1 -> V[2]=1, V[3]=1
        # c00 += p7
        # W[0,0]=1 -> W[0]=1
        u7 = np.array([0,1,0,-1])
        v7 = np.array([0,0,1,1])
        w7 = np.array([1,0,0,0])
        terms.append((u7, v7, w7))
        
        return terms
    
    def schoolbook_block_terms(n_i, m_j, p_k):
        """
        Return rank-1 terms for n_i x m_j times m_j x p_k multiplication.
        Each term is (U_vec, V_vec, W_vec) with lengths n_i*m_j, m_j*p_k, p_k*n_i.
        """
        terms = []
        for i in range(n_i):
            for j in range(m_j):
                for k in range(p_k):
                    u = np.zeros(n_i * m_j)
                    v = np.zeros(m_j * p_k)
                    w = np.zeros(p_k * n_i)
                    u[i * m_j + j] = 1.0
                    v[j * p_k + k] = 1.0
                    w[k * n_i + i] = 1.0
                    terms.append((u, v, w))
        return terms
    
    # Compute block offsets
    block_offsets = [0]
    for s in partition:
        block_offsets.append(block_offsets[-1] + s)
    
    # Build the full decomposition
    all_U = []
    all_V = []
    all_W = []
    
    for bi in range(len(partition)):
        for bj in range(len(partition)):
            for bk in range(len(partition)):
                ni = partition[bi]
                mj = partition[bj]
                pk = partition[bk]
                
                if ni == 2 and mj == 2 and pk == 2:
                    terms = strassen_2x2x2_terms()
                else:
                    terms = schoolbook_block_terms(ni, mj, pk)
                
                for (u, v, w) in terms:
                    # Expand to full size
                    U_full = np.zeros(n * m)
                    V_full = np.zeros(m * p)
                    W_full = np.zeros(p * n)
                    
                    # A block is at rows [block_offsets[bi]:block_offsets[bi+1]],
                    # cols [block_offsets[bj]:block_offsets[bj+1]]
                    # U index: a_row * m + a_col
                    for idx in range(len(u)):
                        i_local = idx // mj
                        j_local = idx % mj
                        a_row = block_offsets[bi] + i_local
                        a_col = block_offsets[bj] + j_local
                        U_full[a_row * m + a_col] = u[idx]
                    
                    # B block is at rows [block_offsets[bj]:block_offsets[bj+1]],
                    # cols [block_offsets[bk]:block_offsets[bk+1]]
                    # V index: b_row * p + b_col
                    for idx in range(len(v)):
                        j_local = idx // pk
                        k_local = idx % pk
                        b_row = block_offsets[bj] + j_local
                        b_col = block_offsets[bk] + k_local
                        V_full[b_row * p + b_col] = v[idx]
                    
                    # C block is at rows [block_offsets[bk]:block_offsets[bk+1]],
                    # cols [block_offsets[bi]:block_offsets[bi+1]]
                    # W index: b_col * n + a_row
                    for idx in range(len(w)):
                        k_local = idx // ni
                        i_local = idx % ni
                        b_col = block_offsets[bk] + k_local
                        a_row = block_offsets[bi] + i_local
                        W_full[b_col * n + a_row] = w[idx]
                    
                    all_U.append(U_full)
                    all_V.append(V_full)
                    all_W.append(W_full)
    
    rank = len(all_U)
    U = np.column_stack(all_U) if rank > 0 else np.zeros((n*m, 1))
    V = np.column_stack(all_V) if rank > 0 else np.zeros((m*p, 1))
    W = np.column_stack(all_W) if rank > 0 else np.zeros((p*n, 1))
    
    # Verify
    def verify(U, V, W, tol=1e-10):
        n, m, p = DIMENSIONS
        max_err = 0.0
        for a_row in range(n):
            for a_col in range(m):
                for b_row in range(m):
                    for b_col in range(p):
                        expected = 1.0 if a_col == b_row else 0.0
                        val = np.dot(U[a_row*m+a_col, :] * V[b_row*p+b_col, :], W[b_col*n+a_row, :])
                        err = abs(val - expected)
                        if err > max_err:
                            max_err = err
        return max_err <= tol, max_err
    
    valid, max_err = verify(U, V, W)
    
    if not valid or rank >= 125:
        # Fall back to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, r] = 1.0
                    V[j * p + k, r] = 1.0
                    W[k * n + i, r] = 1.0
                    r += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
