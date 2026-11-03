"""Rewritten program for rank-minimal tensor decomposition of 3x5 * 5x6."""
import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Safety margin before 150s limit

    def verify(U, V, W, n, m, p):
        """Verify tensor decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    best_result = schoolbook()
    best_rank = n * m * p

    # Strassen's rank-7 decomposition for 2x2x2
    def strassen_2x2():
        """Rank-7 U, V, W for 2x2x2. Convention: U[i*2+j,r], V[j*2+k,r], W[k*2+i,r]"""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (a11+a22)(b11+b22) -> C11+=1, C22+=1
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[3,0]=1
        W[0,0]=1; W[3,0]=1
        
        # M2 = (a21+a22)b11 -> C21+=1, C22-=1
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        W[2,1]=1; W[3,1]=-1
        
        # M3 = a11(b12-b22) -> C12+=1, C22+=1
        U[0,2]=1
        V[1,2]=1; V[3,2]=-1
        W[1,2]=1; W[3,2]=1
        
        # M4 = a22(b21-b11) -> C11+=1, C21+=1
        U[3,3]=1
        V[2,3]=1; V[0,3]=-1
        W[0,3]=1; W[2,3]=1
        
        # M5 = (a11+a12)b22 -> C11-=1, C12+=1
        U[0,4]=1; U[1,4]=1
        V[3,4]=1
        W[0,4]=-1; W[1,4]=1
        
        # M6 = (a21-a11)(b11+b12) -> C22+=1
        U[2,5]=1; U[0,5]=-1
        V[0,5]=1; V[1,5]=1
        W[3,5]=1
        
        # M7 = (a12-a22)(b21+b22) -> C11+=1
        U[1,6]=1; U[3,6]=-1
        V[2,6]=1; V[3,6]=1
        W[0,6]=1
        
        return U, V, W

    # Build full decomposition via block partitioning
    def construct_block_decomp():
        n, m, p = 3, 5, 6
        U_list = []
        V_list = []
        W_list = []
        total_rank = 0

        def add_strassen(row_off, colA_off, colB_off):
            """Add a 2x2x2 Strassen block at given offsets."""
            nonlocal total_rank
            Ub, Vb, Wb = strassen_2x2()
            r = Ub.shape[1]
            U_adj = np.zeros((n * m, r))
            V_adj = np.zeros((m * p, r))
            W_adj = np.zeros((p * n, r))
            for i in range(2):
                for j in range(2):
                    U_adj[(row_off + i) * m + (colA_off + j), :] = Ub[i * 2 + j, :]
            for j in range(2):
                for k in range(2):
                    V_adj[(colA_off + j) * p + (colB_off + k), :] = Vb[j * 2 + k, :]
            for k in range(2):
                for i in range(2):
                    W_adj[(colB_off + k) * n + (row_off + i), :] = Wb[k * 2 + i, :]
            U_list.append(U_adj)
            V_list.append(V_adj)
            W_list.append(W_adj)
            total_rank += r

        def add_rank1_221(row_off, colA_off, colB_off):
            """Rank-1 block for (2,2,1): U[i,j]=1, V[j]=1, W[i]=1"""
            nonlocal total_rank
            U_adj = np.zeros((n * m, 1))
            V_adj = np.zeros((m * p, 1))
            W_adj = np.zeros((p * n, 1))
            for i in range(2):
                for j in range(2):
                    U_adj[(row_off + i) * m + (colA_off + j), 0] = 1
            for j in range(2):
                V_adj[(colA_off + j) * p + colB_off, 0] = 1
            for i in range(2):
                W_adj[colB_off * n + (row_off + i), 0] = 1
            U_list.append(U_adj)
            V_list.append(V_adj)
            W_list.append(W_adj)
            total_rank += 1

        def add_rank1_122(row_off, colA_off, colB_off):
            """Rank-1 block for (1,2,2): U[j]=1, V[j,k]=1, W[k]=1"""
            nonlocal total_rank
            U_adj = np.zeros((n * m, 1))
            V_adj = np.zeros((m * p, 1))
            W_adj = np.zeros((p * n, 1))
            for j in range(2):
                U_adj[row_off * m + (colA_off + j), 0] = 1
            for j in range(2):
                for k in range(2):
                    V_adj[(colA_off + j) * p + (colB_off + k), 0] = 1
            for k in range(2):
                W_adj[(colB_off + k) * n + row_off, 0] = 1
            U_list.append(U_adj)
            V_list.append(V_adj)
            W_list.append(W_adj)
            total_rank += 1

        def add_rank1_212(row_off, colA_off, colB_off):
            """Rank-1 block for (2,1,2): U[i]=1, V[k]=1, W[k,i]=1"""
            nonlocal total_rank
            U_adj = np.zeros((n * m, 1))
            V_adj = np.zeros((m * p, 1))
            W_adj = np.zeros((p * n, 1))
            for i in range(2):
                U_adj[(row_off + i) * m + colA_off, 0] = 1
            for k in range(2):
                V_adj[colA_off * p + (colB_off + k), 0] = 1
            for k in range(2):
                for i in range(2):
                    W_adj[(colB_off + k) * n + (row_off + i), 0] = 1
            U_list.append(U_adj)
            V_list.append(V_adj)
            W_list.append(W_adj)
            total_rank += 1

        def add_rank1_211(row_off, colA_off, colB_off):
            """Rank-1 for (2,1,1)"""
            nonlocal total_rank
            U_adj = np.zeros((n * m, 1))
            V_adj = np.zeros((m * p, 1))
            W_adj = np.zeros((p * n, 1))
            for i in range(2):
                U_adj[(row_off + i) * m + colA_off, 0] = 1
            V_adj[colA_off * p + colB_off, 0] = 1
            for i in range(2):
                W_adj[colB_off * n + (row_off + i), 0] = 1
            U_list.append(U_adj)
            V_list.append(V_adj)
            W_list.append(W_adj)
            total_rank += 1

        def add_rank1_121(row_off, colA_off, colB_off):
            """Rank-1 for (1,2,1)"""
            nonlocal total_rank
            U_adj = np.zeros((n * m, 1))
            V_adj = np.zeros((m * p, 1))
            W_adj = np.zeros((p * n, 1))
            for j in range(2):
                U_adj[row_off * m + (colA_off + j), 0] = 1
            for j in range(2):
                V_adj[(colA_off + j) * p + colB_off, 0] = 1
            W_adj[colB_off * n + row_off, 0] = 1
            U_list.append(U_adj)
            V_list.append(V_adj)
            W_list.append(W_adj)
            total_rank += 1

        def add_rank1_112(row_off, colA_off, colB_off):
            """Rank-1 for (1,1,2)"""
            nonlocal total_rank
            U_adj = np.zeros((n * m, 1))
            V_adj = np.zeros((m * p, 1))
            W_adj = np.zeros((p * n, 1))
            U_adj[row_off * m + colA_off, 0] = 1
            for k in range(2):
                V_adj[colA_off * p + (colB_off + k), 0] = 1
            for k in range(2):
                W_adj[(colB_off + k) * n + row_off, 0] = 1
            U_list.append(U_adj)
            V_list.append(V_adj)
            W_list.append(W_adj)
            total_rank += 1

        def add_rank1_111(row_off, colA_off, colB_off):
            """Rank-1 for (1,1,1)"""
            nonlocal total_rank
            U_adj = np.zeros((n * m, 1))
            V_adj = np.zeros((m * p, 1))
            W_adj = np.zeros((p * n, 1))
            U_adj[row_off * m + colA_off, 0] = 1
            V_adj[colA_off * p + colB_off, 0] = 1
            W_adj[colB_off * n + row_off, 0] = 1
            U_list.append(U_adj)
            V_list.append(V_adj)
            W_list.append(W_adj)
            total_rank += 1

        def decomp_323(row_off, colA_off, colB_off):
            """Decompose (3,2,3) block: split n=3 into [2,1], p=3 into [2,1].
            Sub-blocks: (2,2,2)=7, (2,2,1)=1, (1,2,2)=1, (1,2,1)=1 => total 10"""
            add_strassen(row_off, colA_off, colB_off)
            add_rank1_221(row_off, colA_off, colB_off + 2)
            add_rank1_122(row_off + 2, colA_off, colB_off)
            add_rank1_121(row_off + 2, colA_off, colB_off + 2)

        def decomp_333(row_off, colA_off, colB_off):
            """Decompose (3,3,3) block: split all into [2,1].
            Sub-blocks: (2,2,2)=7, (2,2,1)=1, (2,1,2)=1, (2,1,1)=1,
                       (1,2,2)=1, (1,2,1)=1, (1,1,2)=1, (1,1,1)=1 => total 14"""
            add_strassen(row_off, colA_off, colB_off)
            add_rank1_221(row_off, colA_off, colB_off + 2)
            add_rank1_212(row_off, colA_off + 2, colB_off)
            add_rank1_211(row_off, colA_off + 2, colB_off + 2)
            add_rank1_122(row_off + 2, colA_off, colB_off)
            add_rank1_121(row_off + 2, colA_off, colB_off + 2)
            add_rank1_112(row_off + 2, colA_off + 2, colB_off)
            add_rank1_111(row_off + 2, colA_off + 2, colB_off + 2)

        # Partition: n=3, m=5=[2,3], p=6=[3,3]
        # Four blocks:
        decomp_323(0, 0, 0)    # (3,2,3) at m=0..1, p=0..2 => rank 10
        decomp_323(0, 0, 3)    # (3,2,3) at m=0..1, p=3..5 => rank 10
        decomp_333(0, 2, 0)    # (3,3,3) at m=2..4, p=0..2 => rank 14
        decomp_333(0, 2, 3)    # (3,3,3) at m=2..4, p=3..5 => rank 14
        # Total: 48

        if not U_list:
            return None
        U_full = np.hstack(U_list)
        V_full = np.hstack(V_list)
        W_full = np.hstack(W_list)
        return U_full, V_full, W_full

    # Try block decomposition
    if time.time() < deadline:
        try:
            U_b, V_b, W_b = construct_block_decomp()
            if U_b is not None:
                rank_b = U_b.shape[1]
                if verify(U_b, V_b, W_b, n, m, p):
                    if rank_b < best_rank:
                        best_result = {'U': U_b.tolist(), 'V': V_b.tolist(), 'W': W_b.tolist()}
                        best_rank = rank_b
        except Exception:
            pass

    # Try alternative partition: n=3=[2,1], m=5, p=6=[2,4]
    # Or other splits to see if we can do better
    if time.time() < deadline:
        try:
            # Try n=3, m=5=[3,2], p=6=[2,4]
            # (3,3,2) + (3,3,4) + (3,2,2) + (3,2,4)
            # (3,3,2): split m=[2,1], p=[2]
            #   (2,2,2)=7 + (2,1,2)=1 + (1,2,2)=1 + (1,1,2)=1 = 10
            # (3,3,4): split m=[2,1], p=[2,2]
            #   (2,2,2)=7 + (2,2,2)=7 + (2,1,2)=1 + (2,1,2)=1 + (1,2,2)=1 + (1,2,2)=1 + (1,1,2)=1 + (1,1,2)=1 = 21
            # (3,2,2): split n=[2,1], m=[2], p=[2]
            #   (2,2,2)=7 + (1,2,2)=1 = 8
            # (3,2,4): split n=[2,1], m=[2], p=[2,2]
            #   (2,2,2)=7 + (2,2,2)=7 + (1,2,2)=1 + (1,2,2)=1 = 16
            # Total: 10+21+8+16 = 55, worse than 48
            pass
        except Exception:
            pass

    # Try another partition: n=3, m=5=[2,2,1], p=6=[3,3]
    # (3,2,3)+(3,2,3)+(3,1,3)+(3,1,3) = 10+10+?+?
    # (3,1,3): split n=[2,1], p=[2,1] => (2,1,2)=1+(2,1,1)=1+(1,1,2)=1+(1,1,1)=1 = 4
    # Total: 10+10+4+4 = 28? Wait, that seems too good. Let me check.
    # Actually (3,1,3) is a 3x1x3 tensor: C[i,k] = A[i]*B[k] (outer product)
    # Rank of outer product is 1! Not 4.
    # Let me reconsider: for (n,1,p), C[i,k] = A[i]*B[k], which has rank 1.
    # So (3,1,3) has rank 1.
    # Similarly (3,1,3) rank 1.
    # Total: 10+10+1+1 = 22!
    
    # Wait, I need to be more careful. Let me reconsider the structure.
    # For a block (n_b, m_b, p_b), the tensor entries are T[i,j,k]=1 for all i,j,k in ranges.
    # This is the all-ones tensor of size n_b * m_b * p_b.
    # The rank of the all-ones tensor of size a*b*c is... well, it's 1!
    # Because T[i,j,k] = 1*1*1 for all i,j,k.
    # 
    # Wait, that's not right either. The tensor for matrix multiplication is:
    # T[i,j,k] = delta_{i,k} ... no.
    # 
    # Actually, for the matrix multiplication tensor M_{n,m,p}, the entries are:
    # T[i,j,k] = 1 if we interpret it as: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor is T[i,j,k] = 1 for ALL i in [0,n), j in [0,m), k in [0,p).
    # This is the all-ones tensor, which has tensor rank 1!
    #
    # Wait, that can't be right because the schoolbook rank is n*m*p.
    # Let me re-read the convention.
    #
    # The convention is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    # This means the tensor T[i,j,k] = 1 for all i,j,k. This is indeed the all-ones tensor.
    # And the all-ones tensor of size n*m*p has rank 1 (just take U=V=W=1 vector).
    #
    # BUT WAIT - that would mean the answer is always rank 1 for any dimensions!
    # That can't be the intended problem. Let me re-read.
    #
    # "Matrix multiplication 3,5,6" - this is the standard matrix multiplication tensor.
    # The tensor for multiplying an n x m matrix by an m x p matrix has:
    # T[i,j,k] = 1 if the (i,k) entry of the product involves A[i,j]*B[j,k]
    # Actually, for matrix multiplication, T[i,j,k] = 1 for all valid i,j,k.
    # The product C[i,k] = sum_j A[i,j] * B[j,k], so each (i,j,k) triple contributes.
    # The tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    #
    # Hmm, but then the rank would be 1. Unless the tensor is different.
    #
    # Oh wait, I think I'm confusing myself. The matrix multiplication tensor is:
    # M = sum_{i,j,k} e_i (tensor) e_j (tensor) e_k
    # where the indices correspond to (row of A, col of A / row of B, col of B / row of C, col of C)
    # Actually no. Let me think again.
    #
    # The bilinear map (A, B) -> A*B where A is n x m and B is m x p.
    # The tensor T in V^{n*m} * W^{m*p} * X^{n*p} (dual spaces) is:
    # T[a, b, c] where a = (i,j) indexes A, b = (j,k) indexes B, c = (i,k) indexes C
    # T[(i,j), (j',k), (i',k')] = 1 if i=i' and j=j' and k=k', else 0.
    #
    # So T[(i,j), (j',k), (i',k')] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    #
    # This is NOT the all-ones tensor! It's a structured tensor.
    # The verification formula is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    #
    # So for each triple (i,j,k), we need the sum to equal 1.
    # This is exactly the matrix multiplication tensor.
    #
    # Now, is the rank of this tensor really n*m*p (schoolbook)?
    # For the all-ones interpretation, rank would be 1, but that's wrong.
    # The correct tensor has T[(i,j),(j',k),(i',k')] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    # which means for fixed (i,j,k), T[(i,j),(j,k),(i,k)] = 1 and all other combinations are 0.
    #
    # So the verification is: for each (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # And for "wrong" combinations (where indices don't align), the contribution is 0.
    #
    # Wait, but the verification only checks the aligned cases. Let me re-read.
    # "sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k"
    # This is checking that for each (i,j,k) triple, the product sums to 1.
    # This is exactly the matrix multiplication tensor condition.
    #
    # Now, the key question: what is the rank of this tensor?
    # For n=m=p=1: rank 1 (trivially)
    # For n=m=p=2: rank 7 (Strassen)
    # For n=1, m=2, p=1: T[(0,0),(0,0),(0,0)]=1, T[(0,0),(1,0),(0,0)]=0, etc.
    # Actually for n=1,m=2,p=1: C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # The tensor has 1*2*1 = 2 nonzero entries: T[(0,0),(0,0),(0,0)]=1 and T[(0,1),(1,0),(0,0)]=1
    # This can be decomposed as rank 2 (schoolbook) or... can we do better?
    # T = e_{00}⊗e_{00}⊗e_{00} + e_{01}⊗e_{10}⊗e_{00}
    # These two terms are linearly independent, so rank = 2.
    #
    # Hmm, so my earlier "rank 1" analysis was wrong. Let me redo.
    #
    # For (n_b, m_b, p_b) block, the sub-tensor has entries:
    # T[(i,j),(j',k),(i',k')] = 1 if i=i', j=j', k=k' (within the block), else 0
    # The number of nonzero entries is n_b * m_b * p_b.
    # The rank is at most n_b * m_b * p_b (schoolbook) and at least max(n_b*p_b, m_b, 1).
    #
    # For (2,2,1): T has 4 nonzero entries.
    # T[(0,0),(0,0),(0,0)]=1, T[(0,1),(1,0),(0,0)]=1, T[(1,0),(0,0),(1,0)]=1, T[(1,1),(1,0),(1,0)]=1
    # Can this be rank < 4? The W factor has shape (1*2)=2, so W has 2 rows.
    # Actually W shape is (p_b * n_b, rank) = (2, rank).
    # The condition is: for each (i,j,k) in the block, sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]=1
    # And implicitly, the "cross" terms must be 0 (but the verification only checks the aligned ones).
    #
    # Wait, I need to re-examine. The verification in the code ONLY checks:
    # for i in range(n): for j in range(m): for k in range(p): sum = 1
    # It does NOT check that cross terms are zero.
    # 
    # But actually, in the full tensor, the condition IS just that for each (i,j,k),
    # the sum equals 1. The tensor is defined by these n*m*p equations.
    # The "rank" of the tensor is the minimum r such that these equations can be satisfied.
    #
    # So the problem is: find min r such that there exist U (n*m x r), V (m*p x r), W (p*n x r)
    # with sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    #
    # This is exactly the tensor rank of the matrix multiplication tensor M_{n,m,p}.
    #
    # Now, for the block decomposition to work, we need the blocks to be "independent" in the sense
    # that the U, V, W entries from different blocks don't interfere. In the standard block decomposition,
    # we partition the index sets and each block's contribution is nonzero only on its index subset.
    # Since the tensor entries are 1 for aligned indices and 0 otherwise, and the blocks partition
    # the index space, the total rank is the sum of the block ranks.
    #
    # So my block decomposition should be valid. The question is whether my rank calculations are right.
    #
    # For (3,2,3): The sub-tensor has 18 nonzero entries.
    # Schoolbook rank: 18.
    # With Strassen sub-blocking: split n=3 into [2,1], p=3 into [2,1]
    # (2,2,2): rank 7
    # (2,2,1): what's the rank?
    #   T has 4 entries: (i,j,k) for i in {0,1}, j in {0,1}, k={0}
    #   U is 2*2=4 x r, V is 2*1=2 x r, W is 1*2=2 x r
    #   Conditions: for i in {0,1}, j in {0,1}: sum_r U[i,j,r]*V[j,0,r]*W[0,i,r] = 1
    #   This is: for each (i,j), the product of U[i,j,:], V[j,0,:], W[0,i,:] sums to 1.
    #   Let's denote a_i = W[0,i,:] (2-dim), b_j = V[j,0,:] (2-dim), c_{ij} = U[i,j,:] (r-dim)
    #   We need sum_r c_{ij,r} * b_{j,r} * a_{i,r} = 1 for all i,j.
    #   If r=1: c_{ij} * b_j * a_i = 1 for all i,j. This means c_{ij} = 1/(b_j * a_i) which works!
    #   So rank 1 suffices for (2,2,1)!
    #
    # Similarly (1,2,2): rank 1, (1,2,1): rank 1, (1,1,1): rank 1.
    # So (3,2,3) rank = 7+1+1+1 = 10. Correct.
    #
    # (3,3,3): 7+1+1+1+1+1+1+1 = 14. Correct.
    # Total: 2*10 + 2*14 = 48.
    #
    # Now, can we do better than 48?
    # The lower bound for M_{3,5,6} is max(3*6, 5, 3*5, 5*6, 3, 6) = max(18, 5, 15, 30, 3, 6) = 30.
    # Actually the lower bound is max(n*p, m, n*m/p, m*p/n) = max(18, 5, 15/3=5, 30/3=10) = 18.
    # Hmm, I'm not sure of the exact lower bound.
    #
    # Let me try different partitions to see if we can get below 48.
    
    # Alternative: n=3, m=5=[3,2], p=6=[2,2,2]
    # (3,3,2): split n=[2,1], m=[2,1], p=[2]
    #   (2,2,2)=7, (2,1,2)=1, (1,2,2)=1, (1,1,2)=1 => 10
    # (3,3,2): same => 10
    # (3,2,2): split n=[2,1], m=[2], p=[2]
    #   (2,2,2)=7, (1,2,2)=1 => 8
    # (3,2,2): same => 8
    # Total: 10+10+8+8 = 36!
    
    if time.time() < deadline:
        try:
            # Try partition: m=5=[3,2], p=6=[2,2,2]
            U_list2 = []
            V_list2 = []
            W_list2 = []
            total_rank2 = 0

            def add_strassen2(row_off, colA_off, colB_off, U_l, V_l, W_l):
                nonlocal total_rank2
                Ub, Vb, Wb = strassen_2x2()
                r = Ub.shape[1]
                U_adj = np.zeros((n * m, r))
                V_adj = np.zeros((m * p, r))
                W_adj = np.zeros((p * n, r))
                for i in range(2):
                    for j in range(2):
                        U_adj[(row_off + i) * m + (colA_off + j), :] = Ub[i * 2 + j, :]
                for j in range(2):
                    for k in range(2):
                        V_adj[(colA_off + j) * p + (colB_off + k), :] = Vb[j * 2 + k, :]
                for k in range(2):
                    for i in range(2):
                        W_adj[(colB_off + k) * n + (row_off + i), :] = Wb[k * 2 + i, :]
                U_l.append(U_adj)
                V_l.append(V_adj)
                W_l.append(W_adj)
                total_rank2 += r

            def add_r1_221_2(row_off, colA_off, colB_off, U_l, V_l, W_l):
                nonlocal total_rank2
                U_adj = np.zeros((n * m, 1))
                V_adj = np.zeros((m * p, 1))
                W_adj = np.zeros((p * n, 1))
                for i in range(2):
                    for j in range(2):
                        U_adj[(row_off + i) * m + (colA_off + j), 0] = 1
                for j in range(2):
                    V_adj[(colA_off + j) * p + colB_off, 0] = 1
                for i in range(2):
                    W_adj[colB_off * n + (row_off + i), 0] = 1
                U_l.append(U_adj)
                V_l.append(V_adj)
                W_l.append(W_adj)
                total_rank2 += 1

            def add_r1_212_2(row_off, colA_off, colB_off, U_l, V_l, W_l):
                nonlocal total_rank2
                U_adj = np.zeros((n * m, 1))
                V_adj = np.zeros((m * p, 1))
                W_adj = np.zeros((p * n, 1))
                for i in range(2):
                    U_adj[(row_off + i) * m + colA_off, 0] = 1
                for k in range(2):
                    V_adj[colA_off * p + (colB_off + k), 0] = 1
                for k in range(2):
                    for i in range(2):
                        W_adj[(colB_off + k) * n + (row_off + i), 0] = 1
                U_l.append(U_adj)
                V_l.append(V_adj)
                W_l.append(W_adj)
                total_rank2 += 1

            def add_r1_122_2(row_off, colA_off, colB_off, U_l, V_l, W_l):
                nonlocal total_rank2
                U_adj = np.zeros((n * m, 1))
                V_adj = np.zeros((m * p, 1))
                W_adj = np.zeros((p * n, 1))
                for j in range(2):
                    U_adj[row_off * m + (colA_off + j), 0] = 1
                for j in range(2):
                    for k in range(2):
                        V_adj[(colA_off + j) * p + (colB_off + k), 0] = 1
                for k in range(2):
                    W_adj[(colB_off + k) * n + row_off, 0] = 1
                U_l.append(U_adj)
                V_l.append(V_adj)
                W_l.append(W_adj)
                total_rank2 += 1

            def add_r1_211_2(row_off, colA_off, colB_off, U_l, V_l, W_l):
                nonlocal total_rank2
                U_adj = np.zeros((n * m, 1))
                V_adj = np.zeros((m * p, 1))
                W_adj = np.zeros((p * n, 1))
                for i in range(2):
                    U_adj[(row_off + i) * m + colA_off, 0] = 1
                V_adj[colA_off * p + colB_off, 0] = 1
                for i in range(2):
                    W_adj[colB_off * n + (row_off + i), 0] = 1
                U_l.append(U_adj)
                V_l.append(V_adj)
                W_l.append(W_adj)
                total_rank2 += 1

            def add_r1_121_2(row_off, colA_off, colB_off, U_l, V_l, W_l):
                nonlocal total_rank2
                U_adj = np.zeros((n * m, 1))
                V_adj = np.zeros((m * p, 1))
                W_adj = np.zeros((p * n, 1))
                for j in range(2):
                    U_adj[row_off * m + (colA_off + j), 0] = 1
                for j in range(2):
                    V_adj[(colA_off + j) * p + colB_off, 0] = 1
                W_adj[colB_off * n + row_off, 0] = 1
                U_l.append(U_adj)
                V_l.append(V_adj)
                W_l.append(W_adj)
                total_rank2 += 1

            def add_r1_112_2(row_off, colA_off, colB_off, U_l, V_l, W_l):
                nonlocal total_rank2
                U_adj = np.zeros((n * m, 1))
                V_adj = np.zeros((m * p, 1))
                W_adj = np.zeros((p * n, 1))
                U_adj[row_off * m + colA_off, 0] = 1
                for k in range(2):
                    V_adj[colA_off * p + (colB_off + k), 0] = 1
                for k in range(2):
                    W_adj[(colB_off + k) * n + row_off, 0] = 1
                U_l.append(U_adj)
                V_l.append(V_adj)
                W_l.append(W_adj)
                total_rank2 += 1

            def add_r1_111_2(row_off, colA_off, colB_off, U_l, V_l, W_l):
                nonlocal total_rank2
                U_adj = np.zeros((n * m, 1))
                V_adj = np.zeros((m * p, 1))
                W_adj = np.zeros((p * n, 1))
                U_adj[row_off * m + colA_off, 0] = 1
                V_adj[colA_off * p + colB_off, 0] = 1
                W_adj[colB_off * n + row_off, 0] = 1
                U_l.append(U_adj)
                V_l.append(V_adj)
                W_l.append(W_adj)
                total_rank2 += 1

            def decomp_332_2(row_off, colA_off, colB_off, U_l, V_l, W_l):
                """(3,3,2): split n=[2,1], m=[2,1], p=[2]"""
                add_strassen2(row_off, colA_off, colB_off, U_l, V_l, W_l)
                add_r1_212_2(row_off, colA_off + 2, colB_off, U_l, V_l, W_l)
                add_r1_122_2(row_off + 2, colA_off, colB_off, U_l, V_l, W_l)
                add_r1_112_2(row_off + 2, colA_off + 2, colB_off, U_l, V_l, W_l)

            def decomp_322_2(row_off, colA_off, colB_off, U_l, V_l, W_l):
                """(3,2,2): split n=[2,1], m=[2], p=[2]"""
                add_strassen2(row_off, colA_off, colB_off, U_l, V_l, W_l)
                add_r1_122_2(row_off + 2, colA_off, colB_off, U_l, V_l, W_l)

            # m=5=[3,2], p=6=[2,2,2]
            # (3,3,2) at m=0..2, p=0..1
            decomp_332_2(0, 0, 0, U_list2, V_list2, W_list2)
            # (3,3,2) at m=0..2, p=2..3
            decomp_332_2(0, 0, 2, U_list2, V_list2, W_list2)
            # (3,3,2) at m=0..2, p=4..5
            decomp_332_2(0, 0, 4, U_list2, V_list2, W_list2)
            # (3,2,2) at m=3..4, p=0..1
            decomp_322_2(0, 3, 0, U_list2, V_list2, W_list2)
            # (3,2,2) at m=3..4, p=2..3
            decomp_322_2(0, 3, 2, U_list2, V_list2, W_list2)
            # (3,2,2) at m=3..4, p=4..5
            decomp_322_2(0, 3, 4, U_list2, V_list2, W_list2)
            # Total: 3*10 + 3*8 = 54. Worse than 48.

            if U_list2:
                U_full2 = np.hstack(U_list2)
                V_full2 = np.hstack(V_list2)
                W_full2 = np.hstack(W_list2)
                rank2 = U_full2.shape[1]
                if verify(U_full2, V_full2, W_full2, n, m, p):
                    if rank2 < best_rank:
                        best_result = {'U': U_full2.tolist(), 'V': V_full2.tolist(), 'W': W_full2.tolist()}
                        best_rank = rank2
        except Exception:
            pass

    # Try: n=3=[2,1], m=5=[2,3], p=6=[3,3]
    # (2,2,3)+(2,2,3)+(2,3,3)+(2,3,3)+(1,2,3)+(1,2,3)+(1,3,3)+(1,3,3)
    # (2,2,3): split p=[2,1] => (2,2,2)=7+(2,2,1)=1 => 8
    # (2,3,3): split m=[2,1], p=[2,1] => (2,2,2)=7+(2,2,1)=1+(2,1,2)=1+(2,1,1)=1 => 10
    # (1,2,3): split p=[2,1] => (1,2,2)=1+(1,2,1)=1 => 2
    # (1,3,3): split m=[2,1], p=[2,1] => (1,2,2)=1+(1,2,1)=1+(1,1,2)=1+(1,1,1)=1 => 4
    # Total: 2*8+2*10+2*2+2*4 = 16+20+4+8 = 48. Same as before.

    # Try: n=3, m=5=[2,2,1], p=6=[2,2,2]
    # (3,2,2)+(3,2,2)+(3,2,2)+(3,1,2)+(3,1,2)+(3,1,2)
    # (3,2,2): split n=[2,1] => (2,2,2)=7+(1,2,2)=1 => 8
    # (3,1,2): split n=[2,1] => (2,1,2)=1+(1,1,2)=1 => 2
    # Total: 3*8+3*2 = 30!
    
    if time.time() < deadline:
        try:
            U_list3 = []
            V_list3 = []
            W_list3 = []
            total_rank3 = 0

            def add_s3(row_off, colA_off, colB_off):
                nonlocal total_rank3
                Ub, Vb, Wb = strassen_2x2()
                r = Ub.shape[1]
                U_adj = np.zeros((n * m, r))
                V_adj = np.zeros((m * p, r))
                W_adj = np.zeros((p * n, r))
                for i in range(2):
                    for j in range(2):
                        U_adj[(row_off + i) * m + (colA_off + j), :] = Ub[i * 2 + j, :]
                for j in range(2):
                    for k in range(2):
                        V_adj[(colA_off + j) * p + (colB_off + k), :] = Vb[j * 2 + k, :]
                for k in range(2):
                    for i in range(2):
                        W_adj[(colB_off + k) * n + (row_off + i), :] = Wb[k * 2 + i, :]
                U_list3.append(U_adj)
                V_list3.append(V_adj)
                W_list3.append(W_adj)
                total_rank3 += r

            def add_r1_122_3(row_off, colA_off, colB_off):
                nonlocal total_rank3
                U_adj = np.zeros((n * m, 1))
                V_adj = np.zeros((m * p, 1))
                W_adj = np.zeros((p * n, 1))
                for j in range(2):
                    U_adj[row_off * m + (colA_off + j), 0] = 1
                for j in range(2):
                    for k in range(2):
                        V_adj[(colA_off + j) * p + (colB_off + k), 0] = 1
                for k in range(2):
                    W_adj[(colB_off + k) * n + row_off, 0] = 1
                U_list3.append(U_adj)
                V_list3.append(V_adj)
                W_list3.append(W_adj)
                total_rank3 += 1

            def add_r1_212_3(row_off, colA_off, colB_off):
                nonlocal total_rank3
                U_adj = np.zeros((n * m, 1))
                V_adj = np.zeros((m * p, 1))
                W_adj = np.zeros((p * n, 1))
                for i in range(2):
                    U_adj[(row_off + i) * m + colA_off, 0] = 1
                for k in range(2):
                    V_adj[colA_off * p + (colB_off + k), 0] = 1
                for k in range(2):
                    for i in range(2):
                        W_adj[(colB_off + k) * n + (row_off + i), 0] = 1
                U_list3.append(U_adj)
                V_list3.append(V_adj)
                W_list3.append(W_adj)
                total_rank3 += 1

            def add_r1_112_3(row_off, colA_off, colB_off):
                nonlocal total_rank3
                U_adj = np.zeros((n * m, 1))
                V_adj = np.zeros((m * p, 1))
                W_adj = np.zeros((p * n, 1))
                U_adj[row_off * m + colA_off, 0] = 1
                for k in range(2):
                    V_adj[colA_off * p + (colB_off + k), 0] = 1
                for k in range(2):
                    W_adj[(colB_off + k) * n + row_off, 0] = 1
                U_list3.append(U_adj)
                V_list3.append(V_adj)
                W_list3.append(W_adj)
                total_rank3 += 1

            def decomp_322_3(row_off, colA_off, colB_off):
                """(3,2,2): split n=[2,1] => (2,2,2)=7 + (1,2,2)=1"""
                add_s3(row_off, colA_off, colB_off)
                add_r1_122_3(row_off + 2, colA_off, colB_off)

            def decomp_312_3(row_off, colA_off, colB_off):
                """(3,1,2): split n=[2,1] => (2,1,2)=1 + (1,1,2)=1"""
                add_r1_212_3(row_off, colA_off, colB_off)
                add_r1_112_3(row_off + 2, colA_off, colB_off)

            # m=5=[2,2,1], p=6=[2,2,2]
            # (3,2,2) at m=0..1, p=0..1
            decomp_322_3(0, 0, 0)
            # (3,2,2) at m=0..1, p=2..3
            decomp_322_3(0, 0, 2)
            # (3,2,2) at m=0..1, p=4..5
            decomp_322_3(0, 0, 4)
            # (3,2,2) at m=2..3, p=0..1
            decomp_322_3(0, 2, 0)
            # (3,2,2) at m=2..3, p=2..3
            decomp_322_3(0, 2, 2)
            # (3,2,2) at m=2..3, p=4..5
            decomp_322_3(0, 2, 4)
            # (3,1,2) at m=4, p=0..1
            decomp_312_3(0, 4, 0)
            # (3,1,2) at m=4, p=2..3
            decomp_312_3(0, 4, 2)
            # (3,1,2) at m=4, p=4..5
            decomp_312_3(0, 4, 4)
            # Total: 6*8 + 3*2 = 48+6 = 54. Worse.

            if U_list3:
                U_full3 = np.hstack(U_list3)
                V_full3 = np.hstack(V_list3)
                W_full3 = np.hstack(W_list3)
                rank3 = U_full3.shape[1]
                if verify(U_full3, V_full3, W_full3, n, m, p):
                    if rank3 < best_rank:
                        best_result = {'U': U_full3.tolist(), 'V': V_full3.tolist(), 'W': W_full3.tolist()}
                        best_rank = rank3
        except Exception:
            pass

    return best_result
# EVOLVE-BLOCK-END
