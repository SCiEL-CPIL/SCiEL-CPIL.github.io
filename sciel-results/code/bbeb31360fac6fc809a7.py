import numpy as np
import time
import itertools
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n, m, p = 4, 5, 5
    nm, mp, pn = n * m, m * p, p * n
    
    # Precompute the target tensor T
    # T[i, j, k] = 1 for i in [0,4), j in [0,5), k in [0,5)
    # Indices: U index = i*m + j, V index = j*p + k, W index = k*n + i
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1.0

    def make_block(n_b, m_b, p_b, offset_i, offset_j, offset_k):
        """
        Returns a function that takes a list of terms (u_val, v_val, w_val)
        and adds them to the global U, V, W matrices.
        The terms correspond to the block multiplication:
        A_block (n_b x m_b) * B_block (m_b x p_b) = C_block (n_b x p_b)
        """
        def apply(U, V, W, terms):
            for c, (u_vals, v_vals, w_vals) in enumerate(terms):
                # u_vals: dict or list of (i_local, val)
                # v_vals: dict or list of (j_local, val)
                # w_vals: dict or list of (k_local, val)
                
                for (i_l, u_val) in u_vals:
                    i_global = offset_i + i_l
                    j_global = offset_j + 0 # j is determined by v_vals? No, u depends on i,j.
                    # Wait, the term is a rank-1 component of the tensor.
                    # The tensor entry is T[i,j,k].
                    # The rank-1 term is sum_c U[i,j,c] V[j,k,c] W[k,i,c].
                    # For a block, we have local indices i_l, j_l, k_l.
                    # Global indices: i = offset_i + i_l, j = offset_j + j_l, k = offset_k + k_l.
                    
                    # We need to add to U[i, j, c], V[j, k, c], W[k, i, c].
                    # But U is 2D (nm x rank). Index is i*m + j.
                    # So for a specific term c, U has non-zero entries at (i*m+j, c).
                    # V has non-zero entries at (j*p+k, c).
                    # W has non-zero entries at (k*n+i, c).
                    
                    pass # Logic handled in caller by constructing the full vectors for the term
        
        return apply

    # Helper to add a rank-1 term to U, V, W
    def add_term(U, V, W, c, i, j, k, u_val, v_val, w_val):
        idx_u = i * m + j
        idx_v = j * p + k
        idx_w = k * n + i
        U[idx_u, c] += u_val
        V[idx_v, c] += v_val
        W[idx_w, c] += w_val

    # Known decompositions:
    # 1. 2x2x2 (Strassen) -> Rank 7
    #    A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    #    M1 = (a+d)(e+h)
    #    M2 = (a+d)(f-g)
    #    M3 = a(e-f)
    #    M4 = d(g-h)
    #    M5 = (a+c)(e+g)
    #    M6 = (b+d)(f+h)
    #    M7 = (b-c)(g+h)
    #    C11 = M1+M4-M5-M6
    #    C12 = M2+M3+M5-M7
    #    C21 = M2-M3+M4+M7
    #    C22 = M1-M2+M3+M6
    
    # We represent a term as (u_dict, v_dict, w_dict)
    # u_dict maps (i,j) to val
    # v_dict maps (j,k) to val
    # w_dict maps (k,i) to val
    
    def strassen_2x2x2(offset_i, offset_j, offset_k):
        terms = []
        # M1
        u = {(0,0):1, (1,1):1}
        v = {(0,0):1, (0,1):1, (1,0):1, (1,1):1} # e+h means B[0,0]+B[1,1] -> v[0,0]=1, v[0,1]=1? No.
        # B is m_b x p_b. v[j,k].
        # e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1].
        # M1 = (a+d)(e+h).
        # a=A[0,0], d=A[1,1].
        # u[0,0]=1, u[1,1]=1.
        # v[0,0]=1, v[1,1]=1.
        # w[0,0]=1, w[1,1]=1? No, w[k,i].
        # C[0,0] gets M1. w[0,0] (k=0, i=0) = 1.
        # C[1,1] gets M1. w[1,1] (k=1, i=1) = 1.
        # Let's trace C11 = M1+M4-M5-M6.
        # C11 is at i=0, k=0. So w[0,0] gets +1 from M1.
        # C12 is at i=0, k=1. w[1,0] gets +1 from M2.
        # C21 is at i=1, k=0. w[0,1] gets +1 from M2.
        # C22 is at i=1, k=1. w[1,1] gets +1 from M1.
        
        # Term 1: M1
        terms.append(({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}))
        # Term 2: M2 = (a+d)(f-g)
        # f=B[0,1], g=B[1,0].
        # v[0,1]=1, v[1,0]=-1.
        # C12 (i=0, k=1) gets +1 -> w[1,0]=1.
        # C21 (i=1, k=0) gets +1 -> w[0,1]=1.
        terms.append(({(0,0):1, (1,1):1}, {(0,1):1, (1,0):-1}, {(1,0):1, (0,1):1}))
        # Term 3: M3 = a(e-f)
        # a=A[0,0]. u[0,0]=1.
        # e=B[0,0], f=B[0,1]. v[0,0]=1, v[0,1]=-1.
        # C12 (i=0, k=1) gets +1 -> w[1,0]=1.
        # C21 (i=1, k=0) gets -1 -> w[0,1]=-1.
        terms.append(({(0,0):1}, {(0,0):1, (0,1):-1}, {(1,0):1, (0,1):-1}))
        # Term 4: M4 = d(g-h)
        # d=A[1,1]. u[1,1]=1.
        # g=B[1,0], h=B[1,1]. v[1,0]=1, v[1,1]=-1.
        # C11 (i=0, k=0) gets +1 -> w[0,0]=1.
        # C22 (i=1, k=1) gets +1 -> w[1,1]=1.
        terms.append(({(1,1):1}, {(1,0):1, (1,1):-1}, {(0,0):1, (1,1):1}))
        # Term 5: M5 = (a+c)(e+g)
        # a=A[0,0], c=A[1,0]. u[0,0]=1, u[1,0]=1.
        # e=B[0,0], g=B[1,0]. v[0,0]=1, v[1,0]=1.
        # C11 (i=0, k=0) gets -1 -> w[0,0]=-1.
        # C12 (i=0, k=1) gets +1 -> w[1,0]=1.
        terms.append(({(0,0):1, (1,0):1}, {(0,0):1, (1,0):1}, {(0,0):-1, (1,0):1}))
        # Term 6: M6 = (b+d)(f+h)
        # b=A[0,1], d=A[1,1]. u[0,1]=1, u[1,1]=1.
        # f=B[0,1], h=B[1,1]. v[0,1]=1, v[1,1]=1.
        # C11 (i=0, k=0) gets -1 -> w[0,0]=-1.
        # C22 (i=1, k=1) gets +1 -> w[1,1]=1.
        terms.append(({(0,1):1, (1,1):1}, {(0,1):1, (1,1):1}, {(0,0):-1, (1,1):1}))
        # Term 7: M7 = (b-c)(g+h)
        # b=A[0,1], c=A[1,0]. u[0,1]=1, u[1,0]=-1.
        # g=B[1,0], h=B[1,1]. v[1,0]=1, v[1,1]=1.
        # C12 (i=0, k=1) gets -1 -> w[1,0]=-1.
        # C21 (i=1, k=0) gets +1 -> w[0,1]=1.
        terms.append(({(0,1):1, (1,0):-1}, {(1,0):1, (1,1):1}, {(1,0):-1, (0,1):1}))
        
        return terms

    # 2. 2x3x3 -> Rank 7 (Known decomposition for 2x3x3 matrix multiplication)
    # Reference: "Low rank decompositions for matrix multiplication"
    # A is 2x3, B is 3x3.
    # Let A = [[a1,a2,a3],[b1,b2,b3]], B = [[c1,c2,c3],[d1,d2,d3],[e1,e2,e3]]
    # This one is harder to recall exactly. Let's use 2x2x2 and 1xNxN and Nx1xN blocks if possible.
    # Actually, we can decompose 2x3x3 into 2x2x2 + 2x1x1 + 1x2x2? No, shapes don't match.
    # Let's stick to partitions that allow 2x2x2, 1xNxN, Nx1xN, and 1x1x1 blocks.
    # 1xNxN: Rank N. (Diagonal of A is 1x1, B is 1xN, C is 1xN).
    #   A[i,j] * B[j,k] = C[i,k].
    #   For i=0, sum_j A[0,j] B[j,k] = C[0,k].
    #   Terms: for each j, u[0,j]=1, v[j,k]=1 (for all k), w[k,0]=1?
    #   Wait, if we use schoolbook for 1xNxN, rank is N.
    #   Can we do better? 1xNxN is just dot product. Rank is N (unless N=1).
    #   Actually, 1xNxN multiplication is just a vector-matrix product.
    #   Tensor T[i,j,k] with i=0. T[0,j,k] = 1.
    #   Rank of this tensor is N. (Basis: for each j, term u[0,j]=1, v[j,k]=delta_k? No.
    #   Term c: u[0,j_c]=1, v[j_c, k]=1 for all k? No.
    #   Let's check rank of 1xN x NxP.
    #   If we fix i=0, T[0, j, k] = 1.
    #   This is the all-ones matrix of size N x P. Rank 1?
    #   No, the tensor is T[i,j,k]. i is only 0.
    #   T[0,j,k] = 1.
    #   We need sum_c U[0,j,c] V[j,k,c] W[k,0,c] = 1.
    #   If we set U[0,j,0]=1, V[j,k,0]=1, W[k,0,0]=1, then sum is 1.
    #   So rank 1 is possible for 1xNxP?
    #   Let's verify.
    #   U[0,j] = 1 for all j.
    #   V[j,k] = 1 for all j,k.
    #   W[k,0] = 1 for all k.
    #   Product: 1*1*1 = 1.
    #   Yes! 1xNxP has rank 1.
    #   Similarly Nx1xP has rank 1.
    #   Nx1x1 has rank 1.
    #   1x1xP has rank 1.
    #   1x1x1 has rank 1.
    
    # So we can use Rank 1 for any block with a dimension of size 1.
    # And Rank 7 for 2x2x2.
    # What about 2x3x3?
    # We can split 2x3x3 into 2x2x2 (Rank 7) + 2x1x2 (Rank 1) + 1x3x3? No.
    # Split columns: 2x3x3 = 2x2x3 + 2x1x3.
    # 2x2x3: Split rows? 2x2x3 = 2x2x2 + 2x2x1.
    # 2x2x1: Rank 1?
    #   A 2x2, B 2x1, C 2x1.
    #   T[i,j,k] with k=0.
    #   T[i,j,0] = 1.
    #   U[i,j,0]=1, V[j,0,0]=1, W[0,i,0]=1?
    #   Sum U[i,j] V[j,0] W[0,i] = 1.
    #   If U[i,j]=1, V[j,0]=1, W[0,i]=1, sum=1.
    #   So 2x2x1 has rank 1.
    #   Similarly 1x2x2 has rank 1.
    #   2x1x2 has rank 1.
    
    # So:
    # 2x2x2 -> Rank 7
    # 2x2x1 -> Rank 1
    # 2x1x2 -> Rank 1
    # 1x2x2 -> Rank 1
    # 2x1x1 -> Rank 1
    # 1x2x1 -> Rank 1
    # 1x1x2 -> Rank 1
    # 1x1x1 -> Rank 1
    # 1xNxP -> Rank 1
    # Nx1xP -> Rank 1
    # Nx1x1 -> Rank 1
    # 1x1xP -> Rank 1
    
    # We need to partition 4x5x5 into blocks of these types.
    # To minimize rank, we want to maximize the number of 2x2x2 blocks (cost 7) vs others.
    # Actually, 2x2x2 is cost 7 for 4 elements (2*2*2=8? No, 2x2x2 is 8 entries, rank 7).
    # Cost per entry: 7/8 = 0.875.
    # 1xNxP is cost 1 for N*P entries. Cost per entry: 1/(NP).
    # For 1x5x5, cost 1 for 25 entries. 0.04 per entry. Very cheap!
    # So we want as many 1xNxP blocks as possible.
    
    # Strategy:
    # Partition dimensions into sets.
    # I: {0,1,2,3}
    # J: {0,1,2,3,4}
    # K: {0,1,2,3,4}
    
    # We can group indices.
    # If we group I into {0}, {1,2,3}, we get blocks:
    # 1x5x5 (Rank 1)
    # 3x5x5
    # 3x5x5 can be split further.
    # If we group J into {0}, {1,2,3,4}, and K into {0}, {1,2,3,4}:
    # Block 3x5x5:
    #   Split J: 3x1x5 (Rank 1) + 3x4x5
    #   Split K: 3x4x1 (Rank 1) + 3x4x4
    #   3x4x4:
    #     Split I: 1x4x4 (Rank 1) + 2x4x4
    #     2x4x4:
    #       Split J: 2x2x4 + 2x2x4
    #       2x2x4:
    #         Split K: 2x2x2 (Rank 7) + 2x2x2 (Rank 7)
    #       So 2x4x4 = 2x2x2 + 2x2x2 + 2x2x2 + 2x2x2 = 4 * 7 = 28.
    #       Total for 3x4x4 = 1 (1x4x4) + 28 = 29.
    #       Total for 3x5x5 = 1 (3x1x5) + 1 (3x4x1) + 29 = 31.
    #       Total for 4x5x5 = 1 (1x5x5) + 31 = 32.
    
    # Can we do better?
    # Schoolbook is 100.
    # 32 is a huge improvement.
    # Is it valid?
    # We must ensure the blocks are disjoint and cover all indices.
    # I={0,1,2,3}, J={0,1,2,3,4}, K={0,1,2,3,4}.
    # Partition I: {0}, {1,2,3}
    # Partition J: {0}, {1,2,3,4}
    # Partition K: {0}, {1,2,3,4}
    # This creates a grid of blocks.
    # Block (I={0}, J={0}, K={0}): 1x1x1 -> Rank 1
    # Block (I={0}, J={0}, K={1..4}): 1x1x4 -> Rank 1
    # Block (I={0}, J={1..4}, K={0}): 1x4x1 -> Rank 1
    # Block (I={0}, J={1..4}, K={1..4}): 1x4x4 -> Rank 1
    # Block (I={1..3}, J={0}, K={0}): 3x1x1 -> Rank 1
    # Block (I={1..3}, J={0}, K={1..4}): 3x1x4 -> Rank 1
    # Block (I={1..3}, J={1..4}, K={0}): 3x4x1 -> Rank 1
    # Block (I={1..3}, J={1..4}, K={1..4}): 3x4x4 -> Rank ?
    
    # We calculated 3x4x4 = 29 using 2x2x2 blocks.
    # Let's refine 3x4x4.
    # I={1,2,3}, J={1,2,3,4}, K={1,2,3,4}.
    # Split I: {1}, {2,3}
    # Block 1x4x4 -> Rank 1.
    # Block 2x4x4.
    # Split J: {1,2}, {3,4}
    # Block 2x2x4.
    # Split K: {1,2}, {3,4}
    # Block 2x2x2 (Rank 7) + 2x2x2 (Rank 7) = 14.
    # So 2x4x4 = 14 + 14 = 28.
    # Total 3x4x4 = 1 + 28 = 29.
    # Total 4x5x5 = 1 (1x5x5 part) + 1 (3x1x5 part) + 1 (3x4x1 part) + 29 = 32.
    
    # Wait, my previous sum was:
    # 1x5x5 (I={0}) = 1.
    # 3x5x5 (I={1..3}) = 31.
    # Total 32.
    
    # Can we optimize 3x4x4 further?
    # 3x4x4.
    # Maybe use 3x3x3? Rank 23.
    # 3x3x3 + 3x1x4 + 0x3x1 + 0x1x4?
    # 3x3x3 (Rank 23) + 3x1x4 (Rank 1) + 0...
    # 3x4x4 = 3x3x4 + 0x1x4.
    # 3x3x4 = 3x3x3 + 3x3x1.
    # 3x3x3 (23) + 3x3x1 (1) = 24.
    # 3x4x4 = 24 + 3x1x4 (1) = 25.
    # 25 < 29.
    
    # So if we use 3x3x3 decomposition, we get 25 for the 3x4x4 block.
    # Total 4x5x5 = 1 (1x5x5) + 1 (3x1x5) + 1 (3x4x1) + 25 = 28.
    
    # Do we have a 3x3x3 decomposition?
    # Yes, rank 23 is known (Landsberg, Strassen, etc.).
    # Implementing a specific 3x3x3 rank 23 decomposition is complex but doable.
    # However, we can also just use 2x2x2 blocks and see if we can pack better.
    # Or use 3x3x3 if we can find a standard set of terms.
    
    # Let's try to find a standard 3x3x3 rank 23 decomposition online or in memory.
    # It's often defined by a set of 23 triples.
    # If we don't have it memorized, we can fall back to 2x2x2 (Rank 29 for 3x4x4).
    # Or we can use 2x2x2 for 2x2x2 parts and 1xNxN for the rest.
    
    # Let's stick to the 2x2x2 + 1xNxN strategy for robustness, as 3x3x3 terms are hard to get right without a reference.
    # Rank 32 is already a massive improvement over 100.
    
    # Let's implement the partition:
    # I: [0], [1,2,3]
    # J: [0], [1,2,3,4]
    # K: [0], [1,2,3,4]
    
    # Blocks:
    # 1. I={0}, J={0}, K={0}: 1x1x1 -> Rank 1
    # 2. I={0}, J={0}, K={1,2,3,4}: 1x1x4 -> Rank 1
    # 3. I={0}, J={1,2,3,4}, K={0}: 1x4x1 -> Rank 1
    # 4. I={0}, J={1,2,3,4}, K={1,2,3,4}: 1x4x4 -> Rank 1
    # 5. I={1,2,3}, J={0}, K={0}: 3x1x1 -> Rank 1
    # 6. I={1,2,3}, J={0}, K={1,2,3,4}: 3x1x4 -> Rank 1
    # 7. I={1,2,3}, J={1,2,3,4}, K={0}: 3x4x1 -> Rank 1
    # 8. I={1,2,3}, J={1,2,3,4}, K={1,2,3,4}: 3x4x4 -> Rank 29 (using 2x2x2)
    
    # Total Rank = 1+1+1+1+1+1+1+29 = 36.
    # Wait, earlier I said 32.
    # 1x5x5 (I={0}) covers J={0..4}, K={0..4}.
    # My block decomposition splits J and K.
    # Block 1+2+3+4 covers I={0}.
    # Block 1: 1x1x1.
    # Block 2: 1x1x4.
    # Block 3: 1x4x1.
    # Block 4: 1x4x4.
    # Total for I={0} is 1+1+1+1 = 4 terms.
    # But 1x5x5 can be done in 1 term!
    # Why did I split it?
    # Because I'm using a grid partition.
    # If I use 1x5x5 as a single block, it's Rank 1.
    # Then I={1..3}, J={0..4}, K={0..4} is 3x5x5.
    # 3x5x5:
    #   Split J: 3x1x5 (Rank 1) + 3x4x5.
    #   3x4x5:
    #     Split K: 3x4x1 (Rank 1) + 3x4x4.
    #     3x4x4: Rank 29.
    #   Total 3x5x5 = 1 + 1 + 29 = 31.
    # Total 4x5x5 = 1 + 31 = 32.
    
    # So we should NOT split the 1x5x5 block. We should treat it as a whole.
    # Similarly, 3x1x5 is Rank 1.
    # 3x4x1 is Rank 1.
    # 3x4x4 is the only complex one.
    
    # So the blocks are:
    # 1. I={0}, J={0..4}, K={0..4} -> 1x5x5 -> Rank 1
    # 2. I={1..3}, J={0}, K={0..4} -> 3x1x5 -> Rank 1
    # 3. I={1..3}, J={1..4}, K={0} -> 3x4x1 -> Rank 1
    # 4. I={1..3}, J={1..4}, K={1..4} -> 3x4x4 -> Rank 29
    
    # Total Rank = 32.
    
    # Let's verify 3x4x4 decomposition with 2x2x2.
    # I={1,2,3}, J={1,2,3,4}, K={1,2,3,4}.
    # Split I: {1}, {2,3}.
    # Block A: I={1}, J={1..4}, K={1..4} -> 1x4x4 -> Rank 1.
    # Block B: I={2,3}, J={1..4}, K={1..4} -> 2x4x4.
    # Split J: {1,2}, {3,4}.
    # Block B1: I={2,3}, J={1,2}, K={1..4} -> 2x2x4.
    # Block B2: I={2,3}, J={3,4}, K={1..4} -> 2x2x4.
    # Split K for B1: {1,2}, {3,4}.
    # B1a: 2x2x2 -> Rank 7.
    # B1b: 2x2x2 -> Rank 7.
    # Split K for B2: {1,2}, {3,4}.
    # B2a: 2x2x2 -> Rank 7.
    # B2b: 2x2x2 -> Rank 7.
    # Total Block B = 28.
    # Total 3x4x4 = 1 + 28 = 29.
    
    # This seems correct.
    
    # Implementation Plan:
    # 1. Define a function to add a "Rank 1" block (any block with a dim of 1).
    # 2. Define a function to add a "2x2x2" block (Strassen).
    # 3. Define the specific partition for 4x5x5.
    # 4. Construct U, V, W.
    # 5. Verify.
    
    U = np.zeros((nm, 32))
    V = np.zeros((mp, 32))
    W = np.zeros((pn, 32))
    
    rank = 0
    
    # Helper to add a rank 1 block
    def add_rank1_block(offset_i, offset_j, offset_k, n_b, m_b, p_b):
        nonlocal rank
        # One term
        c = rank
        # U[i,j,c] = 1 for all i in block, j in block
        for i_l in range(n_b):
            for j_l in range(m_b):
                i = offset_i + i_l
                j = offset_j + j_l
                U[i*m + j, c] = 1.0
        # V[j,k,c] = 1 for all j in block, k in block
        for j_l in range(m_b):
            for k_l in range(p_b):
                j = offset_j + j_l
                k = offset_k + k_l
                V[j*p + k, c] = 1.0
        # W[k,i,c] = 1 for all k in block, i in block
        for k_l in range(p_b):
            for i_l in range(n_b):
                k = offset_k + k_l
                i = offset_i + i_l
                W[k*n + i, c] = 1.0
        rank += 1

    # Helper to add a 2x2x2 block (Strassen)
    def add_strassen_block(offset_i, offset_j, offset_k):
        nonlocal rank
        terms = strassen_2x2x2(offset_i, offset_j, offset_k)
        for u_dict, v_dict, w_dict in terms:
            c = rank
            for (i_l, j_l), u_val in u_dict.items():
                i = offset_i + i_l
                j = offset_j + j_l
                U[i*m + j, c] += u_val
            for (j_l, k_l), v_val in v_dict.items():
                j = offset_j + j_l
                k = offset_k + k_l
                V[j*p + k, c] += v_val
            for (k_l, i_l), w_val in w_dict.items():
                k = offset_k + k_l
                i = offset_i + i_l
                W[k*n + i, c] += w_val
            rank += 1

    # Block 1: I={0}, J={0..4}, K={0..4}
    add_rank1_block(0, 0, 0, 1, 5, 5)
    
    # Block 2: I={1..3}, J={0}, K={0..4}
    add_rank1_block(1, 0, 0, 3, 1, 5)
    
    # Block 3: I={1..3}, J={1..4}, K={0}
    add_rank1_block(1, 1, 0, 3, 4, 1)
    
    # Block 4: I={1..3}, J={1..4}, K={1..4}
    # Sub-block 4a: I={1}, J={1..4}, K={1..4}
    add_rank1_block(1, 1, 1, 1, 4, 4)
    
    # Sub-block 4b: I={2,3}, J={1,2}, K={1,2}
    add_strassen_block(2, 1, 1)
    
    # Sub-block 4c: I={2,3}, J={1,2}, K={3,4}
    add_strassen_block(2, 1, 3)
    
    # Sub-block 4d: I={2,3}, J={3,4}, K={1,2}
    add_strassen_block(2, 3, 1)
    
    # Sub-block 4e: I={2,3}, J={3,4}, K={3,4}
    add_strassen_block(2, 3, 3)
    
    # Total rank should be 32
    # Verify
    T_recon = np.einsum('ic,jc,kc->ijk', U, V, W)
    err = np.max(np.abs(T_recon - T))
    
    if err > 1e-10:
        # Fallback to schoolbook if verification fails
        U_school = np.zeros((nm, 100))
        V_school = np.zeros((mp, 100))
        W_school = np.zeros((pn, 100))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    U_school[idx_u, col] = 1.0
                    V_school[idx_v, col] = 1.0
                    W_school[idx_w, col] = 1.0
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
