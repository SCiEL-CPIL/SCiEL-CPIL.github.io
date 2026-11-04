"""Tensor decomposition for 2x5x6 matrix multiplication via hierarchical Kronecker-product structure."""
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6

    # We exploit the structure: 2x5x6 = 2x(2+3)x(2+2+2)
    # We'll decompose m=5 as m1=2, m2=3
    # and p=6 as three blocks of size 2 each: p1=2, p2=2, p3=2
    #
    # Block (m1=2, p1=2): 2x2x2 tensor -> rank 7 (Strassen)
    # Block (m1=2, p2=2): 2x2x2 tensor -> rank 7 (Strassen)
    # Block (m1=2, p3=2): 2x2x2 tensor -> rank 7 (Strassen)
    # Block (m2=3, p1=2): 3x2x2 tensor -> rank 11 (known optimal)
    # Block (m2=3, p2=2): 3x2x2 tensor -> rank 11
    # Block (m2=3, p3=2): 3x2x2 tensor -> rank 11
    #
    # Total rank = 3*7 + 3*11 = 21 + 33 = 54

    rank = 54
    U = [[0] * rank for _ in range(n * m)]  # 10 x 54
    V = [[0] * rank for _ in range(m * p)]  # 30 x 54
    W = [[0] * rank for _ in range(p * n)]  # 12 x 54

    col = 0

    # ---- Strassen 2x2x2 decomposition ----
    # Multiplying A (2x2) by B (2x2) to get C (2x2)
    # A indices: A[i,j] with i in {0,1}, j in {0,1}
    # B indices: B[j,k] with j in {0,1}, k in {0,1}
    # C indices: C[i,k] with i in {0,1}, k in {0,1}
    #
    # Strassen's 7 multiplications (0-indexed):
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    def add_strassen_2x2x2(i_off, j_off, k_off):
        """Add 7 rank terms for a 2x2x2 block at offsets (i_off, j_off, k_off).
        i_off: offset for i in A and C (0 or 1)
        j_off: offset for j in A and B (column offset in m)
        k_off: offset for k in B and C (column offset in p)
        Returns number of columns added.
        """
        global col
        # U index: (i + i_off) * m + (j + j_off)
        # V index: (j + j_off) * p + (k + k_off)
        # W index: (k + k_off) * n + (i + i_off)

        # M1 = (A00 + A11)(B00 + B11)
        u1 = [(0, 0), (1, 1)]  # (i, j) pairs
        v1 = [(0, 0), (1, 1)]  # (j, k) pairs
        w1 = [(0, 0, 1), (1, 1, 1)]  # (k, i, coeff)
        # C00 gets +1, C11 gets +1

        terms = [
            # (u_pairs, v_pairs, w_list)
            # M1
            ([(0,0),(1,1)], [(0,0),(1,1)], [(0,0,1),(1,1,1)]),
            # M2 = (A10+A11)B00
            ([(1,0),(1,1)], [(0,0)], [(0,1,1),(1,1,-1)]),
            # M3 = A00(B10-B11)
            ([(0,0)], [(1,0),(1,1)], [(0,0,1),(1,1,1)]),
            # M4 = A11(B01-B00)
            ([(1,1)], [(0,1),(0,0)], [(0,0,1),(0,1,1)]),
            # M5 = (A00+A01)B11
            ([(0,0),(0,1)], [(1,1)], [(0,0,-1),(0,1,1)]),
            # M6 = (A10-A00)(B00+B01)
            ([(1,0),(0,0)], [(0,0),(0,1)], [(1,1,1)]),
            # M7 = (A01-A11)(B10+B11)
            ([(0,1),(1,1)], [(1,0),(1,1)], [(0,0,1)]),
        ]

        for u_pairs, v_pairs, w_list in terms:
            for (ui, uj) in u_pairs:
                u_idx = (ui + i_off) * m + (uj + j_off)
                U[u_idx][col] += 1
            for (vj, vk) in v_pairs:
                v_idx = (vj + j_off) * p + (vk + k_off)
                V[v_idx][col] += 1
            for (wk, wi, wc) in w_list:
                w_idx = (wk + k_off) * n + (wi + i_off)
                W[w_idx][col] += wc
            col += 1

    # ---- 3x2x2 tensor decomposition (rank 11) ----
    # A is 3x2, B is 2x2, C is 3x2
    # A indices: A[i,j] with i in {0,1,2}, j in {0,1}
    # B indices: B[j,k] with j in {0,1}, k in {0,1}
    # C indices: C[i,k] with i in {0,1,2}, k in {0,1}
    #
    # This can be viewed as 2 independent 3x1x1 multiplications coupled through B.
    # Actually, C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is a tensor of format (3,2)x(2,2)->(3,2)
    # Rank: we can use the fact that this is essentially two rows of B multiplied by 3x2 A.
    # One approach: treat as 2 separate 3x1x2 problems? No.
    # Actually, the tensor [i,j,k] = delta_{j,?} ... no.
    # T[i, (j,k)] = A[i,j] * B[j,k]
    # This is a rank-? tensor. For fixed j, it's outer product of A[:,j] and B[j,:].
    # Sum over j=0,1: T = A[:,0]⊗B[0,:] + A[:,1]⊗B[1,:]
    # This is a sum of 2 rank-1 terms in the (i) and (k) modes, but j is contracted.
    # Wait, the tensor is T_{i,j,k} = ... no.
    # The multiplication tensor for (3x2)(2x2)->(3x2) has indices (i, j) for A, (j, k) for B, (i, k) for C.
    # T[(i,j), (j,k), (i,k)] = 1 if the indices are consistent.
    # Actually T[i,j; j',k; i',k'] = 1 if i=i', j=j', j=j', k=k'
    # So T is supported on (i,j,j',k,i',k') with i=i', j=j'=j, k=k'
    # This is a tensor of rank... For each (j,k), we have A[:,j] ⊗ e_k contribution to C[:,k]
    # Hmm, let me think differently.
    # C[:,k] = A * B[:,k] where B[:,k] is a 2x1 vector.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    #
    # Tensor decomposition: we need to express the tensor T with indices (i,j) for U, (j,k) for V, (i,k) for W.
    # T_{(i,j),(j',k),(i',k')} = δ_{i,i'} δ_{j,j'} δ_{j,j'} δ_{k,k'} = δ_{i,i'} δ_{j,j'} δ_{k,k'} (since j=j' forced)
    # So T = sum over i,j,k of e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(i,k)}
    # That's schoolbook rank = 3*2*2 = 12.
    #
    # But we can do better. Note that for fixed k, C[:,k] = A * b_k where b_k = (B[0,k], B[1,k]).
    # This is a 3x2 matrix A times a 2x1 vector.
    # The tensor for 3x2 times 2x1 to get 3x1 has rank... 
    # Actually, let's use a different decomposition.
    # We can write the 3x2x2 tensor as a combination using the 2x2x2 Strassen-like ideas on the j dimension.
    #
    # Alternative: use the fact that 3 = 2+1.
    # Split A into top 2 rows and bottom 1 row.
    # Top 2x2 times 2x2: rank 7 (Strassen)
    # Bottom 1x2 times 2x2: this is just 1*2*2 = 4 schoolbook terms
    # Total: 7 + 4 = 11. Yes!

    def add_3x2x2(i_off, j_off, k_off):
        """Add 11 rank terms for a 3x2x2 block at offsets (i_off, j_off, k_off).
        i_off: offset for i in A and C (0 or 1, since m2 starts at i=0 or i=2... wait)
        Actually m=5, m1=2 (j=0,1), m2=3 (j=2,3,4)
        For the 3x2x2 block, A has 3 rows (i=0,1,2 relative to the block) and 2 columns
        But wait, A is n x m = 2 x 5. So A has 2 rows (i=0,1) and 5 columns (j=0..4).
        B is m x p = 5 x 6. B has 5 rows (j=0..4) and 6 columns (k=0..5).
        C is p x n = 6 x 2. C has 6 rows (k=0..5) and 2 columns (i=0,1).
        
        So when we split m into m1=2 and m2=3:
        - Block 1: A[:, 0:2] (2x2), B[0:2, :] (2x6), contributes to C
        - Block 2: A[:, 2:5] (2x3), B[2:5, :] (3x6), contributes to C
        
        Wait, I had the dimensions wrong. Let me re-read.
        A is n x m = 2 x 5
        B is m x p = 5 x 6
        C is p x n = 6 x 2
        
        C = A * B
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        So the tensor has indices (i,j) for A, (j,k) for B, (i,k) for C.
        i in {0,1}, j in {0..4}, k in {0..5}
        
        When we split j (the m dimension):
        - j1 = {0,1} (m1=2)
        - j2 = {2,3,4} (m2=3)
        
        And split k (the p dimension):
        - k1 = {0,1} (p1=2)
        - k2 = {2,3} (p2=2)
        - k3 = {4,5} (p3=2)
        
        For block (j1, k1): A[:, 0:2] is 2x2, B[0:2, 0:2] is 2x2, C[0:2, :] gets contribution
        This is a 2x2x2 tensor (i in 2, j in 2, k in 2) -> rank 7
        
        For block (j2, k1): A[:, 2:5] is 2x3, B[2:5, 0:2] is 3x2, C[0:2, :] gets contribution
        This is a 2x3x2 tensor (i in 2, j in 3, k in 2)
        
        So I need a 2x3x2 decomposition, not 3x2x2!
        
        Let me redo. The tensor for (2x3)(3x2)->(2x2) has:
        A: 2x3 (i in 0,1; j in 0,1,2)
        B: 3x2 (j in 0,1,2; k in 0,1)
        C: 2x2 (i in 0,1; k in 0,1)
        
        Schoolbook rank = 2*3*2 = 12.
        Can we do better? Split j into j=2+1:
        - j=0,1: 2x2x2 -> rank 7
        - j=2: 2x1x2 -> rank 4 (schoolbook)
        Total: 7+4 = 11.
        """
        global col
        
        # Sub-block 1: j in {0,1} relative to block start -> 2x2x2 Strassen
        # i_off is the global i offset (0 or 1... but i is only 0,1 so i_off=0 always for this problem)
        # Actually i_off is 0 since n=2 and we don't split i.
        # j_off: offset in j for the start of this 3x2x2 block
        # k_off: offset in k for the start of this 2x2 block
        
        # For the 2x2x2 part (j relative 0,1):
        add_strassen_2x2x2(0, j_off, k_off)
        
        # For the 1x2 part (j relative 2): schoolbook 2*1*2 = 4 terms
        for i in range(2):
            for k in range(2):
                u_idx = i * m + (2 + j_off)
                v_idx = (2 + j_off) * p + (k + k_off)
                w_idx = (k + k_off) * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1

    # Now build the full decomposition
    # j1 = {0,1}, j2 = {2,3,4}
    # k1 = {0,1}, k2 = {2,3}, k3 = {4,5}
    
    # Block (j1, k1): 2x2x2 at j_off=0, k_off=0
    add_strassen_2x2x2(0, 0, 0)
    
    # Block (j1, k2): 2x2x2 at j_off=0, k_off=2
    add_strassen_2x2x2(0, 0, 2)
    
    # Block (j1, k3): 2x2x2 at j_off=0, k_off=4
    add_strassen_2x2x2(0, 0, 4)
    
    # Block (j2, k1): 2x3x2 at j_off=2, k_off=0
    add_3x2x2(0, 2, 0)
    
    # Block (j2, k2): 2x3x2 at j_off=2, k_off=2
    add_3x2x2(0, 2, 2)
    
    # Block (j2, k3): 2x3x2 at j_off=2, k_off=4
    add_3x2x2(0, 2, 4)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
