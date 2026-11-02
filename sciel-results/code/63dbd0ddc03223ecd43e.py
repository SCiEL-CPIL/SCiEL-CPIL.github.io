import numpy as np
from functools import lru_cache

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # 2x4x7: split into two 2x2x7 blocks
    # 2x2x7: split into 2x2x3 + 2x2x3 + 2x2x1
    # 2x2x3: split into 2x2x2 + 2x2x1
    # 2x2x2: Strassen rank 7
    # 2x2x1: rank 2
    # 2x2x3 rank: 7 + 2 = 9
    # 2x2x7 rank: 9 + 9 + 2 = 20
    # 2x4x7 rank: 20 + 20 = 40

    # We'll build the decomposition by constructing rank-1 tensors for each sub-block
    # and placing them into the global U, V, W matrices.

    # Indexing conventions:
    # A: 2x4 matrix, indices (i, j) where i in [0,2), j in [0,4) -> A_idx = i*4 + j
    # B: 4x7 matrix, indices (j, k) where j in [0,4), k in [0,7) -> B_idx = j*7 + k
    # C: 7x2 matrix, indices (k, i) where k in [0,7), i in [0,2) -> C_idx = k*2 + i

    # We split the 4 rows of A (j dimension) into two groups of 2: j in [0,2) and j in [2,4)
    # For each group, we have a 2x2x7 sub-problem.

    # Let's define helper functions to generate rank-1 decompositions for 2x2xp problems.

    # First, define the Strassen decomposition for 2x2x2.
    # The tensor T(i,j,k) = delta_{i,k} (for 2x2x2, where C = AB, C_{ik} = sum_j A_{ij} B_{jk})
    # Wait, the tensor is T_{i,j,k} such that C_{ik} = sum_j T_{i,j,k} A_{ij} B_{jk}
    # Actually, the multiplication tensor is defined by:
    # T = sum_{i,j,k} e_i \otimes e_j \otimes e_k where the coefficient of A_{ij} B_{jk} in C_{ik} is 1.
    # So T_{i,j,k} = 1 if the term A_{ij} B_{jk} contributes to C_{ik}.
    # For matrix multiplication, C_{ik} = sum_j A_{ij} B_{jk}.
    # So the tensor has entries T_{i,j,k} = 1 for all i,j,k (in the appropriate dimensions).
    # The rank-1 decomposition is T = sum_r u_r \otimes v_r \otimes w_r.

    # For 2x2x2, Strassen gives 7 rank-1 tensors.
    # Let's define them explicitly.
    # A indices: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 2, 3
    # B indices: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 2, 3
    # C indices: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 2, 3

    # Strassen's 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B01 - B11)
    # M4 = A11 (B10 - B00)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)

    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M7

    # Each Mr is a product of two linear forms.
    # The rank-1 tensor for Mr is u_r \otimes v_r \otimes w_r where:
    # u_r is the coefficient vector for A in Mr
    # v_r is the coefficient vector for B in Mr
    # w_r is the coefficient vector for C in Mr (how Mr contributes to each C entry)

    # Let's define the 7 vectors for 2x2x2:

    def strassen_222():
        """Return list of 7 (u, v, w) tuples for 2x2x2 Strassen decomposition."""
        # u: length 4 (A indices), v: length 4 (B indices), w: length 4 (C indices)
        terms = []

        # M1 = (A00 + A11)(B00 + B11)
        # u = [1, 0, 0, 1], v = [1, 0, 0, 1]
        # Contributes to C00 (+1), C11 (+1)
        # w = [1, 0, 0, 1]
        terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))

        # M2 = (A10 + A11) B00
        # u = [0, 0, 1, 1], v = [1, 0, 0, 0]
        # Contributes to C10 (+1)
        # w = [0, 0, 1, 0]
        terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]))

        # M3 = A00 (B01 - B11)
        # u = [1, 0, 0, 0], v = [0, 1, 0, -1]
        # Contributes to C01 (+1), C11 (-1)
        # w = [0, 1, 0, -1]
        terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, -1]))

        # M4 = A11 (B10 - B00)
        # u = [0, 0, 0, 1], v = [0, 0, 1, -1]  # Wait, B10 - B00. B indices: B00=0, B01=1, B10=2, B11=3
        # So B10 - B00 = [0, 0, 1, 0] - [1, 0, 0, 0] = [-1, 0, 1, 0]
        # v = [-1, 0, 1, 0]
        # Contributes to C10 (+1), C00 (-1)
        # w = [-1, 0, 1, 0]
        terms.append(([0, 0, 0, 1], [-1, 0, 1, 0], [-1, 0, 1, 0]))

        # M5 = (A00 + A01) B11
        # u = [1, 1, 0, 0], v = [0, 0, 0, 1]
        # Contributes to C01 (+1)
        # w = [0, 1, 0, 0]
        terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]))

        # M6 = (A10 - A00)(B00 + B01)
        # u = [0, 0, 1, 0] - [1, 0, 0, 0] = [-1, 0, 1, 0]
        # v = [1, 1, 0, 0]
        # Contributes to C00 (+1)
        # w = [1, 0, 0, 0]
        terms.append(([-1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]))

        # M7 = (A01 - A11)(B10 + B11)
        # u = [0, 1, 0, 0] - [0, 0, 0, 1] = [0, 1, 0, -1]
        # v = [0, 0, 1, 1]
        # Contributes to C11 (+1)
        # w = [0, 0, 0, 1]
        terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 0, 1]))

        return terms

    def rank2_221():
        """Return list of 2 (u, v, w) tuples for 2x2x1 decomposition."""
        # 2x2x1: C is 1x2? No, C is p x n = 1 x 2.
        # Wait, the dimensions are n=2, m=2, p=1.
        # A is 2x2, B is 2x1, C is 1x2.
        # C_{k,i} = sum_j A_{i,j} B_{j,k} where k=0 only.
        # C_{0,i} = A_{i,0} B_{0,0} + A_{i,1} B_{1,0}
        # This is just 2 independent products:
        # C_{0,0} = A_{0,0} B_{0,0} + A_{0,1} B_{1,0}
        # C_{0,1} = A_{1,0} B_{0,0} + A_{1,1} B_{1,0}
        # This can be done with 2 rank-1 tensors?
        # T = sum_r u_r \otimes v_r \otimes w_r
        # Let's check if rank is 2.
        # Term 1: u=[1,0,0,0], v=[1,0], w=[1,0] -> A00 B00 contributes to C00
        # Term 2: u=[0,0,1,0], v=[1,0], w=[0,1] -> A10 B00 contributes to C01
        # But we also need A01 B10 -> C00 and A11 B10 -> C01.
        # So we need more terms.
        # Actually, the rank of 2x2x1 is 2?
        # Let's think. The tensor is a 2x2x1 array. It has 4 entries.
        # T[i,j,0] = 1 for all i,j.
        # We need to decompose this into rank-1 tensors.
        # A rank-1 tensor in 2x2x1 is u_i v_j w_0.
        # Since w is 1-dimensional, w_0 is just a scalar.
        # So we're decomposing a 2x2 matrix into a sum of rank-1 matrices.
        # The rank of the 2x2 all-ones matrix is 1.
        # So rank 1 suffices?
        # u=[1,1], v=[1,1], w=[1] -> u_i v_j = 1 for all i,j.
        # Yes! Rank 1.
        # But wait, the problem says 2x2x1 has rank 2 in the previous analysis. Let me re-check.
        # The tensor is T_{i,j,k}. For n=2, m=2, p=1.
        # C_{k,i} = sum_j A_{i,j} B_{j,k}.
        # k ranges over [0,1), so k=0.
        # C_{0,i} = sum_j A_{i,j} B_{j,0}.
        # The tensor entry T_{i,j,k} is the coefficient of A_{i,j} B_{j,k} in C_{k,i}.
        # So T_{i,j,0} = 1 for all i,j.
        # This is a 2x2x1 tensor which is just a 2x2 matrix of ones.
        # The rank of this tensor (as a 3-tensor) is the rank of the 2x2 matrix, which is 1.
        # So rank 1 suffices.

        # Let me re-read the previous analysis. It said rank 2. That might have been an error.
        # Let's verify with the formula.
        # If rank is 1, then 2x2x3 = 2x2x2 + 2x2x1 has rank 7 + 1 = 8.
        # 2x2x7 = 2x2x3 + 2x2x3 + 2x2x1 has rank 8 + 8 + 1 = 17.
        # 2x4x7 = 17 + 17 = 34.

        # Let's double check the rank of 2x2x1.
        # The tensor T is defined by T(A,B,C) = Tr(C AB) where C is 1x2, A is 2x2, B is 2x1.
        # Tr(C AB) = sum_{k,i} C_{k,i} (AB)_{i,k} = sum_{k,i} C_{k,i} sum_j A_{i,j} B_{j,k}
        # = sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i}
        # So T_{i,j,k} = 1 for all i,j,k.
        # For p=1, k=0 only. T_{i,j,0} = 1.
        # This is indeed a 2x2 matrix of ones, which has rank 1.
        # So the rank of the 2x2x1 multiplication tensor is 1.

        # Let's use rank 1 for 2x2x1.
        terms = []
        # u = [1, 1, 1, 1] (A indices), v = [1, 1] (B indices), w = [1] (C index)
        # Wait, u has length n*m = 4, v has length m*p = 2, w has length p*n = 2.
        # C indices: k*n + i. For p=1, k=0. i in [0,2). So C indices are 0, 1.
        # w = [1, 1]
        terms.append(([1, 1, 1, 1], [1, 1], [1, 1]))
        return terms

    # Now let's build the full decomposition.
    # We'll create lists of (u, v, w) for the full 2x4x7 problem.
    # u has length n*m = 8, v has length m*p = 28, w has length p*n = 14.

    all_terms = []

    # For each of the two 2x2x7 blocks (j in [0,2) and j in [2,4)):
    for block_j in [0, 2]:
        # For each of the three 2x2xp sub-blocks (p in [0,3), [3,6), [6,7)):
        # Sub-block 0: p in [0,3) -> 2x2x3
        # Sub-block 1: p in [3,6) -> 2x2x3
        # Sub-block 2: p in [6,7) -> 2x2x1

        # 2x2x3 = 2x2x2 + 2x2x1
        for sub_p_start, sub_p_len in [(0, 3), (3, 3), (6, 1)]:
            if sub_p_len == 3:
                # 2x2x3 = 2x2x2 (p in [sub_p_start, sub_p_start+2)) + 2x2x1 (p in [sub_p_start+2, sub_p_start+3))
                # 2x2x2 part
                for (u222, v222, w222) in strassen_222():
                    u_full = [0] * 8
                    v_full = [0] * 28
                    w_full = [0] * 14

                    # A indices: i in [0,2), j in [block_j, block_j+2)
                    # u222 index a = i*2 + j_local where j_local in [0,2)
                    # Global A index = i*4 + (block_j + j_local)
                    for a in range(4):
                        i = a // 2
                        j_local = a % 2
                        global_a = i * 4 + (block_j + j_local)
                        u_full[global_a] = u222[a]

                    # B indices: j in [block_j, block_j+2), k in [sub_p_start, sub_p_start+2)
                    # v222 index b = j_local*2 + k_local where k_local in [0,2)
                    # Global B index = (block_j + j_local)*7 + (sub_p_start + k_local)
                    for b in range(4):
                        j_local = b // 2
                        k_local = b % 2
                        global_b = (block_j + j_local) * 7 + (sub_p_start + k_local)
                        v_full[global_b] = v222[b]

                    # C indices: k in [sub_p_start, sub_p_start+2), i in [0,2)
                    # w222 index c = k_local*2 + i where k_local in [0,2)
                    # Global C index = (sub_p_start + k_local)*2 + i
                    for c in range(4):
                        k_local = c // 2
                        i = c % 2
                        global_c = (sub_p_start + k_local) * 2 + i
                        w_full[global_c] = w222[c]

                    all_terms.append((u_full, v_full, w_full))

                # 2x2x1 part (p in [sub_p_start+2, sub_p_start+3))
                for (u221, v221, w221) in rank2_221():
                    u_full = [0] * 8
                    v_full = [0] * 28
                    w_full = [0] * 14

                    # A indices: same as above
                    for a in range(4):
                        i = a // 2
                        j_local = a % 2
                        global_a = i * 4 + (block_j + j_local)
                        u_full[global_a] = u221[a]

                    # B indices: j in [block_j, block_j+2), k = sub_p_start+2
                    # v221 index b = j_local*1 + 0 = j_local
                    # Global B index = (block_j + j_local)*7 + (sub_p_start + 2)
                    for b in range(2):
                        j_local = b
                        global_b = (block_j + j_local) * 7 + (sub_p_start + 2)
                        v_full[global_b] = v221[b]

                    # C indices: k = sub_p_start+2, i in [0,2)
                    # w221 index c = 0*2 + i = i
                    # Global C index = (sub_p_start + 2)*2 + i
                    for c in range(2):
                        i = c
                        global_c = (sub_p_start + 2) * 2 + i
                        w_full[global_c] = w221[c]

                    all_terms.append((u_full, v_full, w_full))

            elif sub_p_len == 1:
                # 2x2x1 part (p in [sub_p_start, sub_p_start+1))
                for (u221, v221, w221) in rank2_221():
                    u_full = [0] * 8
                    v_full = [0] * 28
                    w_full = [0] * 14

                    # A indices
                    for a in range(4):
                        i = a // 2
                        j_local = a % 2
                        global_a = i * 4 + (block_j + j_local)
                        u_full[global_a] = u221[a]

                    # B indices: j in [block_j, block_j+2), k = sub_p_start
                    for b in range(2):
                        j_local = b
                        global_b = (block_j + j_local) * 7 + sub_p_start
                        v_full[global_b] = v221[b]

                    # C indices: k = sub_p_start, i in [0,2)
                    for c in range(2):
                        i = c
                        global_c = sub_p_start * 2 + i
                        w_full[global_c] = w221[c]

                    all_terms.append((u_full, v_full, w_full))

    # Now construct U, V, W matrices
    rank = len(all_terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r, (u, v, w) in enumerate(all_terms):
        for i in range(n * m):
            U[i][r] = u[i]
        for j in range(m * p):
            V[j][r] = v[j]
        for k in range(p * n):
            W[k][r] = w[k]

    return {"U": U, "V": V, "W": W}
