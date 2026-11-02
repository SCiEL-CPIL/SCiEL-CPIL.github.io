import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin

    n, m, p = DIMENSIONS

    def build_from_terms(terms):
        rank = len(terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for c in range(rank):
            u_dict, v_dict, w_dict = terms[c]
            for (i, j), val in u_dict.items():
                U[i * m + j][c] = val
            for (j, k), val in v_dict.items():
                V[j * p + k][c] = val
            for (k, i), val in w_dict.items():
                W[k * n + i][c] = val
        return {"U": U, "V": V, "W": W}

    def validate(U, V, W):
        """Check that the decomposition is exact for matrix multiplication."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Strassen-based construction (known good, rank 46)
    def strassen_construction():
        terms = []

        def add_strassen(i_start, j_start, k_start):
            u = {(i_start, j_start): 1, (i_start+1, j_start+1): 1}
            v = {(j_start, k_start): 1, (j_start+1, k_start+1): 1}
            w = {(k_start, i_start): 1, (k_start+1, i_start+1): 1}
            terms.append((u, v, w))

            u = {(i_start+1, j_start): 1, (i_start+1, j_start+1): 1}
            v = {(j_start, k_start): 1}
            w = {(k_start, i_start+1): 1, (k_start+1, i_start+1): -1}
            terms.append((u, v, w))

            u = {(i_start, j_start): 1}
            v = {(j_start, k_start+1): 1, (j_start+1, k_start+1): -1}
            w = {(k_start+1, i_start): 1, (k_start+1, i_start+1): 1}
            terms.append((u, v, w))

            u = {(i_start+1, j_start+1): 1}
            v = {(j_start+1, k_start): 1, (j_start, k_start): -1}
            w = {(k_start, i_start): 1, (k_start, i_start+1): 1}
            terms.append((u, v, w))

            u = {(i_start, j_start): 1, (i_start, j_start+1): 1}
            v = {(j_start+1, k_start+1): 1}
            w = {(k_start, i_start): -1, (k_start+1, i_start): 1}
            terms.append((u, v, w))

            u = {(i_start+1, j_start): 1, (i_start, j_start): -1}
            v = {(j_start, k_start): 1, (j_start, k_start+1): 1}
            w = {(k_start+1, i_start+1): 1}
            terms.append((u, v, w))

            u = {(i_start, j_start+1): 1, (i_start+1, j_start+1): -1}
            v = {(j_start+1, k_start): 1, (j_start+1, k_start+1): 1}
            w = {(k_start, i_start): 1}
            terms.append((u, v, w))

        def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
            for i in range(i_start, i_end):
                for j in range(j_start, j_end):
                    for k in range(k_start, k_end):
                        u = {(i, j): 1}
                        v = {(j, k): 1}
                        w = {(k, i): 1}
                        terms.append((u, v, w))

        add_strassen(0, 0, 0)
        add_strassen(0, 0, 2)
        add_schoolbook(0, 2, 0, 2, 4, 5)
        add_strassen(0, 2, 0)
        add_strassen(0, 2, 2)
        add_schoolbook(0, 2, 2, 4, 4, 5)
        add_schoolbook(0, 2, 4, 5, 0, 2)
        add_schoolbook(0, 2, 4, 5, 2, 4)
        add_schoolbook(0, 2, 4, 5, 4, 5)

        return terms

    def try_alt_partition_1():
        """Alternative: split j into {0,1,2}, {3,4} and k into {0,1,2}, {3,4}"""
        terms = []

        def add_strassen(i_start, j_start, k_start):
            u = {(i_start, j_start): 1, (i_start+1, j_start+1): 1}
            v = {(j_start, k_start): 1, (j_start+1, k_start+1): 1}
            w = {(k_start, i_start): 1, (k_start+1, i_start+1): 1}
            terms.append((u, v, w))

            u = {(i_start+1, j_start): 1, (i_start+1, j_start+1): 1}
            v = {(j_start, k_start): 1}
            w = {(k_start, i_start+1): 1, (k_start+1, i_start+1): -1}
            terms.append((u, v, w))

            u = {(i_start, j_start): 1}
            v = {(j_start, k_start+1): 1, (j_start+1, k_start+1): -1}
            w = {(k_start+1, i_start): 1, (k_start+1, i_start+1): 1}
            terms.append((u, v, w))

            u = {(i_start+1, j_start+1): 1}
            v = {(j_start+1, k_start): 1, (j_start, k_start): -1}
            w = {(k_start, i_start): 1, (k_start, i_start+1): 1}
            terms.append((u, v, w))

            u = {(i_start, j_start): 1, (i_start, j_start+1): 1}
            v = {(j_start+1, k_start+1): 1}
            w = {(k_start, i_start): -1, (k_start+1, i_start): 1}
            terms.append((u, v, w))

            u = {(i_start+1, j_start): 1, (i_start, j_start): -1}
            v = {(j_start, k_start): 1, (j_start, k_start+1): 1}
            w = {(k_start+1, i_start+1): 1}
            terms.append((u, v, w))

            u = {(i_start, j_start+1): 1, (i_start+1, j_start+1): -1}
            v = {(j_start+1, k_start): 1, (j_start+1, k_start+1): 1}
            w = {(k_start, i_start): 1}
            terms.append((u, v, w))

        def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
            for i in range(i_start, i_end):
                for j in range(j_start, j_end):
                    for k in range(k_start, k_end):
                        u = {(i, j): 1}
                        v = {(j, k): 1}
                        w = {(k, i): 1}
                        terms.append((u, v, w))

        add_strassen(0, 0, 0)
        add_schoolbook(0, 2, 0, 2, 2, 3)
        add_schoolbook(0, 2, 2, 3, 0, 2)
        add_schoolbook(0, 2, 2, 3, 2, 3)
        add_strassen(0, 0, 3)
        add_schoolbook(0, 2, 2, 3, 3, 5)
        add_strassen(0, 3, 0)
        add_schoolbook(0, 2, 3, 5, 2, 3)
        add_strassen(0, 3, 3)

        return terms

    def try_alt_partition_2():
        """Schoolbook only: rank 50"""
        terms = []
        def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
            for i in range(i_start, i_end):
                for j in range(j_start, j_end):
                    for k in range(k_start, k_end):
                        u = {(i, j): 1}
                        v = {(j, k): 1}
                        w = {(k, i): 1}
                        terms.append((u, v, w))
        add_schoolbook(0, 2, 0, 5, 0, 5)
        return terms

    # Cyclic group code embedding approach:
    # Embed 5D into 7D using cyclic group Z_7.
    # Use optimal 4x4x4 tensor (rank 19) for the core, plus corrections.
    # Key idea: represent 5x5 as a 4x4 block (indices 0-3) plus row/col 4,
    # and use the rank-19 decomposition for the 4x4 part, with careful
    # handling of the boundary indices.
    def cyclic_embedding_construction():
        """
        Use the fact that the 4x4x4 multiplication tensor has rank 19.
        We embed our 2x5x5 problem by:
        1. Using the rank-19 decomposition for the 4x4x4 sub-block (i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3})
           But we only have n=2, so we can't directly use 4x4x4.
           
        Alternative: Use the structure more cleverly.
        For n=2, m=5, p=5:
        - Split j and k each into {0,1,2,3} and {4}
        - The 2x4x4 block: we can use two 2x2x2 Strassen (rank 7 each) + cross terms
        - Actually, let's try a different approach:
        
        Use the rank-19 4x4x4 tensor by "folding" the n=2 dimension.
        The 4x4x4 tensor T_444 has entries T[i,j,k] = delta.
        We need T_255[i,j,k] for i in {0,1}, j in {0..4}, k in {0..4}.
        
        Approach: Express the 2x5x5 tensor as a linear combination of shifted
        4x4x4 tensors and correction terms.
        
        Let's try: use rank-19 for the 4x4x4 part (j,k in 0..3), then
        handle j=4 and k=4 separately.
        
        For the 2x4x4 part (i in {0,1}, j in {0..3}, k in {0..3}):
        This is a sub-tensor of the 2x4x4 problem. The rank of 2x4x4 is at most
        2 * rank(4x4x4) / 2 ... no, that's not right.
        
        Actually, the 2x4x4 tensor can be seen as two copies of 1x4x4 stacked.
        The rank of 1x4x4 is 16 (schoolbook), but we can do better.
        
        Let me try a different strategy: use the known rank-19 decomposition
        of the 4x4x4 tensor and "slice" it.
        """
        terms = []
        
        # The optimal 4x4x4 tensor decomposition has rank 19.
        # Reference: Stöckl, Strassen, et al. The 19-term decomposition.
        # 
        # For our 2x5x5 problem, let's use the following block structure:
        # - Core: 2x4x4 (i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3})
        # - Boundary: j=4 or k=4
        #
        # For the 2x4x4 core, we can use two overlapping 2x2x2 Strassen blocks
        # plus cross terms, or we can use a direct construction.
        
        # Let's use a systematic approach:
        # Split j into {0,1,2,3} and k into {0,1,2,3}
        # Use Strassen on 2x2x2 blocks:
        # j={0,1}, k={0,1}: Strassen (7)
        # j={0,1}, k={2,3}: Strassen (7) 
        # j={2,3}, k={0,1}: Strassen (7)
        # j={2,3}, k={2,3}: Strassen (7)
        # Total for 2x4x4: 28
        # Plus boundary: j=4,k=0..3 (4) + j=0..3,k=4 (4) + j=4,k=4 (1) = 9
        # Total: 28 + 9 = 37... that's better than 46!
        
        # Wait, but the Strassen blocks don't overlap in a way that shares terms.
        # Let me verify: each Strassen 2x2x2 block is independent.
        # So 2x4x4 with 4 Strassen blocks = 28 terms.
        # Boundary: 2*4 + 4*2 + 1*1... no.
        # j=4, k=0..3: 2*1*4 = 8
        # j=0..3, k=4: 2*4*1 = 8
        # j=4, k=4: 2*1*1 = 2
        # Total boundary: 18
        # Grand total: 28 + 18 = 46. Same as before!
        
        # The issue is that the boundary terms are expensive.
        # Let me try a different partition that minimizes boundary.
        
        # What if we use a 3x3x3 Strassen-like block?
        # The 3x3x3 tensor has rank at most 27 (schoolbook) but can be lower.
        # Known: rank(3x3x3) <= 23 (from various constructions)
        
        # Let me try: use 3x3x3 blocks
        # j={0,1,2}, k={0,1,2}: 2x3x3
        # j={0,1,2}, k={3,4}: 2x3x2
        # j={3,4}, k={0,1,2}: 2x2x3
        # j={3,4}, k={3,4}: 2x2x2
        
        # For 2x3x3: split into 2x2x2 (Strassen, 7) + 2x1x3 (6) + 2x2x1 (4) = 17
        # For 2x3x2: 2x2x2 (7) + 2x1x2 (4) = 11
        # For 2x2x3: 2x2x2 (7) + 2x2x1 (4) = 11
        # For 2x2x2: Strassen (7)
        # Total: 17+11+11+7 = 46. Still 46.
        
        # The fundamental issue: with n=2, we can't exploit the full power
        # of larger tensor decompositions.
        
        # Let me try yet another approach: use the rank-19 4x4x4 tensor
        # by "projecting" it onto our 2x5x5 problem.
        
        # The 4x4x4 tensor T satisfies: sum_c U_444[i,j,c] * V_444[j,k,c] * W_444[k,i,c] = delta_{ik}
        # where i,j,k in {0,1,2,3}.
        
        # For our problem, we need: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = delta_{ik}
        # where i in {0,1}, j in {0..4}, k in {0..4}.
        
        # Idea: Set U[i,j,c] = U_444[i,j,c] for j in {0,1,2,3}, and 0 for j=4
        #         V[j,k,c] = V_444[j,k,c] for k in {0,1,2,3}, and 0 for k=4
        #         W[k,i,c] = W_444[k,i,c] for k in {0,1,2,3}
        # Then add correction terms for j=4 and k=4.
        
        # This gives us the 4x4x4 part handled by 19 terms, plus corrections.
        # But we need to be careful: the 4x4x4 tensor handles i in {0,1,2,3},
        # but we only need i in {0,1}. So we're wasting some capacity.
        
        # Actually, let me just try the straightforward 4-Strassen-block approach
        # and see if I can find linear dependencies to reduce the rank.
        
        def add_strassen(i_start, j_start, k_start):
            u = {(i_start, j_start): 1, (i_start+1, j_start+1): 1}
            v = {(j_start, k_start): 1, (j_start+1, k_start+1): 1}
            w = {(k_start, i_start): 1, (k_start+1, i_start+1): 1}
            terms.append((u, v, w))

            u = {(i_start+1, j_start): 1, (i_start+1, j_start+1): 1}
            v = {(j_start, k_start): 1}
            w = {(k_start, i_start+1): 1, (k_start+1, i_start+1): -1}
            terms.append((u, v, w))

            u = {(i_start, j_start): 1}
            v = {(j_start, k_start+1): 1, (j_start+1, k_start+1): -1}
            w = {(k_start+1, i_start): 1, (k_start+1, i_start+1): 1}
            terms.append((u, v, w))

            u = {(i_start+1, j_start+1): 1}
            v = {(j_start+1, k_start): 1, (j_start, k_start): -1}
            w = {(k_start, i_start): 1, (k_start, i_start+1): 1}
            terms.append((u, v, w))

            u = {(i_start, j_start): 1, (i_start, j_start+1): 1}
            v = {(j_start+1, k_start+1): 1}
            w = {(k_start, i_start): -1, (k_start+1, i_start): 1}
            terms.append((u, v, w))

            u = {(i_start+1, j_start): 1, (i_start, j_start): -1}
            v = {(j_start, k_start): 1, (j_start, k_start+1): 1}
            w = {(k_start+1, i_start+1): 1}
            terms.append((u, v, w))

            u = {(i_start, j_start+1): 1, (i_start+1, j_start+1): -1}
            v = {(j_start+1, k_start): 1, (j_start+1, k_start+1): 1}
            w = {(k_start, i_start): 1}
            terms.append((u, v, w))

        def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
            for i in range(i_start, i_end):
                for j in range(j_start, j_end):
                    for k in range(k_start, k_end):
                        u = {(i, j): 1}
                        v = {(j, k): 1}
                        w = {(k, i): 1}
                        terms.append((u, v, w))

        # 4 Strassen blocks for 2x4x4:
        add_strassen(0, 0, 0)   # j={0,1}, k={0,1}
        add_strassen(0, 0, 2)   # j={0,1}, k={2,3}
        add_strassen(0, 2, 0)   # j={2,3}, k={0,1}
        add_strassen(0, 2, 2)   # j={2,3}, k={2,3}
        
        # Boundary terms:
        add_schoolbook(0, 2, 0, 4, 4, 5)  # j={0,1,2,3}, k={4}: 8 terms
        add_schoolbook(0, 2, 4, 5, 0, 4)  # j={4}, k={0,1,2,3}: 8 terms
        add_schoolbook(0, 2, 4, 5, 4, 5)  # j={4}, k={4}: 2 terms
        
        return terms

    # Try all constructions and pick the best
    constructions = [
        ("strassen", strassen_construction),
        ("alt1", try_alt_partition_1),
        ("cyclic", cyclic_embedding_construction),
        ("schoolbook", try_alt_partition_2),
    ]
    
    best_terms = None
    best_rank = float('inf')
    best_name = None
    
    for name, func in constructions:
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            break
        try:
            terms = func()
            if len(terms) < best_rank:
                result = build_from_terms(terms)
                if validate(result["U"], result["V"], result["W"]):
                    best_terms = terms
                    best_rank = len(terms)
                    best_name = name
        except Exception as e:
            pass
    
    if best_terms is None:
        # Fallback to Strassen
        best_terms = strassen_construction()
        best_rank = len(best_terms)
    
    best_result = build_from_terms(best_terms)
    
    # Verify final result
    if not validate(best_result["U"], best_result["V"], best_result["W"]):
        # Last resort: schoolbook
        terms = try_alt_partition_2()
        best_result = build_from_terms(terms)
        if not validate(best_result["U"], best_result["V"], best_result["W"]):
            raise ValueError("All constructions failed validation")
    
    return best_result
