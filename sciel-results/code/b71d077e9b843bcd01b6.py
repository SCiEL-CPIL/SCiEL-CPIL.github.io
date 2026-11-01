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
        # For each (i,j,k), check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
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
    
    # Try alternative constructions
    def try_alternative_1():
        """Try grouping differently: use Strassen on (i,k) pairs for each j"""
        terms = []
        
        def add_strassen_ik(i_start, k_start, j):
            """Strassen-like for fixed j, varying i and k (2x1x2 -> 2x2x2 with j fixed)"""
            # This doesn't directly work since Strassen needs 2x2x2
            pass
        
        # Alternative: try to use rank-6 for 2x2x2 (not possible, Strassen is optimal)
        # Try different block partitioning
        return None
    
    def try_alternative_2():
        """Use a different partition: group j as {0,1,2} and {3,4} won't work for Strassen
        Try: for each pair of i values, use a different strategy on j,k"""
        terms = []
        
        # For i=0 and i=1, we have C[0,k] = sum_j A[0,j]*B[j,k] and C[1,k] = sum_j A[1,j]*B[j,k]
        # These are two independent 1x5 x 5x5 multiplications, each of rank 5
        # But we can share structure between them
        
        # Try: use Strassen on the (i,j) plane for k blocks
        def add_strassen_ij(i_start, j_start, k_start):
            """Standard Strassen on 2x2x2 block"""
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
        
        # Same as before but let's verify
        add_strassen_ij(0, 0, 0)
        add_strassen_ij(0, 0, 2)
        add_schoolbook(0, 2, 0, 2, 4, 5)
        add_strassen_ij(0, 2, 0)
        add_strassen_ij(0, 2, 2)
        add_schoolbook(0, 2, 2, 4, 4, 5)
        add_schoolbook(0, 2, 4, 5, 0, 2)
        add_schoolbook(0, 2, 4, 5, 2, 4)
        add_schoolbook(0, 2, 4, 5, 4, 5)
        
        return terms
    
    # Start with the known good construction
    best_terms = strassen_construction()
    best_result = build_from_terms(best_terms)
    best_rank = len(best_terms)
    
    # Verify it's valid
    if not validate(best_result["U"], best_result["V"], best_result["W"]):
        raise ValueError("Strassen construction failed validation")
    
    # Try to find better via randomized search
    # Strategy: start from the Strassen decomposition and try to merge/eliminate terms
    # by finding linear dependencies
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        # Try a simple optimization: see if we can reduce rank by checking
        # if any column is linearly dependent on others
        U = np.array(best_result["U"], dtype=float)
        V = np.array(best_result["V"], dtype=float)
        W = np.array(best_result["W"], dtype=float)
        
        # Check rank of the combined tensor
        # Flatten: for each term c, the tensor is U[:,c] ⊗ V[:,c] ⊗ W[:,c]
        # We can check if any subset of terms sums to zero in the tensor sense
        # This is expensive, so skip for now
        
    # Try a different approach: use the fact that for n=2, we might be able to
    # use a rank-6 algorithm for 2x2x2 if we allow half-integers
    # Actually Strassen rank 7 is optimal for 2x2x2 over any field
    
    # Let's try: can we do better than 46 by a different block structure?
    # What if we use Strassen on (i, k) for fixed j pairs?
    # That would be: for each pair of j values, apply a 2x2x2 Strassen on (i=2, j=2, k=2)
    # But j only has 5 values, so we can form 2 pairs + 1 leftover
    
    # Actually, let's try a completely different partition:
    # Group k as {0,1}, {2,3}, {4}
    # For k in {0,1}: use Strassen on (i,j,k) where i={0,1}, j={0,1,2,3,4}
    #   This is 2x5x2, not directly Strassen-able
    #   But we can split j into {0,1}, {2,3}, {4}
    #   For j in {0,1}: Strassen 2x2x2 = 7
    #   For j in {2,3}: Strassen 2x2x2 = 7
    #   For j in {4}: schoolbook 2x1x2 = 4
    # Total for k in {0,1}: 18
    # Similarly for k in {2,3}: 18
    # For k in {4}: schoolbook 2x5x1 = 10
    # Total: 18+18+10 = 46 (same as before)
    
    # What about a non-uniform partition?
    # Try: k in {0,1,2} as a 3-wide block, k in {3,4} as a 2-wide block
    # For k in {3,4}: Strassen 2x5x2 -> split j: 7+7+4 = 18
    # For k in {0,1,2}: 2x5x3 -> this is harder
    #   Split j: for each j, 2x1x3 = 6 terms, total 5*6=30
    #   Or: for j in {0,1}: 2x2x3, for j in {2,3}: 2x2x3, for j in {4}: 2x1x3
    #   2x2x3: can we do better than 12? 
    #   2x2x3 tensor: rank is at most 2*2*3=12, lower bound is max(4,6)=6
    #   Actually for 2x2xn, the rank is known. For 2x2x3, I believe rank is 9 or 10.
    #   Let's just use schoolbook: 12 each
    # Total: 12+12+6+18 = 48 (worse)
    
    # Let's try yet another approach: use the "binary splitting" idea
    # View 5 = 4+1, so we have a 2x4x4 part and cross terms
    # 2x4x4: can be decomposed as 4 Strassen 2x2x2 blocks = 28
    # Cross terms: 2x1x4 + 2x4x1 + 2x1x1 = 8+8+2 = 18
    # Total: 28+18 = 46 (same)
    
    # It seems 46 might be close to optimal for this structure.
    # Let's try one more thing: use half-integer coefficients to potentially merge terms
    
    # Actually, let's try a brute-force check for rank 45 by seeing if we can
    # find a valid decomposition. This is very expensive, so we'll do a limited search.
    
    if time.time() - start_time < deadline - 20:
        # Try: modify the Strassen construction slightly to see if we can save 1 term
        # Idea: in the schoolbook parts, see if any term can be absorbed
        # For example, the 2x1x1 term for (j=4, k=4) contributes to both i=0 and i=1
        # Can we merge it with adjacent terms?
        
        # Try a perturbation: use 0.5 coefficients in some places
        terms_alt = []
        
        def add_strassen_v2(i_start, j_start, k_start):
            """Standard Strassen"""
            u = {(i_start, j_start): 1, (i_start+1, j_start+1): 1}
            v = {(j_start, k_start): 1, (j_start+1, k_start+1): 1}
            w = {(k_start, i_start): 1, (k_start+1, i_start+1): 1}
            terms_alt.append((u, v, w))
            
            u = {(i_start+1, j_start): 1, (i_start+1, j_start+1): 1}
            v = {(j_start, k_start): 1}
            w = {(k_start, i_start+1): 1, (k_start+1, i_start+1): -1}
            terms_alt.append((u, v, w))
            
            u = {(i_start, j_start): 1}
            v = {(j_start, k_start+1): 1, (j_start+1, k_start+1): -1}
            w = {(k_start+1, i_start): 1, (k_start+1, i_start+1): 1}
            terms_alt.append((u, v, w))
            
            u = {(i_start+1, j_start+1): 1}
            v = {(j_start+1, k_start): 1, (j_start, k_start): -1}
            w = {(k_start, i_start): 1, (k_start, i_start+1): 1}
            terms_alt.append((u, v, w))
            
            u = {(i_start, j_start): 1, (i_start, j_start+1): 1}
            v = {(j_start+1, k_start+1): 1}
            w = {(k_start, i_start): -1, (k_start+1, i_start): 1}
            terms_alt.append((u, v, w))
            
            u = {(i_start+1, j_start): 1, (i_start, j_start): -1}
            v = {(j_start, k_start): 1, (j_start, k_start+1): 1}
            w = {(k_start+1, i_start+1): 1}
            terms_alt.append((u, v, w))
            
            u = {(i_start, j_start+1): 1, (i_start+1, j_start+1): -1}
            v = {(j_start+1, k_start): 1, (j_start+1, k_start+1): 1}
            w = {(k_start, i_start): 1}
            terms_alt.append((u, v, w))
        
        def add_schoolbook_v2(i_start, i_end, j_start, j_end, k_start, k_end):
            for i in range(i_start, i_end):
                for j in range(j_start, j_end):
                    for k in range(k_start, k_end):
                        u = {(i, j): 1}
                        v = {(j, k): 1}
                        w = {(k, i): 1}
                        terms_alt.append((u, v, w))
        
        # Try: for the j=4, k in {0,1} part (2x1x2 = 4 terms),
        # can we use a Strassen-like trick with the adjacent j=3 terms?
        # This would require modifying the j in {2,3} Strassen block
        
        # For now, let's just verify the current construction is correct and return it
        result_alt = build_from_terms(terms_alt)
        if validate(result_alt["U"], result_alt["V"], result_alt["W"]):
            if len(terms_alt) < best_rank:
                best_result = result_alt
                best_rank = len(terms_alt)
    
    return best_result
