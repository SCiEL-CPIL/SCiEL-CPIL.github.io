"""Rank-42 structural decomposition for 2x2 by 2x11 matrix multiplication."""
import time

def construct():
    n, m, p = 2, 2, 11
    start_time = time.time()
    deadline = start_time + 110  # safety margin

    # Strategy: Use rank 42 by grouping columns of B (size 2x11) into two groups:
    # Group 1: 5 columns, Group 2: 6 columns
    # For each group, we perform a 2x2 * 2xg multiplication.
    # The 2x2 * 2xg tensor has rank at most 2*2*g = 4g if done naively, 
    # but we can share the A-factor structure.
    # Actually, let's look at the tensor T[i,j,k] = delta[i,j,k] where C[i,k] = sum_j A[i,j]B[j,k].
    # The rank of the matrix multiplication tensor MM_{n,m,p} is known to be at most n*m*p.
    # For small dimensions, exact ranks are known.
    # Rank(MM_{2,2,2}) = 7 (Strassen).
    # Rank(MM_{2,2,3})? 
    # Generally, for fixed n,m, as p grows, the rank is linear in p with slope n*m?
    # No, the rank of MM_{n,m,p} is at most n*m*p.
    # Is it possible to do better than n*m*p for large p?
    # The asymptotic rank (exponent) is the key. For fixed n,m, rank ~ p * n*m?
    # Actually, for 2x2 * 2xp, we can view it as p independent 2x2 multiplications? No.
    # C is 2xp. C[:,k] = A * B[:,k].
    # This is a linear map from (A, B) to C.
    # We can write C = A B.
    # Let's try to construct a rank 42 decomposition.
    # 42 = 44 - 2. Can we save 2 terms?
    # Strassen saves 1 term for 2x2x2 (rank 7 vs 8).
    # If we split 11 columns into 5 and 6.
    # 5 columns: 2x2 * 2x5. Naive rank 20. Can we use Strassen-like sharing?
    # 6 columns: 2x2 * 2x6. Naive rank 24.
    # Total naive 44.
    # If we treat the 11 columns as a single block, the tensor is MM_{2,2,11}.
    # Known lower bound for MM_{2,2,p} is at least 2p+? No.
    # Let's look at the structure.
    # T = sum_k T_k where T_k is the tensor for column k.
    # Each T_k is MM_{2,2,1} which is just the outer product of A and B[:,k] projected?
    # No, T_k[i,j,k'] = delta[i,j,k'] is not right.
    # The tensor is T[i,j,k] such that C[i,k] = sum_{j,l} T[i,j,k] A[i,j] B[j,k]?
    # No, standard tensor decomposition for MM_{n,m,p}:
    # T in V_n^* x V_m^* x V_p^* ? No.
    # T[i,j,k] = 1 if the term A[i,j]B[j,k] contributes to C[i,k].
    # So T[i,j,k] is non-zero only when the indices match the multiplication structure.
    # Specifically, for each (i,j,k), the term is A[i,j] * B[j,k] * (E_{i,k} in output space).
    # So the tensor is sum_{i,j,k} A_i^* \otimes B_j^* \otimes C_{i,k}^* ?
    # The decomposition is:
    # T = sum_{r=1}^R u_r \otimes v_r \otimes w_r
    # such that sum_r u_r[i] v_r[j] w_r[k] = delta_{ij} delta_{jk}? No.
    # The condition is: For all A, B, sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,k} C[i,k] (E_{i,k})
    # where C = AB.
    # So T[i,j,k] must satisfy: sum_{j} T[i,j,k] A[i,j] B[j,k] = C[i,k] = sum_{j} A[i,j] B[j,k].
    # This implies that for fixed i,k, the vector T[i,:,k] must act as the identity on the j-index?
    # No, T is a 3-tensor. The decomposition terms are rank-1 tensors.
    # A rank-1 term is u \otimes v \otimes w.
    # Contribution to C[i,k] from term r is u_r[i] * (sum_j v_r[j] B[j,k]) * w_r[k]? No.
    # The standard form:
    # C[i,k] = sum_{r=1}^R U[r,i] * V[r,j] * W[r,k] ? No, indices must contract.
    # Correct form:
    # C[i,k] = sum_{r=1}^R U[r,i] * (sum_j V[r,j] B[j,k]) * W[r,k] ? No.
    # Let's stick to the seed program's logic which is correct:
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Sum over r.
    # For the seed program (rank 44), each r corresponds to a unique (i,j,k).
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1.
    # Then sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 if (i,j,k) matches the term, else 0.
    # This exactly reconstructs the multiplication.
    
    # Can we reduce rank?
    # For 2x2x2, Strassen reduces 8 to 7.
    # The idea is to combine terms.
    # For 2x2xp, can we use Strassen on subsets?
    # If we group columns into sets of 2, we can apply Strassen to each pair.
    # 11 columns -> 5 pairs + 1 single.
    # Each pair (2x2 * 2x2) can be done with rank 7 instead of 8. Saving 1 per pair.
    # 5 pairs save 5 terms. 1 single column (2x2 * 2x1) has rank 4 (naive) or can it be less?
    # Rank of 2x2 * 2x1 is 4? 
    # C[i] = sum_j A[i,j] B[j]. This is a linear map.
    # Tensor T[i,j,k] where k=0,1.
    # T[i,j,k] = delta_{ik}? No.
    # C[i,0] = A[i,0]B[0,0] + A[i,1]B[1,0].
    # C[i,1] = A[i,0]B[0,1] + A[i,1]B[1,1].
    # Terms: (0,0,0), (0,1,0), (1,0,0), (1,1,0), (0,0,1), (0,1,1), (1,0,1), (1,1,1).
    # Rank 4? No, rank 4 is the naive bound. Is it reducible?
    # For 2x2 * 2x1, the rank is 4. (It is the sum of two 2x1 multiplications, each rank 2? No.)
    # Actually, 2x2 * 2x1 is equivalent to two 2x1 multiplications? No.
    # It is a bilinear map.
    # Let's assume rank 4 for the single column.
    # Total rank = 5 * 7 + 4 = 35 + 4 = 39.
    # Wait, is the Strassen decomposition for 2x2x2 applicable to 2x2 * 2x2 block?
    # Yes, if we treat the 2 columns as a 2x2 matrix.
    # So for columns 0,1: use Strassen rank 7.
    # Columns 2,3: use Strassen rank 7.
    # Columns 4,5: use Strassen rank 7.
    # Columns 6,7: use Strassen rank 7.
    # Columns 8,9: use Strassen rank 7.
    # Column 10: use naive rank 4.
    # Total rank = 5*7 + 4 = 39.
    
    # Let's implement this.
    # We need to map the Strassen terms to the global indices.
    
    # Strassen 2x2x2 decomposition:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # In our tensor decomposition framework:
    # Each term r has U vector (size 4), V vector (size 22), W vector (size 22).
    # For a block of 2 columns (indices k0, k1), we use 7 terms.
    # The U entries are combinations of A[i,j] for i,j in {0,1}.
    # The V entries are combinations of B[j,k] for j in {0,1}, k in {k0, k1}.
    # The W entries are combinations of output indices (i, k).
    
    # Let's define the Strassen terms for a generic 2x2 block.
    # Let col_offset be the starting column index in B and W.
    
    def strassen_block(col_offset):
        """
        Returns list of 7 tuples (U, V, W) for the 2x2 * 2x2 block starting at col_offset.
        U is dict { (i,j): coeff }
        V is dict { (j,k): coeff }
        W is dict { (i,k): coeff }
        """
        terms = []
        
        # M1 = (A00 + A11)(B00 + B11)
        # U: A00:1, A11:1
        # V: B00:1, B11:1 (relative to block)
        # W: C00:1, C11:1 (relative to block)
        u = {(0,0): 1, (1,1): 1}
        v = {(0, col_offset): 1, (1, col_offset+1): 1}
        w = {(0, col_offset): 1, (1, col_offset+1): 1}
        terms.append((u, v, w))
        
        # M2 = (A10 + A11)B00
        u = {(1,0): 1, (1,1): 1}
        v = {(0, col_offset): 1}
        w = {(1, col_offset): 1} # C10 gets M2
        terms.append((u, v, w))
        
        # M3 = A00(B10 - B11)
        u = {(0,0): 1}
        v = {(1, col_offset): 1, (1, col_offset+1): -1}
        w = {(0, col_offset+1): 1, (1, col_offset): 1} # C01 gets M3, C10 gets M3? 
        # Check: C01 = M3 + M5. C10 = M2 + M4.
        # Wait, standard Strassen:
        # C00 = M1 - M5 + M4 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # So M3 contributes to C01 and C11?
        # Let's re-verify Strassen formulas.
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 - M5 + M4 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's re-map M3.
        # M3 = A00 * (B10 - B11).
        # Contributes to C01 (+1) and C11 (+1).
        # So W for M3: {(0, col_offset+1): 1, (1, col_offset+1): 1}
        
        # Let's re-map M4.
        # M4 = A11 * (B01 - B00).
        # Contributes to C00 (+1) and C10 (+1).
        # So W for M4: {(0, col_offset): 1, (1, col_offset): 1}
        
        # Let's re-map M5.
        # M5 = (A00+A01) * B11.
        # Contributes to C00 (-1) and C01 (+1).
        # So W for M5: {(0, col_offset): -1, (0, col_offset+1): 1}
        
        # Let's re-map M6.
        # M6 = (A10-A00) * (B00+B01).
        # Contributes to C11 (+1).
        # So W for M6: {(1, col_offset+1): 1}
        
        # Let's re-map M7.
        # M7 = (A01-A11) * (B10+B11).
        # Contributes to C00 (+1).
        # So W for M7: {(0, col_offset): 1}
        
        # Let's re-map M2.
        # M2 = (A10+A11) * B00.
        # Contributes to C10 (+1) and C11 (-1).
        # So W for M2: {(1, col_offset): 1, (1, col_offset+1): -1}
        
        # Let's re-map M1.
        # M1 = (A00+A11) * (B00+B11).
        # Contributes to C00 (+1) and C11 (+1).
        # So W for M1: {(0, col_offset): 1, (1, col_offset+1): 1}
        
        # Corrected terms:
        terms = []
        
        # M1
        u = {(0,0): 1, (1,1): 1}
        v = {(0, col_offset): 1, (1, col_offset+1): 1}
        w = {(0, col_offset): 1, (1, col_offset+1): 1}
        terms.append((u, v, w))
        
        # M2
        u = {(1,0): 1, (1,1): 1}
        v = {(0, col_offset): 1}
        w = {(1, col_offset): 1, (1, col_offset+1): -1}
        terms.append((u, v, w))
        
        # M3
        u = {(0,0): 1}
        v = {(1, col_offset): 1, (1, col_offset+1): -1}
        w = {(0, col_offset+1): 1, (1, col_offset+1): 1}
        terms.append((u, v, w))
        
        # M4
        u = {(1,1): 1}
        v = {(0, col_offset+1): 1, (0, col_offset): -1}
        w = {(0, col_offset): 1, (1, col_offset): 1}
        terms.append((u, v, w))
        
        # M5
        u = {(0,0): 1, (0,1): 1}
        v = {(1, col_offset+1): 1}
        w = {(0, col_offset): -1, (0, col_offset+1): 1}
        terms.append((u, v, w))
        
        # M6
        u = {(1,0): 1, (0,0): -1}
        v = {(0, col_offset): 1, (0, col_offset+1): 1}
        w = {(1, col_offset+1): 1}
        terms.append((u, v, w))
        
        # M7
        u = {(0,1): 1, (1,1): -1}
        v = {(1, col_offset): 1, (1, col_offset+1): 1}
        w = {(0, col_offset): 1}
        terms.append((u, v, w))
        
        return terms

    # For the single column (index 10), use naive rank 4.
    def naive_block(col_offset):
        """
        Returns list of 4 tuples for 2x2 * 2x1 block at col_offset.
        """
        terms = []
        # C00 = A00 B00 + A01 B10
        # C10 = A10 B00 + A11 B10
        # Term 1: A00 * B00 -> C00
        u = {(0,0): 1}
        v = {(0, col_offset): 1}
        w = {(0, col_offset): 1}
        terms.append((u, v, w))
        
        # Term 2: A01 * B10 -> C00
        u = {(0,1): 1}
        v = {(1, col_offset): 1}
        w = {(0, col_offset): 1}
        terms.append((u, v, w))
        
        # Term 3: A10 * B00 -> C10
        u = {(1,0): 1}
        v = {(0, col_offset): 1}
        w = {(1, col_offset): 1}
        terms.append((u, v, w))
        
        # Term 4: A11 * B10 -> C10
        u = {(1,1): 1}
        v = {(1, col_offset): 1}
        w = {(1, col_offset): 1}
        terms.append((u, v, w))
        
        return terms

    all_terms = []
    
    # 5 Strassen blocks for columns 0-9
    for block_idx in range(5):
        col_offset = block_idx * 2
        terms = strassen_block(col_offset)
        all_terms.extend(terms)
        
    # 1 Naive block for column 10
    terms = naive_block(10)
    all_terms.extend(terms)
    
    rank = len(all_terms)
    # rank should be 5*7 + 4 = 39
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_dict, v_dict, w_dict) in enumerate(all_terms):
        for (i, j), val in u_dict.items():
            U[i * m + j][r] = val
        for (j, k), val in v_dict.items():
            V[j * p + k][r] = val
        for (i, k), val in w_dict.items():
            W[k * n + i][r] = val
            
    return {"U": U, "V": V, "W": W}
