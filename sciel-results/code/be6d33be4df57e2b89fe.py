import numpy as np
import time

def construct():
    n, m, p = 2, 2, 7
    start_time = time.time()
    deadline = start_time + 100  # safety margin

    # Trivial rank-28 schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p  # 28
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, r] = 1.0
                    V[j * p + k, r] = 1.0
                    W[k * n + i, r] = 1.0
                    r += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Verify a decomposition is correct
    def verify(U_mat, V_mat, W_mat):
        # Check matching entries (should be 1)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    val = np.dot(U_mat[a_idx, :], V_mat[b_idx, :] * W_mat[c_idx, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        # Check non-matching entries (should be 0)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for j2 in range(m):
                        if j2 != j:
                            a_idx = i * m + j
                            b_idx = j2 * p + k
                            c_idx = k * n + i
                            val = np.dot(U_mat[a_idx, :], V_mat[b_idx, :] * W_mat[c_idx, :])
                            if abs(val) > 1e-9:
                                return False
        return True

    # Build a decomposition from explicit rank-1 terms
    # Each term is (a_indices, b_indices, c_indices, value)
    # where a_indices, b_indices, c_indices are lists of (index, coeff) pairs
    def build_from_terms(terms):
        rank = len(terms)
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        for r, (a_terms, b_terms, c_terms, val) in enumerate(terms):
            for idx, coeff in a_terms:
                U[idx, r] = coeff * val
            for idx, coeff in b_terms:
                V[idx, r] = coeff * val
            for idx, coeff in c_terms:
                W[idx, r] = coeff * val
        return U, V, W

    # Strategy: Use the isomorphism T_{2,2,7} ≅ T_{2,7,2}
    # This reinterprets as multiplying 2x7 by 7x2
    # Split columns 0-3 (rank 7 via Strassen) + columns 4-6 (rank 7 via Strassen-like)
    # Total target rank: 14 (if we can find a rank-7 for the 3-column part)
    
    # Actually, let me try a different approach:
    # Use rank-7 Strassen for columns (0,1), (2,3), and a clever rank-4 for columns (4,5,6)
    # That gives 7+7+4 = 18, not better.
    
    # Let me try: rank-7 for (0,1), rank-7 for (2,3), rank-7 for (4,5), rank-4 for (6)
    # That's 7+7+7+4 = 25 (current)
    
    # Better: Can we do rank-6 for two columns? Or rank-7 for three columns?
    
    # Let me try a rank-11 decomposition for the full 2x7x2 tensor
    # Using the isomorphism to T_{2,7,2}
    
    # For T_{2,7,2}: multiply 2x7 by 7x2
    # This is equivalent to: C = A * B where A is 2x7, B is 7x2
    # C[i,j] = sum_k A[i,k] * B[k,j]
    
    # The tensor entries are T[i,j,k,l] = delta_{i,k} * delta_{k,j}... 
    # Wait, let me be more careful.
    
    # Original tensor T_{2,2,7}[a,b,c] where:
    # a = i*m + j (i in {0,1}, j in {0,1})
    # b = j*p + k (j in {0,1}, k in {0,...,6})
    # c = k*n + i (k in {0,...,6}, i in {0,1})
    # T[a,b,c] = 1 if the indices are consistent (i.e., the j in a matches the j in b), else 0
    
    # Under isomorphism T_{2,2,7} ≅ T_{2,7,2}:
    # New indices: i' in {0,1}, j' in {0,...,6}, k' in {0,1}
    # This corresponds to: A is 2x7, B is 7x2, C is 2x2
    # T'[i',j',k'] = 1 if i'=k' (no, that's not right)
    
    # Let me think again. The tensor for matrix mult T_{m,n,p} has:
    # T[i,j,k] = 1 if the "flow" is consistent
    # For T_{2,2,7}: T[a,b,c] where a=(i,j), b=(j,k), c=(k,i)
    # T=1 iff the j in a matches the j in b, and the i in c matches the i in a, and k in c matches k in b.
    
    # Under T_{2,2,7} ≅ T_{2,7,2}:
    # We relabel so that the "middle" dimension is 7
    # New: i' in {0,1} (was i), j' in {0,...,6} (was k), k' in {0,1} (was j)
    # T'[i',j',k'] = T[i',k',j'] (just permuting the "flow" interpretation)
    # T'[i',j',k'] = 1 iff i'=k' (the i in c matches i in a) AND the k in b matches j' AND the j in a matches k'
    
    # Actually, the isomorphism is: T_{n,m,p} ≅ T_{n,p,m} by swapping the roles
    # T_{2,2,7}[a,b,c] with a=i*2+j, b=j*7+k, c=k*2+i
    # ≅ T_{2,7,2}[a',b',c'] with a'=i*7+k, b'=k*2+j, c'=j*2+i
    # where T'[a',b',c'] = 1 iff consistent flow
    
    # So in the new tensor T_{2,7,2}:
    # a' = i*7+k (i in {0,1}, k in {0,...,6}) - 14 values
    # b' = k*2+j (k in {0,...,6}, j in {0,1}) - 14 values  
    # c' = j*2+i (j in {0,1}, i in {0,1}) - 4 values
    # T'[a',b',c'] = 1 iff the k in a' matches k in b' AND j in b' matches j in c' AND i in c' matches i in a'
    
    # Now for T_{2,7,2}, we can try to find a low-rank decomposition.
    # The tensor has 14*14*4 = 784 entries, but only 2*7*2=28 are non-zero (the "consistent" ones).
    
    # For the 2x7x2 tensor, let me try to construct a decomposition.
    # C[i,j] = sum_{k=0}^{6} A[i,k] * B[k,j]
    
    # Rank-1 terms: each term picks (i,k) from A, (k,j) from B, (i,j) from C
    # But we need sums, so we use the standard technique.
    
    # For 2xN x 2, the minimal rank is at least 2N-1 for even N? 
    # Actually, for 2xN x 2, the rank is known to be 2N for N>=2 in general,
    # but can be less with special structure.
    
    # Let me try a direct construction for rank 14 (2*7):
    # For each k, have rank-1 terms for (0,k,0), (1,k,1) diagonal
    # Plus terms for off-diagonal...
    
    # Actually, let me just try to beat rank 25 by using a better block decomposition.
    
    # New idea: Use rank-7 Strassen for columns (0,1), rank-7 for (2,3), rank-7 for (4,5)
    # And handle column 6 with rank 4. Total: 25.
    
    # To improve, I need to either:
    # 1. Find a rank-6 decomposition for 2 columns (impossible for 2x2x2, min is 7)
    # 2. Find a rank-7 decomposition for 3 columns
    # 3. Find a rank-3 decomposition for 1 column (impossible, min is 4 for 2x1x2)
    
    # Let me try option 2: rank-7 for 2x2x3 (multiply 2x2 by 2x3)
    
    # For 2x2x3: C = A(2x2) * B(2x3)
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for k=0,1,2
    # 6 output entries.
    
    # Can we do this in rank 7? Let me try.
    
    # Terms for the 2x2x3 tensor:
    # We need: C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    #          C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
    #          C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2]
    #          C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
    #          C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1]
    #          C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2]
    
    # Let me try to find a rank-7 decomposition.
    # This is 2x2x3, which should have rank at most 2*2*3=12, and I believe rank 7 or 8 is achievable.
    
    # Let me try a specific construction:
    
    def try_2x2x3_decomposition():
        """Try to find a rank-7 decomposition for 2x2x3 matrix multiplication."""
        # C = A(2x2) * B(2x3)
        # Index: a = i*2+j (4 values), b = j*3+k (6 values), c = k*2+i (6 values)
        
        # Let me try to construct this.
        # Using a modified Strassen approach for the 3 columns.
        
        # For columns 0,1: use Strassen (rank 7)
        # For column 2: need additional terms
        # But we want total rank 7 for all 3 columns.
        
        # Alternative: Use a different approach.
        # Let's try rank 8 first (easier), then see if we can reduce to 7.
        
        # Rank-8 for 2x2x3:
        # Terms 0-6: Strassen for columns 0,1 (as before)
        # Term 7: Handle column 2 with one additional term? No, we need 4 terms for a single column.
        
        # Actually, for a single column k, we need 4 rank-1 terms:
        # C[0,k] = A[0,0]B[0,k] + A[0,1]B[1,k] -> 2 terms
        # C[1,k] = A[1,0]B[0,k] + A[1,1]B[1,k] -> 2 terms
        # Total 4 terms for 1 column.
        
        # So rank 7 for 3 columns seems impossible with this approach.
        # Let me try rank 10: 7 (for cols 0,1) + 3 (for col 2)? No, need 4 for col 2.
        
        # Actually, we can share terms between columns. Let me think differently.
        
        # For 2x2x3, the tensor has 6 non-zero entries (C[0,0], C[0,1], C[0,2], C[1,0], C[1,1], C[1,2]).
        # The border rank is at least the "slice rank" which is max over slices.
        
        # Let me just try to construct a rank-8 decomposition and see if it works.
        
        terms = []
        
        # Strassen for columns 0,1:
        # Term 0: (A[0,0]+A[1,1])*(B[0,0]+B[1,1]) -> C[0,0], C[1,1]
        terms.append([(0,1),(3,1)], [(0,1),(1*3+1,1)], [(0*2+0,1),(1*2+1,1)], 1)
        
        # Term 1: (A[1,0]+A[1,1])*(B[0,0]) -> C[1,0], C[1,1]
        terms.append([(2,1),(3,1)], [(0,1)], [(0*2+1,1),(1*2+1,-1)], 1)
        
        # Term 2: A[0,0]*(B[0,1]-B[1,1]) -> C[0,1], C[1,1]
        terms.append([(0,1)], [(1*3+1,1),(1*3+1,-1)], [(1*2+0,1),(1*2+1,1)], 1)
        
        # Term 3: A[1,1]*(B[1,0]-B[0,0]) -> C[0,0], C[1,0]
        terms.append([(3,1)], [(1*3+0,1),(0*3+0,-1)], [(0*2+0,1),(0*2+1,1)], 1)
        
        # Term 4: (A[0,0]+A[0,1])*(B[1,1]) -> -C[0,0], C[0,1]
        terms.append([(0,1),(1,1)], [(1*3+1,1)], [(0*2+0,-1),(1*2+0,1)], 1)
        
        # Term 5: (A[1,0]-A[0,0])*(B[0,0]+B[0,1]) -> C[1,1]
        terms.append([(2,1),(0,-1)], [(0,1),(1*3+0,1)], [(1*2+1,1)], 1)
        
        # Term 6: (A[0,1]-A[1,1])*(B[1,0]+B[1,1]) -> C[0,0]
        terms.append([(1,1),(3,-1)], [(1*3+0,1),(1*3+1,1)], [(0*2+0,1)], 1)
        
        # Term 7: Handle column 2: A[0,0]*B[0,2] -> C[0,2]
        terms.append([(0,1)], [(1*3+2,1)], [(2*2+0,1)], 1)
        
        # Term 8: A[0,1]*B[1,2] -> C[0,2]
        terms.append([(1,1)], [(1*3+2,1)], [(2*2+0,1)], 1)
        
        # Term 9: A[1,0]*B[0,2] -> C[1,2]
        terms.append([(2,1)], [(1*3+2,1)], [(2*2+1,1)], 1)
        
        # Term 10: A[1,1]*B[1,2] -> C[1,2]
        terms.append([(3,1)], [(1*3+2,1)], [(2*2+1,1)], 1)
        
        # That's rank 11, not good. Let me try a different approach.
        
        # Actually, for the full 2x2x7 problem, let me just try to use the current
        # approach but with a better decomposition for the last column.
        
        # Or, let me try to find a rank-24 decomposition by improving the last block.
        
        # For now, let me just return the current rank-25 solution but verify it carefully.
        return None
    
    # Let me just build the rank-25 solution carefully and verify it.
    
    def build_rank25():
        rank = 25
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        def add_strassen_block(col_start, k_start):
            # Term 0: (A[0,0]+A[1,1])*(B[0,k0]+B[1,k1]) -> C[0,k0], C[1,k1]
            U[0, col_start+0] += 1
            U[3, col_start+0] += 1
            V[7*0 + k_start, col_start+0] += 1
            V[7*1 + (k_start+1), col_start+0] += 1
            W[2*k_start + 0, col_start+0] += 1
            W[2*(k_start+1) + 1, col_start+0] += 1

            # Term 1: (A[1,0]+A[1,1])*(B[0,k0]) -> C[1,k0], C[1,k1]
            U[2, col_start+1] += 1
            U[3, col_start+1] += 1
            V[7*0 + k_start, col_start+1] += 1
            W[2*k_start + 1, col_start+1] += 1
            W[2*(k_start+1) + 1, col_start+1] -= 1

            # Term 2: A[0,0]*(B[0,k1] - B[1,k1]) -> C[0,k1], C[1,k1]
            U[0, col_start+2] += 1
            V[7*0 + (k_start+1), col_start+2] += 1
            V[7*1 + (k_start+1), col_start+2] -= 1
            W[2*(k_start+1) + 0, col_start+2] += 1
            W[2*(k_start+1) + 1, col_start+2] += 1

            # Term 3: A[1,1]*(B[1,k0] - B[0,k0]) -> C[0,k0], C[1,k0]
            U[3, col_start+3] += 1
            V[7*1 + k_start, col_start+3] += 1
            V[7*0 + k_start, col_start+3] -= 1
            W[2*k_start + 0, col_start+3] += 1
            W[2*k_start + 1, col_start+3] += 1

            # Term 4: (A[0,0]+A[0,1])*(B[1,k1]) -> -C[0,k0], C[0,k1]
            U[0, col_start+4] += 1
            U[1, col_start+4] += 1
            V[7*1 + (k_start+1), col_start+4] += 1
            W[2*k_start + 0, col_start+4] -= 1
            W[2*(k_start+1) + 0, col_start+4] += 1

            # Term 5: (A[1,0]-A[0,0])*(B[0,k0]+B[0,k1]) -> C[1,k1]
            U[2, col_start+5] += 1
            U[0, col_start+5] -= 1
            V[7*0 + k_start, col_start+5] += 1
            V[7*0 + (k_start+1), col_start+5] += 1
            W[2*(k_start+1) + 1, col_start+5] += 1

            # Term 6: (A[0,1]-A[1,1])*(B[1,k0]+B[1,k1]) -> C[0,k0]
            U[1, col_start+6] += 1
            U[3, col_start+6] -= 1
            V[7*1 + k_start, col_start+6] += 1
            V[7*1 + (k_start+1), col_start+6] += 1
            W[2*k_start + 0, col_start+6] += 1

        # Apply Strassen blocks for column pairs (0,1), (2,3), (4,5)
        add_strassen_block(0, 0)
        add_strassen_block(7, 2)
        add_strassen_block(14, 4)

        # Handle column 6 with 4 separate terms
        col_start = 21
        k = 6
        U[0, col_start+0] += 1
        V[7*0 + k, col_start+0] += 1
        W[2*k + 0, col_start+0] += 1

        U[1, col_start+1] += 1
        V[7*1 + k, col_start+1] += 1
        W[2*k + 0, col_start+1] += 1

        U[2, col_start+2] += 1
        V[7*0 + k, col_start+2] += 1
        W[2*k + 1, col_start+2] += 1

        U[3, col_start+3] += 1
        V[7*1 + k, col_start+3] += 1
        W[2*k + 1, col_start+3] += 1

        return U, V, W
    
    # Build and verify rank-25
    U, V, W = build_rank25()
    
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    return schoolbook()
