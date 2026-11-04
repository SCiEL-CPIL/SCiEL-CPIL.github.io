import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # The schoolbook rank is n*m*p = 24.
    # We aim for rank 20, which is known to be optimal for 2x3x4.
    # Strategy: Use the tensor product structure.
    # 2x3x4 can be viewed as combining 2x2x2 (Strassen rank 7) with 2x1x2 (rank 2) or similar splits?
    # Actually, a known result is that R(2,3,4) = 20.
    # One way to construct it is by viewing the 2x3x4 multiplication as:
    # Split the 3 dimension into 2+1 and the 4 dimension into 2+2?
    # Or simply use the known explicit decomposition.
    
    # Let's build the rank-20 decomposition explicitly.
    # The multiplication tensor T_{2,3,4} has entries T_{i,j,k} = 1 for (i,j,k) in {0,1}x{0,1,2}x{0,1,2,3}.
    # Indices: A is 2x3 (indices i in {0,1}, j in {0,1,2}), B is 3x4 (j in {0,1,2}, k in {0,1,2,3}).
    # Output C is 2x4 (i in {0,1}, k in {0,1,2,3}).
    # C_{i,k} = sum_{j=0}^2 A_{i,j} B_{j,k}.
    
    # We can split B into B1 (3x2, cols 0,1) and B2 (3x2, cols 2,3).
    # C1 (2x2) = A * B1
    # C2 (2x2) = A * B2
    # This splits the problem into two 2x3x2 multiplications.
    # R(2,3,2) = ?
    # 2x3x2 is equivalent to 2x2x3 by symmetry? No, dimensions are fixed.
    # R(2,3,2) = 10. (Schoolbook 12, Strassen-like reduction?)
    # If R(2,3,2) = 10, then R(2,3,4) <= 10 + 10 = 20.
    # Is R(2,3,2) = 10?
    # Let's verify R(2,3,2).
    # 2x3x2 multiplication: C (2x2) = A (2x3) * B (3x2).
    # C_{i,k} = sum_{j=0}^2 A_{i,j} B_{j,k}.
    # This is equivalent to multiplying two vectors of length 3? No.
    # It is equivalent to 2x2x2 with an extra dimension?
    # Actually, R(2,3,2) = 10 is a known result.
    # Reference: Bini et al., or standard tensor rank tables.
    # R(2,2,2) = 7.
    # R(2,3,2) can be seen as 2x2x2 (rank 7) + 2x1x2 (rank 2) + 2x1x2 (rank 2)? No, that's 11.
    # Wait, 2x3x2 = 2x(2+1)x2 = 2x2x2 + 2x1x2.
    # R(2,2,2) = 7.
    # R(2,1,2) = 2.
    # So R(2,3,2) <= 7 + 2 = 9?
    # Let's check if R(2,3,2) = 9 or 10.
    # If R(2,3,2) = 9, then R(2,3,4) <= 18.
    # But literature says R(2,3,4) = 20.
    # Why? Because the splits might not be independent in a simple additive way if we just sum ranks?
    # Actually, tensor rank is sub-multiplicative but not strictly additive in a simple way for direct sums?
    # Direct sum of tensors: rank(T1 + T2) <= rank(T1) + rank(T2).
    # So if we split 2x3x4 into two 2x3x2 tensors, rank <= R(2,3,2) + R(2,3,2).
    # If R(2,3,2) = 10, then 20.
    # If R(2,3,2) = 9, then 18.
    # Let's check R(2,3,2).
    # 2x3x2: A_{i,j} B_{j,k}. i in {0,1}, j in {0,1,2}, k in {0,1}.
    # This is isomorphic to 2x2x3? No.
    # Let's look up R(2,3,2).
    # According to "The rank of the 2x3x2 matrix multiplication tensor", it is 10.
    # Wait, some sources say 10.
    # Let's assume R(2,3,2) = 10.
    # Then we need a rank-10 decomposition for 2x3x2.
    
    # Let's construct the rank-10 decomposition for 2x3x2.
    # A is 2x3, B is 3x2, C is 2x2.
    # We can use Strassen for the 2x2x2 part and handle the extra j=2 term.
    # C_{i,k} = A_{i,0}B_{0,k} + A_{i,1}B_{1,k} + A_{i,2}B_{2,k}.
    # Let A' = [A_{i,0}, A_{i,1}] (2x2), B' = [B_{0,k}, B_{1,k}] (2x2).
    # Let A2 = [A_{i,2}] (2x1), B2 = [B_{2,k}] (1x2).
    # C = A'B' + A2B2.
    # Rank(A'B') = 7 (Strassen).
    # Rank(A2B2) = 2 (since A2 is 2x1 and B2 is 1x2, the tensor is sum_{i,k} A2_i B2_k C_{i,k}, which is rank 2? 
    # Actually, A2B2 is a 2x1x2 tensor. Rank is 2.
    # So total rank <= 7 + 2 = 9?
    # Why is it 10 then?
    # Maybe the Strassen decomposition and the extra term can be combined?
    # Or maybe my assumption that R(2,1,2)=2 is too simple when added to Strassen?
    # No, direct sum rank is additive.
    # Let's re-verify R(2,3,2).
    # Actually, R(2,3,2) = 10 is widely cited.
    # Why not 9?
    # Perhaps the 2x1x2 tensor has rank 2, but the interaction prevents reduction?
    # No, direct sum is disjoint supports.
    # Support of A'B' is j in {0,1}. Support of A2B2 is j=2.
    # They are disjoint. So rank is sum of ranks.
    # So R(2,3,2) <= 7 + 2 = 9.
    # Is R(2,3,2) = 9?
    # If R(2,3,2) = 9, then R(2,3,4) <= 18.
    # But the task says "Find a rank-minimal...". And the inspiration mentions "rank-19" in the directive?
    # "Use the explicit rank-19 decomposition... derived from tensor product of rank-7 Strassen... and trivial rank-3"
    # 7 * 3 = 21? No.
    # 7 + 3 + ...?
    # Let's look at the directive again: "rank-19 decomposition ... tensor product of rank-7 Strassen ... and trivial rank-3 decomposition for the remaining linear dimension"
    # This phrasing is confusing. Tensor product of ranks multiplies. 7 * 3 = 21.
    # Maybe it means direct sum?
    # Or maybe the decomposition is for 2x3x4 directly.
    
    # Let's try to find a rank-20 solution first, as it's safer than guessing 18 or 19.
    # Actually, if I can find 18, that's better.
    # Let's check the literature for R(2,3,4).
    # R(2,3,4) = 20 is the known bound.
    # R(2,3,3) = 14?
    # R(2,2,4) = 14?
    
    # Let's implement the rank-20 construction based on two rank-10 2x3x2 blocks.
    # We need a rank-10 decomposition for 2x3x2.
    # If R(2,3,2) is actually 9, we should use 9.
    # Let's assume the standard result R(2,3,4) = 20 is correct and optimal.
    # I will construct the rank-20 solution.
    
    # Construction of 2x3x2 rank-10 (or 9) decomposition:
    # Let's use the explicit Strassen formulas for 2x2x2.
    # Strassen 2x2x2:
    # m1 = (A00+A11)(B00+B11)
    # m2 = (A10+A11)B00
    # m3 = A00(B01-B11)
    # m4 = A11(B10-B00)
    # m5 = (A00+A01)B11
    # m6 = (A10-A00)(B00+B10)
    # m7 = (A01-A11)(B01+B11)
    # C00 = m1 - m4 + m6 + m7
    # C01 = m3 + m5 + m7
    # C10 = m2 + m4 + m6
    # C11 = m1 - m3 + m2 + m5
    
    # For 2x3x2:
    # C_{i,k} = sum_{j=0}^2 A_{i,j} B_{j,k}
    # Split into j=0,1 (Strassen) and j=2 (extra).
    # Extra term: A_{i,2} B_{2,k}.
    # This is a 2x1x2 tensor.
    # Decomposition of 2x1x2:
    # Term 1: A_{0,2} B_{2,0} -> C_{0,0}
    # Term 2: A_{0,2} B_{2,1} -> C_{0,1}
    # Term 3: A_{1,2} B_{2,0} -> C_{1,0}
    # Term 4: A_{1,2} B_{2,1} -> C_{1,1}
    # Wait, rank of 2x1x2 is 2?
    # T_{i,j,k} = A_i B_k.
    # This is a rank-1 tensor? No.
    # T_{i,k} = A_i B_k.
    # This is an outer product of vector A and vector B.
    # As a 3-tensor T_{i, j, k}, the middle index j is fixed (j=2).
    # So T_{i, 2, k} = A_i B_k.
    # This is a sum of 2 rank-1 tensors?
    # No, it is a single rank-1 tensor in the space of matrices?
    # Rank of a tensor T_{i,j,k} is min r such that T = sum_{l=1}^r u_l v_l w_l.
    # Here, T_{i,2,k} = A_i B_k.
    # We can write this as u_1 v_1 w_1 where u_1 = A, v_1 = e_2 (basis vector for j=2), w_1 = B.
    # So it is rank 1!
    # Wait, is it?
    # u_1[i] = A_i.
    # v_1[j] = 1 if j=2 else 0.
    # w_1[k] = B_k.
    # Sum u_1[i] v_1[j] w_1[k] = A_i * (j==2) * B_k.
    # Yes, this is exactly the term A_{i,2} B_{2,k} when j=2, and 0 otherwise.
    # So the extra term has rank 1.
    # Therefore R(2,3,2) <= 7 + 1 = 8.
    # Is R(2,3,2) = 8?
    # If so, R(2,3,4) <= 8 + 8 = 16.
    # This seems too good. Why is the bound 20?
    
    # Let's re-read the tensor definition.
    # T_{i,j,k} = 1 if ...
    # In matrix multiplication, the tensor is T_{i,j,k} = delta_{i,j,k} ? No.
    # The multiplication tensor is defined by C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The tensor T has components T_{i,j,k} such that C_{i,k} = sum_{j} T_{i,j,k} A_{i,j} B_{j,k}?
    # No, the standard definition is:
    # T(A,B) = C.
    # The tensor is the linear map.
    # The rank is the number of multiplications.
    # My decomposition of the extra term A_{i,2} B_{2,k} as rank 1 is correct for that sub-tensor.
    # So why is R(2,3,2) not 8?
    # Maybe the Strassen decomposition and the extra term can be merged?
    # Or maybe my assumption that R(2,1,2)=1 is correct?
    # Yes, a 2x1x2 tensor where the middle index is fixed is rank 1.
    # So R(2,3,2) <= 8.
    # Let's check if R(2,3,2) = 8 is a known result.
    # Some sources say R(2,3,2) = 10.
    # Why?
    # Ah, the "trivial" decomposition of the extra term might not be compatible with the Strassen structure in a way that allows simple addition?
    # No, direct sum is always additive.
    # Unless... the indices don't align?
    # In 2x3x2, j goes 0,1,2.
    # Strassen uses j=0,1.
    # Extra uses j=2.
    # They are disjoint.
    # So rank should be sum.
    # Is it possible that R(2,2,2) is not 7 in this context? No, it's 7.
    # Is it possible that R(2,1,2) is not 1?
    # Let's check the rank of the tensor T where T_{i,j,k} = 1 if j=2, else 0.
    # T = sum_{i,k} A_i B_k e_2.
    # This is u v w^T?
    # u is vector in R^2 (A), v is vector in R^3 (e_2), w is vector in R^2 (B).
    # T_{i,j,k} = u_i v_j w_k.
    # This is a rank-1 tensor.
    # So R(2,3,2) <= 8.
    
    # Let's try to find a rank-16 solution for 2x3x4.
    # If R(2,3,4) = 16, that's a huge improvement over 20.
    # However, I suspect there is a catch.
    # Maybe the "schoolbook" rank is not the only baseline.
    # Let's look at the inspiration again: "rank-19 ... tensor product of rank-7 Strassen ... and trivial rank-3".
    # 7 * 3 = 21.
    # Maybe it's not a tensor product.
    
    # Let's assume the known optimal rank for 2x3x4 is 20.
    # Why would it be 20 and not 16?
    # Perhaps the decomposition of 2x3x2 is not 8?
    # Let's check "rank of 2x3x2 matrix multiplication tensor".
    # According to a 2015 paper, R(2,3,2) = 10.
    # Why?
    # Maybe the Strassen decomposition uses linear combinations that "leak" into the j=2 space? No, it's exact.
    # Maybe the extra term is not rank 1?
    # Let's re-evaluate T_{i,2,k} = A_{i,2} B_{2,k}.
    # This is a bilinear form.
    # As a 3-tensor, it is indeed rank 1.
    # So why 10?
    # Perhaps the question implies that we must use a specific structure?
    # No, "exact tensor decomposition".
    
    # Let's try to construct the rank-20 solution explicitly, as it is the safe "known" bound.
    # If I can construct rank 16, I will.
    # Let's try to construct rank 16.
    
    # Step 1: Construct 2x3x2 rank-8 decomposition.
    # Step 2: Duplicate for the two halves of 2x3x4.
    
    # Let's implement the rank-8 decomposition for 2x3x2.
    # Indices: i in {0,1}, j in {0,1,2}, k in {0,1}.
    # U, V, W matrices.
    # U: 2x3 x 8
    # V: 3x2 x 8
    # W: 2x2 x 8
    
    # Strassen part (j=0,1):
    # 7 terms.
    # Extra part (j=2):
    # 1 term.
    
    # Let's build the matrices.
    
    def build_2x3x2_rank8():
        # U: 6x8, V: 6x8, W: 4x8
        # Wait, dimensions:
        # A is 2x3 -> 6 entries.
        # B is 3x2 -> 6 entries.
        # C is 2x2 -> 4 entries.
        
        U = np.zeros((6, 8), dtype=float)
        V = np.zeros((6, 8), dtype=float)
        W = np.zeros((4, 8), dtype=float)
        
        # Strassen 2x2x2 part (j=0,1)
        # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) ... wait, A is 2x3.
        # Let's map A_{i,j} to index i*3+j.
        # A_{0,0}=0, A_{0,1}=1, A_{0,2}=2
        # A_{1,0}=3, A_{1,1}=4, A_{1,2}=5
        
        # B indices: B_{j,k} to index j*2+k.
        # B_{0,0}=0, B_{0,1}=1
        # B_{1,0}=2, B_{1,1}=3
        # B_{2,0}=4, B_{2,1}=5
        
        # C indices: C_{i,k} to index i*2+k.
        # C_{0,0}=0, C_{0,1}=1
        # C_{1,0}=2, C_{1,1}=3
        
        # Strassen uses A_{0,0}, A_{0,1}, A_{1,0}, A_{1,1} and B_{0,0}, B_{0,1}, B_{1,0}, B_{1,1}.
        # These correspond to A indices 0,1,3,4 and B indices 0,1,2,3.
        
        # m1 = (A00+A11)(B00+B11)
        # A00=0, A11=4. B00=0, B11=3.
        U[0, 0] = 1; U[4, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        # C00 += m1, C11 += m1
        W[0, 0] = 1; W[3, 0] = 1
        
        # m2 = (A10+A11)B00
        # A10=3, A11=4. B00=0.
        U[3, 1] = 1; U[4, 1] = 1
        V[0, 1] = 1
        # C10 += m2
        W[2, 1] = 1
        
        # m3 = A00(B01-B11)
        # A00=0. B01=1, B11=3.
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        # C01 += m3
        W[1, 2] = 1
        
        # m4 = A11(B10-B00)
        # A11=4. B10=2, B00=0.
        U[4, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        # C00 -= m4, C10 += m4
        W[0, 3] = -1; W[2, 3] = 1
        
        # m5 = (A00+A01)B11
        # A00=0, A01=1. B11=3.
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        # C01 += m5, C11 += m5
        W[1, 4] = 1; W[3, 4] = 1
        
        # m6 = (A10-A00)(B00+B10)
        # A10=3, A00=0. B00=0, B10=2.
        U[3, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[2, 5] = 1
        # C00 += m6, C10 += m6
        W[0, 5] = 1; W[2, 5] = 1
        
        # m7 = (A01-A11)(B01+B11)
        # A01=1, A11=4. B01=1, B11=3.
        U[1, 6] = 1; U[4, 6] = -1
        V[1, 6] = 1; V[3, 6] = 1
        # C00 += m7, C01 += m7
        W[0, 6] = 1; W[1, 6] = 1
        
        # Extra term: A_{i,2} B_{2,k}
        # A_{0,2}=2, A_{1,2}=5.
        # B_{2,0}=4, B_{2,1}=5.
        # Term: A_{0,2} B_{2,0} -> C_{0,0}
        # Term: A_{0,2} B_{2,1} -> C_{0,1}
        # Term: A_{1,2} B_{2,0} -> C_{1,0}
        # Term: A_{1,2} B_{2,1} -> C_{1,1}
        # This is rank 1?
        # T_{i,2,k} = A_{i,2} B_{2,k}.
        # We need to represent this as sum of rank-1 tensors.
        # It is a single rank-1 tensor u v w where:
        # u = [A_{0,2}, A_{1,2}]^T = [U[2], U[5]]
        # v = [0, 0, 1]^T (for j=2) -> V[4], V[5]
        # w = [B_{2,0}, B_{2,1}]^T = [V[4], V[5]]? No, w is for C indices.
        # w = [1, 1, 1, 1]? No.
        # C_{i,k} gets A_{i,2} B_{2,k}.
        # So W[C_{0,0}] += A_{0,2} B_{2,0}
        # W[C_{0,1}] += A_{0,2} B_{2,1}
        # W[C_{1,0}] += A_{1,2} B_{2,0}
        # W[C_{1,1}] += A_{1,2} B_{2,1}
        
        # This is not a single rank-1 tensor in the (A,B,C) space?
        # It is a rank-1 tensor in the (i,k) space for fixed j?
        # Yes, but as a 3-tensor T_{i,j,k}, it is rank 1.
        # u_i = A_{i,2} (vector in R^2)
        # v_j = delta_{j,2} (vector in R^3)
        # w_k = B_{2,k} (vector in R^2)
        # T_{i,j,k} = u_i v_j w_k.
        # So we add one column to U, V, W.
        
        U[2, 7] = 1  # A_{0,2}
        U[5, 7] = 1  # A_{1,2}
        
        V[4, 7] = 1  # B_{2,0}
        V[5, 7] = 1  # B_{2,1}
        
        W[0, 7] = 1  # C_{0,0}
        W[1, 7] = 1  # C_{0,1}
        W[2, 7] = 1  # C_{1,0}
        W[3, 7] = 1  # C_{1,1}
        
        return U, V, W

    # Now, 2x3x4 is two 2x3x2 blocks.
    # Block 1: k=0,1. B indices 0,1,2,3,4,5 (j=0,1,2).
    # Block 2: k=2,3. B indices 6,7,8,9,10,11 (j=0,1,2).
    
    # We can just concatenate the two rank-8 decompositions.
    # Total rank = 8 + 8 = 16.
    
    U1, V1, W1 = build_2x3x2_rank8()
    
    # Block 1 uses B indices 0-5.
    # Block 2 uses B indices 6-11.
    # We need to shift the V indices for Block 2.
    
    U2, V2, W2 = build_2x3x2_rank8()
    
    # Shift V2 indices by 6.
    V2_shifted = np.zeros_like(V2)
    for r in range(8):
        for idx in range(6):
            if V2[idx, r] != 0:
                V2_shifted[idx+6, r] = V2[idx, r]
                
    # Shift W2 indices by 2 (C has 4 entries, first 2 are block 1, last 2 are block 2).
    W2_shifted = np.zeros_like(W2)
    for r in range(8):
        for idx in range(4):
            if W2[idx, r] != 0:
                W2_shifted[idx+2, r] = W2[idx, r]

    # Concatenate
    U_full = np.concatenate([U1, U2], axis=1)
    V_full = np.concatenate([V1, V2_shifted], axis=1)
    W_full = np.concatenate([W1, W2_shifted], axis=1)
    
    # Verify
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    
    # Check a few entries
    # C_{0,0} = sum_j A_{0,j} B_{j,0}
    # a=0 (A_{0,0}), b=0 (B_{0,0}), c=0 (C_{0,0})
    # Should be 1.
    
    # Let's do a full check
    nm, mp, pn = 6, 12, 8
    
    def check_solution(U, V, W, r):
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    val = 0
                    for r_idx in range(r):
                        val += U[a, r_idx] * V[b, r_idx] * W[c, r_idx]
                    # Target:
                    # a = i*3+j
                    # b = j*4+k
                    # c = k*2+i
                    # If they are consistent, val should be 1.
                    i_a, j_a = divmod(a, 3)
                    j_b, k_b = divmod(b, 4)
                    k_c, i_c = divmod(c, 2)
                    
                    if i_a == i_c and j_a == j_b and k_b == k_c:
                        target = 1
                    else:
                        target = 0
                        
                    if abs(val - target) > 1e-9:
                        return False
        return True

    if check_solution(U_full, V_full, W_full, 16):
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    else:
        # Fallback to schoolbook
        rank = 24
        U = np.zeros((nm, rank), dtype=int)
        V = np.zeros((mp, rank), dtype=int)
        W = np.zeros((pn, rank), dtype=int)
        
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    U[i*3+j, col] = 1
                    V[j*4+k, col] = 1
                    W[k*2+i, col] = 1
                    col += 1
                    
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
