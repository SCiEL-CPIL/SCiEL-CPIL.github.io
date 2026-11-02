import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 6
    
    # We are looking for a decomposition of the matrix multiplication tensor M_{2,2,6}
    # T = sum_{k=0}^{rank-1} U_k \otimes V_k \otimes W_k
    # where U_k is a 2x2 matrix, V_k is a 2x6 matrix, W_k is a 6x2 matrix.
    # The condition is: sum_k U_k * (V_k * W_k) = M_{2,2,6}
    # Equivalently, for all i,j,l: sum_k U_k[i,j] V_k[j,l] W_k[l,i] = 1 if (i,j,l) is in the support, else 0.
    # Actually, the tensor M_{2,2,6} has entries M_{i,j,l} = 1 if i in {0,1}, j in {0,1}, l in {0,5} and the multiplication is standard.
    # Wait, the tensor for matrix multiplication A(2x2) * B(2x6) -> C(2x6)
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor T_{i,j,k} is 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # So T_{i,j,k} = 1 for all i,j,k? No.
    # The tensor is T_{i,j,k} = 1 if the indices match the multiplication.
    # Specifically, the multiplication tensor is sum_{i,j,k} e_i \otimes e_j \otimes e_k * delta_{i,k}?? No.
    # A_{i,j} B_{j,k} -> C_{i,k}.
    # So the tensor is sum_{i,j,k} (e_i \otimes e_j \otimes e_k) * 1.
    # Wait, C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So the tensor is T = sum_{i,j,k} e_i \otimes e_j \otimes e_k.
    # This is the "all ones" tensor? No.
    # Let's check the definition.
    # The matrix multiplication tensor M_{n,m,p} is defined such that M_{n,m,p}(A,B) = A B.
    # In coordinates: (A B)_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So the tensor is T = sum_{i,j,k} e_i \otimes e_j \otimes e_k.
    # This tensor has rank at least max(n*m, m*p, p*n) = max(4, 12, 12) = 12.
    # But we know rank of 2x2x2 is 7.
    # For 2x2x6, we can view it as 3 blocks of 2x2x2.
    # B is 2x6, so B = [B1, B2, B3] where B1, B2, B3 are 2x2.
    # A B = [A B1, A B2, A B3].
    # The tensor for A B1 is M_{2,2,2}.
    # The tensor for A B2 is M_{2,2,2}.
    # The tensor for A B3 is M_{2,2,2}.
    # These tensors act on disjoint slices of the output (columns 0-1, 2-3, 4-5).
    # And they act on disjoint slices of the input B (columns 0-1, 2-3, 4-5).
    # They all act on the same input A.
    # So the total tensor is the direct sum of three M_{2,2,2} tensors, but sharing the A-space.
    # Let T_total = sum_{t=0}^2 T_t.
    # T_t = sum_{i,j,k=0}^1 e_i \otimes (e_j \otimes e_{t*2+k}) \otimes (e_{t*2+k} \otimes e_i) ?
    # Wait, the output index is k' = t*2+k.
    # The tensor entry is T_{i, j, k'} where k' is the column index of B and C.
    # T_{i, j, k'} = 1 if k' < 6 and j is the row index of B.
    # Yes, T_{i, j, k'} = 1 for all i, j, k'.
    # So the tensor is just the all-ones tensor of size 2x2x6?
    # No!
    # (A B)_{i, k'} = sum_j A_{i,j} B_{j,k'}.
    # The term A_{i,j} B_{j,k'} appears in the sum.
    # So the tensor is indeed sum_{i,j,k'} e_i \otimes e_j \otimes e_{k'}.
    # This is the tensor with all entries equal to 1.
    # Let's verify.
    # If T = sum_{i,j,k} e_i \otimes e_j \otimes e_k, then T(A,B) = sum_{i,j,k} A_{i,j} B_{j,k} e_i \otimes e_k.
    # The coefficient of e_i \otimes e_k is sum_j A_{i,j} B_{j,k}.
    # This is exactly (A B)_{i,k}.
    # So the tensor is the all-ones tensor of dimensions 2x2x6.
    
    # What is the rank of the all-ones tensor of size n x m x p?
    # It is the rank of the matrix multiplication tensor? No.
    # The matrix multiplication tensor M_{n,m,p} is NOT the all-ones tensor.
    # M_{n,m,p} is defined by M_{n,m,p}(A,B) = AB.
    # The coordinates of M_{n,m,p} are 1 if the indices correspond to a valid multiplication term.
    # In M_{n,m,p}, the indices are (i, j, k) where i is row of A, j is col of A / row of B, k is col of B.
    # The term is A_{i,j} B_{j,k}.
    # So the tensor entry at (i, j, k) is 1.
    # This is true for ALL i, j, k.
    # So M_{n,m,p} IS the all-ones tensor!
    
    # Wait, is the rank of the all-ones tensor of size 2x2x2 equal to 7?
    # The all-ones tensor of size 2x2x2 is J_2 \otimes J_2 \otimes J_2? No.
    # The all-ones tensor has T_{ijk} = 1 for all i,j,k.
    # The rank of the all-ones tensor of size 2x2x2 is 1?
    # T = e_0+e_1 \otimes e_0+e_1 \otimes e_0+e_1.
    # This is a rank 1 tensor!
    # But the rank of 2x2 matrix multiplication is 7.
    # So M_{2,2,2} is NOT the all-ones tensor.
    
    # Where is my mistake?
    # M_{n,m,p}(A,B) = AB.
    # (AB)_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So the tensor is sum_{i,j,k} A_{i,j} B_{j,k} e_i \otimes e_k.
    # The coefficient of e_i \otimes e_k in the output is sum_j A_{i,j} B_{j,k}.
    # The tensor T is such that T(A,B) = sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} e_i \otimes e_k.
    # Comparing coefficients:
    # T_{i,j,k} e_i \otimes e_k must match the contribution to (AB)_{i,k}.
    # The contribution of A_{i,j} B_{j,k} to (AB)_{i,k} is 1.
    # So T_{i,j,k} = 1.
    # This implies M_{n,m,p} is the all-ones tensor.
    
    # But this contradicts the known rank of 2x2x2 being 7.
    # Let's re-read the definition of tensor rank for matrix multiplication.
    # The matrix multiplication tensor M_{n,m,p} is an element of V^* \otimes W^* \otimes Z.
    # It is defined by M_{n,m,p} = sum_{i,j,k} e_i^* \otimes e_j^* \otimes e_k \cdot \delta_{...}
    # No, it's defined by the bilinear map.
    # The tensor is T = sum_{i,j,k} e_i \otimes e_j \otimes e_k.
    # If this is the case, the rank is 1.
    # But Strassen's algorithm gives rank 7 for 2x2x2.
    # Strassen's algorithm computes AB with 7 multiplications.
    # This means the tensor can be written as a sum of 7 rank-1 tensors.
    # If the tensor were rank 1, we could do it with 1 multiplication.
    # So the tensor CANNOT be the all-ones tensor.
    
    # Let's look at the indices again.
    # A is n x m. B is m x p. C is n x p.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The tensor T lives in (R^n \otimes R^m \otimes R^p)^*?
    # Or R^n \otimes R^m \otimes R^p?
    # Usually, the multiplication tensor is in V^* \otimes W^* \otimes Z.
    # Let's use the standard basis.
    # T = sum_{i,j,k} e_i \otimes e_j \otimes e_k.
    # If we contract T with A (in V^*) and B (in W^*), we get a vector in Z.
    # <T, A \otimes B> = sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} e_k.
    # If T_{i,j,k} = 1, then <T, A \otimes B> = sum_{i,j,k} A_{i,j} B_{j,k} e_k.
    # The coefficient of e_k is sum_{i,j} A_{i,j} B_{j,k}.
    # This is NOT (AB)_{i,k}. It is the sum over i as well!
    # So T = sum_{i,j,k} e_i \otimes e_j \otimes e_k is NOT the multiplication tensor.
    
    # The multiplication tensor is T = sum_{i,j,k} e_i \otimes e_j \otimes e_k \cdot \delta_{i,k}? No.
    # We want the coefficient of e_i \otimes e_k in the output to be sum_j A_{i,j} B_{j,k}.
    # But the output is a matrix C, which is a vector in R^{n \times p}.
    # So the tensor should map A \otimes B to C.
    # C = sum_{i,k} C_{i,k} e_i \otimes e_k.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So C = sum_{i,k} (sum_j A_{i,j} B_{j,k}) e_i \otimes e_k.
    # C = sum_{i,j,k} A_{i,j} B_{j,k} e_i \otimes e_k.
    # So the tensor T must satisfy:
    # T(A,B) = sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} e_i \otimes e_k = sum_{i,j,k} A_{i,j} B_{j,k} e_i \otimes e_k.
    # This implies T_{i,j,k} = 1 for all i,j,k.
    # But wait, the output basis is e_i \otimes e_k.
    # The input basis is e_i \otimes e_j for A and e_j \otimes e_k for B?
    # No, A is a matrix, so it's a vector in R^{n \times m}. Basis is e_i \otimes e_j.
    # B is a matrix, so it's a vector in R^{m \times p}. Basis is e_j \otimes e_k.
    # So A = sum_{i,j} A_{i,j} (e_i \otimes e_j).
    # B = sum_{j,k} B_{j,k} (e_j \otimes e_k).
    # The tensor T is in (R^{n \times m})^* \otimes (R^{m \times p})^* \otimes (R^{n \times p}).
    # T = sum_{i,j,k} e_{i,j}^* \otimes e_{j,k}^* \otimes e_{i,k}.
    # Note the indices: i, j, k.
    # The term is e_{i,j}^* \otimes e_{j,k}^* \otimes e_{i,k}.
    # This is NOT a rank 1 tensor. A rank 1 tensor would be u \otimes v \otimes w.
    # Here, the third index (i,k) depends on the first (i) and second (k) indices.
    # So T = sum_{i,j,k} e_{i,j}^* \otimes e_{j,k}^* \otimes e_{i,k}.
    # This is the correct tensor.
    
    # Now, what is the rank of this tensor?
    # For n=2, m=2, p=2:
    # T = sum_{i,j,k \in \{0,1\}} e_{i,j}^* \otimes e_{j,k}^* \otimes e_{i,k}.
    # This is the 2x2x2 matrix multiplication tensor.
    # Its rank is 7.
    
    # For n=2, m=2, p=6:
    # T = sum_{i \in \{0,1\}, j \in \{0,1\}, k \in \{0,\dots,5\}} e_{i,j}^* \otimes e_{j,k}^* \otimes e_{i,k}.
    # We can split the sum over k into three groups: k=0,1; k=2,3; k=4,5.
    # T = T_0 + T_1 + T_2.
    # T_t = sum_{i,j \in \{0,1\}, k' \in \{0,1\}} e_{i,j}^* \otimes e_{j, t*2+k'}^* \otimes e_{i, t*2+k'}.
    # Let k = t*2+k'.
    # T_t = sum_{i,j,k'} e_{i,j}^* \otimes e_{j, t*2+k'}^* \otimes e_{i, t*2+k'}.
    # This looks like the 2x2x2 tensor, but the middle index is shifted.
    # Specifically, the middle vector space is R^{2 \times 6}.
    # The basis is e_{j, k}.
    # In T_t, the middle index is j, t*2+k'.
    # So T_t acts on the subspace of B corresponding to columns 2t and 2t+1.
    # And it produces the subspace of C corresponding to columns 2t and 2t+1.
    # The A-space is the same for all t.
    # So T = T_0 + T_1 + T_2.
    # T_0, T_1, T_2 are "disjoint" in the B and C spaces, but share the A-space.
    # Can we combine them?
    # A decomposition of T is a sum of rank-1 tensors u \otimes v \otimes w.
    # u is in (R^{2x2})^*, v is in (R^{2x6})^*, w is in R^{2x6}.
    # If we have a decomposition for T_0 of rank 7, it uses u_l \otimes v_l \otimes w_l.
    # v_l is non-zero only in the first 2 columns.
    # w_l is non-zero only in the first 2 columns.
    # Similarly for T_1 and T_2.
    # If we just take the union of the 3 decompositions, we get rank 21.
    # Can we do better?
    # Suppose we have a rank 1 tensor u \otimes v \otimes w.
    # u = sum_{i,j} u_{i,j} e_{i,j}^*.
    # v = sum_{j,k} v_{j,k} e_{j,k}^*.
    # w = sum_{i,k} w_{i,k} e_{i,k}.
    # The contribution to T is u \otimes v \otimes w.
    # The coefficient of e_{i,j}^* \otimes e_{j',k'}^* \otimes e_{i',k} in u \otimes v \otimes w is u_{i,j} v_{j',k'} w_{i',k}.
    # For this to match T, we need:
    # u_{i,j} v_{j,k} w_{i,k} = 1 if (i,j,k) is in the support, else 0.
    # But T has support for ALL i,j,k.
    # So we need u_{i,j} v_{j,k} w_{i,k} = 1 for all i,j,k.
    # This implies u_{i,j} = 1, v_{j,k} = 1, w_{i,k} = 1.
    # But this is only 1 term.
    # Wait, this implies the tensor is rank 1?
    # No, because the indices must match.
    # In T, the term is e_{i,j}^* \otimes e_{j,k}^* \otimes e_{i,k}.
    # The middle index j in the first factor MUST equal the middle index j in the second factor.
    # In u \otimes v \otimes w, the factors are independent.
    # u_{i,j} v_{j',k'} w_{i',k}.
    # We want this to be non-zero only if j=j' and i=i' and k=k'.
    # So u_{i,j} v_{j,k} w_{i,k} = 1.
    # And u_{i,j} v_{j',k} w_{i,k} = 0 if j != j'.
    # This is impossible for a single rank 1 term unless the dimensions are 1.
    # So the rank is > 1.
    
    # The rank of M_{2,2,6} is at least the rank of M_{2,2,2} which is 7.
    # It is at most 3 * 7 = 21.
    # Is it possible to get rank < 21?
    # This is an open question in general, but for small sizes it is known.
    # For 2x2x6, the rank is likely 21.
    # However, there might be a "shared" structure.
    # But since the B and C blocks are disjoint, any rank 1 term can only contribute to one block?
    # No, a rank 1 term u \otimes v \otimes w can have v and w with support in multiple blocks.
    # But if v has support in block 0 and block 1, and w has support in block 0 and block 1,
    # then u \otimes v \otimes w will have cross terms.
    # Specifically, if v = v_0 + v_1 and w = w_0 + w_1,
    # then u \otimes v \otimes w = u \otimes v_0 \otimes w_0 + u \otimes v_0 \otimes w_1 + u \otimes v_1 \otimes w_0 + u \otimes v_1 \otimes w_1.
    # The cross terms u \otimes v_0 \otimes w_1 will have non-zero entries where j is in block 0 and k is in block 1.
    # But in T, the entry (i, j, k) is non-zero only if j and k are in the SAME block?
    # No, in T, the entry is e_{i,j}^* \otimes e_{j,k}^* \otimes e_{i,k}.
    # The index j is the column of A and row of B.
    # The index k is the column of B.
    # There is NO constraint that j and k are in the same block.
    # j is always 0 or 1.
    # k is 0 to 5.
    # So the term is e_{i,j}^* \otimes e_{j,k}^* \otimes e_{i,k}.
    # This term is non-zero for ALL i, j, k.
    # So the cross terms are actually REQUIRED.
    # Wait, if I use a decomposition for block 0, it only produces terms where k is in block 0.
    # But the tensor T has terms where k is in block 0, 1, or 2.
    # So I need to cover all k.
    # If I use 7 terms for block 0, they cover k=0,1.
    # If I use 7 terms for block 1, they cover k=2,3.
    # If I use 7 terms for block 2, they cover k=4,5.
    # Total 21 terms.
    # Can I use fewer?
    # Suppose I use a term that covers k=0 and k=2.
    # Then v must have support at (j,0) and (j,2).
    # And w must have support at (i,0) and (i,2).
    # Then the term contributes to (i,j,0) and (i,j,2) and (i,j,0) with w(i,2)??
    # No, the term is u_{i,j} v_{j,k} w_{i,k}.
    # If v has support at (j,0) and (j,2), and w has support at (i,0) and (i,2),
    # then the term contributes to (i,j,0) with value u_{i,j} v_{j,0} w_{i,0}.
    # And to (i,j,2) with value u_{i,j} v_{j,2} w_{i,2}.
    # And to (i,j,0) with value u_{i,j} v_{j,2} w_{i,0}?? No, v_{j,2} is paired with k=2.
    # The index k in v_{j,k} and w_{i,k} must be the same.
    # So if v has support at k=0 and k=2, and w has support at k=0 and k=2,
    # then the term contributes to k=0 and k=2.
    # It does NOT contribute to cross terms like k=0 in v and k=2 in w.
    # Because the index k is summed over? No, it's a tensor product.
    # u \otimes v \otimes w = sum_{i,j,k} u_{i,j} v_{j,k} w_{i,k} e_{i,j}^* \otimes e_{j,k}^* \otimes e_{i,k}.
    # So the index k is the same in v and w.
    # So a single rank 1 term can cover multiple k values, as long as v and w have support at those k values.
    # So I could potentially cover k=0 and k=2 with the same set of u, v, w terms?
    # If I take the 7 Strassen terms for block 0, they have v and w with support only at k=0,1.
    # If I take the 7 Strassen terms for block 1, they have v and w with support only at k=2,3.
    # If I combine them, I get 14 terms.
    # Can I merge them?
    # The u terms are the same (Strassen coefficients for 2x2).
    # The v terms are different (shifted).
    # The w terms are different (shifted).
    # So the rank 1 terms are distinct.
    # So the rank is 21.
    
    # I will stick with rank 21.

    rank = 21
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for 2x2x2
    S_A = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, 1], # P2
        [1, 0, 0, 0], # P3
        [0, 0, 0, 1], # P4
        [1, 1, 0, 0], # P5
        [-1, 0, 1, 0],# P6
        [0, 1, 0, -1] # P7
    ]
    S_B = [
        [1, 0, 0, 1], # P1
        [1, 0, 0, 0], # P2
        [0, 1, 0, -1],# P3
        [-1, 0, 1, 0],# P4
        [0, 0, 0, 1], # P5
        [1, 1, 0, 0], # P6
        [0, 0, 1, 1]  # P7
    ]
    
    # S_C is the matrix for C = P1 ... P7
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # W index is k*n + i.
    # k is the column of C (0..5). i is the row of C (0..1).
    # For block t, k = t*2 + k_local.
    
    S_C_vals = []
    # P1
    s = [[0]*2 for _ in range(2)]
    s[0][0] += 1; s[1][1] += 1
    S_C_vals.append(s)
    # P2
    s = [[0]*2 for _ in range(2)]
    s[1][0] += 1; s[1][1] -= 1
    S_C_vals.append(s)
    # P3
    s = [[0]*2 for _ in range(2)]
    s[0][1] += 1; s[1][1] += 1
    S_C_vals.append(s)
    # P4
    s = [[0]*2 for _ in range(2)]
    s[0][0] += 1; s[1][0] += 1
    S_C_vals.append(s)
    # P5
    s = [[0]*2 for _ in range(2)]
    s[0][0] -= 1; s[0][1] += 1
    S_C_vals.append(s)
    # P6
    s = [[0]*2 for _ in range(2)]
    s[1][1] += 1
    S_C_vals.append(s)
    # P7
    s = [[0]*2 for _ in range(2)]
    s[0][0] += 1
    S_C_vals.append(s)
    
    for t in range(3):
        for l in range(7):
            k_global = t * 7 + l
            
            # Update U
            for i in range(n):
                for j in range(m):
                    U[i * m + j, k_global] = S_A[l][i * m + j]
            
            # Update V
            for j in range(m):
                for k_local in range(2):
                    k_global_b = t * 2 + k_local
                    V[j * p + k_global_b, k_global] = S_B[l][j * 2 + k_local]
            
            # Update W
            for i in range(n):
                for k_local in range(2):
                    k_global_c = t * 2 + k_local
                    W[k_global_c * n + i, k_global] = S_C_vals[l][i][k_local]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
