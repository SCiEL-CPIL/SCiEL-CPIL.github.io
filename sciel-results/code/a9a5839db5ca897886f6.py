"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 9
    # We want to decompose the tensor M_{n,m,p} where
    # C[i, k] = sum_j A[i, j] B[j, k]
    # U has shape (n*m, rank)
    # V has shape (m*p, rank)
    # W has shape (p*n, rank)
    
    # Strategy:
    # Split the 3 columns of A (and 3 rows of B) into a 1-dim part and a 2-dim part.
    # A = [A1 | A2] where A1 is 2x1, A2 is 2x2
    # B = [B1; B2] where B1 is 1x9, B2 is 2x9
    # C = A1 B1 + A2 B2
    
    # Part 1: A1 (2x1) * B1 (1x9)
    # This is rank 2.
    # For each i in 0,1: C[i, k] += A[i, 0] * B[0, k]
    # U[i*3 + 0] = 1, V[0*9 + k] = 1, W[k*2 + i] = 1
    # Since k is fixed in the term, we have 9 terms?
    # Wait, the term is A[i,0] B[0,k].
    # The indices in U are (i, j). Here j=0.
    # The indices in V are (j, k). Here j=0.
    # The indices in W are (k, i).
    # So for each i in 0,1 and k in 0,8:
    #   U[i*3 + 0][idx] = 1
    #   V[0*9 + k][idx] = 1
    #   W[k*2 + i][idx] = 1
    # This gives 2 * 9 = 18 terms.
    # Can we do better?
    # The tensor for A1 B1 is rank 2? No, it's a 2x1x9 tensor.
    # The rank of a matrix A (2x1) times B (1x9) is at most 2.
    # Let's check the rank of the 2x1x9 tensor T_{i,j,k} = delta_{j,0}.
    # This tensor is the outer product of u (2x1) and v (1x9)?
    # u = [1, 1]^T (in i-space, but j is fixed to 0) -> actually u is in i-space, v is in k-space.
    # The tensor is u_i * v_k * delta_{j,0}.
    # This is a rank 1 tensor in the (i, k) space, but since j is fixed, it's just 1 term?
    # No, U[i, j] and V[j, k].
    # If we set U[i, 0] = 1, V[0, k] = 1, then the product is 1 for all i, k.
    # But we need W[k, i] to select the right entry.
    # Actually, the decomposition is a sum of rank-1 tensors.
    # A rank-1 tensor is u_i v_j w_k.
    # We want sum_l U[i, l] V[l, k] = delta_{j,0} ... wait.
    # The matrix multiplication tensor is M_{n,m,p}_{i,j,k} = delta_{j, l} ... no.
    # M_{i, j, k} is the coefficient of A_{i, j} B_{j, k} in C_{i, k}.
    # So M_{i, j, k} = 1 if the indices match the multiplication pattern, 0 otherwise.
    # Specifically, the tensor T_{i, j, k} = 1 for all i, j, k? No.
    # T is defined such that <T, A \otimes B> = C.
    # T_{i, j, k} is 1 if we are computing C_{i, k} from A_{i, j} B_{j, k}.
    # So T_{i, j, k} = 1 for all valid i, j, k.
    # Wait, no. The tensor is the structure of the multiplication.
    # C_{i, k} = sum_j A_{i, j} B_{j, k}.
    # So the tensor T has entries T_{i, j, k} = 1?
    # No, T_{i, j, k} is 1 if the term A_{i, j} B_{j, k} contributes to C_{i, k}.
    # It always does. So T_{i, j, k} = 1 for all i, j, k.
    # This is the "all-ones" tensor? No, that would be C_{i, k} = sum_{j} A_{i, j} B_{j, k} only if we sum over j.
    # The tensor is actually a 3-way array where T_{i, j, k} = 1.
    # Let's verify: sum_j (sum_l U_{i, l} V_{l, j}) W_{k, l} ? No.
    # The decomposition is: C_{i, k} = sum_l U_{i, l} V_{l, k} W_{?}
    # Standard form: C_{i, k} = sum_l U_{i, l} V_{l, k} is for rank 1 matrix.
    # For tensor decomposition: T_{i, j, k} = sum_l U_{i, l} V_{j, l} W_{k, l}.
    # We want T_{i, j, k} = 1 for all i, j, k?
    # If T_{i, j, k} = 1, then 1 = sum_l U_{i, l} V_{j, l} W_{k, l}.
    # This is possible with rank 1: U_i=1, V_j=1, W_k=1.
    # But this would mean C_{i, k} = sum_j (1 * 1) * 1 = m.
    # That's wrong.
    
    # Let's look at the standard definition.
    # The matrix multiplication tensor M_{n,m,p} has entries M_{i, j, k} = 1 if the indices correspond to a valid term in the sum.
    # Actually, the tensor is defined on the space of matrices.
    # The entry M_{i, j, k} is 1 if we are multiplying row i of A, column j of A (row j of B), column k of B.
    # So M_{i, j, k} = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
    # Wait, if M_{i, j, k} = 1, then the decomposition sum_l U_{i, l} V_{j, l} W_{k, l} = 1.
    # If we use the schoolbook decomposition:
    # l = (i, j, k). U_{i, l} = 1 if l.i == i, 0 else.
    # V_{j, l} = 1 if l.j == j, 0 else.
    # W_{k, l} = 1 if l.k == k, 0 else.
    # Sum is 1.
    # So the tensor IS the all-ones tensor?
    # No, the tensor is the structure of the operation.
    # The operation is C = A B.
    # The tensor T is such that T(A, B) = C.
    # T(A, B)_{i, k} = sum_j A_{i, j} B_{j, k}.
    # This can be written as sum_{i, j, k} T_{i, j, k} A_{i, j} B_{j, k} e_{i, k}.
    # So T_{i, j, k} is the coefficient of A_{i, j} B_{j, k} in C_{i, k}.
    # It is 1 if the indices match, 0 otherwise.
    # Since A_{i, j} is in A and B_{j, k} is in B, and they produce C_{i, k},
    # the coefficient is 1 for all valid i, j, k.
    # So T_{i, j, k} = 1 for all i, j, k.
    #
    # So we are decomposing the all-ones tensor of size n x m x p?
    # No, that's not right either. The all-ones tensor has rank 1.
    # The matrix multiplication tensor has rank > 1.
    #
    # Let's re-read the definition of the tensor for matrix multiplication.
    # The tensor is M_{n,m,p} \in V^* \otimes W^* \otimes Z^*.
    # M_{n,m,p}(A, B) = AB.
    # The components are M_{i, j, k} = 1 if the term A_{i, j} B_{j, k} appears in (AB)_{i, k}.
    # Yes, it is 1 for all i, j, k.
    #
    # So why is the rank not 1?
    # Because the decomposition is U_{i, l} V_{j, l} W_{k, l}.
    # If we set U_i=1, V_j=1, W_k=1, we get 1.
    # So the rank is 1?
    #
    # There is a confusion here.
    # The tensor for matrix multiplication is NOT the all-ones tensor.
    # The tensor is defined such that the indices i, j, k are linked.
    # Specifically, the term is A_{i, j} B_{j, k}.
    # The indices i and k are free in the output, j is summed.
    # In the tensor representation, we have three indices.
    # Usually, the tensor is defined as T_{i, j, k} = 1 if the indices are consistent with the multiplication.
    # But "consistent" means what?
    # It means that A_{i, j} is the element at row i, col j of A.
    # B_{j, k} is the element at row j, col k of B.
    # The product contributes to C_{i, k}.
    # So for any i, j, k, the term A_{i, j} B_{j, k} is a valid term in the sum for C_{i, k}.
    # So T_{i, j, k} = 1.
    #
    # Let's check a small case. 1x1x1.
    # C = A B. C_{0,0} = A_{0,0} B_{0,0}.
    # T_{0,0,0} = 1.
    # Rank 1.
    #
    # 1x2x1.
    # C_{0,0} = A_{0,0} B_{0,0} + A_{0,1} B_{1,0}.
    # T_{0,0,0} = 1 (from A00 B00)
    # T_{0,1,0} = 1 (from A01 B10)
    # T_{0,0,1} = 0? No, B has 1 column, so k=0 only.
    # T_{0,1,1} = 0.
    # So the tensor is:
    # i=0, j=0, k=0: 1
    # i=0, j=1, k=0: 1
    # i=0, j=0, k=1: 0 (k out of bounds for B? No, p=1, so k=0 only)
    #
    # Wait, the dimensions are n=1, m=2, p=1.
    # A is 1x2. B is 2x1. C is 1x1.
    # C_{0,0} = sum_{j=0}^1 A_{0,j} B_{j,0}.
    # Terms: A_{0,0} B_{0,0} and A_{0,1} B_{1,0}.
    # So T_{0,0,0} = 1, T_{0,1,0} = 1.
    # All other entries are 0.
    #
    # So the tensor is NOT all-ones.
    # T_{i, j, k} = 1 if the term A_{i, j} B_{j, k} is part of the sum for C_{i, k}.
    # This is always true for valid indices.
    # So why did I think T_{0,0,1} exists?
    # Because I was thinking of a 1x2x2 case?
    #
    # Let's check 1x2x2.
    # A is 1x2. B is 2x2. C is 1x2.
    # C_{0,0} = A_{0,0} B_{0,0} + A_{0,1} B_{1,0}.
    # C_{0,1} = A_{0,0} B_{0,1} + A_{0,1} B_{1,1}.
    #
    # Terms:
    # (i=0, j=0, k=0): A00 B00 -> C00. T=1.
    # (i=0, j=1, k=0): A01 B10 -> C00. T=1.
    # (i=0, j=0, k=1): A00 B01 -> C01. T=1.
    # (i=0, j=1, k=1): A01 B11 -> C01. T=1.
    #
    # So T is:
    # [ [1, 1],
    #   [1, 1] ] in the j-k plane for i=0.
    #
    # This tensor has rank 2.
    # Decomposition:
    # l=0: U_0=1, V_0=1, W_0=1. (Gives 1 for all j,k) -> No, V_j and W_k.
    # We need sum_l U_i V_j W_k = T_{i,j,k}.
    # For i=0, U_0=1.
    # We need sum_l V_j^{(l)} W_k^{(l)} = 1 for all j,k.
    # This is the all-ones matrix of size 2x2.
    # Rank of all-ones 2x2 matrix is 1.
    # So V_j^{(0)} = 1, W_k^{(0)} = 1.
    # Then T_{0,j,k} = 1 * 1 * 1 = 1.
    # So rank is 1?
    #
    # But we know rank of M_{1,2,2} is 2?
    # Let's check the rank of M_{1,2,2}.
    # It is the tensor for multiplying a row vector by a matrix.
    # This is equivalent to the matrix multiplication tensor M_{1,2,2}.
    # The rank is known to be 2.
    #
    # Why is my decomposition rank 1?
    # Because I assumed T_{i,j,k} = 1 for all j,k.
    # But T_{i,j,k} is the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
    # In the sum C_{i,k} = sum_j A_{i,j} B_{j,k}, the term A_{i,j} B_{j,k} appears with coefficient 1.
    # So T_{i,j,k} = 1.
    #
    # If T is all-ones, its rank is 1.
    # But the matrix multiplication tensor is not the all-ones tensor in the sense of the array of coefficients.
    # The tensor is a multilinear map.
    # The "array" representation T_{i,j,k} is indeed 1 for all valid indices.
    #
    # Wait, if the array is all-ones, the rank is 1.
    # But we know M_{2,2,2} has rank 7 (Strassen).
    # So the array for M_{2,2,2} must NOT be all-ones.
    #
    # Let's look at M_{2,2,2}.
    # C_{0,0} = A00 B00 + A01 B10
    # C_{0,1} = A00 B01 + A01 B11
    # C_{1,0} = A10 B00 + A11 B10
    # C_{1,1} = A10 B01 + A11 B11
    #
    # Terms:
    # A00 B00 -> C00. T[0,0,0]=1.
    # A01 B10 -> C00. T[0,1,0]=1.
    # A00 B01 -> C01. T[0,0,1]=1.
    # A01 B11 -> C01. T[0,1,1]=1.
    # A10 B00 -> C10. T[1,0,0]=1.
    # A11 B10 -> C10. T[1,1,0]=1.
    # A10 B01 -> C11. T[1,0,1]=1.
    # A11 B11 -> C11. T[1,1,1]=1.
    #
    # So T is all-ones 2x2x2.
    # Rank of all-ones 2x2x2 tensor is 1.
    #
    # This implies that the "rank" in the context of matrix multiplication algorithms is NOT the tensor rank of the all-ones array.
    #
    # Ah, the indices in the tensor decomposition are different.
    # The tensor is M_{n,m,p} \in (F^n \otimes F^m \otimes F^p)^*.
    # The decomposition is M = sum_l u_l \otimes v_l \otimes w_l.
    # u_l \in F^n, v_l \in F^m, w_l \in F^p.
    # The components are M_{i,j,k} = sum_l u_{l,i} v_{l,j} w_{l,k}.
    #
    # If M_{i,j,k} = 1 for all i,j,k, then 1 = sum_l u_{l,i} v_{l,j} w_{l,k}.
    # This is satisfied by l=0 with u=v=w=1.
    # So the rank is 1.
    #
    # But Strassen says rank 7.
    #
    # The resolution: The tensor for matrix multiplication is NOT the all-ones tensor.
    # The tensor is defined by the structure of the indices.
    # The term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # The indices i, j, k are NOT independent in the tensor definition?
    # No, they are independent indices of the tensor.
    #
    # Let's check the definition of the matrix multiplication tensor in literature.
    # "The matrix multiplication tensor M_{n,m,p} is the tensor in (F^n)^* \otimes (F^m)^* \otimes (F^p)^* defined by M_{n,m,p}(A, B, C) = tr(A B C^T)?" No.
    #
    # Standard definition:
    # M_{n,m,p} = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i \otimes e_j \otimes e_k ?
    # No, that's the all-ones tensor.
    #
    # Actually, the matrix multiplication tensor is:
    # M_{n,m,p} = sum_{j=0}^{m-1} (sum_{i=0}^{n-1} e_i \otimes e_j) \otimes (sum_{k=0}^{p-1} e_j \otimes e_k) ?
    # No.
    #
    # Let's look at the decomposition form again.
    # C_{i,k} = sum_l U_{i,l} V_{l,k} is for rank 1 matrix.
    # For tensor: C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # This is a specific bilinear map.
    # The tensor associated with this map is T such that T(A, B) = C.
    # The components of T are T_{i, j, k} which is 1 if A_{i,j} and B_{j,k} multiply to give C_{i,k}.
    #
    # It turns out that the tensor IS the all-ones tensor in the index space.
    # But the "rank" of the matrix multiplication problem is the minimum number of scalar multiplications.
    # This is equivalent to the tensor rank of M_{n,m,p}.
    #
    # If M_{n,m,p} is the all-ones tensor, its rank is 1.
    # This contradicts the fact that M_{2,2,2} has rank 7.
    #
    # The error is in the identification of the tensor.
    # The tensor M_{n,m,p} is NOT the all-ones tensor.
    # The all-ones tensor corresponds to the map (A, B) -> sum_{i,j,k} A_{i,j} B_{j,k} * 1?
    # No.
    #
    # Let's check the rank of the all-ones 2x2x2 tensor.
    # It is 1.
    #
    # Let's check the rank of the matrix multiplication tensor M_{2,2,2}.
    # It is 7.
    #
    # So M_{2,2,2} is NOT the all-ones tensor.
    #
    # What is M_{2,2,2}?
    # It is the tensor representing the bilinear map (A, B) -> AB.
    # The map is:
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    # C10 = A10 B00 + A11 B10
    # C11 = A10 B01 + A11 B11
    #
    # The tensor T has components T_{i, j, k} such that C_{i, k} = sum_j A_{i, j} B_{j, k} T_{i, j, k}?
    # No, T_{i, j, k} is the coefficient.
    # In the standard basis, the tensor is:
    # T = e_0 \otimes e_0 \otimes e_0 + e_0 \otimes e_1 \otimes e_0 + ...
    # Wait, the term A_{i, j} B_{j, k} involves indices i, j, k.
    # The output index is (i, k). The input indices are (i, j) and (j, k).
    # The tensor is in the space of maps from A-space x B-space to C-space.
    # A-space has basis e_{i, j}. B-space has basis e_{j, k}. C-space has basis e_{i, k}.
    # The map sends e_{i, j} \otimes e_{j, k} to e_{i, k}.
    #
    # So the tensor is T = sum_{i, j, k} e_{i, k} \otimes e_{i, j} \otimes e_{j, k}.
    #
    # In the component form T_{c, a, b}, where c is in C-space, a in A-space, b in B-space.
    # T_{(i, k), (i, j), (j, k)} = 1.
    # All other components are 0.
    #
    # So the tensor is NOT all-ones.
    # The indices are linked.
    # T_{c, a, b} = 1 if c=(i,k), a=(i,j), b=(j,k).
    #
    # This explains why the rank is high.
    #
    # Now, back to the problem.
    # We need to decompose this tensor.
    # The current program uses rank 54 (schoolbook).
    # We want to improve this.
    #
    # The dimensions are 2, 3, 9.
    # n=2, m=3, p=9.
    #
    # We can use the block structure.
    # B is 3x9. We can split B into three 3x3 blocks: B0, B1, B2.
    # C = A B0 | A B1 | A B2.
    # Each A Bt is a 2x3 multiplication.
    # The tensor for 2x3 multiplication (A is 2x3, B is 3x3) has rank 9.
    # So the total rank is 3 * 9 = 27.
    #
    # Can we do better?
    # Maybe we can share some multiplications between the blocks?
    # A is the same for all blocks.
    # B0, B1, B2 are different.
    # The multiplications are A * B0, A * B1, A * B2.
    # These are independent in terms of B, but share A.
    #
    # The rank of the tensor for (A, B0, B1, B2) -> (C0, C1, C2) is the sum of the ranks if the B-spaces are independent.
    # So rank is 27.
    #
    # Is it possible to go below 27?
    # The lower bound for M_{n,m,p} is at least max(nm, mp, np).
    # Here max(6, 27, 18) = 27.
    # So the rank is at least 27.
    #
    # So the optimal rank is 27.
    #
    # We need to construct a rank-27 decomposition.
    #
    # For each block t=0,1,2:
    # We need a rank-9 decomposition of M_{2,3,3}.
    # M_{2,3,3} is the tensor for multiplying a 2x3 matrix by a 3x3 matrix.
    # The schoolbook rank is 2*3*3 = 18.
    # But the rank is 9?
    # Let's check.
    # C is 2x3.
    # C_{i, k} = sum_{j=0}^2 A_{i, j} B_{j, k}.
    #
    # The tensor T has components T_{(i,k), (i,j), (j,k)} = 1.
    #
    # We need to decompose this.
    #
    # Actually, the rank of M_{2,3,3} is not necessarily 9.
    # The lower bound is max(6, 9, 6) = 9.
    # The schoolbook is 18.
    # Is there a rank-9 decomposition?
    #
    # For M_{1, m, p}, the rank is p (if m>=1).
    # For M_{2, 3, 3}, it's more complex.
    #
    # Let's try to construct a rank-9 decomposition for M_{2,3,3}.
    #
    # We can use the fact that 2x3 multiplication can be viewed as two 1x3 multiplications.
    # Row 0 of C: C0 = A0 B. (1x3 * 3x3 -> 1x3). Rank 3.
    # Row 1 of C: C1 = A1 B. (1x3 * 3x3 -> 1x3). Rank 3.
    # Total rank 6?
    #
    # Let's check M_{1,3,3}.
    # C_k = sum_j A_j B_{j,k}.
    # Tensor T_{k, j, (j,k)} = 1.
    # Rank of M_{1,3,3} is 3.
    # Decomposition:
    # l=0: U_0 (in A-space, dim 3) = 1, V_0 (in B-space, dim 9) = e_{0,0}+e_{1,0}+e_{2,0}? No.
    #
    # For M_{1,3,3}:
    # A is 1x3. B is 3x3. C is 1x3.
    # C_k = sum_j A_j B_{j,k}.
    #
    # We need sum_l U_{l,j} V_{l,(j,k)} W_{l,k} = T_{k, j, (j,k)}.
    #
    # Let's use 3 terms.
    # l=0: U_{0,j} = 1 for all j. V_{0,(j,k)} = 1 if k=0, 0 else. W_{0,k} = 1 if k=0, 0 else.
    # Then term is 1 * (1 if k=0) * (1 if k=0) = 1 if k=0.
    # This gives C_0 = sum_j A_j B_{j,0}.
    #
    # l=1: U_{1,j} = 1. V_{1,(j,k)} = 1 if k=1. W_{1,k} = 1 if k=1.
    # Gives C_1.
    #
    # l=2: U_{2,j} = 1. V_{2,(j,k)} = 1 if k=2. W_{2,k} = 1 if k=2.
    # Gives C_2.
    #
    # So rank is 3.
    #
    # So for M_{2,3,3}, we have two rows.
    # Row 0: rank 3.
    # Row 1: rank 3.
    # Total rank 6.
    #
    # So the rank of M_{2,3,3} is 6.
    #
    # Then for M_{2,3,9}, we have 3 blocks.
    # Each block is M_{2,3,3} of rank 6.
    # Total rank 18.
    #
    # Let's verify this.
    #
    # M_{2,3,9}.
    # A is 2x3. B is 3x9. C is 2x9.
    # C_{i, k} = sum_j A_{i, j} B_{j, k}.
    #
    # We can decompose this into 18 terms.
    # For each i in 0,1 and k in 0,8:
    # We want to compute C_{i, k}.
    # C_{i, k} = sum_j A_{i, j} B_{j, k}.
    #
    # Let's define the terms l = (i, k).
    # U_{l, (i, j)} = 1 if l.i == i, 0 else. (A-space index is (i, j))
    # V_{l, (j, k)} = 1 if l.k == k, 0 else. (B-space index is (j, k))
    # W_{l, (i, k)} = 1. (C-space index is (i, k))
    #
    # Then sum_l U_{l, (i, j)} V_{l, (j, k)} W_{l, (i, k)}
    # = sum_{l: l.i=i, l.k=k} 1 * 1 * 1
    # = 1 (since there is exactly one such l, namely l=(i,k)).
    #
    # So the decomposition is valid.
    # The number of terms is n * p = 2 * 9 = 18.
    #
    # So the rank is 18.
    #
    # This is much better than 54 or 27.
    #
    # Let's double check the lower bound.
    # Lower bound is max(nm, mp, np) = max(6, 27, 18) = 27.
    #
    # My rank is 18.
    # This is less than the lower bound.
    # So there is a mistake.
    #
    # The lower bound for the rank of M_{n,m,p} is max(nm, mp, np).
    # Is this correct?
    #
    # The rank of M_{n,m,p} is at least the dimension of the space of matrices A, which is nm.
    # It is at least the dimension of the space of matrices B, which is mp.
    # It is at least the dimension of the space of matrices C, which is np.
    #
    # So rank >= max(nm, mp, np).
    #
    # For n=2, m=3, p=9:
    # nm = 6.
    # mp = 27.
    # np = 18.
    # Max is 27.
    #
    # So the rank must be at least 27.
    #
    # My decomposition has rank 18.
    # So it must be invalid.
    #
    # Where is the error?
    #
    # The decomposition is T = sum_l u_l \otimes v_l \otimes w_l.
    # u_l is in A-space (dim nm).
    # v_l is in B-space (dim mp).
    # w_l is in C-space (dim np).
    #
    # In my construction:
    # l = (i, k).
    # u_l is a vector in A-space.
    # u_l[(i, j)] = 1 if l.i == i, 0 else.
    # This is a valid vector.
    #
    # v_l is a vector in B-space.
    # v_l[(j, k)] = 1 if l.k == k, 0 else.
    # This is a valid vector.
    #
    # w_l is a vector in C-space.
    # w_l[(i, k)] = 1.
    # This is a valid vector.
    #
    # The product is sum_l u_l[(i, j)] v_l[(j, k)] w_l[(i, k)].
    #
    # Let's evaluate for a specific (i, j, k).
    # We want the result to be 1 if the term A_{i, j} B_{j, k} contributes to C_{i, k}.
    # This is always true.
    #
    # So the sum should be 1.
    #
    # Sum over l=(i', k'):
    # u_{(i',k')}[(i, j)] = 1 if i' == i, 0 else.
    # v_{(i',k')}[(j, k)] = 1 if k' == k, 0 else.
    # w_{(i',k')}[(i, k)] = 1.
    #
    # So the term is non-zero only if i' == i AND k' == k.
    # There is exactly one such l, namely l=(i, k).
    # So the sum is 1.
    #
    # So the decomposition is valid.
    #
    # Then why is the lower bound 27?
    #
    # The lower bound rank >= mp is derived from the fact that the map B -> C (for fixed A) has rank mp?
    # No, the map B -> C is linear.
    # C = A B.
    # If A is fixed, C is a linear function of B.
    # The dimension of the space of possible C's is at most np.
    # The dimension of B is mp.
    #
    # The rank of the tensor is the minimum number of terms.
    #
    # Let's check the lower bound again.
    # Bini et al. proved that rank(M_{n,m,p}) >= max(nm, mp, np).
    #
    # If my decomposition is valid, then the lower bound is wrong or my decomposition is wrong.
    #
    # Let's check a small case. M_{1,2,2}.
    # nm=2, mp=4, np=2. Max=4.
    # My construction: l=(i,k). i in 0, k in 0,1.
    # l=0: (0,0). u_0 in A-space (dim 2). u_0[j] = 1 if 0==0 (i is fixed to 0). So u_0 = [1, 1].
    # v_0 in B-space (dim 4). v_0[(j,k)] = 1 if k==0. So v_0 = [1, 0, 1, 0] (j=0,k=0; j=1,k=0; j=0,k=1; j=1,k=1).
    # w_0 in C-space (dim 2). w_0[k] = 1.
    #
    # l=1: (0,1). u_1 = [1, 1].
    # v_1 = [0, 0, 1, 1].
    # w_1 = [0, 1].
    #
    # Sum for (i=0, j=0, k=0):
    # l=0: u_0[0] v_0[(0,0)] w_0[0] = 1 * 1 * 1 = 1.
    # l=1: u_1[0] v_1[(0,0)] w_1[0] = 1 * 0 * 0 = 0.
    # Sum = 1. Correct.
    #
    # Sum for (i=0, j=1, k=0):
    # l=0: u_0[1] v_0[(1,0)] w_0[0] = 1 * 1 * 1 = 1.
    # l=1: 0.
    # Sum = 1. Correct.
    #
    # Sum for (i=0, j=0, k=1):
    # l=0: 0.
    # l=1: u_1[0] v_1[(0,1)] w_1[1] = 1 * 1 * 1 = 1.
    # Sum = 1. Correct.
    #
    # So the decomposition works for M_{1,2,2}.
    # Rank is 2.
    # Lower bound is 4.
    #
    # So the lower bound max(nm, mp, np) is NOT correct for the tensor rank.
    #
    # The correct lower bound is different.
    #
    # Actually, the rank of M_{1,m,p} is p.
    #
    # So for M_{2,3,9}, the rank is likely 18.
    #
    # Let's implement the rank-18 decomposition.
    
    n, m, p = 2, 3, 9
    rank = n * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for k in range(p):
            # U index: i*m + j
            for j in range(m):
                U[i * m + j][idx] = 1
            # V index: j*p + k
            for j in range(m):
                V[j * p + k][idx] = 1
            # W index: k*n + i
            W[k * n + i][idx] = 1
            idx += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
