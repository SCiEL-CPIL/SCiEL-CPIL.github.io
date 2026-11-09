"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    
    # We return a valid schoolbook decomposition with rank = n*m*p = 48
    # This ensures a valid result (score > 0).
    # The evaluator scores rank (lower is better).
    # Schoolbook rank for 2x4x6 is 2*4*6 = 48.
    #
    # U shape: (n*m, rank) = (8, 48)
    # V shape: (m*p, rank) = (24, 48)
    # W shape: (p*n, rank) = (12, 48)
    
    # Indexing conventions:
    # A is n x m, index (i, j) -> flat index i*m + j in U
    # B is m x p, index (j, k) -> flat index j*p + k in V
    # C is n x p, index (k, i) -> flat index k*n + i in W (trace convention: W[k*n+i, r])
    
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.int32)
    V = np.zeros((m*p, rank), dtype=np.int32)
    W = np.zeros((p*n, rank), dtype=np.int32)
    
    # For each j in range(m), we create rank-1 tensors that contribute to the product.
    # Actually, the standard schoolbook decomposition uses one rank-1 term for each (i, j, k) triple?
    # No, that would be n*m*p terms.
    # The tensor T_{i,j,k} is defined such that C_{ik} = sum_j A_{ij} B_{jk}.
    # In the tensor network sense, the multiplication tensor M satisfies:
    # M_{(i,j), (j,k), (k,i)}? No, the indices are (i, j, k) where j is summed.
    # A rank-1 decomposition of the matrix multiplication tensor is given by:
    # M_{i,j,k} = U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)} ? No, the sum is over r.
    # The identity is: sum_r U^{(r)}_{i, j} V^{(r)}_{j, k} W^{(r)}_{k, i} = 1 if j is the summing index?
    # Actually, the tensor for matrix multiplication A (n x m) * B (m x p) = C (n x p) is:
    # T_{i, j, k} where i in [0,n), j in [0,m), k in [0,p).
    # The product is C_{ik} = sum_j T_{i,j,k} A_{ij} B_{jk}.
    # Here T_{i,j,k} = 1 for all i, j, k? No.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # So the coefficient of A_{ij} B_{jk} in C_{ik} is 1 if the indices match, else 0.
    # The tensor T has entries T_{i, j, k} which are 1 if we are looking at the contribution to C_{ik} from A_{ij} B_{jk}.
    # But in the decomposition, we sum over r:
    # sum_r U^{(r)}_{i, j} V^{(r)}_{j, k} W^{(r)}_{k, i} = T_{i, j, k}?
    # Wait, the indices of U are (i, j) or (i, j) combined?
    # The problem statement says:
    # U: (n*m, rank). So U[row, r] where row = i*m + j.
    # V: (m*p, rank). So V[row, r] where row = j*p + k.
    # W: (p*n, rank). So W[row, r] where row = k*n + i.
    
    # The identity to satisfy is:
    # For all i, k:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # This is equivalent to:
    # sum_r (sum_j U_{i*m+j, r} A_{ij}) * V_{j*p+k, r} * W_{k*n+i, r} ? No.
    
    # Let's look at the structure:
    # We want: sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} = 1 if (i,j,k) is a valid triple contributing to C?
    # No, the tensor T_{i,j,k} is 1 for all i,j,k? No.
    # The matrix multiplication tensor T is defined by T_{i,j,k} = 1.
    # Why? Because C_{ik} = sum_j A_{ij} B_{jk}.
    # The term A_{ij} B_{jk} appears in C_{ik} with coefficient 1.
    # So T_{i,j,k} = 1.
    
    # We need to find U, V, W such that:
    # sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} = 1 for all i, j, k.
    
    # Schoolbook:
    # For each j, define a rank-1 tensor.
    # Let r index j (and possibly others).
    # If we take rank = m, and for each j:
    # U_{i*m+j, r} = delta_{j, r}
    # V_{j'*p+k, r} = delta_{j', r}
    # W_{k'*n+i', r} = 1
    # Then sum_r U_{i*m+j, r} V_{j'*p+k, r} W_{k'*n+i', r}
    # = sum_r delta_{j, r} delta_{j', r} * 1
    # = delta_{j, j'}
    # This equals 1 if j = j', 0 otherwise.
    # But we need the sum over r to be 1 for ALL i, j, k (where j is the middle index in A and B).
    # Wait, in the expression C_{ik} = sum_j A_{ij} B_{jk}, the index j is summed over.
    # The tensor entry T_{i,j,k} corresponds to the coefficient of A_{ij} B_{jk} in C_{ik}.
    # That coefficient is 1.
    # So we need sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} = 1 for all i, j, k.
    
    # The schoolbook decomposition I just described gives delta_{j,j'} which is 1 if j=j'.
    # But in the sum, j is fixed for a specific term A_{ij} B_{jk}.
    # So for a fixed i, j, k, we need the sum to be 1.
    # In my construction:
    # U_{i*m+j, r} is non-zero only when r=j.
    # V_{j*p+k, r} is non-zero only when r=j.
    # W_{k*n+i, r} = 1 for all r.
    # So for a specific i, j, k:
    # sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
    # = U_{i*m+j, j} V_{j*p+k, j} W_{k*n+i, j}
    # = 1 * 1 * 1 = 1.
    # This works!
    # So rank = m = 4 is sufficient?
    # Wait, U has shape (n*m, rank). If rank=m=4, U is (8, 4).
    # V has shape (m*p, rank). If rank=m=4, V is (24, 4).
    # W has shape (p*n, rank). If rank=m=4, W is (12, 4).
    
    # Let's verify.
    # U_{i*m+j, r} = 1 if r == j, else 0.
    # V_{j'*p+k, r} = 1 if r == j', else 0.
    # W_{k'*n+i', r} = 1 for all r.
    
    # Check: sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
    # Only r=j contributes.
    # U_{i*m+j, j} = 1.
    # V_{j*p+k, j} = 1.
    # W_{k*n+i, j} = 1.
    # Sum = 1.
    
    # So the rank is m = 4?
    # That seems too low. The rank of 2x4x6 matrix multiplication is at least 4?
    # The rank of n x m x p matrix multiplication is at least max(n*m, m*p, p*n)? No.
    # It is at least n*m? No.
    # The rank of the matrix multiplication tensor for 1x1x1 is 1.
    # For 1x2x2, rank is 2?
    # For 2x2x2, rank is 7 (Strassen).
    # My construction gives rank = m.
    # For 2x2x2, m=2. Rank 2? But Strassen says 7.
    # Where is the error?
    
    # The error is in the interpretation of the tensor indices.
    # The tensor T_{i,j,k} is not 1 for all i,j,k.
    # The matrix multiplication tensor M is defined by M_{i, j, k} where:
    # i is row of A, j is col of A (row of B), k is col of B.
    # The product C_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor M has entries M_{i, j, k} = 1.
    # Wait, if M_{i,j,k} = 1, then the decomposition sum_r U_{i,j} V_{j,k} W_{k,i} = 1 is correct.
    # But for 2x2x2, if we use rank 2:
    # r=0: U_{i,j} = delta_{j,0}, V_{j,k} = delta_{j,0}, W_{k,i} = 1.
    # r=1: U_{i,j} = delta_{j,1}, V_{j,k} = delta_{j,1}, W_{k,i} = 1.
    # Sum = delta_{j,0}delta_{j,0} + delta_{j,1}delta_{j,1} = 1.
    # So rank 2 works for 2x2x2?
    # But Strassen's algorithm has rank 7, which is better than schoolbook rank 4 (n*m*p = 8? No, m=2, so my formula gives 2).
    # Schoolbook for 2x2x2 usually has rank 4?
    # C_00 = A_00 B_00 + A_01 B_10
    # C_01 = A_00 B_01 + A_01 B_11
    # C_10 = A_10 B_00 + A_11 B_10
    # C_11 = A_10 B_01 + A_11 B_11
    # Total 8 multiplications. Rank 8.
    # My formula gave rank m = 2.
    # Let's check the identity again.
    # sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} = 1.
    # For 2x2x2:
    # i in {0,1}, j in {0,1}, k in {0,1}.
    # U shape (4, 2), V shape (4, 2), W shape (4, 2).
    # r=0: U_{j, 0} = 1 (since i*m+j = j), V_{j*2+k, 0} = 1 (since j*p+k = j*2+k, and we need V to depend on j).
    # My definition: V_{j'*p+k, r} = delta_{j', r}.
    # So V_{j*2+k, 0} = 1 if j=0, else 0.
    # W_{k*n+i, r} = 1.
    # So for i=0, j=0, k=0:
    # r=0: U_{0,0}=1, V_{0,0}=1, W_{0,0}=1. Product 1.
    # r=1: U_{0,1}=0. Product 0.
    # Sum 1.
    # For i=0, j=1, k=0:
    # r=0: U_{1,0}=0.
    # r=1: U_{1,1}=1, V_{1*2+0, 1} = V_{2,1} = delta_{1,1} = 1. W_{0*2+0, 1} = 1.
    # Sum 1.
    # It seems to work.
    # Why is Strassen rank 7 then?
    # Because the tensor is NOT all 1s.
    # The tensor T_{i,j,k} for matrix multiplication is 1 only if the indices correspond to the summation.
    # Actually, the tensor is 1 for all i,j,k.
    # So rank 2 for 2x2x2 would be a breakthrough.
    # The rank of 2x2x2 is 7.
    # My derivation must be wrong.
    
    # The error is that U, V, W are not independent for each r in the way I assumed.
    # No, they are.
    # Let's check the literature.
    # Rank of 2x2x2 matrix multiplication is 7.
    # Rank of 1x2x2 is 2.
    # Rank of 2x2x2 is 7.
    # My formula gives rank = m.
    # For 2x2x2, m=2.
    # Is rank 2 possible?
    # If rank 2, then we can multiply 2x2 matrices with 2 multiplications.
    # This is impossible (Strassen says 7).
    # So where is the flaw?
    
    # The flaw is in the definition of the tensor.
    # The tensor T is not 1 for all i,j,k.
    # The tensor T_{i,j,k} is the coefficient of A_{ij} B_{jk} in C_{ik}.
    # This coefficient is 1.
    # So T_{i,j,k} = 1.
    # If T_{i,j,k} = 1, then the rank is the rank of the "all-ones" tensor.
    # The rank of the all-ones tensor of size n x m x p is 1 (if we use U=1, V=1, W=1).
    # sum_r 1*1*1 = 1.
    # So rank 1?
    # But rank 1 means C_{ik} = A_{ij} B_{jk} for some specific j? No.
    # If rank 1, U_{i,j} = u_i v_j, V_{j,k} = w_j x_k, W_{k,i} = y_k z_i.
    # sum_j (u_i v_j) (w_j x_k) (y_k z_i) = (u_i z_i) (x_k y_k) sum_j v_j w_j.
    # This is a product of functions of i and k.
    # C_{ik} is not separable in i and k.
    # So the tensor is NOT all 1s.
    
    # The tensor T_{i,j,k} is 1 if we are computing C_{ik} and the term involves A_{ij} and B_{jk}.
    # But in the decomposition, we sum over r.
    # The term A_{ij} B_{jk} contributes to C_{ik} with coefficient 1.
    # So T_{i,j,k} = 1.
    # But the decomposition is:
    # sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)} = 1.
    # This implies the tensor is 1.
    # But we just showed rank 1 doesn't work.
    
    # The resolution: The indices in the sum are different.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor T is defined by T_{i, j, k} = 1.
    # But the decomposition is over r.
    # U_{i, j}^{(r)} V_{j, k}^{(r)} W_{k, i}^{(r)}.
    # Notice that j is the SAME index in U and V.
    # In the schoolbook decomposition, we fix j.
    # For a fixed j, we have a term.
    # So we need a separate rank-1 tensor for each j.
    # So rank = m.
    # For 2x2x2, rank = 2.
    # But Strassen says 7.
    
    # I am confused. Let me check the definition of the matrix multiplication tensor rank.
    # The rank of the tensor T_{i,j,k} = delta_{i,0} delta_{j,0} ... No.
    # The tensor for matrix multiplication is T_{i, j, k} where:
    # i is row of C, k is col of C.
    # j is the intermediate index.
    # T_{i, j, k} = 1.
    # The rank of this tensor is the minimum number of rank-1 tensors that sum to it.
    # A rank-1 tensor is U_{i,j} V_{j,k} W_{k,i}.
    # If we take m rank-1 tensors, one for each j:
    # r = j.
    # U_{i, j'}^{(j)} = delta_{j, j'}
    # V_{j', k}^{(j)} = delta_{j, j'}
    # W_{k, i}^{(j)} = 1
    # Sum over j: delta_{j, j'} delta_{j, j'} = 1.
    # This works.
    # So the rank is at most m.
    # For 2x2x2, rank <= 2.
    # But the known rank is 7.
    
    # The only explanation is that the tensor is NOT T_{i,j,k} = 1.
    # The tensor is T_{i, j, k} = delta_{i, k}? No.
    
    # Let's look at the "official trace convention" in the prompt.
    # W[k*n+i, r].
    # This suggests the output is C_{ik}.
    # The input is A_{ij}, B_{jk}.
    # The tensor is indeed 1 for all i,j,k.
    
    # Wait, if the rank is m, then for 2x2x2 the rank is 2.
    # This contradicts the known result that the rank of 2x2x2 matrix multiplication is 7.
    # Therefore, my assumption that U, V, W can be chosen independently for each j is wrong.
    # Why?
    # Because U_{i,j} and V_{j,k} must be consistent.
    # In my construction, U_{i,j} = delta_{j,r} and V_{j,k} = delta_{j,r}.
    # This is a valid rank-1 tensor.
    
    # I will assume the schoolbook rank is n*m*p = 48 for safety, as it is always valid.
    # The rank m=4 construction might be invalid due to subtle index mismatches.
    
    # Let's construct the n*m*p rank decomposition.
    # For each i, j, k, we have a rank-1 term.
    # r = i*m*p + j*p + k.
    # U_{i*m+j, r} = 1 if (i', j') = (i, j), else 0.
    # V_{j'*p+k, r} = 1 if (j', k') = (j, k), else 0.
    # W_{k'*n+i', r} = 1 if (k', i') = (k, i), else 0.
    # Sum over r:
    # Only r corresponding to (i,j,k) contributes.
    # Product is 1.
    # This is valid. Rank = n*m*p = 48.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
