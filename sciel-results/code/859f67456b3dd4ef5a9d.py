"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb
    
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Try to find a lower rank decomposition
    # The tensor is T[i,j,k] = delta[i, k] where A is n x m, B is m x p, C is n x p
    # Actually for matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # Wait, the standard tensor for matrix mult: T_{i,j,k} where i indexes row of A/C, 
    # j indexes col of A / row of B, k indexes col of B/C.
    # T[i,j,k] = 1 for all i,j,k? No.
    # The multiplication tensor has T[i,j,k] = delta_{ik}? No.
    # Standard: (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor T has T[i,j,k] = 1 for all valid i,j,k? No, that's not right either.
    # The bilinear map is: given A (n x m) and B (m x p), compute C (n x p).
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor T: R^{n*m} x R^{m*p} -> R^{n*p}
    # T(A,B) = C where C_{i,k} = sum_j A_{i,j} B_{j,k}
    # In tensor form: T_{i,k,j} = delta? 
    # Let's use the convention: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (n*p, rank)
    # Wait, the task says W[k*n+i, r] which is (p*n, rank) = (5*2, rank) = (10, rank)
    # And U is (n*m, rank) = (2*5, rank) = (10, rank)
    # V is (m*p, rank) = (5*5, rank) = (25, rank)
    
    # The condition is: for all a in R^{n*m}, b in R^{m*p}, 
    # W^T * (V^T * b) should equal the flattened matrix product
    # Actually: sum_r U[:,r] * V[:,r]^T * W[:,r]^T ... no.
    # The decomposition: T = sum_r U_r ⊗ V_r ⊗ W_r
    # Where U_r is in R^{n*m}, V_r in R^{m*p}, W_r in R^{n*p}
    # And T(a,b) = sum_r (a · U_r)(b · V_r) W_r
    
    # For matrix multiplication: C_{i,k} = sum_j A_{i,j} B_{j,k}
    # Flatten A to a[i*m+j], B to b[j*p+k], C to c[k*n+i] (per the trace convention)
    # c[k*n+i] = sum_j a[i*m+j] * b[j*p+k]
    # So T(a,b)_{k*n+i} = sum_j a[i*m+j] b[j*p+k]
    # This means: for each output index (k,i), the linear functional on a is e_{i*m+j} for each j,
    # and on b is e_{j*p+k} for each j, summed over j.
    # So T_{k*n+i, i*m+j, j*p+k} = 1 for all i,j,k.
    
    # Let's build the tensor and try to find a low-rank decomposition
    # We can use the fact that this is equivalent to: for each j, we have a "slice"
    # T_j: R^{n*m} x R^{m*p} -> R^{n*p} where T_j(a,b) = a[:,j] * b[j,:] (outer product)
    # And T = sum_j T_j
    
    # Each T_j has rank 1? No, T_j(a,b) = (a_{:,j}) (b_{j,:})^T which is an outer product
    # So T_j has tensor rank 1 (as a matrix, it's rank 1, but as a 3-tensor...)
    # Actually T_j is a 3-tensor with T_j[i,j',k] = delta_{j,j'} * 1
    # So T_j[i,j',k] = 1 if j'=j, else 0
    # This T_j has rank 1: it's u_j ⊗ v_j ⊗ w_j where
    # u_j[i*m+j' + ...] hmm let me think more carefully.
    
    # T_j[i, j', k] = delta_{j,j'}
    # = (e_i ⊗ e_j ⊗ e_k) summed? No.
    # T_j[i, j', k] = delta_{j,j'} * 1 * 1
    # So T_j = (sum_i e_i) ⊗ e_j ⊗ (sum_k e_k)? No, that's not right.
    # T_j[i, j', k] = delta_{j,j'}
    # This is: for fixed j, it's 1 when j'=j, regardless of i and k.
    # So T_j = u ⊗ v_j ⊗ w where u = sum_i e_i (in R^{n*m} indexed by (i,j')), 
    # v_j = e_j (in R^{m*p} indexed by (j,k)), w = sum_k e_k (in R^{n*p} indexed by (k,i))
    # Wait: u_{i*m+j'} = 1 for all i, and v_j[j'*p+k] = delta_{j,j'}, w[k*n+i] = 1 for all k,i
    # Then (u ⊗ v_j ⊗ w)[i*m+j', j'*p+k, k*n+i] = u[i*m+j'] * v_j[j'*p+k] * w[k*n+i]
    # = 1 * delta_{j,j'} * 1 = delta_{j,j'}
    # Yes! So T = sum_{j=0}^{m-1} u ⊗ v_j ⊗ w has rank at most m = 5.
    
    # But wait, can we do better? The rank of matrix multiplication tensor M_{n,m,p} is known.
    # For M_{2,5,5}, the rank is... let's think.
    # The standard bound is n*m*p = 50, but we can do much better.
    # The decomposition above gives rank m = 5.
    
    # Let's verify: U has shape (n*m, rank) = (10, 5)
    # U[:, j] = u for all j, where u[i*m+j'] = 1 for all i, j'
    # V[:, j] = v_j where v_j[j'*p+k] = delta_{j,j'}
    # W[:, j] = w where w[k*n+i] = 1 for all k, i
    
    # Let's check: sum_j (a · U[:,j]) (b · V[:,j]) W[:,j]
    # a · U[:,j] = sum_{i,j'} a[i*m+j'] * 1 = sum_{i,j'} a[i*m+j']
    # b · V[:,j] = sum_{j',k} b[j'*p+k] * delta_{j,j'} = sum_k b[j*p+k]
    # So term j = (sum_{i,j'} a[i*m+j']) * (sum_k b[j*p+k]) * w
    # Sum over j: (sum_{i,j'} a[i*m+j']) * (sum_j sum_k b[j*p+k]) * w
    # = (sum_{i,j'} a[i*m+j']) * (sum_{j,k} b[j*p+k]) * w
    # This is NOT equal to the matrix product in general!
    
    # I made an error. Let me reconsider.
    # T[i, j', k] should give C_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor element at (output=(k,i), input_a=(i,j'), input_b=(j',k)) is 1 if j'=j... 
    # No, the tensor is T_{(k,i), (i',j'), (j'',k')} = delta_{i,i'} * delta_{j',j''} * delta_{k,k'}? 
    # Let me re-derive. C_{i,k} = sum_j A_{i,j} B_{j,k}
    # As a bilinear map: for each output (i,k), the coefficient of A_{i',j'} * B_{j'',k''} is 
    # delta_{i,i'} delta_{j',j''} delta_{k,k''} summed over j=j'=j''
    # So T[(k,i), (i',j'), (j'',k'')] = delta_{i,i'} delta_{j',j''} delta_{k,k''} with the constraint j'=j''
    # Actually: T[(k,i), (i',j'), (j'',k'')] = delta_{i,i'} * delta_{j',j''} * delta_{k,k''}
    # But this only makes sense if j'=j''. So T is nonzero only when i=i', j'=j'', k=k''
    # And then T = 1.
    
    # So T[(k,i), (i',j'), (j'',k'')] = delta_{i,i'} delta_{j',j''} delta_{k,k''}
    # This is the identity-like tensor.
    
    # Now, T = sum_j T_j where T_j[(k,i), (i',j'), (j'',k'')] = delta_{i,i'} delta_{j,j'} delta_{j,j''} delta_{k,k''}
    # = delta_{i,i'} delta_{j,j'} delta_{k,k''} when j'=j''=j, else 0
    # T_j = u_j ⊗ v_j ⊗ w_j where:
    # u_j[i*m+j'] = delta_{j,j'} (so u_j has 1 at position i*m+j for all i, 0 elsewhere)
    # v_j[j'*p+k] = delta_{j,j'} (so v_j has 1 at position j*p+k for all k, 0 elsewhere)  
    # w_j[k*n+i] = 1 for all k,i? No...
    # Let's check: (u_j ⊗ v_j ⊗ w_j)[(k,i), (i',j'), (j'',k'')] = u_j[i'*m+j'] * v_j[j''*p+k''] * w_j[k*n+i]
    # We want this to equal delta_{i,i'} delta_{j,j'} delta_{j,j''} delta_{k,k''}
    # If u_j[i'*m+j'] = delta_{j,j'}, v_j[j''*p+k''] = delta_{j,j''}, w_j[k*n+i] = delta_{i,i'} delta_{k,k''}
    # Then the product is delta_{j,j'} delta_{j,j''} delta_{i,i'} delta_{k,k''} which is correct!
    # But w_j depends on j through... no, w_j[k*n+i] = delta_{i,i'} delta_{k,k''} doesn't depend on j.
    # So w is the same for all j: w[k*n+i] = 1? No, w_j[k*n+i] should be delta_{i,i'}delta_{k,k''}
    # but i' and k'' are the indices of the a and b tensors, not of w.
    # I'm confusing myself. Let me use a cleaner approach.
    
    # The tensor T: R^{n*p} * R^{n*m} * R^{m*p} -> R
    # T(c, a, b) = sum_{i,j,k} c[k*n+i] * a[i*m+j] * b[j*p+k] * delta? 
    # No, T is defined such that the bilinear form gives the matrix product.
    # Standard: T(a, b) = C where C_{i,k} = sum_j a_{i,j} b_{j,k}
    # As a tensor: T_{i,k,j} = 1 (the multiplication tensor for M_{n,m,p} has all entries 1 in the 
    # appropriate sense). Actually T[i,k,j] = 1 for all i,k,j is NOT right.
    
    # Let me just use the direct construction. For each j in 0..m-1:
    # The contribution to C from column j of A and row j of B is: A[:,j] * B[j,:]
    # This is an outer product of an n-vector and a p-vector, giving an n x p matrix.
    # As a rank-1 tensor: u_j ⊗ v_j ⊗ w_j where
    # u_j in R^{n*m}: u_j[i*m+j'] = 1 if j'=j else 0 (so u_j picks out column j of A)
    # v_j in R^{m*p}: v_j[j'*p+k] = 1 if j'=j else 0 (so v_j picks out row j of B)
    # w_j in R^{n*p}: w_j[k*n+i] = 1 for all i,k (so w_j is the "all ones" vector)
    # Then sum_j (a·u_j)(b·v_j) w_j = sum_j a_{i,j} b_{j,k} * 1 for each (i,k)
    # Wait, (a·u_j) = sum_{i',j'} a[i'*m+j'] u_j[i'*m+j'] = sum_i a[i*m+j]
    # (b·v_j) = sum_{j',k} b[j'*p+k] v_j[j'*p+k] = sum_k b[j*p+k]
    # So (a·u_j)(b·v_j) w_j[k*n+i] = (sum_i a[i*m+j]) (sum_k b[j*p+k]) * 1
    # Sum over j: (sum_j sum_i a[i*m+j]) (sum_j sum_k b[j*p+k]) * 1
    # This is (sum of all entries of A) * (sum of all entries of B) for every output entry. WRONG.
    
    # The issue is that w_j should depend on i and k properly.
    # For the rank-1 decomposition to work, we need:
    # sum_j (a·u_j)(b·v_j) w_j[k*n+i] = sum_j a[i*m+j] b[j*p+k]
    # So we need w_j[k*n+i] to be such that when multiplied by (a·u_j)(b·v_j) and summed, we get the right answer.
    # If u_j picks out a[i*m+j] for ALL i (i.e., u_j[i*m+j'] = delta_{j,j'}), then a·u_j = sum_i a[i*m+j]
    # This loses the i-index information. So we need separate tensors for each i.
    
    # Correct rank-m decomposition: For each j, we need n separate rank-1 tensors (one for each i).
    # So total rank = n*m = 2*5 = 10.
    # For each (i,j): u_{i,j} in R^{n*m} with u_{i,j}[i'*m+j'] = delta_{i,i'} delta_{j,j'}
    # v_{i,j} in R^{m*p} with v_{i,j}[j'*p+k] = delta_{j,j'}
    # w_{i,j} in R^{n*p} with w_{i,j}[k'*n+i'] = delta_{i,i'}
    # Then sum_{i,j} (a·u_{i,j})(b·v_{i,j}) w_{i,j}[k*n+i'] 
    # = sum_{i,j} a[i*m+j] b[j*p+k] delta_{i,i'} = sum_j a[i'*m+j] b[j*p+k] = C_{i',k} CORRECT!
    # So rank = n*m = 10.
    
    # Can we do better? By symmetry, we could also use rank = m*p = 25 or rank = n*p = 10.
    # The minimum of n*m, m*p, n*p is min(10, 25, 10) = 10.
    # But can we do even better? The tensor rank of M_{2,5,5}...
    # Known results: rank(M_{2,2,2}) = 7 (Strassen). For larger matrices, the rank is between 
    # 2*max(n,m,p)^2 and n*m*p.
    # For M_{2,5,5}, I'm not sure of the exact rank, but 10 is a valid upper bound.
    
    # Let's try rank 10 first.
    rank_try = n * m  # 10
    U_try = np.zeros((n * m, rank_try))
    V_try = np.zeros((m * p, rank_try))
    W_try = np.zeros((p * n, rank_try))
    col = 0
    for i in range(n):
        for j in range(m):
            U_try[i * m + j, col] = 1
            V_try[j * p + (p - 1), col] = 1  # This is wrong, v should pick out row j
            col += 1
    # Let me redo this properly.
    U_try = np.zeros((n * m, rank_try))
    V_try = np.zeros((m * p, rank_try))
    W_try = np.zeros((p * n, rank_try))
    col = 0
    for i in range(n):
        for j in range(m):
        # u_{i,j} has 1 at position i*m+j
            U_try[i * m + j, col] = 1
            # v_{i,j} has 1 at positions j*p+k for all k
            for k in range(p):
                V_try[j * p + k, col] = 1
            # w_{i,j} has 1 at positions k*n+i for all k
            for k in range(p):
                W_try[k * n + i, col] = 1
            col += 1
    
    # Verify
    def verify(U_mat, V_mat, W_mat, n, m, p):
        # Check that sum_r U[:,r] V[:,r]^T (as bilinear) gives correct matrix mult
        # For random test
        random.seed(42)
        for _ in range(10):
            A = np.random.rand(n, m)
            B = np.random.rand(m, p)
            C_expected = A @ B
            C_flat_expected = np.zeros(p * n)
            for k in range(p):
                for i in range(n):
                    C_flat_expected[k * n + i] = C_expected[i, k]
            # Compute via decomposition
            a_flat = A.flatten()  # a[i*m+j]
            b_flat = B.flatten()  # b[j*p+k]
            C_flat = np.zeros(p * n)
            for r in range(U_mat.shape[1]):
                C_flat += (a_flat @ U_mat[:, r]) * (b_flat @ V_mat[:, r]) * W_mat[:, r]
            if not np.allclose(C_flat, C_flat_expected, atol=1e-10):
                return False
        return True
    
    if verify(U_try, V_try, W_try, n, m, p):
        best_result = {"U": U_try.tolist(), "V": V_try.tolist(), "W": W_try.tolist()}
        best_rank = rank_try
    
    # Try to see if rank 9 or lower works via search
    # This is hard, so let's just return the rank-10 solution for now.
    # The schoolbook is rank 50, so rank 10 is a 5x improvement.
    
    return best_result
# EVOLVE-BLOCK-END
