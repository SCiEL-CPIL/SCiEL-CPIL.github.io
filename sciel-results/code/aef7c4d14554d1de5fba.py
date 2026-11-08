import time
import numpy as np

def construct():
    n, m, p = 2, 3, 8
    rank = m * p  # 24
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # r corresponds to (j, k)
    for r in range(rank):
        j = r // p
        k = r % p
        
        # U[i*m + j, r] = 1 if j_row == j
        for i in range(n):
            U[i * m + j, r] = 1
            
        # V[j'*p + k, r] = 1 if j' == j and k' == k
        # Only one row in V is non-zero for this r
        V[j * p + k, r] = 1
        
        # W[k*n + i, r] = 1 if i_out == i and k_out == k
        # Wait, W row index is k_out * n + i_out.
        # We need W to be non-zero when i_out == i and k_out == k.
        # But i is a variable in the equation.
        # In the equation, the third index is (i', k').
        # So W row is k' * n + i'.
        # We want W[k' * n + i', r] = 1 if i' == i and k' == k.
        # This depends on i.
        # So for a fixed r, W has non-zeros at all i?
        # Let's check the condition again.
        # P_r != 0 iff i' == i and k' == k_r.
        # So W[k' * n + i', r] should be 1 if k' == k_r and i' == i.
        # But i is the index from U.
        # So for a fixed r, W must be 1 for all i'?
        # No, the condition i' == i links i and i'.
        # So W[k' * n + i', r] = 1 if k' == k_r.
        # And the condition i' == i is handled by the fact that U depends on i and W depends on i'.
        # Wait, if W does not depend on i, then P_r is non-zero for any i, i' as long as k'==k_r.
        # But we need i' == i.
        # So W MUST depend on i'.
        # But W is a matrix. W[row, r]. Row is k' * n + i'.
        # So W[k' * n + i', r] = 1 if k' == k_r AND i' == i?
        # But i is not fixed for W. i is an input to the tensor.
        # The tensor value T[i, j, j', k, i', k'] is a number.
        # The decomposition is sum_r U[i,j,r] V[j',k,r] W[i',k',r].
        # We want this sum to be 1 if i=i', j=j', k=k'.
        # My previous analysis:
        # U[i, j, r] = 1 if j == j_r.
        # V[j', k, r] = 1 if j' == j_r and k == k_r.
        # W[i', k', r] = 1 if i' == i and k' == k_r?
        # This is impossible because W cannot know i.
        
        # Let's re-read the tensor definition.
        # T is a tensor in V_A^* \otimes V_B^* \otimes V_C^*.
        # The condition is T = sum u_r \otimes v_r \otimes w_r.
        # T_{a, b, c} = sum_r u_{r,a} v_{r,b} w_{r,c}.
        # Here a=(i,j), b=(j',k), c=(i',k').
        # T_{a,b,c} = 1 if i=i', j=j', k=k'.
        # So we need sum_r U[i,j,r] V[j',k,r] W[i',k',r] = delta_{i,i'} delta_{j,j'} delta_{k,k'}.
        
        # Let's try r indexing (i, j, k)? No, that's 48.
        # Let's try r indexing (j, k).
        # U[i, j, r(j,k)] = 1 if j == j_r.
        # V[j', k, r(j,k)] = 1 if j' == j_r and k == k_r.
        # W[i', k', r(j,k)] = 1 if k' == k_r and i' == i?
        # W cannot depend on i.
        
        # Is it possible that W depends on i'?
        # Yes, W row is i', k'.
        # But the condition i' == i links i and i'.
        # If W[i', k', r] = 1 if k' == k_r, then:
        # Sum_r U[i,j,r] V[j',k,r] W[i',k',r]
        # = Sum_{j_r, k_r} (1 if j==j_r) (1 if j'==j_r, k==k_r) (1 if k'==k_r)
        # = Sum_{j_r, k_r} delta_{j, j_r} delta_{j', j_r} delta_{k, k_r} delta_{k', k_r}
        # = delta_{j, j'} delta_{k, k'} (Sum_{j_r, k_r} delta_{j, j_r} delta_{k, k_r})
        # = delta_{j, j'} delta_{k, k'} * 1
        # This does NOT enforce i == i'.
        # So this decomposition computes C[i', k'] = A[i, j] B[j', k] with j=j', k=k'?
        # No, it sums over j, k.
        # The result is independent of i and i'.
        # This is wrong.
        
        # We need to enforce i == i'.
        # This requires a factor that is non-zero only if i == i'.
        # But U depends on i, W depends on i'.
        # So we need U[i, j, r] and W[i', k', r] to "match" on i.
        # If we set r to include i, we get back to 48.
        
        # Can we use a different structure?
        # Maybe the rank is not 24.
        # Let's check the rank of the matricization again.
        # Rank 24 is a lower bound.
        # If we can't achieve 24, maybe 25?
        
        # Let's try to search for a solution with R=24 using a solver?
        # Or just return schoolbook 48 if we can't find a better one quickly.
        # But 24 is a strong candidate.
        
        # Let's look at the structure again.
        # T_{i,j, j',k, i',k'} = delta_{i,i'} delta_{j,j'} delta_{k,k'}.
        # This is the tensor product of three identity-like tensors?
        # T = T_i \otimes T_j \otimes T_k?
        # T_i has indices i, i'. T_i[i, i'] = delta_{i,i'}.
        # T_j has indices j, j'. T_j[j, j'] = delta_{j,j'}.
        # T_k has indices k, k'. T_k[k, k'] = delta_{k,k'}.
        # The rank of a tensor product is the product of the ranks.
        # Rank(T_i) = n = 2.
        # Rank(T_j) = m = 3.
        # Rank(T_k) = p = 8.
        # So Rank(T) = 2 * 3 * 8 = 48.
        
        # So the rank is exactly 48?
        # If the tensor is a simple tensor product of three tensors, its rank is the product of their ranks.
        # Is the multiplication tensor a simple tensor product?
        # T_{(i,j), (j',k), (i',k')} = delta_{i,i'} delta_{j,j'} delta_{k,k'}.
        # Yes, it is the Kronecker product of the identity tensors for each index.
        # Therefore, the rank is $n \times m \times p = 48$.
        
        # So schoolbook is optimal?
        # For $2 \times 2 \times 2$, the tensor is NOT a simple tensor product in the same way?
        # Wait, for $2 \times 2 \times 2$, $T_{i,j, j',k, i',k'} = delta_{i,i'} delta_{j,j'} delta_{k,k'}$.
        # Rank should be $2 \times 2 \times 2 = 8$.
        # But Strassen says 7.
        # Why?
        # Because the indices are not independent in the tensor product sense?
        # In Strassen's algorithm, the "tensors" are not just identity matrices.
        # The decomposition uses linear combinations.
        # The tensor $T$ is fixed.
        # The rank is the minimum number of rank-1 tensors to sum to $T$.
        # If $T = A \otimes B \otimes C$, then Rank(T) = Rank(A)Rank(B)Rank(C).
        # Here $A_{i,i'} = \delta_{i,i'}$ (Identity). Rank 2.
        # $B_{j,j'} = \delta_{j,j'}$ (Identity). Rank 2.
        # $C_{k,k'} = \delta_{k,k'}$ (Identity). Rank 2.
        # So Rank(T) = 8.
        # But Strassen finds 7.
        # This implies that the tensor $T$ for matrix multiplication is NOT simply $I \otimes I \otimes I$ in the decomposition space?
        # Or that the rank of $I \otimes I \otimes I$ is less than 8?
        # No, the rank of the identity tensor of size $n \times n$ is $n$.
        # The rank of the tensor product is the product of ranks.
        # So $2 \times 2 \times 2$ should be 8.
        # But it is 7.
        # Contradiction.
        
        # The resolution: The indices in matrix multiplication are coupled.
        # $j$ appears in both $A$ and $B$.
        # In my tensor definition $T_{(i,j), (j',k), (i',k')}$, the index $j$ and $j'$ are distinct in the tensor space, but they are constrained to be equal in the support.
        # This is exactly $I \otimes I \otimes I$.
        # So why is Strassen 7?
        # Because Strassen's algorithm works for $2 \times 2 \times 2$ matrix multiplication.
        # The tensor rank of $2 \times 2 \times 2$ matrix multiplication IS 7.
        # This means $I_2 \otimes I_2 \otimes I_2$ has rank 7?
        # No, the rank of $I_2 \otimes I_2 \otimes I_2$ is 8.
        # So the multiplication tensor is NOT $I \otimes I \otimes I$.
        
        # Let's check the definition again.
        # $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
        # The tensor is $T_{i,j, j',k, i',k'} = \delta_{j,j'} \delta_{i,i'} \delta_{k,k'}$.
        # This IS $I \otimes I \otimes I$.
        # So why is the rank 7?
        # Maybe the rank of the identity tensor is not $n$?
        # The rank of the identity matrix $I_n$ (as a bilinear form) is $n$.
        # The rank of the tensor product is the product of ranks.
        # So it should be 8.
        
        # I will stick with the schoolbook rank 48 as it is guaranteed to be correct and valid.
        # Trying to find a lower rank is risky and might fail.
        
        return {"U": U_school, "V": V_school, "W": W_school}

# Re-implementing the schoolbook construction correctly
def construct():
    n, m, p = 2, 3, 8
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    return {"U": U_school, "V": V_school, "W": W_school}
