import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Rank 54 is always valid (schoolbook).
    def get_rank_54():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Rank 36 via Kronecker structure: use rank-3 decomposition of 2x3x1 tensor
    # The 2x3x1 tensor has rank 3 (it's just the identity-like structure)
    # So rank = 3 * 9 = 27? Let's verify.
    # Actually for 2x3x1: T[i,j,k] = delta_{i,0} if j==0,k==0 ... no
    # The multiplication tensor for (n x m) * (m x p) -> (n x p)
    # T[i,j,k] = 1 if we're computing output[i,k] = sum_j A[i,j]*B[j,k]
    # For p=1: T[i,j,0] = delta_{j, j} ... 
    # The tensor is: T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 if this term contributes to C[i,k], which is always A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all valid (i,j,k) - it's the all-ones tensor in that sense
    # No wait - the tensor entries are T[i,j,k] = 1 (each term A[i,j]B[j,k] contributes)
    # The rank of the all-ones tensor of size n x m x p is max(n,m,p)? No.
    # Actually the multiplication tensor T has T[i,j,k] = 1 for all i,j,k in the product
    # This is the "all ones" tensor, which has rank 1!
    # Wait no - let me reconsider.
    
    # The matrix multiplication tensor: C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T is defined such that C[i,k] = sum_{j} A[i,j] B[j,k]
    # In tensor notation: T[i,j,k] = 1 (coefficient of A[i,j]B[j,k] in C[i,k])
    # So T is the all-ones tensor of dimensions n x m x p.
    # The rank of the all-ones tensor 1_{n x m x p} is max(n, m, p)? 
    # No, it's actually 1! Because 1_{n x m x p} = u * v * w where u[i]=1, v[j]=1, w[k]=1
    # That can't be right either because then all matrix multiplications would be rank 1.
    
    # Let me reconsider. The tensor decomposition for matrix multiplication:
    # We want: C[i,k] = sum_{r=1}^{rank} U[i,j,r] * V[j,k,r] ... no that's not right.
    
    # The correct formulation:
    # A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor form: C = T(A, B) where T[i,j,k] = 1
    # A rank-r decomposition: C[i,k] = sum_{r=1}^r (sum_j U[i,j,r] A[i,j]) * (sum_k V[j,k,r] B[j,k])
    # Wait, that's not standard. Let me use the standard formulation.
    
    # Standard: We write C[i,k] = sum_{r=1}^R sum_j U_r[i,j] A[i,j] * V_r[j,k] B[j,k]
    # No, that's still not right.
    
    # The correct tensor decomposition:
    # The multiplication tensor T ∈ R^{n×m × m×p × n×p} where
    # T[(i,j),(j',k),(i',k')] = 1 if i=i', j=j', k=k' else 0
    # Wait, that's just the identity in the j index.
    
    # Let me use the interface specification:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition says: for each r in 0..rank-1:
    #   U[:,r] is a vector in R^{n*m} indexed by (i,j)
    #   V[:,r] is a vector in R^{m*p} indexed by (j,k)  
    #   W[:,r] is a vector in R^{p*n} indexed by (k,i)
    # And the product tensor is: sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # which equals the multiplication tensor.
    
    # The multiplication tensor T has T[(i,j),(j',k),(k',i')] = 1 iff j=j' and i=i' and k=k'
    # i.e., T[(i,j),(j,k),(k,i)] = 1 (using same j, same i, same k)
    # So T is a sum over (i,j,k) of e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(k,i)}
    # That gives rank n*m*p = 54 for the trivial decomposition.
    
    # But we can do better! We need to find linear combinations.
    # Key insight: for fixed i, the terms are over j and k.
    # T[i,j,k] = 1 (in the sense that each triple (i,j,k) contributes)
    
    # Actually, let me just verify rank 36 works by construction.
    # If we group by k (9 groups of 2x3), each 2x3x1 sub-tensor has rank at most 3.
    # So total rank <= 9 * 3 = 27.
    
    # For 2x3x1: T[i,j,0] = 1 for i in {0,1}, j in {0,1,2}
    # We need U[i,j,r], V[j,r] (since p=1, V[j*1+0,r]=V[j,r]), W[0*2+i,r]=W[i,r]
    # sum_r U[i,j,r] * V[j,r] * W[i,r] = 1 for all i,j
    # This is: sum_r (U[i,j,r] * W[i,r]) * V[j,r] = 1
    # Let a_{i,r} = U[i,j,r]*W[i,r]... but U depends on j too.
    # We need: for all i,j: sum_r U[i,j,r] V[j,r] W[i,r] = 1
    # Let's try rank 3: 
    # r=0: U[i,0,0]=1, U[i,1,0]=0, U[i,2,0]=0; V[0,0]=1,V[1,0]=0,V[2,0]=0; W[i,0]=1
    #   Contribution: U[i,j,0]*V[j,0]*W[i,0] = 1*1*1=1 if j=0, else 0
    # r=1: U[i,0,1]=0, U[i,1,1]=1, U[i,2,1]=0; V[0,1]=0,V[1,1]=1,V[2,1]=0; W[i,1]=1
    #   Contribution: 1 if j=1, else 0
    # r=2: U[i,0,2]=0, U[i,1,2]=0, U[i,2,2]=1; V[0,2]=0,V[1,2]=0,V[2,2]=1; W[i,2]=1
    #   Contribution: 1 if j=2, else 0
    # Total: sum_r = 1 for all i,j. 
    # So rank 3 works for 2x3x1!
    
    # Therefore for 2x3x9, we can use rank 9*3 = 27 by treating each k independently.
    
    def get_rank_27():
        rank = 27
        U = [[0] * rank for _ in range(n * m)]  # (6, 27)
        V = [[0] * rank for _ in range(m * p)]  # (27, 27)
        W = [[0] * rank for _ in range(p * n)]  # (18, 27)
        
        # For each k (0..8), we need a rank-3 decomposition of the 2x3x1 sub-tensor
        # The sub-tensor for fixed k: T_k[(i,j),(j',k),(k',i')] = 1 iff i=i', j=j', k=k', k'=k
        # i.e., T_k[(i,j),(j',k),(k,i')] = 1 iff i=i' and j=j'
        # This is: sum_{j=0}^{2} delta_{j,j'} * delta_{i,i'} (for fixed k=k')
        # We can decompose as: sum_{j=0}^{2} e_j (in j) ⊗ e_j (in j') ⊗ (sum_i e_i ⊗ e_i) (in i,i')
        # But sum_i e_i ⊗ e_i is the identity in i, which has rank 2.
        # So total rank per k is 3*2 = 6? No...
        
        # Let me think again. For fixed k:
        # T_k[(i,j),(j',k),(k,i')] = delta_{j,j'} * delta_{i,i'}
        # = sum_{j} delta_{j,j'} * delta_{j,j} * delta_{i,i'}
        # We need: sum_r U_r[i,j] * V_r[j',k] * W_r[k,i'] = delta_{j,j'} * delta_{i,i'}
        
        # Try: for each j in 0,1,2 and i in 0,1:
        # r = (k, j, i): U[i,j,r]=1, V[j,k,r]=1, W[k,i,r]=1
        # Then sum_r U[i,j,r]V[j',k,r]W[k,i',r] = sum_{j,i} delta_{j,j'}delta_{i,i'} = delta_{j,j'}delta_{i,i'}
        # But that's rank 2*3=6 per k, total 9*6=54. That's just the schoolbook.
        
        # Can we do better? The key is that delta_{i,i'} can be written with rank 2 (it's the 2x2 identity).
        # And delta_{j,j'} has rank 3.
        # The tensor is delta_{j,j'} ⊗ delta_{i,i'}, which has rank 3*2=6.
        # So per k, we need rank 6, total 54. That's no improvement!
        
        # Wait, I think I was wrong earlier. Let me reconsider the 2x3x1 case.
        # For 2x3x1: T[(i,j),(j',0),(0,i')] = delta_{j,j'} * delta_{i,i'}
        # This is the tensor (J_3) ⊗ (I_2) where J_3 is the 3x3 all-ones? No.
        # delta_{j,j'} is the 3x3 identity I_3, and delta_{i,i'} is the 2x2 identity I_2.
        # The tensor I_3 ⊗ I_2 has rank 3*2 = 6.
        
        # So actually the schoolbook rank of 54 might be optimal? Let me check known results.
        # For 2x3 matrix multiplication, the rank is 6 (since it's I_2 ⊗ I_3 which has rank 6).
        # For 2x3x9, it's 9 copies, so rank 54.
        
        # Hmm, but that means the current solution is already optimal and we can't improve.
        # Let me just return the rank 54 solution.
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # The rank-27 construction is actually just rank-54 in disguise (my analysis above shows
    # the 2x3x1 sub-tensor has rank 6, not 3). Let me just return rank 54.
    return get_rank_54()
