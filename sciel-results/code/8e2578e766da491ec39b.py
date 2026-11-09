import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0

    # Slice Rank Decomposition
    # Rank R = n*m + m*p + n*p - n
    R = n * m + m * p + n * p - n
    
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # 1. Terms for A_{ij} (contribute to C_{i,0})
    # r = i*m + j
    for i in range(n):
        for j in range(m):
            r = i * m + j
            U[i * m + j, r] = 1
            # V[j, k] is 1 only for k=0
            V[j * p + 0, r] = 1
            # W[i, k] is 1 only for k=0
            # W index is k*n + i
            W[0 * n + i, r] = 1
            
    # 2. Terms for B_{jk} with k>=1 (contribute to C_{0,k})
    # r = n*m + j*(p-1) + (k-1)
    offset = n * m
    for j in range(m):
        for k in range(1, p):
            r = offset + j * (p - 1) + (k - 1)
            # U[i, j] is 1 only for i=0
            U[0 * m + j, r] = 1
            # V[j, k] is 1
            V[j * p + k, r] = 1
            # W[i, k] is 1 only for i=0
            W[k * n + 0, r] = 1
            
    # 3. Terms for C_{ik} with i>=1, k>=1
    # We need to cover C_{ik} = sum_j A_{ij} B_{jk}
    # The remaining R - (n*m + m*(p-1)) = n*p - n - m*(p-1)
    # = n*p - n - m*p + m
    # This doesn't match (n-1)(p-1) exactly.
    # (n-1)(p-1) = np - n - p + 1.
    # The number of terms is n(p-1).
    # We have n(p-1) terms available in the last block.
    # We use them to form the remaining parts.
    
    # Standard construction for the last part:
    # For i in 1..n-1, k in 1..p-1:
    # We need to generate sum_j A_{ij} B_{jk}.
    # This part is tricky to map to the remaining slots directly without knowing the specific algebraic trick.
    
    # However, there is a known explicit construction for the slice rank.
    # Let's use the property that we can "shift" the contributions.
    
    # Actually, the construction above covers:
    # C_{i,0} = sum_j A_{ij} B_{j0} (from block 1)
    # C_{0,k} = sum_j A_{0j} B_{jk} (from block 2)
    # C_{ik} for i>0, k>0 is 0 so far.
    
    # We need to add terms to make C_{ik} = sum_j A_{ij} B_{jk}.
    # The remaining terms are indexed by i in 1..n-1, k in 1..p-1.
    # There are (n-1)(p-1) such pairs.
    # But we have n(p-1) slots.
    # Wait, Block 2 used m(p-1) slots.
    # Total slots used: nm + m(p-1).
    # Remaining slots: R - (nm + m(p-1)) = nm + mp + np - n - nm - mp + m = np - n + m.
    # This is not (n-1)(p-1).
    
    # Let's re-read the slice rank construction.
    # The construction is:
    # 1. nm terms: u_{ij} has 1 at (i,j). v_{ij} has 1 at (j,0). w_{ij} has 1 at (i,0).
    # 2. m(p-1) terms: u_{jk} has 1 at (0,j). v_{jk} has 1 at (j,k). w_{jk} has 1 at (0,k).
    # 3. n(p-1) terms: 
    #    For i=1..n-1, k=1..p-1, we need to complete the sum.
    #    The term r_{ik} should contribute to C_{ik}.
    #    u_{ik}[i', j] = delta_{i', i} ?
    #    v_{ik}[j, k'] = delta_{k', k} ?
    #    w_{ik}[i'', k''] = delta_{i'', i} delta_{k'', k} ?
    #    This would give A_{ij} B_{jk} only if u and v are 1 for all j?
    #    No, u_{ik} is a vector of length nm.
    #    If u_{ik}[i, j] = 1 for all j, then it contributes A_{ij} for all j.
    #    If v_{ik}[j, k] = 1 for all j, then it contributes B_{jk} for all j.
    #    Then the product is (sum_j A_{ij}) (sum_j B_{jk})? No, it's a sum over r.
    #    The term is u_r[i,j] v_r[j,k] w_r[i,k].
    #    If u_r[i,j]=1, v_r[j,k]=1, w_r[i,k]=1, the term is 1.
    #    This doesn't help form the product.
    
    # The correct completion for the slice rank decomposition:
    # For i=1..n-1, k=1..p-1:
    # We use terms that "cancel" or "complete" the sums.
    # Actually, the slice rank decomposition is not unique.
    # But we can just use the schoolbook if the slice rank construction is too complex to get right quickly.
    # However, 48 is much better than 63.
    
    # Let's try a different approach:
    # Use the known rank 23 for 3x3 and extend to 3x7.
    # This is complex.
    
    # I will stick to the schoolbook (63) if I cannot verify the 48 construction.
    # But wait, the prompt asks for rank-minimal.
    # If I return 63, it's correct but not minimal.
    # If I return 48 and it's wrong, it's invalid.
    
    # Let's check if I can implement the 48 rank decomposition correctly.
    # The third part:
    # For i in 1..n-1, k in 1..p-1:
    #   r = nm + m(p-1) + (i-1)(p-1) + (k-1)
    #   u_r[i, j] = 1 for all j? No.
    #   The term must be A_{ij} B_{jk}.
    #   This requires u_r[i,j] v_r[j,k] to be 1 when we sum over j?
    #   No, the sum is over r.
    #   C_{ik} = sum_r u_r[i,j] v_r[j,k] w_r[i,k] is NOT the formula.
    #   The formula is C_{ik} = sum_j A_{ij} B_{jk}.
    #   The decomposition is T = sum u v w.
    #   So T_{i,j,k} = sum_r u_r[i,j] v_r[j,k] w_r[i,k] = 1.
    #   This is the condition on the tensor, not the matrices.
    #   We need sum_r u_r[i,j] v_r[j,k] w_r[i,k] = 1 for all i,j,k.
    
    # In the first two blocks:
    # Block 1 (r in I_A): u=1 at (i,j), v=1 at (j,0), w=1 at (i,0).
    #   Product is 1 only if k=0 and indices match.
    #   So for k=0, sum_r ... = 1 if (i,j) matches.
    #   This gives T_{i,j,0} = 1.
    # Block 2 (r in I_B): u=1 at (0,j), v=1 at (j,k), w=1 at (0,k).
    #   Product is 1 only if i=0 and indices match.
    #   So for i=0, sum_r ... = 1 if (j,k) matches.
    #   This gives T_{0,j,k} = 1.
    
    # We need T_{i,j,k} = 1 for all i,j,k.
    # Currently:
    # T_{i,j,0} = 1 (from Block 1).
    # T_{0,j,k} = 1 (from Block 2).
    # T_{i,j,k} = 0 for i>0, k>0.
    
    # We need to add terms to make T_{i,j,k} = 1 for i>0, k>0.
    # We have n(p-1) slots.
    # We need to cover (n-1)m(p-1) entries? No, T_{i,j,k} is 1 for all j.
    # So for each i>0, k>0, we need the sum over r to be 1 for all j.
    # This means we need to add a term that is 1 for all j?
    # If we add a term u_r[i,j]=1, v_r[j,k]=1, w_r[i,k]=1, it contributes 1 to all j.
    # So we can just use one term for each (i,k) pair with i>0, k>0.
    # Number of such pairs is (n-1)(p-1).
    # We have n(p-1) slots.
    # So we have enough slots.
    
    # Construction for Block 3:
    # For i in 1..n-1, k in 1..p-1:
    #   r = nm + m(p-1) + (i-1)(p-1) + (k-1)
    #   u_r[i, j] = 1 for all j in 0..m-1.
    #   v_r[j, k] = 1 for all j in 0..m-1.
    #   w_r[i, k] = 1.
    #   (w_r is 1 only at index k*n+i).
    
    # Let's verify T_{i', j', k'} for this term.
    # u_r[i', j'] v_r[j', k'] w_r[i', k']
    # = (1 if i'=i else 0) * (1 if k'=k else 0) * (1 if i'=i and k'=k else 0)
    # = 1 if i'=i, k'=k, for any j'.
    # So this term adds 1 to T_{i,j,k} for all j.
    # This is exactly what we need!
    
    # So the construction is valid.
    
    # Implement Block 3
    offset2 = n * m + m * (p - 1)
    for i in range(1, n):
        for k in range(1, p):
            r = offset2 + (i - 1) * (p - 1) + (k - 1)
            # u_r[i, j] = 1 for all j
            for j in range(m):
                U[i * m + j, r] = 1
            # v_r[j, k] = 1 for all j
            for j in range(m):
                V[j * p + k, r] = 1
            # w_r[i, k] = 1
            W[k * n + i, r] = 1

    # Verify
    def verify(U, V, W, tol=1e-10):
        r = U.shape[1]
        i_vals = np.repeat(np.arange(n), m*p)
        j_vals = np.tile(np.repeat(np.arange(m), p), n)
        k_vals = np.tile(np.arange(p), n*m)
        
        u_idx = i_vals * m + j_vals
        v_idx = j_vals * p + k_vals
        w_idx = k_vals * n + i_vals
        
        U_sel = U[u_idx, :]
        V_sel = V[v_idx, :]
        W_sel = W[w_idx, :]
        
        products = U_sel * V_sel * W_sel
        sums = products.sum(axis=1)
        
        return np.all(np.abs(sums - 1) < tol)

    if verify(U, V, W):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to schoolbook
        rank_school = n * m * p
        U_school = np.zeros((n * m, rank_school))
        V_school = np.zeros((m * p, rank_school))
        W_school = np.zeros((p * n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, col] = 1
                    V_school[j * p + k, col] = 1
                    W_school[k * n + i, col] = 1
                    col += 1
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }

# EVOLVE-BLOCK-END
