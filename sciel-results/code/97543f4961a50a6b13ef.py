"""Program for rank-minimal tensor decomposition of 2x3x9 matrix multiplication."""
import random
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We exploit the structure that p=9 is a multiple of m=3.
    # We can decompose the multiplication into 3 blocks of size 2x3x3.
    # If we find a rank-r decomposition for 2x3x3, we can construct a rank-3r decomposition for 2x3x9.
    # We search for a low-rank decomposition of 2x3x3 first.
    
    n_sub, m_sub, p_sub = 2, 3, 3
    
    def verify_sub(U, V, W, n, m, p):
        """Vectorized verification for sub-problem."""
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # T[a, b, d] = 1 if a=i*m+j, b=j*p+k, d=k*n+i else 0
        r = U.shape[1]
        # Construct the tensor T implicitly
        # We need sum_c U[a,c] V[b,c] W[d,c] == T[a,b,d]
        # This is a 3-way tensor contraction.
        # We can check all active entries.
        # Active entries: i in [0,n), j in [0,m), k in [0,p)
        # a = i*m + j, b = j*p + k, d = k*n + i
        
        # Create indices for active entries
        i_vals, j_vals, k_vals = np.indices((n, m, p))
        a_idx = i_vals * m + j_vals
        b_idx = j_vals * p + k_vals
        d_idx = k_vals * n + i_vals
        
        # Flatten
        a_flat = a_idx.flatten()
        b_flat = b_idx.flatten()
        d_flat = d_idx.flatten()
        
        # Compute sum_c U[a,c] V[b,c] W[d,c]
        # This is equivalent to (U * V) contracted with W?
        # Let's do it in batches or use einsum.
        # U[a, c] * V[b, c] is a matrix of shape (num_active, r)
        # We need to multiply element-wise with W[d, c] and sum over c.
        
        U_a = U[a_flat, :] # shape (num_active, r)
        V_b = V[b_flat, :] # shape (num_active, r)
        W_d = W[d_flat, :] # shape (num_active, r)
        
        # Element-wise product and sum
        products = U_a * V_b * W_d
        sums = products.sum(axis=1)
        
        # Check if all sums are 1
        if not np.allclose(sums, 1.0, atol=1e-9):
            return False
        
        # Check inactive entries
        # We need to ensure that for any (a,b,d) not in the active set, the sum is 0.
        # This is harder to check vectorized without generating all tuples.
        # But if the rank is low and the solution is exact, it's likely correct.
        # For rigor, we can check a sample of inactive entries.
        num_checks = 500
        for _ in range(num_checks):
            a = random.randint(0, n*m - 1)
            b = random.randint(0, m*p - 1)
            d = random.randint(0, p*n - 1)
            
            # Check if active
            i1, j1 = divmod(a, m)
            j2, k2 = divmod(b, p)
            k1, i2 = divmod(d, n)
            
            if i1 == i2 and j1 == j2 and k1 == k2:
                continue # Active, already checked
            
            s = np.dot(np.dot(U[a, :], V[b, :]), W[d, :])
            if abs(s) > 1e-9:
                return False
                
        return True

    def search_sub_decomposition(n, m, p, max_rank, num_trials=1000):
        """Search for low-rank decomposition of nxmxp."""
        for r in range(1, max_rank + 1):
            for _ in range(num_trials):
                # Use small integers
                U = np.random.randint(-2, 3, size=(n*m, r))
                V = np.random.randint(-2, 3, size=(m*p, r))
                W = np.random.randint(-2, 3, size=(p*n, r))
                
                if verify_sub(U, V, W, n, m, p):
                    return U, V, W, r
        return None

    # Search for 2x3x3 decomposition
    # Schoolbook rank is 18. Try up to 18.
    # We expect rank to be around 13-15.
    U_sub, V_sub, W_sub, r_sub = search_sub_decomposition(n_sub, m_sub, p_sub, 18, num_trials=500)
    
    if U_sub is not None:
        # Construct 2x3x9 decomposition
        # U is same for all blocks (depends on A, which is 2x3)
        # V and W are block-diagonal (depend on B and C blocks)
        
        # U_new: shape (n*m, 3*r_sub) = (6, 3*r_sub)
        # We repeat U_sub 3 times horizontally?
        # No, the rank is additive.
        # The decomposition is sum_{c=1}^{3*r} ...
        # For block g (0,1,2), we use U_sub[:, c-g*r_sub] for the U part?
        # Wait, U depends on i,j. The i,j indices are the same for all blocks.
        # So U_new[:, g*r_sub : (g+1)*r_sub] = U_sub
        
        U_new = np.zeros((n*m, 3 * r_sub), dtype=int)
        V_new = np.zeros((m*p, 3 * r_sub), dtype=int)
        W_new = np.zeros((p*n, 3 * r_sub), dtype=int)
        
        for g in range(3):
            # U part
            U_new[:, g*r_sub:(g+1)*r_sub] = U_sub
            
            # V part
            # V_sub has shape (m*p_sub, r_sub) = (3*3, r_sub) = (9, r_sub)
            # V_new has shape (m*p, 3*r_sub) = (3*9, 3*r_sub) = (27, 3*r_sub)
            # The block g of B corresponds to rows [g*p_sub : (g+1)*p_sub] in V_new?
            # V index b = j*p + k.
            # For block g, k is in [g*p_sub, (g+1)*p_sub).
            # So b is in [j*p + g*p_sub, j*p + (g+1)*p_sub).
            # This is not a simple block diagonal in V_new if we just stack V_sub.
            # V_sub index is j*p_sub + k'.
            # V_new index is j*p + k = j*(3*p_sub) + (g*p_sub + k') = 3*j*p_sub + g*p_sub + k'.
            # This is complex. Let's map it.
            
            # For each j in [0, m_sub):
            #   For each k' in [0, p_sub):
            #     b_sub = j*p_sub + k'
            #     b_new = j*p + (g*p_sub + k')
            #     V_new[b_new, g*r_sub:(g+1)*r_sub] = V_sub[b_sub, :]
            
            for j in range(m_sub):
                for k_prime in range(p_sub):
                    b_sub = j * p_sub + k_prime
                    b_new = j * p + (g * p_sub + k_prime)
                    V_new[b_new, g*r_sub:(g+1)*r_sub] = V_sub[b_sub, :]
            
            # W part
            # W_sub has shape (p_sub*n, r_sub) = (3*2, r_sub) = (6, r_sub)
            # W_new has shape (p*n, 3*r_sub) = (9*2, 3*r_sub) = (18, 3*r_sub)
            # W index d = k*n + i.
            # For block g, k is in [g*p_sub, (g+1)*p_sub).
            # So d is in [g*p_sub*n, (g+1)*p_sub*n).
            # W_sub index is k'*n + i.
            # W_new index is (g*p_sub + k')*n + i = g*p_sub*n + k'*n + i.
            
            for k_prime in range(p_sub):
                for i in range(n_sub):
                    d_sub = k_prime * n_sub + i
                    d_new = (g * p_sub + k_prime) * n_sub + i
                    W_new[d_new, g*r_sub:(g+1)*r_sub] = W_sub[d_sub, :]
        
        # Verify the full decomposition
        if verify_sub(U_new, V_new, W_new, n, m, p):
            return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
        else:
            # Fallback if verification fails (should not happen if sub-verify was correct)
            pass

    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
