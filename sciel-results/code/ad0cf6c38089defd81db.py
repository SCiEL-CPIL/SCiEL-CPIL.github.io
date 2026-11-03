import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # n=3, m=4, p=8
    rank = 88
    
    U = np.zeros((n * m, rank))  # (12, 88)
    V = np.zeros((m * p, rank))  # (32, 88)
    W = np.zeros((p * n, rank))  # (24, 88)
    
    col = 0
    
    # Strassen factors for 2x2x2
    strassen_factors = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # For the 2x4x8 part (rows 0,1 of A):
    for a_group in range(2):
        for b_group in range(4):
            for factor in strassen_factors:
                a_coeffs, b_coeffs, c_coeffs = factor
                
                for (ai, aj), au in a_coeffs.items():
                    u_idx = ai * 4 + (a_group * 2 + aj)
                    U[u_idx, col] += au
                
                for (bi, bj), bv in b_coeffs.items():
                    v_idx = (a_group * 2 + bi) * 8 + (b_group * 2 + bj)
                    V[v_idx, col] += bv
                
                for (ci, cj), cw in c_coeffs.items():
                    w_idx = (b_group * 2 + cj) * 3 + ci
                    W[w_idx, col] += cw
                
                col += 1
    
    # For the 1x4x8 part (row 2 of A):
    for j in range(4):
        for k in range(8):
            u_idx = 2 * 4 + j
            v_idx = j * 8 + k
            w_idx = k * 3 + 2
            
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
    
    # --- Local search: identify and merge duplicate/linearly dependent rank terms ---
    deadline = time.time() + 130  # safety margin before 150s limit
    
    # Build the full tensor T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # and verify it matches the matrix multiplication tensor
    def build_tensor(U, V, W):
        n_, m_, p_ = n, m, p
        T = np.zeros((n_, m_, p_))
        for r in range(U.shape[1]):
            T += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        return T
    
    def verify(U, V, W):
        """Check if the decomposition exactly matches matrix multiplication."""
        T_decomp = build_tensor(U, V, W)
        # Build reference tensor: T[i,j,k] = 1 if i==j==k in the multiplication sense
        # Actually, T[i,j,k] should be A[i,j]*B[j,k] -> C[i,k]
        # The tensor for matmul is: T[i,j,k] = 1 if we're computing C[i,k] += A[i,j]*B[j,k]
        # So T[i,j,k] = 1 for all valid (i,j,k) where i<n, j<m, k<p
        # Wait, no. The multiplication tensor has T[i,j,k] = 1 iff the term A[i,j]*B[j,k] appears in C[i,k]
        # That's always 1 for all (i,j,k) in range.
        # Actually the standard matmul tensor: T[i,j,k] = 1 for all i,j,k (it's the identity in the sense that each triple contributes)
        # Let me reconsider: the tensor T has T[i,j,k] = 1 if and only if the product A[i,j]*B[j,k] contributes to C[i,k]
        # In standard matmul, every (i,j,k) triple contributes, so T[i,j,k] = 1 for all valid indices.
        # But that's the "unweighted" tensor. The decomposition satisfies:
        # sum_r U[i,r] V[j,r] W[k,r] = T[i,j,k]
        # where T[i,j,k] = 1 for all (i,j,k) in the multiplication pattern.
        # For matmul C = AB, T[i,j,k] = 1 iff j is the shared index, which is always the case here.
        # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
        
        # Verify: for each (i,k), sum_j A[i,j]B[j,k] = C[i,k]
        # The tensor equation is: sum_r U[i,r]V[j,r]W[k,r] = 1 for all (i,j,k)
        # This means: for fixed (i,k), sum_j (sum_r U[i,r]V[j,r]) W[k,r] ... no that's not right.
        # 
        # The correct check: the decomposition gives C[i,k] = sum_r U[i,r] * (sum_j V[j,r] * A[j,:]) ... 
        # Actually, let's just check: for random A and B, does C = A@B?
        
        # Simpler check: T_decomp[i,j,k] should equal 1 for all (i,j,k)
        T_ref = np.ones((n_, m_, p_))
        if not np.allclose(T_decomp, T_ref, atol=1e-10):
            return False, np.max(np.abs(T_decomp - T_ref))
        return True, 0.0
    
    # First verify the initial decomposition
    valid, err = verify(U, V, W)
    if not valid:
        # Return as-is if invalid (shouldn't happen)
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Try to find linear dependencies among the rank terms
    # A rank term r is a triple (U[:,r], V[:,r], W[:,r])
    # Two terms are "duplicate" if they produce the same contribution
    # More generally, we can check if any column in the stacked matrix is linearly dependent
    
    # Approach: form the Kronecker product matrix
    # Flatten each rank term as a vector of length n*m*p
    # The tensor T is the sum of rank-1 tensors, each flattened is u_r ⊗ v_r ⊗ w_r
    # We want to find the minimal number of such terms that sum to T
    
    # Alternative: check if any two rank terms are identical
    # For each pair, check if U[:,r1]==U[:,r2] and V[:,r1]==V[:,r2] and W[:,r1]==W[:,r2]
    
    # More powerful: check linear dependencies in the 3-way tensor space
    # The space of rank-1 tensors has dimension n*m*p = 96
    # We have 88 rank-1 terms. Check if they're linearly independent.
    
    # Flatten each rank term
    def flatten_rank_term(u, v, w):
        return np.kron(np.kron(u, v), w)
    
    # Build the matrix whose columns are the flattened rank terms
    M = np.zeros((n * m * p, rank))
    for r in range(rank):
        M[:, r] = flatten_rank_term(U[:, r], V[:, r], W[:, r])
    
    # Find the rank of M
    # Use SVD to find linear dependencies
    try:
        U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
    except:
        U_svd, s, Vt = np.linalg.svd(M, full_matrices=True)
    
    # Count how many singular values are above tolerance
    tol = 1e-10
    effective_rank = np.sum(s > tol)
    
    # If effective_rank < rank, we have linear dependencies
    # We can potentially reduce the number of terms
    # But we need to be careful: the sum must still equal T_ref
    
    # For now, just identify which columns are linearly dependent
    # and try to merge them
    
    # A simpler approach: check for exact duplicates
    seen = {}
    duplicates = []
    for r in range(rank):
        key = (tuple(U[:, r]), tuple(V[:, r]), tuple(W[:, r]))
        if key in seen:
            duplicates.append((seen[key], r))
        else:
            seen[key] = r
    
    # If there are duplicates, we can combine them
    # But in our construction, there shouldn't be exact duplicates
    
    # Try a different approach: use the fact that the 1x4x8 part has 32 terms
    # and see if some can be combined with the Strassen terms
    
    # Actually, let's try to reduce rank by checking if we can express
    # some rank terms as linear combinations of others in the tensor space
    
    # For each term r, check if it's in the span of the others
    # This is expensive, so only do it if we have time
    
    if time.time() < deadline - 5:
        # Try to find the minimal subset
        # Use greedy: start with all terms, try removing one at a time
        # and check if the sum still equals T_ref
        
        current_rank = rank
        U_curr = U.copy()
        V_curr = V.copy()
        W_curr = W.copy()
        T_target = np.ones((n, m, p))
        
        # Try removing terms one by one
        removed = []
        for r in range(rank):
            if time.time() > deadline - 10:
                break
            
            # Temporarily remove term r
            U_temp = np.delete(U_curr, r, axis=1)
            V_temp = np.delete(V_curr, r, axis=1)
            W_temp = np.delete(W_curr, r, axis=1)
            
            # Check if the sum still equals T_target
            T_check = build_tensor(U_temp, V_temp, W_temp)
            if np.allclose(T_check, T_target, atol=1e-10):
                # We can remove this term
                U_curr = U_temp
                V_curr = V_temp
                W_curr = W_temp
                removed.append(r)
                current_rank -= 1
        
        if current_rank < rank:
            U = U_curr
            V = V_curr
            W = W_curr
            rank = current_rank
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
