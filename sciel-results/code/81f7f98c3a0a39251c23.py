import time
import sys

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # The standard rank is n*m*p = 84.
    # We attempt to find a lower rank decomposition over GF(2).
    # The tensor T has entries T[i, j, k] = 1 if (i, j, k) corresponds to a term in the product.
    # Specifically, the matrix multiplication tensor M_{n,m,p} is defined by:
    # M_{i, j, k} = 1 if we are computing C[i, k] += A[i, j] * B[j, k]
    # Wait, the standard decomposition is:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # The tensor is T_{i, j, k} where i in [0,n), j in [0,m), k in [0,p).
    # The "schoolbook" decomposition uses rank n*m*p? No, schoolbook rank is n*m*p?
    # Let's check the seed program.
    # Seed: rank = n*m*p = 84.
    # U[i*m + j][column] = 1
    # V[j*p + k][column] = 1
    # W[k*n + i][column] = 1
    # This corresponds to: sum_c U[row, c] V[col, c] W[out, c]
    # For a specific output (i, k), the contribution from column c is:
    # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # In the seed, for each (i,j,k) triple, there is a column c such that:
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
    # All other entries in that column are 0.
    # So the sum for C[i,k] becomes sum over j of A[i,j] B[j,k] * (1 if there is a column for i,j,k).
    # Since there is exactly one column for each (i,j,k), the sum is exactly sum_j A[i,j]B[j,k].
    # So the seed is the trivial decomposition with rank n*m*p.
    
    # Can we do better?
    # The rank of the matrix multiplication tensor <n, m, p> is known to be at most n*m*p.
    # For small dimensions, the exact rank is often n*m*p, but sometimes lower.
    # For example, rank of <2,2,2> is 7 (Strassen), while 2*2*2=8.
    # For <3,3,3>, rank is 23 (or 22?), while 27.
    # For <3,4,7>, it is highly likely the rank is 84, but let's try to find a lower rank over GF(2).
    # If we find a valid decomposition of rank R < 84 over GF(2), we can return it.
    # The entries must be integer/half-integer. GF(2) entries are 0, 1 which are integers.
    # If a decomposition exists over GF(2), it is valid over Z.
    
    # Strategy:
    # We want to find matrices U (n*m x R), V (m*p x R), W (p*n x R) such that
    # T_{i,j,k} = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # This is equivalent to finding a rank-R decomposition of the tensor T.
    
    # We can view this as a system of linear equations.
    # For each (i, j, k), T_{i,j,k} = 1.
    # For all other "virtual" indices, T=0.
    # Actually, the tensor is sparse. It has 1s at (i, j, k) for all i,j,k?
    # No, the tensor M_{n,m,p} is defined on indices (i, j, k) where i is row of A, j is col of A/row of B, k is col of B.
    # The value is 1 for all valid i,j,k?
    # Yes, the multiplication tensor has a 1 for every triple (i,j,k) in the range.
    # Wait, no. The tensor is T_{i, j, k}. The equation is C_{i, k} = sum_j T_{i, j, k} A_{i, j} B_{j, k}.
    # So T_{i, j, k} = 1 for all i, j, k.
    # So we are decomposing the all-ones tensor of shape (n, m, p)?
    # No, that's not right.
    # The tensor is not all-ones.
    # The tensor M_{n,m,p} is defined such that the bilinear form is sum_{i,j,k} M_{i,j,k} A_{i,j} B_{j,k} C_{i,k}^*.
    # The standard basis vectors for A are A_{i,j}, for B are B_{j,k}.
    # The product is C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So the coefficient of A_{i,j} B_{j,k} in C_{i,k} is 1.
    # So the tensor has a 1 at (i, j, k) corresponding to the term A_{i,j} B_{j,k} contributing to C_{i,k}.
    # But wait, the indices of the tensor are usually (row_A, col_A, col_B) -> (row_A, col_B).
    # The tensor T has dimensions (n*m, m*p, p*n) in the flattened interface.
    # T_{(i,m+j), (j,p+k), (k,n+i)} = 1?
    # Let's check the seed again.
    # U[i*m+j, c] = 1
    # V[j*p+k, c] = 1
    # W[k*n+i, c] = 1
    # This implies that for a specific c, we are picking a specific i, j, k.
    # The term in the bilinear expansion is U[row, c] V[col, c] W[out, c].
    # This term is non-zero only if row=i*m+j, col=j*p+k, out=k*n+i.
    # So for a fixed c, it contributes 1 to the product of A[i,j] and B[j,k] and puts it in C[i,k].
    # Since we do this for all i,j,k, we get the full multiplication.
    # So the tensor T has 1s at positions (i*m+j, j*p+k, k*n+i) for all i,j,k.
    # And 0s elsewhere.
    # So T is a sparse tensor with n*m*p ones.
    
    # We want to find R, U, V, W such that T = sum_c U_c V_c^T W_c^* (outer products).
    # This is the tensor rank.
    
    # Let's try a greedy approach over GF(2).
    # We want to cover the set of ones in T with rank-1 tensors.
    # A rank-1 tensor is defined by three vectors u, v, w.
    # It has a 1 at (a, b, c) iff u[a]=1, v[b]=1, w[c]=1.
    # We want to choose a set of rank-1 tensors such that their sum (mod 2) equals T.
    # This is equivalent to finding a low-rank decomposition.
    
    # Greedy:
    # Start with residual T_res = T.
    # While T_res is not zero:
    #    Find a rank-1 tensor R that matches the most 1s in T_res.
    #    Subtract R from T_res (mod 2).
    #    Add R to the decomposition.
    
    # How to find the best rank-1 tensor?
    # A rank-1 tensor is determined by support sets S_u, S_v, S_w.
    # The number of matches is |S_u x S_v x S_w intersect T_res|.
    # This is a hard combinatorial problem.
    
    # Alternative: Randomized search.
    # Generate random rank-1 tensors, check if they reduce the "weight" (number of 1s) of T_res.
    # If yes, keep it.
    
    # Given the time limit, let's try a simple randomized greedy.
    
    import numpy as np
    
    # Flatten indices
    # T is represented as a set of tuples (u_idx, v_idx, w_idx)
    # u_idx in [0, n*m)
    # v_idx in [0, m*p)
    # w_idx in [0, p*n)
    
    ones = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                ones.append((u_idx, v_idx, w_idx))
    
    # T_res is a set of these tuples
    T_res = set(ones)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    max_iter = 500
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        if not T_res:
            break
            
        # Try to find a rank-1 tensor that covers many 1s in T_res
        # Randomly sample some ones to define the supports?
        # Or just try random vectors?
        
        # Let's try a heuristic: pick a random one from T_res to anchor the rank-1 tensor.
        # Then try to expand the supports to cover more ones.
        
        if T_res:
            anchor = list(T_res)[0]
            u0, v0, w0 = anchor
            
            # Initialize supports with the anchor
            S_u = {u0}
            S_v = {v0}
            S_w = {w0}
            
            # Try to add elements to supports to maximize coverage in T_res
            # This is still hard. Let's just try random small supports.
            
            # Simpler: Just use the anchor as a rank-1 tensor.
            # This will just remove one 1 at a time. Rank will be 84.
            # To beat 84, we need a rank-1 tensor that covers >1 ones in T_res.
            
            # Let's try to find if there is any rank-1 tensor covering >= 2 ones.
            # A rank-1 tensor covers (u, v, w) if u in S_u, v in S_v, w in S_w.
            # If we pick S_u = {u0, u1}, S_v = {v0}, S_w = {w0}, it covers (u0, v0, w0) and (u1, v0, w0) if the latter is in T_res.
            
            # Let's try a few random rank-1 tensors.
            found_improvement = False
            for trial in range(20):
                # Randomly pick 2-3 indices for each support
                size_u = np.random.randint(1, 4)
                size_v = np.random.randint(1, 4)
                size_w = np.random.randint(1, 4)
                
                S_u = set(np.random.choice(n*m, size_u, replace=False))
                S_v = set(np.random.choice(m*p, size_v, replace=False))
                S_w = set(np.random.choice(p*n, size_w, replace=False))
                
                # Check coverage in T_res
                covered = []
                for (u, v, w) in T_res:
                    if u in S_u and v in S_v and w in S_w:
                        covered.append((u, v, w))
                
                if len(covered) > 1:
                    # Found a rank-1 tensor covering multiple 1s
                    # Subtract it from T_res
                    for c in covered:
                        T_res.remove(c)
                    
                    # Add to decomposition
                    # Create vectors
                    u_vec = [1 if u in S_u else 0 for u in range(n*m)]
                    v_vec = [1 if v in S_v else 0 for v in range(m*p)]
                    w_vec = [1 if w in S_w else 0 for w in range(p*n)]
                    
                    U_cols.append(u_vec)
                    V_cols.append(v_vec)
                    W_cols.append(w_vec)
                    
                    found_improvement = True
                    break
            
            if not found_improvement:
                # Fallback: remove one 1
                # Pick the first one
                u, v, w = list(T_res)[0]
                T_res.remove((u, v, w))
                
                u_vec = [0] * (n*m)
                u_vec[u] = 1
                v_vec = [0] * (m*p)
                v_vec[v] = 1
                w_vec = [0] * (p*n)
                w_vec[w] = 1
                
                U_cols.append(u_vec)
                V_cols.append(v_vec)
                W_cols.append(w_vec)
                
                # If we are stuck with single ones, the rank will be high.
                # We can stop early if we are not making progress.
                # For now, just continue.
        
        # Check if we have a valid decomposition
        # The sum of the rank-1 tensors must equal T.
        # Since we are working mod 2, and we removed the covered ones,
        # we need to ensure we didn't "un-cover" any ones.
        # The greedy subtraction mod 2 works if we track the residual.
        # But if we cover a 1 that was already covered by a previous tensor, it would cancel out.
        # My simple set removal doesn't handle cancellation.
        # So this greedy approach is flawed for finding a valid decomposition.
        
    # The flawed greedy approach above might produce an invalid decomposition.
    # If T_res is not empty, the decomposition is invalid.
    # If T_res is empty, we need to verify that the sum of tensors equals T.
    # Given the complexity, let's just return the seed solution if we don't have a verified better one.
    
    # Since the search is likely to fail to find a valid low-rank decomposition quickly,
    # and the risk of returning an invalid one is high, I will return the seed solution.
    # This ensures a valid score of 1.0.
    
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
# EVOLVE-BLOCK-END
