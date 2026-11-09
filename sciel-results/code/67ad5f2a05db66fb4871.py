import time
import numpy as np
import random

DIMENSIONS = [2, 5, 5]
DEADLINE = 120.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Initialize with schoolbook decomposition (rank 50)
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
                
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Helper to check if a candidate decomposition is valid
    # We need to check if the tensor defined by U, V, W matches the matmul tensor
    # The matmul tensor T has T[i, j, k] = 1 if i, j, k are valid indices for C[i,k] = sum A[i,j]B[j,k]
    # Actually, the tensor for matrix multiplication is defined by:
    # T_{ijk} = 1 if the term A_{ij} B_{jk} contributes to C_{ik}
    # In our indexing:
    # A index: i*m + j
    # B index: j*p + k
    # C index: k*n + i
    # The decomposition is sum_r U[a, r] V[b, r] W[c, r] = T_{a,b,c}
    # where a=i*m+j, b=j*p+k, c=k*n+i.
    
    def is_valid(U_cand, V_cand, W_cand, tol=1e-9):
        # Efficient check: construct the tensor from factors and compare to known ones
        # However, constructing the full tensor (2*5*5*5*5*5*5*5*5) is 2*5*5*5*5*5*5*5*5? No.
        # Tensor size is nm * mp * pn = 10 * 25 * 10 = 2500.
        # Rank is R.
        # We can compute the contraction for each (a, b) pair and check if it matches W[:, r] contributions?
        # Actually, for a fixed (a, b), the resulting vector over c is sum_r U[a, r] V[b, r] W[:, r].
        # This is equivalent to (U[a, :] * V[b, :]) @ W.T
        # We can precompute the target tensor T as a 3D array of shape (nm, mp, pn)
        
        # But we can be smarter.
        # For each r, the term is u_r v_r^T w_r.
        # The sum is sum_r u_r v_r^T w_r.
        # This is a tensor.
        
        # Let's just build the target tensor once outside the loop.
        return True # Placeholder, logic moved outside
        
    # Build target tensor T_target of shape (nm, mp, pn)
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
                
    # Randomized Search Strategy:
    # Try to find a lower rank R < 50.
    # We can use a heuristic: try to merge terms.
    # Or use a random linear combination of existing terms to see if we can eliminate one.
    # A more robust approach for small sizes:
    # Iterate R from 49 down to, say, 40.
    # For each R, try to solve for U, V, W with that rank.
    # This is a non-convex problem.
    # However, we can use the fact that if we have a valid decomposition of rank R,
    # we can try to perturb it or find a new one.
    
    # Given the time limit (120s) and the complexity of finding a rank < 50 for 2x5x5
    # (which is likely 50 or close), a heavy numerical optimization might not find a better solution
    # quickly. But the directive asks for a STRUCTURAL change: "structured random search".
    
    # Let's implement a simple local search / random restart for rank reduction.
    # We start with rank 50.
    # We try to "compress" the decomposition.
    # One way: pick two columns r1, r2. Try to find linear combinations that preserve the tensor.
    # This is essentially checking if the columns are linearly dependent in the tensor space.
    # Since the schoolbook decomposition is minimal for "diagonal" terms, they are likely independent.
    
    # Alternative: Use a random projection to reduce rank.
    # If we pick a random vector x of length R, and form new terms based on combinations,
    # we might find a lower rank representation if one exists.
    
    # Let's try a specific algorithm:
    # 1. Start with R = 49.
    # 2. Randomly initialize U, V, W with size (nm, R), (mp, R), (pn, R).
    # 3. Use gradient descent or alternating least squares to fit the tensor T_target.
    # 4. If it converges to a low error, we found a rank-49 decomposition.
    
    # ALS (Alternating Least Squares) for Tensor Decomposition (CP Decomposition)
    # Minimize || T - sum_r u_r v_r w_r ||_F^2
    
    def als_decompose(T, rank, iters=50, lr=0.1):
        nm, mp, pn = T.shape
        U = np.random.randn(nm, rank) * 0.01
        V = np.random.randn(mp, rank) * 0.01
        W = np.random.randn(pn, rank) * 0.01
        
        for it in range(iters):
            # Update U
            # For each i, U[i, :] should minimize || T[i, :, :] - sum_r U[i, r] V[:, r] W[:, r]^T ||
            # This is a linear regression problem for each i.
            # T[i] is a matrix (mp, pn).
            # We want U[i, r] such that sum_r U[i, r] (V[:, r] W[:, r]^T) approx T[i].
            # Let M_r = V[:, r] W[:, r]^T.
            # T[i] = sum_r U[i, r] M_r.
            # Vectorize: T[i].flatten() = sum_r U[i, r] M_r.flatten().
            # So we solve a linear system: A_U U[i, :] = T[i].flatten()
            # where A_U is (mp*pn, rank) with columns M_r.flatten().
            
            # Construct A_U
            # M_r is (mp, pn). Flatten to (mp*pn,).
            A_U = np.zeros((mp * pn, rank))
            for r in range(rank):
                M_r = np.outer(V[:, r], W[:, r])
                A_U[:, r] = M_r.flatten()
            
            # Solve least squares for each i
            for i in range(nm):
                b = T[i].flatten()
                # U[i, :] = A_U^+ b
                # Using np.linalg.lstsq
                try:
                    U[i, :], _, _, _ = np.linalg.lstsq(A_U, b, rcond=None)
                except:
                    pass
                    
            # Update V
            A_V = np.zeros((nm * pn, rank))
            for r in range(rank):
                M_r = np.outer(U[:, r], W[:, r])
                A_V[:, r] = M_r.flatten()
                
            for j in range(mp):
                b = T[:, j, :].flatten()
                try:
                    V[j, :], _, _, _ = np.linalg.lstsq(A_V, b, rcond=None)
                except:
                    pass
                    
            # Update W
            A_W = np.zeros((nm * mp, rank))
            for r in range(rank):
                M_r = np.outer(U[:, r], V[:, r])
                A_W[:, r] = M_r.flatten()
                
            for k in range(pn):
                b = T[:, :, k].flatten()
                try:
                    W[k, :], _, _, _ = np.linalg.lstsq(A_W, b, rcond=None)
                except:
                    pass
            
            # Check error
            # Error = || T - sum_r U[:, r] V[:, r] W[:, r]^T ||
            # Compute reconstructed tensor
            T_rec = np.zeros_like(T)
            for r in range(rank):
                T_rec += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                
            err = np.linalg.norm(T - T_rec)
            if err < 1e-9:
                return U, V, W, True
                
        return U, V, W, False

    # Try ranks from 49 down to 40
    for R in range(49, 39, -1):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        found = False
        # Multiple random restarts
        for restart in range(5):
            if time.time() - start_time > DEADLINE - 5:
                break
                
            U_cand, V_cand, W_cand, success = als_decompose(T_target, R, iters=20, lr=0.1)
            if success:
                # Verify exactly
                T_rec = np.zeros_like(T_target)
                for r in range(R):
                    T_rec += np.einsum('i,j,k->ijk', U_cand[:, r], V_cand[:, r], W_cand[:, r])
                if np.allclose(T_rec, T_target, atol=1e-9):
                    best_rank = R
                    best_U, best_V, best_W = U_cand, V_cand, W_cand
                    found = True
                    break
        
        if found:
            # If we found rank R, we might be able to find R-1, but it's harder.
            # Continue searching for lower ranks.
            pass
        else:
            # If we couldn't find rank R, it might be impossible or we just failed.
            # Given the time, we stop if we fail a few times?
            # For now, just try the next lower rank.
            pass

    # Return the best found
    if best_rank < 50:
        U_out = best_U.tolist()
        V_out = best_V.tolist()
        W_out = best_W.tolist()
    else:
        U_out = U.tolist()
        V_out = V.tolist()
        W_out = W.tolist()
        
    return {"U": U_out, "V": V_out, "W": W_out}
