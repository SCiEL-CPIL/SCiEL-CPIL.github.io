"""Tensor decomposition search for 3x4 x 4x7 matrix multiplication."""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 7]

def construct():
    deadline = time.time() + 130.0  # 20s safety margin
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    best_result = None

    # Schoolbook fallback (rank 84)
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Build the multiplication tensor T[i,j,k] = 1
    # T is indexed as (i*m+j, k*n+i) -> wait, let me think about the tensor structure.
    # The multiplication tensor T has entries T[i,j,k] = 1 for all i,j,k.
    # Matrix U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    # The decomposition requires: for all (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # And for "wrong" indices, the sum should be 0.
    # Actually, the tensor is: T[a, b, c] where a = i*m+j (input), b = k*n+i (output), c = j*p+k (middle)
    # Wait, let me re-read: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # The condition is: sum_c U[a,c] * V[b,c] * W[c_idx,c] = T[a, b, c_idx]
    # where a = i*m+j, b = j*p+k, c_idx = k*n+i
    # T[a, b, c_idx] = 1 if the indices are consistent (same i,j,k), 0 otherwise.
    
    # Let me build the constraint system.
    # For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # For all other combinations of (a, b, c_idx) that don't correspond to a valid (i,j,k): sum = 0
    
    # Actually, the tensor T is a 3-tensor with dimensions (n*m) x (m*p) x (p*n)
    # T[a, b, c] = 1 if there exist i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i
    # Otherwise T[a,b,c] = 0
    
    # Let me enumerate all valid (i,j,k) triples and the invalid ones.
    # Total entries in the tensor: (n*m) * (m*p) * (p*n) = 84 * 28 * 21 = 49392
    # Valid entries: n*m*p = 84
    
    # For a rank-r decomposition, we need:
    # For valid (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # For invalid (a,b,c): sum_c U[a,c] * V[b,c] * W[c,c] = 0
    
    # This is a system of polynomial equations. Let me use a different approach.
    # 
    # Approach: Randomized search for low-rank decompositions.
    # 1. Try ranks from low to high (starting from lower bound ~12-15)
    # 2. For each rank, try random initializations and use iterative refinement.
    
    # Let me use the approach from "Tensor Decomposition" literature:
    # For a given rank r, we want to find U, V, W such that the tensor T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # 
    # I'll use a randomized approach:
    # 1. For rank r, randomly initialize U, V, W
    # 2. Use alternating least squares or gradient-based optimization
    # 3. Check if the solution is exact (within tolerance)
    
    # Actually, for exact solutions, let me try a different approach.
    # The tensor T has a specific structure. Let me use the fact that:
    # T = sum_{i,j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # 
    # This is a sum of 84 rank-1 tensors. We want to find a shorter sum.
    # 
    # For small matrices, we can use known results:
    # - 2x2x2: rank 7 (Strassen)
    # - 3x3x3: rank 23 (Landsberg) or 23 (Bini et al.)
    # - For 3x4x4: known rank is around 15-18
    # - For 3x4x7: we need to search
    
    # Let me try a systematic approach with random restarts.
    
    def try_rank(r, max_iters=50, seed_offset=0):
        """Try to find a rank-r decomposition."""
        if time.time() > deadline - 5:
            return None
        
        rng = random.Random(seed_offset)
        np_rng = np.random.RandomState(seed_offset)
        
        # Try multiple random initializations
        for attempt in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            # Random initialization
            U = np_rng.randn(nm, r)
            V = np_rng.randn(mp, r)
            W = np_rng.randn(pn, r)
            
            # Build the constraint matrix for linearization
            # For each valid (i,j,k): U[i*m+j,:] . (V[j*p+k,:] * W[k*n+i,:]) = 1
            # This is nonlinear. Let me use a different approach.
            
            # Use the "random subspace" method:
            # Project the tensor onto a random subspace and solve.
            
            # Alternative: use the fact that for rank r, we can write:
            # T = U @ diag(s) @ V^T for each "slice"
            # Actually, let me use a simpler approach: 
            # For fixed W, the problem becomes linear in U and V.
            
            # Let me use alternating minimization:
            # 1. Fix V, W -> solve for U (linear least squares)
            # 2. Fix U, W -> solve for V (linear least squares)
            # 3. Fix U, V -> solve for W (linear least squares)
            
            # But this gives approximate solutions, not exact ones.
            
            # For exact solutions, I need to use a different method.
            # Let me try the "random linear constraints" approach:
            # Add random linear constraints to make the system solvable.
            
            # Actually, let me just try to verify known decompositions.
            # For 3x4x4, there are known rank-15 decompositions.
            # For 3x4x7, I'll need to search.
            
            # Let me try a greedy approach:
            # Start with the schoolbook decomposition and try to merge terms.
            
            break  # This simple approach won't work, let me try something else
        
        return None
    
    # Let me try a more systematic approach using the structure of the problem.
    # 
    # Key insight: The tensor T for matrix multiplication (n,m,p) has the property that:
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k)
    # T[a,b,c] = 0 otherwise
    # 
    # For the 3x4x7 case, let me try to find a decomposition by:
    # 1. Using known results for smaller subproblems
    # 2. Combining them using block structure
    
    # For 3x4x7, we can think of it as:
    # C (3x7) = A (3x4) * B (4x7)
    # 
    # We can split B into B1 (4x4) and B2 (4x3):
    # C1 (3x4) = A * B1
    # C2 (3x3) = A * B2
    # 
    # If we have rank-r1 decomposition for 3x4x4 and rank-r2 for 3x4x3,
    # then the total rank is r1 + r2.
    
    # Known: 3x4x4 has rank at most 15 (from literature)
    # Known: 3x4x3 has rank at most 12 (schoolbook) or less
    
    # Let me try to construct explicit decompositions.
    
    # For 3x4x4, let me try to find a rank-15 decomposition.
    # I'll use a randomized search with verification.
    
    def build_tensor(n, m, p):
        """Build the multiplication tensor as a 3D array."""
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        return T
    
    T = build_tensor(n, m, p)
    
    # For verification, I need to check that the decomposition is exact.
    # Given U (nm x r), V (mp x r), W (pn x r):
    # The reconstructed tensor is: sum_c U[:,c] * V[:,c] * W[:,c] (outer product)
    # This should equal T.
    
    def verify_decomposition(U, V, W, T, tol=1e-10):
        """Check if U, V, W exactly decompose T."""
        r = U.shape[1]
        # Reconstruct tensor
        T_recon = np.zeros(T.shape)
        for c in range(r):
            T_recon += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
        return np.allclose(T_recon, T, atol=tol)
    
    # Try to find low-rank decompositions using random search.
    # For each rank r, try random initializations and use ALS to converge.
    
    def als_decompose(T, r, max_sweeps=100, tol=1e-12, seed=0):
        """Alternating least squares for tensor decomposition."""
        if time.time() > deadline - 5:
            return None
        
        np_rng = np.random.RandomState(seed)
        U = np_rng.randn(T.shape[0], r)
        V = np_rng.randn(T.shape[1], r)
        W = np_rng.randn(T.shape[2], r)
        
        for sweep in range(max_sweeps):
            if time.time() > deadline - 5:
                return None
            
            # Update U: for each i, solve (V * W) @ U[:,c] = T[i,:,:]
            # T[i,:,:] = sum_c U[i,c] * V[:,c] * W[:,c]
            # So T[i,:,:] = (V * W) @ U[i,:] where V*W is element-wise product
            # Actually: T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
            # For fixed i: T[i,:,:] = sum_c U[i,c] * (V[:,c] * W[:,c])
            # So T[i,:,:] = (V * W) @ U[i,:] where V*W is (mp*pn, r) matrix with entries V[j,c]*W[k,c]
            
            VW = np.einsum('jc,kc->jkc', V, W).reshape(T.shape[1]*T.shape[2], r)
            # Solve: VW @ U[i,:] = T[i,:,:].reshape(-1)
            U, _, _, _ = np.linalg.lstsq(VW, T.reshape(T.shape[0], -1).T, rcond=None)
            
            # Update V
            UW = np.einsum('ic,kc->ikc', U, W).reshape(T.shape[0]*T.shape[2], r)
            V, _, _, _ = np.linalg.lstsq(UW, T.reshape(-1, T.shape[1]).T, rcond=None)
            
            # Update W
            UV = np.einsum('ic,jc->ijc', U, V).reshape(T.shape[0]*T.shape[1], r)
            W, _, _, _ = np.linalg.lstsq(UV, T.reshape(-1, T.shape[2]).T, rcond=None)
            
            # Check convergence
            if sweep > 0:
                T_recon = np.einsum('ic,jc,kc->ijk', U, V, W)
                err = np.linalg.norm(T_recon - T) / np.linalg.norm(T)
                if err < tol:
                    break
        
        # Verify
        if verify_decomposition(U, V, W, T):
            return U, V, W
        return None
    
    # Try ranks from low to high
    min_rank = n * m  # lower bound
    max_rank = n * m * p  # schoolbook
    
    # Try a range of ranks
    for r in range(max_rank, min_rank - 1, -1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple seeds for this rank
        for seed in range(20):
            if time.time() > deadline - 5:
                break
            
            result = als_decompose(T, r, max_sweeps=50, seed=seed)
            if result is not None:
                U, V, W = result
                # Convert to lists
                U_list = [[float(U[i,j]) for j in range(r)] for i in range(nm)]
                V_list = [[float(V[i,j]) for j in range(r)] for i in range(mp)]
                W_list = [[float(W[i,j]) for j in range(r)] for i in range(pn)]
                if best_result is None or r < best_result[0]:
                    best_result = (r, U_list, V_list, W_list)
        
        # If we found a solution, try lower ranks
        if best_result is not None and r < best_result[0]:
            continue
    
    # Return best result or schoolbook
    if best_result is not None:
        r, U_list, V_list, W_list = best_result
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        return schoolbook()
