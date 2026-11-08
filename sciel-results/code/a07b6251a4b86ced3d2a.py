"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Schoolbook decomposition as baseline (rank 60)
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
    
    best_rank = rank_school
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    def verify(U, V, W, tol=1e-8):
        """Check if the decomposition is valid using the tensor convention."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m + j, :] * V[j*p + k, :] * W[k*n + i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def verify_fast(U, V, W, tol=1e-8):
        """Vectorized verification."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        r = U.shape[1]
        # Build tensor: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # Vectorize over c
        for c in range(r):
            uc = U[:, c]  # (n*m,)
            vc = V[:, c]  # (m*p,)
            wc = W[:, c]  # (p*n,)
            # Reshape to 3D
            uc3 = uc.reshape(n, m)
            vc3 = vc.reshape(m, p)
            wc3 = wc.reshape(p, n)
            # Compute contribution to tensor
            # T_contrib[i,j,k] = uc3[i,j]*vc3[j,k]*wc3[k,i]
            # Use einsum
            contrib = np.einsum('ij,jk,ki->ijk', uc3, vc3, wc3)
            # Check all entries should sum to 1
            # We accumulate
            if not hasattr(verify_fast, '_accum'):
                verify_fast._accum = np.zeros((n, m, p))
            verify_fast._accum += contrib
        # After all columns, check
        result = np.all(np.abs(verify_fast._accum - 1.0) < tol)
        verify_fast._accum = np.zeros((n, m, p))
        return result
    
    def verify_accum(U, V, W, tol=1e-8):
        """Accumulate and verify."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        r = U.shape[1]
        accum = np.zeros((n, m, p))
        for c in range(r):
            uc3 = U[:, c].reshape(n, m)
            vc3 = V[:, c].reshape(m, p)
            wc3 = W[:, c].reshape(p, n)
            accum += np.einsum('ij,jk,ki->ijk', uc3, vc3, wc3)
        return np.all(np.abs(accum - 1.0) < tol)
    
    rng = np.random.default_rng(42)
    
    # Strategy: Use structured search with better initialization
    # Key insight: For 2x5*5x6, we can use the fact that the two rows of A 
    # allow some sharing. Try to find decompositions with rank < 60.
    
    # Approach: Start from schoolbook and try to merge columns
    # Or use random initialization with structured patterns
    
    # Try various ranks
    for target_rank in range(59, 40, -1):
        if time.time() > deadline - 20:
            break
        
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Structured initialization: each column corresponds to a (i,j,k) pattern
            # with some probability of including other indices
            for c in range(target_rank):
                # Primary (i,j,k)
                i0 = rng.integers(0, n)
                j0 = rng.integers(0, m)
                k0 = rng.integers(0, p)
                
                # U: mainly i0*m+j0, possibly others
                U[i0 * m + j0, c] = 1.0
                if rng.random() < 0.3:
                    i1 = rng.integers(0, n)
                    j1 = rng.integers(0, m)
                    if i1 != i0 or j1 != j0:
                        U[i1 * m + j1, c] = rng.choice([-1.0, 1.0])
                
                # V: mainly j0*p+k0, possibly others
                V[j0 * p + k0, c] = 1.0
                if rng.random() < 0.3:
                    j1 = rng.integers(0, m)
                    k1 = rng.integers(0, p)
                    if j1 != j0 or k1 != k0:
                        V[j1 * p + k1, c] = rng.choice([-1.0, 1.0])
                
                # W: mainly k0*n+i0, possibly others
                W[k0 * n + i0, c] = 1.0
                if rng.random() < 0.3:
                    k1 = rng.integers(0, p)
                    i1 = rng.integers(0, n)
                    if k1 != k0 or i1 != i0:
                        W[k1 * n + i1, c] = rng.choice([-1.0, 1.0])
            
            if verify_accum(U, V, W):
                best_rank = target_rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
        
        if best_rank < rank_school:
            break
    
    # If still at schoolbook, try a different structured approach
    if best_rank == rank_school:
        # Use the identity: for 2 rows, we can potentially reduce rank
        # by combining computations. Try a specific construction.
        
        # Construction idea: Split j into two groups {0,1,2} and {3,4}
        # Or use the fact that rank(M_{2,m,p}) <= m*p + rank(M_{2,m,p} with some structure)
        
        # Actually, let's try: for each k, compute C[0,k] and C[1,k]
        # C[0,k] = sum_j A[0,j] B[j,k]
        # C[1,k] = sum_j A[1,j] B[j,k]
        
        # We can write these as:
        # Let S_j = A[0,j] + A[1,j], D_j = A[0,j] - A[1,j]
        # Then C[0,k] + C[1,k] = sum_j S_j B[j,k]
        #      C[0,k] - C[1,k] = sum_j D_j B[j,k]
        
        # This uses 2*p multiplications for the sums, but we still need to 
        # recover C[0,k] and C[1,k] which are free (addition/subtraction).
        # But wait, that's still 2*p = 12 multiplications, same as schoolbook for 
        # the "sum" part. The schoolbook uses 2*5*6 = 60.
        
        # Hmm, that's not right. Schoolbook uses m*p per row = 5*6 = 30 per row, 
        # total 60. The sum/diff approach uses 5*6 = 30 for sums, but then we need
        # to get C[0] and C[1] back, which is free. So total 30? No, that can't be right.
        
        # Let me reconsider. The tensor rank counts the number of bilinear terms.
        # In schoolbook: for each (i,j,k), we have one term A[i,j]*B[j,k]*C[i,k].
        # That's 2*5*6 = 60 terms.
        
        # In the sum/diff approach:
        # Term 1: (A[0,j]+A[1,j])*B[j,k] contributes to C[0,k]+C[1,k]
        # This is not a standard tensor term because it involves sums of A entries.
        
        # The tensor decomposition requires terms of the form u_i * v_j * w_k where
        # u, v, w are vectors. So we can't directly use sums.
        
        # Let me try a different approach: use the known result that for M_{2,m,p},
        # if we can find a rank-r decomposition, it corresponds to r triples of vectors.
        
        # For now, let's just return the best we found.
        pass
    
    return best_result
# EVOLVE-BLOCK-END
