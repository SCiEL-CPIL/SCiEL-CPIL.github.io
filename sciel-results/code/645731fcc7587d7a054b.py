"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build the target tensor T: T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(i,k) else 0
    T = np.zeros((nm, mp, pn), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    def verify_full(U, V, W):
        """Full verification - check exact decomposition."""
        rank = U.shape[1]
        # Use float64 for computation, check exactness
        T_recon = np.zeros((nm, mp, pn), dtype=np.float64)
        for r in range(rank):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            # Only process non-zero entries
            nz_u = np.nonzero(u)[0]
            nz_v = np.nonzero(v)[0]
            nz_w = np.nonzero(w)[0]
            for a in nz_u:
                for b in nz_v:
                    for c in nz_w:
                        T_recon[a, b, c] += u[a] * v[b] * w[c]
        # Check exact match (allow tiny floating point errors)
        return np.allclose(T_recon, T, atol=1e-10, rtol=0)
    
    def verify_quick(U, V, W):
        """Quick verification - check only the required 1s and a sample of 0s."""
        rank = U.shape[1]
        # Check all required 1s
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = 0.0
                    for r in range(rank):
                        val += U[a, r] * V[b, r] * W[c, r]
                    if abs(val - 1.0) > 1e-10:
                        return False
        # Check a random sample of 0s
        np.random.seed(42)
        checked = 0
        while checked < 200:
            a = np.random.randint(0, nm)
            b = np.random.randint(0, mp)
            c = np.random.randint(0, pn)
            if T[a, b, c] == 0:
                val = 0.0
                for r in range(rank):
                    val += U[a, r] * V[b, r] * W[c, r]
                if abs(val) > 1e-10:
                    return False
                checked += 1
        return True
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school), dtype=np.float64)
    V_school = np.zeros((mp, rank_school), dtype=np.float64)
    W_school = np.zeros((pn, rank_school), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    
    # Strategy: Try structured decompositions with rank < 54
    # The tensor has structure: for each (i,j,k), we need term (i*m+j, j*p+k, k*n+i)
    # Try to find linear combinations that cancel unwanted terms
    
    # Approach 1: Try rank R decompositions using ALS-like optimization
    # Start from schoolbook and try to merge terms
    
    def try_als(rank, max_iter=500, tol=1e-8):
        """Try to find rank-`rank` decomposition using ALS."""
        if rank >= best_rank:
            return None
        
        # Initialize with random or schoolbook-based init
        np.random.seed(rank)  # Deterministic for reproducibility
        
        # Use schoolbook as base, then random perturbation
        U = np.zeros((nm, rank), dtype=np.float64)
        V = np.zeros((mp, rank), dtype=np.float64)
        W = np.zeros((pn, rank), dtype=np.float64)
        
        # Initialize: use first `rank` schoolbook terms as starting point
        for r in range(min(rank, rank_school)):
            i, j, k = r // (m*p), (r // p) % m, r % p
            U[i * m + j, r] = 1.0
            V[j * p + k, r] = 1.0
            W[k * n + i, r] = 1.0
        
        # Add random components for remaining rank
        for r in range(rank):
            U[:, r] += np.random.randn(nm) * 0.1
            V[:, r] += np.random.randn(mp) * 0.1
            W[:, r] += np.random.randn(pn) * 0.1
        
        prev_err = float('inf')
        for it in range(max_iter):
            if time.time() > deadline - 30:
                break
            
            # Update U: for each column a, solve least squares
            # T[a, :, :] = sum_r U[a,r] * outer(V[:,r], W[:,r])
            for a in range(nm):
                # Build system: sum_r U[a,r] * V[:,r] * W[:,r] = T[a,:,:]
                # This is a linear system in U[a, :]
                # A = [outer(V[:,0],W[:,0]), outer(V[:,1],W[:,1]), ...]
                # A @ U[a,:] = T[a,:,:].flatten()
                
                # For efficiency, only build if needed
                if it < 5 or it % 10 == 0:  # Only full update sometimes
                    A = np.zeros((mp * pn, rank))
                    for r in range(rank):
                        A[:, r] = np.outer(V[:, r], W[:, r]).flatten()
                    b = T[a, :, :].flatten()
                    try:
                        U[a, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
            
            # Update V: for each column b
            for b in range(mp):
                if it < 5 or it % 10 == 0:
                    A = np.zeros((nm * pn, rank))
                    for r in range(rank):
                        A[:, r] = np.outer(U[:, r], W[:, r]).flatten()
                    b_vec = T[:, b, :].flatten()
                    try:
                        V[b, :] = np.linalg.lstsq(A, b_vec, rcond=None)[0]
                    except:
                        pass
            
            # Update W: for each column c
            for c in range(pn):
                if it < 5 or it % 10 == 0:
                    A = np.zeros((nm * mp, rank))
                    for r in range(rank):
                        A[:, r] = np.outer(U[:, r], V[:, r]).flatten()
                    b_vec = T[:, :, c].flatten()
                    try:
                        W[c, :] = np.linalg.lstsq(A, b_vec, rcond=None)[0]
                    except:
                        pass
            
            # Check convergence
            if it % 20 == 0:
                if verify_quick(U, V, W):
                    if verify_full(U, V, W):
                        return (U.copy(), V.copy(), W.copy())
        
        # Final check
        if verify_quick(U, V, W) and verify_full(U, V, W):
            return (U.copy(), V.copy(), W.copy())
        
        return None
    
    # Try various ranks
    for target_rank in [53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40]:
        if time.time() > deadline - 60:
            break
        
        result = try_als(target_rank, max_iter=200)
        if result is not None:
            U_new, V_new, W_new = result
            best_rank = target_rank
            best_U, best_V, best_W = U_new, V_new, W_new
            print(f"Found valid decomposition with rank {target_rank}")
            break
    
    # Approach 2: Structured decomposition exploiting the 3x3x6 structure
    # The output has 18 elements (3x6), input has 9x9=81 possible pairs
    # But only 54 are non-zero in the tensor
    
    # Try a different structure: group by output index
    # For output (i,k), the contribution is sum_j A[i,j] * B[j,k]
    # We can try to share computation across different (i,k) pairs
    
    if time.time() < deadline - 30 and best_rank > 40:
        # Try structured rank-48 decomposition
        # Idea: For each pair (i,j), we need to compute contributions to all k
        # B[j,:] is a 1x6 vector, so A[i,j] * B[j,:] gives a 1x6 row
        # Sum over j gives the i-th row of output
        
        # Try to factor B[j,:] in a shared way
        # B has 3 rows of 6 elements = 18 elements
        # If we can represent each B[j,:] as sum of few rank-1 terms, we can save
        
        # For now, try a specific structured approach
        rank_struct = 48
        U_s = np.zeros((nm, rank_struct), dtype=np.float64)
        V_s = np.zeros((mp, rank_struct), dtype=np.float64)
        W_s = np.zeros((pn, rank_struct), dtype=np.float64)
        
        # Structure: for each (i,j,k), we have a term
        # But try to merge some terms
        # Merge terms that have same (i,j) but different k: they share U[i*m+j, :]
        # Merge terms that have same (j,k) but different i: they share V[j*p+k, :]
        # Merge terms that have same (k,i) but different j: they share W[k*n+i, :]
        
        # This is complex; skip for now and rely on ALS
        
    # Ensure best solution is valid
    if not verify_full(best_U, best_V, best_W):
        # Fallback to schoolbook
        best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
        best_rank = rank_school
    
    # Convert to list of lists for output
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
