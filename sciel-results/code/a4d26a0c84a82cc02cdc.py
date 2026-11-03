"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s deadline for safety
    
    # Build the target tensor as a 3D array T[c, a, b]
    # T[k*n+i, i*m+j, j*p+k] = 1
    T = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = k * n + i
                a = i * m + j
                b = j * p + k
                T[c, a, b] = 1.0
    
    # Try to find a low-rank decomposition
    # Start with a good lower bound estimate
    # For M_{n,m,p}, known bounds... let's try from a reasonable range
    
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # Schoolbook as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best_result = {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}
    best_rank = rank_sb
    
    # Try to find better decompositions using local search
    # Strategy: start with a random low-rank init and iteratively improve
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W give an exact decomposition of T."""
        r = U.shape[1]
        # Compute T_reconstructed = sum_r U[:,r] outer V[:,r] outer W[:,r]
        # T_reconstructed[c,a,b] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Check against T
        max_err = 0.0
        for c in range(p * n):
            for a in range(n * m):
                for b in range(m * p):
                    val = 0.0
                    for r_idx in range(r):
                        val += U[a, r_idx] * V[b, r_idx] * W[c, r_idx]
                    target = T[c, a, b]
                    err = abs(val - target)
                    if err > max_err:
                        max_err = err
        return max_err < tol
    
    def verify_fast(U, V, W, tol=1e-9):
        """Faster verification using matrix operations."""
        r = U.shape[1]
        # For each r, the contribution is U[:,r] outer V[:,r] outer W[:,r]
        # T_reconstructed[c,a,b] = sum_r W[c,r] * U[a,r] * V[b,r]
        # = sum_r W[c,r] * (U[a,r] * V[b,r])
        # For fixed c: T_reconstructed[c,:,:] = sum_r W[c,r] * (U[:,r] outer V[:,r])
        # Let's compute M = U.T @ V  (shape r x ... no)
        # Actually: for each (a,b), sum_r U[a,r]*V[b,r]*W[c,r]
        # = (U * V.T) @ W ... not quite
        # Let P[a,b,r] = U[a,r]*V[b,r], then T_reconstructed[c,a,b] = sum_r P[a,b,r]*W[c,r]
        # = (P reshaped) @ W
        # P has shape (n*m, m*p, r)
        # Reshape to (n*m*m*p, r) and multiply by W.T
        # This is expensive for large r but let's try
        P = U[:, None, :] * V[None, :, :]  # (n*m, m*p, r)
        P_flat = P.reshape(-1, r)  # (n*m*m*p, r)
        T_recon = (P_flat @ W.T).reshape(p*n, n*m, m*p)  # (p*n, n*m, m*p)
        return np.max(np.abs(T_recon - T)) < tol
    
    def try_rank(r, num_trials=10, seed=42):
        """Try to find a rank-r decomposition."""
        np.random.seed(seed)
        random.seed(seed)
        
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Random initialization with small values
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            
            # Iteratively improve: for each factor, given the other two, solve for the best
            # This is like alternating minimization for tensor rank approximation
            # But we need EXACT decomposition, so this is tricky
            
            # Alternative: use a greedy approach
            # Start with schoolbook-like structure but try to merge terms
            
            # Let's try a different approach: start from schoolbook and try to eliminate
            # columns by finding linear dependencies
            
            # For now, let's try a structured approach:
            # Use the fact that for fixed i, the rows U[i*m+j, :] for j=0..m-1
            # and for fixed j, columns V[j*p+k, :] for k=0..p-1 have structure
            
            # Try: rank = n*p + m*p + n*m - something
            # This is getting complex. Let me try a simple local search on the rank.
            
            # Actually, let me try a direct approach for small ranks:
            # Assume rank r, and try to find U, V, W such that the decomposition holds.
            # This is a system of polynomial equations. Hard to solve directly.
            
            # Instead, let's use a heuristic: start with a slightly reduced rank
            # and try to fix up errors.
            
            pass
        
        return None
    
    # Try specific known constructions or bounds
    # For M_{4,5,5}, let's try ranks from 70 to 99
    # and use a simple verification
    
    # Actually, let me try a more systematic approach:
    # Use the "border rank" or known algorithms
    
    # For now, let's try to find ANY improvement over 100
    # by using a structured low-rank approximation
    
    # Try rank = 90, 85, 80, etc. with random init and check
    for test_rank in [95, 90, 85, 80, 75, 70, 65, 60]:
        if time.time() > deadline:
            break
        
        found = False
        for trial in range(5):
            if time.time() > deadline:
                break
            
            np.random.seed(test_rank * 100 + trial)
            
            # Initialize with a structured pattern
            # Try to use the "Landsberg" type construction or similar
            U = np.zeros((n * m, test_rank))
            V = np.zeros((m * p, test_rank))
            W = np.zeros((p * n, test_rank))
            
            # Fill with random sparse patterns
            for c in range(test_rank):
                # Each column picks some non-zero entries
                for a in range(n * m):
                    if random.random() < 0.3:
                        U[a, c] = random.choice([-1, 1])
                for b in range(m * p):
                    if random.random() < 0.3:
                        V[b, c] = random.choice([-1, 1])
                for cc in range(p * n):
                    if random.random() < 0.3:
                        W[cc, c] = random.choice([-1, 1])
            
            # Verify
            if verify_fast(U, V, W):
                # Convert to half-integers if needed
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                best_rank = test_rank
                found = True
                break
        
        if found:
            break
    
    # Convert to proper format
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    return {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}
# EVOLVE-BLOCK-END
