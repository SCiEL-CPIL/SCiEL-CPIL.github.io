"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """Find a low-rank exact tensor decomposition for 3x3 * 3x8 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Tensor T[i,j,k] = 1 for matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
    # Convention: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Build the full tensor T of shape (n, m, p) where T[i,j,k] = 1
    T = np.ones((n, m, p))
    
    # Schoolbook rank as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Strategy: Use structured low-rank decomposition based on the known 
    # structure of matrix multiplication. We'll try to find a decomposition
    # with rank significantly below n*m*p = 72.
    
    # Key insight: For matrix multiplication, we can use the fact that
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This is a bilinear form. The tensor T[i,j,k] = delta(i,j,k) in some sense.
    
    # Let's try a different approach: use the fact that we can decompose
    # the multiplication using fewer terms by exploiting structure.
    
    # Approach 1: Try to find a decomposition where we group operations.
    # For 3x3 * 3x8, we have 3*3*8 = 72 scalar multiplications in schoolbook.
    # 
    # One known approach: use the fact that (a+b)(c+d) = ac + ad + bc + bd
    # can be done with 3 multiplications instead of 4 (Karatsuba-like).
    # 
    # But for matrix multiplication, the rank is known to be less than n*m*p
    # for many cases. For 3x3, the rank is known to be 23 (Landsberg).
    # For 3x3 * 3x8, we can potentially do better than 72.
    
    # Let's try a systematic search using a greedy approach with better initialization.
    
    # First, let's try a simple structured decomposition.
    # We can write:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # One approach: use the identity that allows us to combine terms.
    # For example, if we define:
    # S1 = A[0,0] + A[1,1]
    # S2 = A[0,1] + A[1,0]
    # etc.
    
    # Actually, let's use a more principled approach.
    # The matrix multiplication tensor has a known structure. Let's try
    # to find a low-rank decomposition using a combination of:
    # 1. Structured patterns (based on known algorithms)
    # 2. Greedy refinement
    
    # Try target ranks from low to high
    # Known: 3x3 matrix mult has rank 23 (best known upper bound)
    # For 3x3 * 3x8, a reasonable target might be around 40-50
    
    max_rank_to_try = 60
    
    for target_rank in range(30, max_rank_to_try + 1):
        if time.time() > deadline - 30:
            break
            
        # Try greedy construction with target_rank terms
        success = _greedy_construct_v2(T, n, m, p, target_rank, deadline)
        if success is not None:
            U, V, W = success
            if target_rank < best_rank:
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                if best_rank <= 35:  # Good enough, stop early
                    break
        else:
            # If greedy fails, continue trying higher ranks
            continue
    
    # If we didn't find anything better than schoolbook, return schoolbook
    # But let's also try a structured approach as a fallback
    
    if best_rank == rank_school:
        # Try a structured decomposition
        structured = _structured_decomposition(n, m, p)
        if structured is not None:
            U, V, W, rank = structured
            if rank < best_rank:
                best_U, best_V, best_W = U, V, W
                best_rank = rank
    
    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

def _greedy_construct_v2(T, n, m, p, target_rank, deadline):
    """
    Improved greedy algorithm to find a rank-`target_rank` decomposition.
    Uses better initialization and local search.
    Returns (U, V, W) arrays or None if failed.
    """
    T_residual = T.copy().astype(float)
    U = np.zeros((n*m, target_rank))
    V = np.zeros((m*p, target_rank))
    W = np.zeros((p*n, target_rank))
    
    # Precompute index mappings for efficiency
    u_indices = [(i*m+j) for i in range(n) for j in range(m)]
    v_indices = [(j*p+k) for j in range(m) for k in range(p)]
    w_indices = [(k*n+i) for k in range(p) for i in range(n)]
    
    for r in range(target_rank):
        if time.time() > deadline - 15:
            return None
        
        # Find the best rank-1 tensor to subtract
        # We'll use a combination of random search and local optimization
        
        best_score = -1e18
        best_u, best_v, best_w = None, None, None
        
        # Try multiple random initializations with different strategies
        np.random.seed(1000 + r)
        
        num_attempts = 200
        for attempt in range(num_attempts):
            if time.time() > deadline - 15:
                return None
            
            # Different initialization strategies
            if attempt < num_attempts // 3:
                # Strategy 1: Sparse binary vectors
                u = np.zeros(n*m)
                v = np.zeros(m*p)
                w = np.zeros(p*n)
                # Set random subset to 1
                u[np.random.choice(n*m, size=max(1, n*m//4), replace=False)] = 1
                v[np.random.choice(m*p, size=max(1, m*p//4), replace=False)] = 1
                w[np.random.choice(p*n, size=max(1, p*n//4), replace=False)] = 1
            elif attempt < 2 * num_attempts // 3:
                # Strategy 2: Dense with small values
                u = np.random.choice([-1, 0, 1], size=n*m)
                v = np.random.choice([-1, 0, 1], size=m*p)
                w = np.random.choice([-1, 0, 1], size=p*n)
            else:
                # Strategy 3: Structured patterns
                u = np.zeros(n*m)
                v = np.zeros(m*p)
                w = np.zeros(p*n)
                # Try block patterns
                for i in range(n):
                    for j in range(m):
                        if np.random.random() < 0.5:
                            u[i*m+j] = 1
                for j in range(m):
                    for k in range(p):
                        if np.random.random() < 0.5:
                            v[j*p+k] = 1
                for k in range(p):
                    for i in range(n):
                        if np.random.random() < 0.5:
                            w[k*n+i] = 1
            
            # Compute the rank-1 tensor and score
            score = 0.0
            for i in range(n):
                for j in range(m):
                    ui = u[i*m+j]
                    if ui == 0:
                        continue
                    for k in range(p):
                        score += T_residual[i,j,k] * ui * v[j*p+k] * w[k*n+i]
            
            if score > best_score:
                best_score = score
                best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
        
        if best_u is None or best_score <= 1e-10:
            return None
        
        # Local search: try to improve the current solution
        # Try flipping individual elements
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            # Try flipping u elements
            for idx in range(n*m):
                old_val = best_u[idx]
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    best_u[idx] = new_val
                    score = 0.0
                    for i in range(n):
                        for j in range(m):
                            ui = best_u[i*m+j]
                            if ui == 0:
                                continue
                            for k in range(p):
                                score += T_residual[i,j,k] * ui * best_v[j*p+k] * best_w[k*n+i]
                    if score > best_score + 1e-10:
                        best_score = score
                        improved = True
                    else:
                        best_u[idx] = old_val
            # Try flipping v elements
            for idx in range(m*p):
                old_val = best_v[idx]
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    best_v[idx] = new_val
                    score = 0.0
                    for i in range(n):
                        for j in range(m):
                            ui = best_u[i*m+j]
                            if ui == 0:
                                continue
                            for k in range(p):
                                score += T_residual[i,j,k] * ui * best_v[j*p+k] * best_w[k*n+i]
                    if score > best_score + 1e-10:
                        best_score = score
                        improved = True
                    else:
                        best_v[idx] = old_val
            # Try flipping w elements
            for idx in range(p*n):
                old_val = best_w[idx]
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                        best_w[idx] = new_val
                    score = 0.0
                    for i in range(n):
                        for j in range(m):
                            ui = best_u[i*m+j]
                            if ui == 0:
                                continue
                            for k in range(p):
                                score += T_residual[i,j,k] * ui * best_v[j*p+k] * best_w[k*n+i]
                    if score > best_score + 1e-10:
                        best_score = score
                        improved = True
                    else:
                        best_w[idx] = old_val
        
        # Subtract the best rank-1 tensor
        for i in range(n):
            for j in range(m):
                ui = best_u[i*m+j]
                if ui == 0:
                    continue
                for k in range(p):
                    T_residual[i,j,k] -= ui * best_v[j*p+k] * best_w[k*n+i]
        
        U[:, r] = best_u
        V[:, r] = best_v
        W[:, r] = best_w
        
        # Check if residual is close to zero
        if np.allclose(T_residual, 0, atol=1e-10):
            # Success! Trim unused columns
            for r2 in range(r+1, target_rank):
                U[:, r2] = 0
                V[:, r2] = 0
                W[:, r2] = 0
            return U[:, :r+1], V[:, :r+1], W[:, :r+1]
    
    # Check final residual
    if np.allclose(T_residual, 0, atol=1e-10):
        return U, V, W
    
    return None

def _structured_decomposition(n, m, p):
    """
    Try a structured decomposition based on known matrix multiplication algorithms.
    Returns (U, V, W, rank) or None.
    """
    # For 3x3 * 3x8, let's try to use a structured approach.
    # One idea: use the fact that we can combine some terms.
    
    # Actually, let's just return None for now and rely on the greedy approach.
    # The structured approach would require more careful design.
    return None

# EVOLVE-BLOCK-END
