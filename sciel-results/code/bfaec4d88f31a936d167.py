"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-6):
    """
    Verify exact decomposition.
    Checks:
    1. sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1 for all i,j,k
    2. sum_r U[i,j,r] V[j',k,r] W[k,i,r] == 0 for all i,k and j != j'
    """
    rank = len(U[0]) if U else 0
    # Convert to numpy for speed in verification if needed, but lists are fine for small sizes
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    # Precompute sums for efficiency? 
    # For each r, we have vectors u_r, v_r, w_r.
    # Condition 1: For each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # Condition 2: For each (i,k,j,j'), j!=j', sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] == 0
    
    # We can compute the "product tensor" P[i, j, k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # And Q[i, j, j', k] = sum_r U[i,j,r] V[j',k,r] W[k,i,r]
    
    # Efficient check using numpy
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_3d = U_np.reshape(n, m, rank)
    V_3d = V_np.reshape(m, p, rank)
    W_3d = W_np.reshape(p, n, rank)
    
    # Check Condition 1:
    # For each i, k, we need sum_j ( ... )? No, the condition is for each specific j.
    # Actually, the decomposition computes C_ik = sum_j A_ij B_jk.
    # The term A_ij B_jk should have coefficient 1.
    # The term A_ij B_j'k should have coefficient 0 for j != j'.
    
    # Let's compute the coefficient matrix for each i, k.
    # Coeff[i, k, j, j'] = sum_r U[i, j, r] V[j', k, r] W[k, i, r]
    
    # We can compute this by iterating over r?
    # Or using einsum.
    # U[i, j, r] * V[j', k, r] * W[k, i, r]
    # Sum over r.
    
    # Let's construct the full tensor of coefficients.
    # This might be memory heavy if rank is large, but rank <= 90, n,m,p small.
    
    # Coeff[i, k, j, j']
    # We can loop over i, k.
    for i in range(n):
        for k in range(p):
            # w_ik is a vector of length rank: W[k*n+i, :]
            w_vec = W_np[k * n + i]
            
            # For each j, j', we need sum_r U[i, j, r] * V[j', k, r] * w_vec[r]
            # U_i is a matrix of shape (m, rank)
            U_i = U_3d[i] # shape (m, rank)
            # V_k is a matrix of shape (m, rank) -- wait, V is indexed by j, k.
            # V[j, k, r]. For fixed k, V[:, k, :] is shape (m, rank).
            V_k = V_3d[:, k, :] # shape (m, rank)
            
            # We want M[j, j'] = sum_r U_i[j, r] * V_k[j', r] * w_vec[r]
            # This is equivalent to:
            # M = (U_i * w_vec[None, :]) @ V_k.T
            # U_i * w_vec: shape (m, rank)
            # V_k.T: shape (rank, m)
            # M: shape (m, m)
            
            M = (U_i * w_vec) @ V_k.T
            
            # Check diagonal: M[j, j] should be 1
            # Check off-diagonal: M[j, j'] should be 0
            for j in range(m):
                if abs(M[j, j] - 1.0) > tol:
                    return False
                for j_prime in range(m):
                    if j_prime != j and abs(M[j, j_prime]) > tol:
                        return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _sa_search(n, m, p, target_rank, max_iters=20000, seed=42):
    """
    Simulated Annealing to find a decomposition of target_rank.
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Initialize with random integers {-1, 0, 1}
    def rand_int():
        return random.choice([-1, 0, 1])
        
    U = [[rand_int() for _ in range(target_rank)] for _ in range(n * m)]
    V = [[rand_int() for _ in range(target_rank)] for _ in range(m * p)]
    W = [[rand_int() for _ in range(target_rank)] for _ in range(p * n)]
    
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    def compute_error(U_np, V_np, W_np):
        # Compute error as sum of squares of violations
        err = 0.0
        U_3d = U_np.reshape(n, m, target_rank)
        V_3d = V_np.reshape(m, p, target_rank)
        W_3d = W_np.reshape(p, n, target_rank)
        
        for i in range(n):
            for k in range(p):
                w_vec = W_np[k * n + i]
                U_i = U_3d[i]
                V_k = V_3d[:, k, :]
                M = (U_i * w_vec) @ V_k.T
                # Diagonal should be 1
                diag = np.diag(M)
                err += np.sum((diag - 1.0) ** 2)
                # Off-diagonal should be 0
                off_diag_mask = ~np.eye(m, dtype=bool)
                err += np.sum(M[off_diag_mask] ** 2)
        return err

    current_err = compute_error(U_np, V_np, W_np)
    best_err = current_err
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    T = 10.0
    cool_rate = 0.999
    
    for it in range(max_iters):
        if time.time() > start_time + 110:
            break
            
        # Pick a random variable to flip
        # 1/3 chance for U, 1/3 for V, 1/3 for W
        choice = random.random()
        if choice < 1/3:
            idx = random.randint(0, n * m - 1)
            col = random.randint(0, target_rank - 1)
            old_val = U[idx][col]
            U[idx][col] = random.choice([-1, 0, 1])
            U_np[idx, col] = U[idx][col]
            new_err = compute_error(U_np, V_np, W_np)
        elif choice < 2/3:
            idx = random.randint(0, m * p - 1)
            col = random.randint(0, target_rank - 1)
            old_val = V[idx][col]
            V[idx][col] = random.choice([-1, 0, 1])
            V_np[idx, col] = V[idx][col]
            new_err = compute_error(U_np, V_np, W_np)
        else:
            idx = random.randint(0, p * n - 1)
            col = random.randint(0, target_rank - 1)
            old_val = W[idx][col]
            W[idx][col] = random.choice([-1, 0, 1])
            W_np[idx, col] = W[idx][col]
            new_err = compute_error(U_np, V_np, W_np)
            
        delta = new_err - current_err
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_err = new_err
            if current_err < best_err:
                best_err = current_err
                best_U = [row[:] for row in U]
                best_V = [row[:] for row in V]
                best_W = [row[:] for row in W]
                if best_err < 1e-6:
                    return best_U, best_V, best_W
        else:
            # Revert
            if choice < 1/3:
                U[idx][col] = old_val
                U_np[idx, col] = old_val
            elif choice < 2/3:
                V[idx][col] = old_val
                V_np[idx, col] = old_val
            else:
                W[idx][col] = old_val
                W_np[idx, col] = old_val
                
        T *= cool_rate
        
    if best_err < 1e-6:
        return best_U, best_V, best_W
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Best solution starts as schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 90
    
    # Try to find lower ranks
    # We try ranks from 89 down to 70 (arbitrary cutoff)
    # We use multiple seeds for each rank
    for rank in range(best_rank - 1, 69, -1):
        if time.time() > deadline - 10:
            break
            
        found = None
        # Try a few seeds
        for seed in range(3):
            if time.time() > deadline - 10:
                break
            result = _sa_search(n, m, p, rank, max_iters=5000, seed=seed)
            if result is not None:
                U, V, W = result
                # Final verification
                if _verify(U, V, W, n, m, p):
                    found = (U, V, W)
                    break
        
        if found is not None:
            U, V, W = found
            best_rank = rank
            best = {"U": U, "V": V, "W": W}
            # We found a lower rank. Can we go lower?
            # Yes, continue loop.
            
    return best
# EVOLVE-BLOCK-END
