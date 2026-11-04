"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 6]
n, m, p = DIMENSIONS

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return the standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W, tol=1e-6):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    U_reshaped = U.reshape(n, m, -1)
    V_reshaped = V.reshape(m, p, -1)
    W_reshaped = W.reshape(p, n, -1)
    try:
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    except Exception:
        return False
    if T.shape != (n, m, p):
        return False
    return np.allclose(T, 1.0, atol=tol, rtol=tol)

def _to_integer_or_half(U, V, W, tol=1e-6):
    """Convert entries to integer or half-integer if possible."""
    def snap(arr):
        arr2 = 2.0 * arr
        rounded = np.round(arr2)
        if np.allclose(arr2, rounded, atol=tol):
            return (rounded / 2.0).astype(float)
        return arr
    U = snap(U)
    V = snap(V)
    W = snap(W)
    return U, V, W

def _compute_error(U, V, W):
    """Compute the squared error between the tensor reconstruction and the all-ones tensor."""
    U_reshaped = U.reshape(n, m, -1)
    V_reshaped = V.reshape(m, p, -1)
    W_reshaped = W.reshape(p, n, -1)
    T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    return np.sum((T - 1.0) ** 2)

def _local_search_integer(r, max_steps=5000, time_limit=10.0):
    """
    Perform a local search on the integer/half-integer grid to minimize the tensor error.
    Returns (U, V, W, success).
    """
    start_time = time.time()
    
    # Initialize with random small integers/half-integers
    # Using values from {-1, 0, 1, 0.5, -0.5} is a good start for rank-1 components
    vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    U = np.random.choice(vals, size=(n * m, r))
    V = np.random.choice(vals, size=(m * p, r))
    W = np.random.choice(vals, size=(p * n, r))
    
    current_err = _compute_error(U, V, W)
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
        
        if current_err < 1e-10:
            return U, V, W, True
            
        # Randomly pick one matrix, one row, and one column to perturb
        mat_choice = np.random.randint(3)
        
        if mat_choice == 0:
            i = np.random.randint(n * m)
            j = np.random.randint(r)
            old_val = U[i, j]
            # Try perturbing to a nearby integer/half-integer
            # Neighbors: +/- 0.5, +/- 1.0
            delta = np.random.choice([-1.0, -0.5, 0.5, 1.0])
            new_val = old_val + delta
            # Snap to grid (it should already be on grid if we start on grid and use grid steps)
            U[i, j] = new_val
            new_err = _compute_error(U, V, W)
            if new_err >= current_err:
                U[i, j] = old_val
            else:
                current_err = new_err
                
        elif mat_choice == 1:
            i = np.random.randint(m * p)
            j = np.random.randint(r)
            old_val = V[i, j]
            delta = np.random.choice([-1.0, -0.5, 0.5, 1.0])
            new_val = old_val + delta
            V[i, j] = new_val
            new_err = _compute_error(U, V, W)
            if new_err >= current_err:
                V[i, j] = old_val
            else:
                current_err = new_err
                
        else:
            i = np.random.randint(p * n)
            j = np.random.randint(r)
            old_val = W[i, j]
            delta = np.random.choice([-1.0, -0.5, 0.5, 1.0])
            new_val = old_val + delta
            W[i, j] = new_val
            new_err = _compute_error(U, V, W)
            if new_err >= current_err:
                W[i, j] = old_val
            else:
                current_err = new_err
                
    return U, V, W, False

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    U_school, V_school, W_school = _schoolbook()
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    best_rank = n * m * p
    
    np.random.seed(42)
    
    # We know the rank is at most n*m*p = 120.
    # Known lower bounds for 4x5x6 are around 60-70? 
    # Let's try to find a solution with rank < 120.
    # We will search for ranks from a plausible minimum up to 120.
    # Since exact integer solutions are hard to find via random search for high ranks,
    # we will try a few ranks. If we fail, we return the schoolbook solution.
    
    # Strategy: Try ranks 100, 110, 120. 
    # Actually, for 4x5x6, the schoolbook is 120. 
    # There are known decompositions with lower rank (e.g. Strassen-like).
    # However, finding them via random integer search is hard.
    # Let's try to improve the schoolbook rank slightly if possible, or just return schoolbook if search fails.
    
    # To maximize score, we want the lowest rank.
    # If we can't find a better one quickly, schoolbook (120) is a safe valid answer.
    
    # Let's try a few random initializations for rank 119, 118, etc.
    # But local search on a high-dimensional integer grid is slow.
    
    # A better strategy for "integer/half-integer" requirement:
    # The schoolbook solution is already integer. 
    # Can we find a rank < 120?
    # Let's try rank 119, 118, 117 with a few restarts.
    
    for target_rank in [119, 118, 117, 116]:
        if time.time() > deadline - 10.0:
            break
            
        for trial in range(5):
            if time.time() > deadline - 10.0:
                break
            
            np.random.seed(42 + target_rank * 100 + trial)
            U, V, W, success = _local_search_integer(target_rank, max_steps=2000, time_limit=5.0)
            
            if success:
                # Verify strictly
                if _verify(U, V, W, tol=1e-6):
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    best_rank = target_rank
                    break # Move to next lower rank? No, we want minimal. 
                         # But we are iterating downwards. If we find 119, we break inner loop and try 118.
                         # Wait, the loop is `for target_rank in [119, 118...]`. 
                         # If we find a solution for 119, we should keep looking for 118.
                         # So we should NOT break the outer loop immediately.
                         # We just update best_rank and continue.
                         # But if we find 118, that's better.
                         # So we continue the loop.
                    # However, if we find a solution, we should probably stop trying higher ranks? 
                    # We are trying lower ranks. So we continue.
                    # But we need to update best_rank.
                    # If we find a solution for rank R, we set best_rank = R.
                    # Then we try R-1.
                    # If we find a solution for R-1, best_rank = R-1.
                    # This is correct.
                    # The `break` inside `for trial` breaks the trial loop for this rank.
                    # Then the outer loop continues to the next rank.
                    # This is fine.
        
        # If we found a solution for this rank, we don't need to try more trials for this rank.
        # The outer loop will try the next lower rank.
    
    # Ensure output is in the correct format
    # The search produces integer/half-integer values by construction.
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
