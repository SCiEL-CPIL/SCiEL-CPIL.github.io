"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from collections import defaultdict

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify exact decomposition."""
    r = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p):
    """Faster verification using numpy."""
    r = len(U[0])
    U_arr = np.array(U, dtype=np.float64).reshape(n, m, r)
    V_arr = np.array(V, dtype=np.float64).reshape(m, p, r)
    W_arr = np.array(W, dtype=np.float64).reshape(p, n, r)
    
    Ue = U_arr.reshape(n, m, 1, r)
    Ve = V_arr.reshape(1, m, p, r)
    We = W_arr.T.reshape(n, 1, p, r)
    
    T = np.sum(Ue * Ve * We, axis=3)
    return np.allclose(T, np.ones((n, m, p)), atol=1e-9)

def _schoolbook(n, m, p):
    """Standard O(nmp) decomposition."""
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
    return U, V, W, rank

def _try_random_uv_solve_w(n, m, p, target_rank, rng, max_attempts=1000):
    """
    Try to find a decomposition by randomly choosing U and V with small integer entries,
    and then solving for W using linear systems for each (k, i) pair.
    """
    for _ in range(max_attempts):
        # Generate random U and V with entries in {-1, 0, 1}
        U = np.random.randint(-1, 2, size=(n, m, target_rank))
        V = np.random.randint(-1, 2, size=(m, p, target_rank))
        
        # We need to solve for W[k, i, c] for all k, i, c.
        # For each fixed k, i, we have a system:
        # sum_c U[i, j, c] * V[j, k, c] * W[k, i, c] = 1 for all j in 0..m-1
        # Let A[j, c] = U[i, j, c] * V[j, k, c]
        # Then A @ W_vec = 1_vec
        # W_vec is of size target_rank.
        
        W_solution = np.zeros((p, n, target_rank))
        valid = True
        
        for k in range(p):
            for i in range(n):
                # Construct A (m x target_rank)
                A = np.zeros((m, target_rank))
                for j in range(m):
                    for c in range(target_rank):
                        A[j, c] = U[i, j, c] * V[j, k, c]
                
                b = np.ones(m)
                
                # Solve A @ x = b
                # Check if A has full column rank? Or just use lstsq.
                # If target_rank == m, we can use solve if invertible.
                # If target_rank > m, we have infinite solutions. We want one that is "nice".
                # If target_rank < m, we need consistency.
                
                # Let's use lstsq to get the minimum norm solution.
                try:
                    x, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                # Check if the solution is close to integer or half-integer.
                # The problem allows integer/half-integer.
                # So 2*x should be close to integer.
                two_x = 2 * x
                rounded_two_x = np.round(two_x)
                if not np.allclose(two_x, rounded_two_x, atol=1e-6):
                    valid = False
                    break
                
                W_solution[k, i, :] = x
        
        if not valid:
            continue
            
        # Verify the solution
        if _verify_fast(U.tolist(), V.tolist(), W_solution.tolist(), n, m, p):
            return U.tolist(), V.tolist(), W_solution.tolist(), target_rank
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    rng = random.Random(42)
    np.random.seed(42)
    
    # Baseline: schoolbook
    U, V, W, rank = _schoolbook(n, m, p)
    best = (U, V, W, rank)
    best_rank = rank
    
    # Theoretical lower bound
    min_possible = max(n * p, m * p, n * m)  # max(12, 16, 12) = 16
    
    # Strategy: Random U,V -> Solve W
    # Try ranks from lower bound up to best_rank
    for target_rank in range(min_possible, best_rank):
        if time.time() - start_time > deadline - 5:
            break
        
        # Increase attempts as rank increases? Or keep fixed?
        # Lower ranks are harder to find.
        max_attempts = 2000 if target_rank < 30 else 5000
        
        result = _try_random_uv_solve_w(n, m, p, target_rank, rng, max_attempts)
        if result is not None:
            U_new, V_new, W_new, r_new = result
            if r_new < best_rank:
                best = (U_new, V_new, W_new, r_new)
                best_rank = r_new
                # If we found a solution, we can try to go lower?
                # But we are iterating upwards. We break and restart?
                # No, we just update best and continue to see if we can find even lower?
                # Wait, the loop is `for target_rank in range(min_possible, best_rank)`.
                # If we find a solution at R, best_rank becomes R.
                # The loop condition `target_rank < best_rank` will stop the loop for this R.
                # But we want to find the MINIMAL rank.
                # So if we find a solution at R=20, we should check if R=19 is possible.
                # But we are iterating 16, 17, 18...
                # So if we fail at 16, 17, 18, 19 and succeed at 20, we have rank 20.
                # We should break because we found the smallest so far?
                # Yes, because we are iterating from low to high.
                break
                
    # Final verification
    U_best, V_best, W_best, _ = best
    
    # Ensure we're within time limit
    if time.time() - start_time > 140:
        U, V, W, rank = _schoolbook(n, m, p)
        U_best, V_best, W_best = U, V, W
    
    return {"U": U_best, "V": V_best, "W": W_best}
# EVOLVE-BLOCK-END
