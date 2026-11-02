import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 3]
DEADLINE_SECONDS = 140.0  # Safety margin for the 150s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    R_school = n * m * p  # 18
    
    # Build schoolbook solution as fallback
    U_school = np.zeros((n * m, R_school))
    V_school = np.zeros((m * p, R_school))
    W_school = np.zeros((p * n, R_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    best_rank = R_school

    # The tensor T is defined such that C = A * B where A is n x m, B is m x p.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The decomposition is C = sum_r (u_r * v_r) * w_r^T (conceptually)
    # Specifically, the tensor T has indices (i, j, k).
    # T[i, j, k] = 1 if the term A[i,j] B[j,k] contributes to C[i,k].
    # In the factorization: C[i,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]?
    # Wait, the standard tensor decomposition for matrix multiplication (n x m) * (m x p) -> (n x p)
    # is T_{i, j, k} where i in [0,n), j in [0,m), k in [0,p).
    # The factors are U (n*m x R), V (m*p x R), W (p*n x R).
    # The condition is: for all i, k, sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for all i, k, j' != j, the cross terms must vanish?
    # Actually, the standard condition is:
    # sum_r U[i, j, r] V[j, k, r] W[k, i, r] = delta_{i,k} (in the sense of the product structure)
    # More precisely, the tensor T_{i, j, k} is 1 for all valid i, j, k.
    # The decomposition requires:
    # T_{i, j, k} = sum_r U_{i, j, r} V_{j, k, r} W_{k, i, r}
    # where U is indexed by (i, j), V by (j, k), W by (k, i).
    
    # Let's verify indices from the prompt:
    # U: (n*m, rank) -> index i*m + j
    # V: (m*p, rank) -> index j*p + k
    # W: (p*n, rank) -> index k*n + i
    # Target: T[i, j, k] = 1 for all i, j, k.
    
    # We want to find minimal R such that there exist U, V, W with:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k.
    
    # This is a 3-way tensor decomposition.
    # We can search for integer solutions.
    # Known lower bound for 2x3x3 is 15.
    # We will try to find a rank 15 solution.
    
    def check_solution(U, V, W):
        # U: (6, R), V: (9, R), W: (6, R)
        # Check if sum_r U[i, r] V[j, r] W[k, r] == 1 for all i, j, k
        # Note: U index is i*m+j, V is j*p+k, W is k*n+i
        # We need to iterate i, j, k
        # For a fixed i, j, k:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # To speed up, we can precompute for each (i, k) the vector of W[k*n+i, :]
        # But R is small, so direct check is fine.
        
        R = U.shape[1]
        # Construct the tensor from factors
        # T_reconstructed[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        
        # We can do this efficiently using numpy einsum or loops.
        # Since dimensions are small, loops are okay.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    if not np.isclose(val, 1.0, atol=1e-9):
                        return False
        return True

    # Search for rank 15 solution
    # We can fix the scale by setting U[0,0] = 1 (if non-zero)
    # Or use a grid search for small integers.
    # Range: -3 to 3 is likely enough for rank 15.
    
    # To make search tractable, we can use a randomized local search or simulated annealing.
    # But the directive says "hardcode" or "bypass heuristic search" in the current program's comments,
    # but the general directive allows search if within budget.
    # "COMPUTE BUDGET: ... Keep an EXPLICIT deadline guard... Spare time does not establish a bottleneck..."
    # "REFINEMENT: ... investigate improvements within its family."
    
    # Let's try a simple randomized search with local refinement.
    # Or, better, use the fact that we can solve for W given U and V.
    # For each (k, i), the equation is:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all j.
    # This is a system of linear equations for W[k*n+i, :].
    # Let A_{j, r} = U[i*m+j, r] V[j*p+k, r].
    # We want A * W_col = 1_vector (length m).
    # If A is invertible (or has full rank m), we can solve for W_col.
    # Since m=3, we need rank(A) >= 3.
    
    # Algorithm:
    # 1. Randomly initialize U and V with small integers.
    # 2. For each (k, i), solve for W[k*n+i, :].
    # 3. Check if the solution W is "nice" (e.g., integers or half-integers, or simply valid).
    # 4. If valid, we have a solution.
    
    # Actually, we just need ANY valid solution, not necessarily integer, but the problem says "integer/half-integer entries".
    # The evaluator checks "exactly decompose", so floating point errors might be an issue if we don't round.
    # But if we solve linear systems, the W entries will be rational.
    # We can check if they are close to integers or half-integers.
    
    # Let's implement this solver.
    
    def try_random_solution(R, num_trials=1000):
        for trial in range(num_trials):
            if time.time() - start_time > DEADLINE_SECONDS:
                return None
            
            # Initialize U and V with small integers
            U = np.random.randint(-2, 3, size=(n*m, R))
            V = np.random.randint(-2, 3, size=(m*p, R))
            
            W = np.zeros((p*n, R))
            valid = True
            
            for k in range(p):
                for i in range(n):
                    # Build matrix A (m x R)
                    # A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    A = np.zeros((m, R))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    
                    # Solve A * w = 1
                    # Use least squares
                    try:
                        w, residuals, rank, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    except:
                        valid = False
                        break
                    
                    # Check if solution is exact
                    if not np.allclose(A @ w, np.ones(m), atol=1e-9):
                        valid = False
                        break
                    
                    # Check if w is close to integer or half-integer
                    # The problem requires integer/half-integer.
                    # Let's check if 2*w is close to integer.
                    w_scaled = 2 * w
                    if not np.allclose(w_scaled, np.round(w_scaled), atol=1e-6):
                        # Maybe relax this? The prompt says "integer/half-integer entries that exactly decompose".
                        # If we find a rational solution, we can scale U, V, W to make them integers?
                        # If U, V are integers, and W is rational, we can clear denominators.
                        # But the rank might change? No, scaling columns doesn't change rank.
                        # However, the evaluator might check for integer/half-integer specifically.
                        # Let's assume we can return rational numbers if they are exact?
                        # "integer/half-integer entries" suggests the output must be of that type.
                        # So we should try to find solutions where W is half-integer.
                        valid = False
                        break
                    
                    W[k*n+i, :] = w
            
            if valid:
                # Verify the whole solution
                if check_solution(U, V, W):
                    return U, V, W
            
            # Local search idea: if close, refine?
            # For now, just random restarts.
            
        return None

    # Try ranks from 15 up to 17
    for R in range(15, best_rank):
        if time.time() - start_time > DEADLINE_SECONDS - 10:
            break
            
        # Increase trials as we go?
        trials = 500 if R == 15 else 200
        
        sol = try_random_solution(R, trials)
        if sol is not None:
            U, V, W = sol
            best_U, best_V, best_W = U, V, W
            best_rank = R
            break  # Found minimal rank
    
    # If no better solution found, return schoolbook
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
