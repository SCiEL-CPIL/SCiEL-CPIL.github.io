import numpy as np
import time
import random
import itertools

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit

    def verify(U, V, W):
        """Verify that the tensor decomposition is exact."""
        r = len(U[0]) if isinstance(U, list) else U.shape[1]
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        U_3d = U_arr.reshape(n, m, r)
        V_3d = V_arr.reshape(m, p, r)
        W_3d = W_arr.reshape(p, n, r)
        T_prime = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        # Check exactness: all entries should be exactly 1
        if not np.allclose(T_prime, 1.0, atol=1e-10):
            return False
        return True

    def _schoolbook():
        """Return the schoolbook (rank 36) decomposition."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def _rank_from_matrices(U, V, W):
        """Return rank from the number of columns."""
        if isinstance(U, list):
            return len(U[0])
        return U.shape[1]

    def _try_rank(r, time_budget=25):
        """
        Attempt to find a rank-r decomposition using a structured search.
        Strategy: 
        1. Initialize with a random or structured pattern.
        2. Use coordinate descent with rounding to integers/half-integers.
        3. Use a smooth surrogate (log-sum-exp) for the error to guide local search.
        """
        end_time = time.time() + time_budget
        best_error = np.inf
        best_U = None
        best_V = None
        best_W = None

        # Number of restarts
        n_restarts = max(1, int(time_budget / 5))
        
        for restart in range(n_restarts):
            if time.time() > end_time:
                break
            
            # Initialize with random integers in [-2, 2]
            U = np.random.randint(-2, 3, size=(n*m, r)).astype(float)
            V = np.random.randint(-2, 3, size=(m*p, r)).astype(float)
            W = np.random.randint(-2, 3, size=(p*n, r)).astype(float)
            
            # Local search with coordinate descent
            for iteration in range(500):
                if time.time() > end_time:
                    break
                
                # Compute current error using log-sum-exp for smoothness
                U_3d = U.reshape(n, m, r)
                V_3d = V.reshape(m, p, r)
                W_3d = W.reshape(p, n, r)
                T_prime = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
                errors = (T_prime - 1.0) ** 2
                # Use log-sum-exp for a smooth surrogate
                max_err = np.max(errors)
                if max_err > 1000:
                    current_error = np.sum(errors)
                else:
                    current_error = np.log(np.sum(np.exp(errors - max_err))) + max_err
                
                if current_error < best_error:
                    best_error = current_error
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    
                    # Check for exact solution
                    if current_error < 1e-8:
                        # Try rounding to integers
                        U_int = np.round(U)
                        V_int = np.round(V)
                        W_int = np.round(W)
                        if verify(U_int, V_int, W_int):
                            return U_int, V_int, W_int
                        # Try half-integers
                        U_half = np.round(U * 2) / 2
                        V_half = np.round(V * 2) / 2
                        W_half = np.round(W * 2) / 2
                        if verify(U_half, V_half, W_half):
                            return U_half, V_half, W_half
                
                # Coordinate descent: pick a random entry and perturb
                mat = random.choice(['U', 'V', 'W'])
                if mat == 'U':
                    idx = random.randrange(n*m)
                    col = random.randrange(r)
                elif mat == 'V':
                    idx = random.randrange(m*p)
                    col = random.randrange(r)
                else:
                    idx = random.randrange(p*n)
                    col = random.randrange(r)
                
                # Try several perturbations
                best_perturbation = None
                best_perturbation_error = current_error
                
                for delta in [-1.0, -0.5, 0.5, 1.0, -2.0, 2.0]:
                    if mat == 'U':
                        U[idx, col] += delta
                    elif mat == 'V':
                        V[idx, col] += delta
                    else:
                        W[idx, col] += delta
                    
                    U_3d = U.reshape(n, m, r)
                    V_3d = V.reshape(m, p, r)
                    W_3d = W.reshape(p, n, r)
                    T_prime = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
                    errors = (T_prime - 1.0) ** 2
                    max_err = np.max(errors)
                    if max_err > 1000:
                        new_error = np.sum(errors)
                    else:
                        new_error = np.log(np.sum(np.exp(errors - max_err))) + max_err
                    
                    if new_error < best_perturbation_error:
                        best_perturbation_error = new_error
                        best_perturbation = delta
                    else:
                        if mat == 'U':
                            U[idx, col] -= delta
                        elif mat == 'V':
                            V[idx, col] -= delta
                        else:
                            W[idx, col] -= delta
                
                # Apply the best perturbation
                if best_perturbation is not None:
                    if mat == 'U':
                        U[idx, col] += best_perturbation
                    elif mat == 'V':
                        V[idx, col] += best_perturbation
                    else:
                        W[idx, col] += best_perturbation
                else:
                    # No improvement, try a larger random jump
                    if mat == 'U':
                        U[idx, col] += random.uniform(-2, 2)
                    elif mat == 'V':
                        V[idx, col] += random.uniform(-2, 2)
                    else:
                        W[idx, col] += random.uniform(-2, 2)
        
        # Check if we found a valid solution
        if best_U is not None:
            U_int = np.round(best_U)
            V_int = np.round(best_V)
            W_int = np.round(best_W)
            if verify(U_int, V_int, W_int):
                return U_int, V_int, W_int
            U_half = np.round(best_U * 2) / 2
            V_half = np.round(best_V * 2) / 2
            W_half = np.round(best_W * 2) / 2
            if verify(U_half, V_half, W_half):
                return U_half, V_half, W_half
        
        return None

    # Try ranks in increasing order, starting from the theoretical minimum
    # For 3x3x4, the rank is at least max(n*m, m*p, p*n) = 12, but likely higher.
    # Known results suggest rank 20-24 might be achievable.
    for r in [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]:
        if time.time() > deadline - 5:
            break
        
        result = _try_rank(r, time_budget=min(20, deadline - time.time() - 5))
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: Schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-END
