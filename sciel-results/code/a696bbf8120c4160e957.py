import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, leave margin for overhead
    
    n, m, p = DIMENSIONS
    # Dimensions of U, V, W
    dim_U = n * m  # 12
    dim_V = m * p  # 28
    dim_W = p * n  # 21
    
    # Total number of equations: n * m * p = 84
    num_eqs = n * m * p
    
    # Lower bound for rank
    min_rank = max(dim_U, dim_V, dim_W) # 28
    
    # We will search for ranks starting from min_rank
    # Known schoolbook is 84.
    # We hope to find something lower.
    
    # Pre-compute indices for the linear system construction
    # Equation for (i, j, k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # Variables are W[k*n+i, c]
    # Coefficient for variable W[k*n+i, c] in equation (i,j,k) is U[i*m+j, c] * V[j*p+k, c]
    
    # We will try different ranks
    # Given the complexity, we might not find a solution for r < 84 easily.
    # However, we must try.
    
    # Fallback: Schoolbook
    def get_schoolbook():
        rank = num_eqs
        U = np.zeros((dim_U, rank))
        V = np.zeros((dim_V, rank))
        W = np.zeros((dim_W, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Search for better rank
    best_result = None
    
    # Try ranks from 28 to 83
    # To save time, we might skip some or limit iterations per rank
    max_rank_to_try = 83
    
    for r in range(min_rank, max_rank_to_try + 1):
        if time.time() > deadline - 10:
            break
            
        # Number of random tries per rank
        # Higher ranks are easier to solve (more variables), so fewer tries needed?
        # Lower ranks are harder.
        num_tries = 20 if r < 50 else 5
        
        for _ in range(num_tries):
            if time.time() > deadline - 5:
                break
                
            # Initialize U and V with random integers {-1, 0, 1}
            U = np.random.choice([-1, 0, 1], size=(dim_U, r))
            V = np.random.choice([-1, 0, 1], size=(dim_V, r))
            
            # Build the linear system A W = b
            # A has shape (num_eqs, dim_W * r)
            # b has shape (num_eqs,)
            A = np.zeros((num_eqs, dim_W * r))
            b = np.ones(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Equation index eq_idx
                        # Variables are W[row, c] where row = k*n+i, c in 0..r-1
                        # Coeff for W[k*n+i, c] is U[i*m+j, c] * V[j*p+k, c]
                        row_W = k * n + i
                        for c in range(r):
                            A[eq_idx, row_W * r + c] = U[i*m+j, c] * V[j*p+k, c]
                        eq_idx += 1
            
            # Solve A W_flat = b
            # Use least squares
            try:
                W_flat, residuals, rank_sol, s = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                continue
                
            # Check residual
            if residuals.size > 0 and residuals[0] > 1e-9:
                continue
                
            # Reshape W_flat to W
            W_sol = W_flat.reshape((dim_W, r))
            
            # Check if W_sol is close to half-integers
            # Multiply by 2 and check closeness to integers
            W_scaled = W_sol * 2
            W_rounded = np.round(W_scaled)
            
            # Check error after rounding
            if np.max(np.abs(W_scaled - W_rounded)) < 1e-6:
                # Candidate found!
                W_int = W_rounded / 2.0
                
                # Verify exactness with integer/half-integer arithmetic
                # We should use exact arithmetic if possible, or high precision float
                # Since entries are small, float64 should be exact for integers up to 2^53
                is_valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # Calculate sum
                            val = 0.0
                            for c in range(r):
                                val += U[i*m+j, c] * V[j*p+k, c] * W_int[k*n+i, c]
                            if abs(val - 1.0) > 1e-9:
                                is_valid = False
                                break
                    if not is_valid:
                        break
                if not is_valid:
                    break
                    
                if is_valid:
                    # Convert to lists for output
                    result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W_int.tolist()
                    }
                    # We found a valid decomposition with rank r
                    # Since we search from low to high, this is the minimal found
                    return result
            else:
                # Not close to half-integers. 
                # We could try to project to nearest half-integer and check, 
                # but it's likely to fail.
                pass

    # If no better rank found, return schoolbook
    return get_schoolbook()

# EVOLVE-BLOCK-START
# The construct function is defined above. 
# We need to ensure the interface is correct.
# The prompt says: Interface: construct() -> {'U', 'V', 'W'}
# My construct returns exactly that.
# EVOLVE-BLOCK-END
