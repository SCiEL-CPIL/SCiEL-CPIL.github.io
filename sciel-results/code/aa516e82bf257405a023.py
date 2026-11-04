import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 115s limit, leave 5s margin for overhead and output
    
    n, m, p = DIMENSIONS
    # Dimensions of U, V, W
    dim_U = n * m  # 12
    dim_V = m * p  # 28
    dim_W = p * n  # 21
    
    # Total number of equations: n * m * p = 84
    num_eqs = n * m * p
    
    # Lower bound for rank
    min_rank = max(dim_U, dim_V, dim_W) # 28
    
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

    # Pre-compute indices for equations
    # Equation (i, j, k) corresponds to index eq_idx
    # Variables are W[k*n+i, c]
    # We can vectorize the construction of A for given U, V
    
    # Strategy:
    # 1. Try ranks from min_rank upwards.
    # 2. For each rank, try random integer initializations for U and V.
    # 3. Solve for W using least squares.
    # 4. Check if W is close to half-integers.
    # 5. If so, verify with exact Fraction arithmetic.
    
    # Optimization:
    # Instead of building A in a loop, we can use broadcasting if we reshape.
    # But U and V change every iteration.
    # A[i,j,k, row_W, c] = U[i*m+j, c] * V[j*p+k, c]
    # This is a bit complex to vectorize directly without large intermediate tensors.
    # dim_U=12, dim_V=28, dim_W=21.
    # r is small (28-83).
    # num_eqs = 84.
    # A size: 84 x (21*r).
    # For r=50, A is 84 x 1050.
    # Building A in Python loop: 84 * 50 = 4200 ops. Very fast.
    # Solving lstsq: 84x1050. Fast.
    
    # The bottleneck is the number of tries.
    # We can increase the number of tries per rank if we are fast.
    
    # Let's implement a helper to build A and b
    def build_system(U, V, r):
        A = np.zeros((num_eqs, dim_W * r))
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row_W = k * n + i
                    # Vectorized update for this equation
                    # A[eq_idx, row_W*r : row_W*r+r] = U[i*m+j, :] * V[j*p+k, :]
                    A[eq_idx, row_W*r : (row_W+1)*r] = U[i*m+j, :] * V[j*p+k, :]
                    eq_idx += 1
        b = np.ones(num_eqs)
        return A, b

    def verify(U, V, W2, r, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(r):
                        total += U[i*m+j, c] * V[j*p+k, c] * W2[k*n+i, c]
                    if total != 2:
                        return False
        return True

    # Search for better rank
    # We start from min_rank.
    # We limit the total number of attempts or time per rank.
    
    # Known lower bounds:
    # For 3x4x7, the rank is likely > 28.
    # We'll try to find any rank < 84.
    
    # To improve search efficiency, we can use random integers from a larger set?
    # Or stick to {-1, 0, 1} which is good for sparsity and exactness.
    # Maybe {-1, 1} is better? Or {-2, -1, 0, 1, 2}?
    # Let's stick to {-1, 0, 1} first, but maybe try {-1, 1} as well.
    
    # Let's iterate ranks.
    # We will allocate time budget to each rank.
    # Lower ranks are harder, so maybe give them more tries?
    # Or just do a fixed number of tries per rank.
    
    # Let's try to be smarter.
    # We can stop early if we find a solution.
    
    # To maximize chance of finding a solution, we should try ranks that are "plausible".
    # But we don't know the minimal rank.
    # Let's just sweep from min_rank to 83.
    
    # Improved Search Strategy:
    # 1. Use sparse random initialization for U and V.
    # 2. Focus on ranks 27 to 40 first, as the minimal rank is likely in this range.
    # 3. Increase the number of tries for lower ranks.
    # 4. Use a tighter tolerance for the half-integer check.
    
    # We will try ranks from 27 to 83.
    # Note: min_rank is 28, but we can try 27 just in case (though lower bound is 28).
    # Actually, let's stick to 28 to be safe with the lower bound.
    
    for r in range(min_rank, 84):
        if time.time() > deadline - 5:
            break
        
        # Determine number of tries
        # More tries for lower ranks
        if r < 35:
            num_tries = 200
        elif r < 50:
            num_tries = 50
        else:
            num_tries = 10
            
        for _ in range(num_tries):
            if time.time() > deadline - 2:
                break
            
            # Sparse Random Initialization
            # Density 0.1 to 0.3
            density = np.random.uniform(0.1, 0.3)
            U = np.random.choice([-1, 1], size=(dim_U, r)) * (np.random.rand(dim_U, r) < density)
            V = np.random.choice([-1, 1], size=(dim_V, r)) * (np.random.rand(dim_V, r) < density)
            
            # Ensure U and V are not all zeros
            if np.all(U == 0) or np.all(V == 0):
                continue
            
            # Build system
            A, b = build_system(U, V, r)
            
            # Solve
            try:
                W_flat, residuals, rank_sol, s = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                continue
                
            # Check residual
            if residuals.size > 0 and residuals[0] > 1e-8:
                continue
                
            # Reshape
            W_sol = W_flat.reshape((dim_W, r))
            
            # Check half-integer
            W_scaled = W_sol * 2
            W_rounded = np.round(W_scaled)
            
            # Tighter tolerance
            if np.max(np.abs(W_scaled - W_rounded)) < 1e-6:
                # Potential candidate
                W_int = W_rounded / 2.0
                
                # Verify with Fraction
                if verify(U, V, W_rounded, r, n, m, p):
                    # Valid!
                    result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W_int.tolist()
                    }
                    return result

    # If no better rank found, return schoolbook
    return get_schoolbook()

# EVOLVE-BLOCK-START
# The construct function is defined above. 
# We need to ensure the interface is correct.
# The prompt says: Interface: construct() -> {'U', 'V', 'W'}
# My construct returns exactly that.
# EVOLVE-BLOCK-END
