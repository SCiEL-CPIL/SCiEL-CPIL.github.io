import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n = 24
    # Use a dense grid for approximation of max |P(z)| on |z|=1
    # 2048 points should be sufficient to distinguish near-optimal solutions
    num_points = 2048
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    z = np.exp(1j * angles)
    
    # Precompute powers of z for fast evaluation? 
    # P(z) = sum c_k z^k. 
    # We can use np.vdot or matrix multiplication.
    # Let Z be a matrix where Z[i, k] = z_i^k.
    # Then P(z_i) = dot(c, Z[i, :]).
    # Constructing Z is O(N^2) which is 2048*24 ~ 50k ops, very fast.
    # But we can also just use iterative multiplication or np.polynomial.
    # Let's use a precomputed power matrix for speed in the inner loop.
    
    # Z[k, i] = z_i^k for k in 0..23, i in 0..num_points-1
    # Shape: (24, 2048)
    Z = np.zeros((n, num_points), dtype=np.complex128)
    Z[0, :] = 1.0
    for k in range(1, n):
        Z[k, :] = Z[k-1, :] * z
        
    def get_cost(c):
        # c is a numpy array of shape (n,) with values 1 or -1
        # P(z) = c @ Z
        p_vals = c @ Z
        return np.max(np.abs(p_vals))
    
    best_cost = float('inf')
    best_c = np.random.choice([-1, 1], n)
    
    # Initial random restarts to find a good starting point
    # and then hill climb.
    
    def hill_climb(c):
        c = c.copy()
        current_cost = get_cost(c)
        improved = True
        while improved:
            improved = False
            # Try flipping each bit
            # To make it faster, we can try random order or all. n=24 is small.
            for i in range(n):
                c[i] *= -1
                new_cost = get_cost(c)
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                else:
                    c[i] *= -1
        return c, current_cost

    # Strategy: Multiple restarts.
    # Each restart: random init -> hill climb.
    # Keep the best overall.
    
    num_restarts = 0
    while time.time() < deadline:
        # Random initialization
        c = np.random.choice([-1, 1], n)
        
        # Hill climb
        c, cost = hill_climb(c)
        
        if cost < best_cost:
            best_cost = cost
            best_c = c.copy()
        
        num_restarts += 1
        
        # If we found a very good cost, we might want to do more local searches 
        # or just continue with restarts. 
        # The optimum is ~1.2583758 * sqrt(24) ~ 6.16.
        # Let's see how close we get.
        
        # Check time again to be safe
        if time.time() > deadline - 5.0:
            break
            
    return best_c.tolist()
