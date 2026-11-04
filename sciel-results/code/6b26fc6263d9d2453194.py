"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    best_h = None
    best_cost = float('inf')
    
    # Pre-allocate
    h = np.zeros(N, dtype=np.int8)
    R = np.zeros(2*N - 1, dtype=np.float64)
    
    def compute_R(h):
        # h is int8, -1 or 1
        # R[k] = sum h[i] h[i+k]
        # Use FFT for initial computation? Or direct loop?
        # Direct loop for N=400 is 160k ops, fast enough for init.
        # But we will update incrementally.
        R_local = np.zeros(2*N - 1, dtype=np.float64)
        for k in range(N):
            if k == 0:
                R_local[N-1] = N
            else:
                # R[k] and R[-k]
                # R[k] = sum_{i=0}^{N-1-k} h[i] h[i+k]
                # R[-k] = sum_{i=k}^{N-1} h[i] h[i-k] = R[k]
                R_local[N-1+k] = np.dot(h[:N-k], h[k:])
                R_local[N-1-k] = R_local[N-1+k]
        return R_local

    def get_cost(R):
        # Sum of squares of off-peak correlations
        # R[N] to R[2N-2] are lags 1 to N-1
        # R[0] to R[N-2] are lags -(N-1) to -1
        return np.sum(R[N:]**2) + np.sum(R[:N-1]**2)

    # Simulated Annealing / Hill Climbing
    rng = np.random.RandomState(42)
    
    # Number of restarts
    # We don't know how long each run takes, so loop until deadline
    restart_count = 0
    
    while time.time() < deadline:
        # Initialize
        h = rng.choice([-1, 1], size=N).astype(np.int8)
        R = compute_R(h)
        current_cost = get_cost(R)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_h = h.copy()
        
        # Simulated Annealing parameters
        T = 50.0
        T_min = 0.01
        cooling_rate = 0.995
        steps_per_run = 50000 # Adjust based on speed
        
        for _ in range(steps_per_run):
            if time.time() > deadline:
                break
            
            i = rng.randint(N)
            
            # Calculate delta cost
            # We need to update R for flipping h[i]
            # Save old R segments
            # Indices affected:
            # Pos lags: 1 to N-1-i
            # Neg lags: -1 to -i
            
            h_i = h[i]
            
            # Delta for positive lags
            # R[N-1+k] changes by -2 * h[i] * h[i+k]
            # k ranges 1 to N-1-i
            if N-1-i > 0:
                delta_pos = -2 * h_i * h[i+1:N]
                idx_pos = slice(N, N + (N-1-i))
                old_R_pos = R[idx_pos].copy()
                new_R_pos = old_R_pos + delta_pos
                cost_delta_pos = np.sum(new_R_pos**2) - np.sum(old_R_pos**2)
            else:
                cost_delta_pos = 0.0
                new_R_pos = None # Not used
            
            # Delta for negative lags
            # R[N-1-k] changes by -2 * h[i] * h[i-k]
            # k ranges 1 to i
            if i > 0:
                # h[i-1], h[i-2], ..., h[0]
                delta_neg = -2 * h_i * h[i-1::-1]
                idx_neg = slice(N-1-i, N-1)
                old_R_neg = R[idx_neg].copy()
                new_R_neg = old_R_neg + delta_neg
                cost_delta_neg = np.sum(new_R_neg**2) - np.sum(old_R_neg**2)
            else:
                cost_delta_neg = 0.0
                new_R_neg = None
            
            delta_cost = cost_delta_pos + cost_delta_neg
            
            # Acceptance criterion
            if delta_cost < 0 or rng.rand() < np.exp(-delta_cost / T):
                # Apply flip
                h[i] = -h_i
                
                # Update R
                if new_R_pos is not None:
                    R[idx_pos] = new_R_pos
                if new_R_neg is not None:
                    R[idx_neg] = new_R_neg
                
                current_cost += delta_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_h = h.copy()
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
        
        restart_count += 1

    if best_h is None:
        # Fallback to seed if no improvement found or error
        return np.ones(N)
        
    return best_h

# EVOLVE-BLOCK-END
