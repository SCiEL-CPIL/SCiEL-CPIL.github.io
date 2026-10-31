import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    best_x = None
    best_score = float('inf')
    
    # Initial candidates
    candidates = []
    candidates.append(np.ones(N))
    candidates.append(np.random.choice([-1, 1], N))
    candidates.append(np.tile([1, -1], N//2))
    
    for x in candidates:
        score = compute_score(x)
        if score < best_score:
            best_score = score
            best_x = x.copy()
            
    # Local search
    # Use a random start for the main search to avoid local minima of all-ones
    x = np.random.choice([-1, 1], N)
    c = compute_conv(x)
    max_side = np.max(np.abs(c[1:N]))
    
    while time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            # Flip x[i]
            x[i] *= -1
            new_c = compute_conv(x)
            new_max_side = np.max(np.abs(new_c[1:N]))
            
            if new_max_side < max_side:
                max_side = new_max_side
                improved = True
                # Update best if better than global
                if max_side / N < best_score:
                    best_score = max_side / N
                    best_x = x.copy()
            else:
                x[i] *= -1 # Revert
                
        if not improved:
            # Random restart or perturbation
            x = np.random.choice([-1, 1], N)
            c = compute_conv(x)
            max_side = np.max(np.abs(c[1:N]))
            
    return best_x
