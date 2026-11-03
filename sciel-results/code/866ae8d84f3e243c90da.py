"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START

def compute_max_overlap(h):
    """Compute max over shifts s of sum_i h_i * (1 - h_{i+s}) for i where both exist."""
    N = len(h)
    max_ov = 0.0
    for s in range(1, N):
        ov = np.sum(h[:N-s] * (1.0 - h[s:]))
        if ov > max_ov:
            max_ov = ov
    return max_ov

def construct():
    deadline = time.time() + 140.0  # 10s safety margin
    np.random.seed(42)
    
    # Start with Thue-Morse inspired binary sequence
    # Thue-Morse: t(n) = popcount(n) % 2
    h = np.zeros(N)
    for i in range(N):
        # Thue-Morse bit
        cnt = bin(i).count('1')
        h[i] = 1.0 if cnt % 2 == 1 else 0.0
    
    # Ensure sum is N/2 (it should be approximately, but let's check)
    # Thue-Morse has exactly N/2 ones for N power of 2, close for others
    # Adjust: if sum != N/2, fix by flipping
    target_sum = N / 2.0
    current_sum = h.sum()
    
    # If not exactly balanced, adjust
    if abs(current_sum - target_sum) > 1e-10:
        diff = target_sum - current_sum
        if diff > 0:
            # Need more ones: flip zeros to ones
            zero_indices = np.where(h == 0.0)[0]
            for idx in zero_indices[:int(diff)]:
                h[idx] = 1.0
        else:
            one_indices = np.where(h == 1.0)[0]
            for idx in one_indices[:int(-diff)]:
                h[idx] = 0.0
    
    best_h = h.copy()
    best_score = compute_max_overlap(best_h)
    
    # Local search: try flipping individual bits or small perturbations
    step_size = 0.1
    lr = 0.3
    
    while time.time() < deadline:
        improved = False
        # Try perturbing each position
        for i in range(N):
            if time.time() > deadline:
                break
            old_val = h[i]
            
            # Try small perturbations
            for delta in [-lr, lr, -step_size, step_size, 1.0 - old_val, 0.0, 1.0]:
                new_val = np.clip(old_val + delta, 0.0, 1.0)
                if new_val == old_val:
                    continue
                h[i] = new_val
                
                # Check sum constraint
                s = h.sum()
                if abs(s - target_sum) > 1e-6:
                    h[i] = old_val
                    continue
                
                score = compute_max_overlap(h)
                if score < best_score - 1e-9:
                    best_h = h.copy()
                    best_score = score
                    improved = True
                else:
                    h[i] = old_val
        
        if not improved:
            # Try random restarts or larger moves
            if np.random.random() < 0.3:
                # Random perturbation of a few positions
                for _ in range(3):
                    i = np.random.randint(N)
                    h[i] = np.clip(h[i] + np.random.uniform(-0.5, 0.5), 0.0, 1.0)
                # Fix sum
                diff = target_sum - h.sum()
                if abs(diff) > 1e-6:
                    # Distribute correction
                    for i in range(N):
                        if abs(diff) < 1e-6:
                            break
                        if diff > 0:
                            room = 1.0 - h[i]
                            adj = min(room, diff)
                            h[i] += adj
                            diff -= adj
                        else:
                            room = h[i]
                            adj = max(-room, diff)
                            h[i] += adj
                            diff -= adj
                
                score = compute_max_overlap(h)
                if score < best_score:
                    best_h = h.copy()
                    best_score = score
    
    return best_h

# EVOLVE-BLOCK-END
