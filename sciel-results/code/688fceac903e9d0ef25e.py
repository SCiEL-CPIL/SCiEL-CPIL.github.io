"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = 30
    
    # Calculate LCM(1..30)
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    lcm_val = 1
    for i in range(1, n + 1):
        lcm_val = lcm(lcm_val, i)
        
    # Precompute weights
    weights = [lcm_val // (i + 1) for i in range(n)]
    
    # Initial state: all zeros
    coeffs = [0] * n
    current_sum = 0
    best_sum = None
    best_coeffs = None
    
    # Helper to update sum and check
    def try_update(idx, new_val):
        nonlocal current_sum, best_sum, best_coeffs, coeffs
        old_val = coeffs[idx]
        if old_val == new_val:
            return False
        
        delta = (new_val - old_val) * weights[idx]
        new_sum = current_sum + delta
        
        if new_sum == 0:
            # We must avoid zero. 
            # If we are currently at a non-zero best, we don't want to move to zero.
            # If we are currently at zero (start), we must move away.
            # Actually, the problem states "signed sum ... must be nonzero".
            # So we cannot return a state where sum is 0.
            # We can pass through 0 in search? No, if we land on 0, it's an invalid state for the final answer.
            # However, local search can traverse through 0 if we just reject it as a *final* candidate.
            # But if we reject the move, we stay at old state.
            # Let's just reject moves that result in sum == 0.
            return False
        
        if best_sum is None or abs(new_sum) < abs(best_sum):
            best_sum = new_sum
            # Deep copy is needed if we want to save the exact config
            best_coeffs = coeffs.copy()
            # Note: we haven't updated coeffs[idx] yet in this function logic if we return True?
            # Let's update coeffs and current_sum if we accept the move.
            coeffs[idx] = new_val
            current_sum = new_sum
            return True
        else:
            # If it's not better, we might still want to accept it for diversity?
            # Standard hill climbing: only accept if better.
            # To escape local minima, we could add simulated annealing, but let's keep it simple first.
            # Given the landscape is sparse, simple hill climbing from multiple starts might work.
            return False

    # Run local search from multiple random starts
    num_starts = 200000 # High number, but each step is O(1)
    # Estimate steps per start: 1000 steps
    # Total ops: 200M. Might be too slow in Python?
    # Let's optimize the loop.
    
    # Alternative: One long run with random restarts if stuck.
    
    # Let's do a loop that runs until deadline.
    # We will perform random restarts every few seconds or if stuck for a while.
    
    global_best_sum = None
    global_best_coeffs = None
    
    while time.time() < deadline:
        # Random start
        coeffs = [random.randint(-1, 1) for _ in range(n)]
        # Ensure non-zero sum? If zero, flip a random bit.
        current_sum = sum(c * w for c, w in zip(coeffs, weights))
        if current_sum == 0:
            # Flip first non-zero? Or just randomize again.
            # Let's just randomize one element to break zero.
            idx = random.randint(0, n - 1)
            coeffs[idx] = 1 if coeffs[idx] != 1 else -1
            current_sum = sum(c * w for c, w in zip(coeffs, weights))
            if current_sum == 0:
                # Extremely unlikely to be zero again, but if so, skip this start
                continue
        
        # Local search
        steps = 0
        max_steps = 5000
        while steps < max_steps and time.time() < deadline:
            steps += 1
            idx = random.randint(0, n - 1)
            # Choose a new value different from current
            curr_val = coeffs[idx]
            if curr_val == 0:
                new_val = random.choice([-1, 1])
            else:
                # curr_val is -1 or 1
                # Options: 0 and the other sign
                options = [v for v in [-1, 0, 1] if v != curr_val]
                new_val = random.choice(options)
            
            old_val = coeffs[idx]
            delta = (new_val - old_val) * weights[idx]
            new_sum = current_sum + delta
            
            if new_sum == 0:
                continue
            
            # Accept if better or with some probability (Simulated Annealing light?)
            # Let's stick to greedy for speed, but maybe allow neutral moves?
            # No, neutral moves don't help minimize.
            # Let's try to accept if abs(new_sum) <= abs(current_sum) to allow plateaus?
            # No, strictly better is safer.
            
            if global_best_sum is None or abs(new_sum) < abs(global_best_sum):
                global_best_sum = new_sum
                global_best_coeffs = coeffs.copy()
                coeffs[idx] = new_val
                current_sum = new_sum
            else:
                # Try to accept if it's a "small" increase to escape local minima?
                # For now, let's just restart if we don't improve for a while.
                # Simple strategy: if no improvement in last 100 steps, break and restart.
                # We need a counter for this.
                pass
        
        # Check if we can break early if we found 1?
        # The minimum non-zero integer sum in LCM units is 1.
        if global_best_sum is not None and abs(global_best_sum) == 1:
            break

    if global_best_coeffs is None:
        # Fallback: find any non-zero sum
        # e.g. 1/1
        return [1] + [0] * 29
        
    return global_best_coeffs
# EVOLVE-BLOCK-END
