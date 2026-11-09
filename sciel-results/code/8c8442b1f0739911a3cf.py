import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    N = 51975
    # Get divisors
    divisors = [d for d in range(2, N + 1) if N % d == 0]
    # Sort? Greedy usually works better with small moduli first?
    # Seed used increasing order.
    
    # counts[x] = number of classes covering x
    counts = [0] * N
    uncovered = N
    
    # current_r[m] = chosen residue for m
    current_r = {}
    
    # Precompute ranges? 
    # range(r, N, m) is fast enough.
    
    # 1. Greedy Initialization
    # Sort divisors by size? 
    # Smaller moduli cover more. 
    # Seed did increasing order.
    for m in divisors:
        best_r = 0
        best_gain = -1
        # For each r, calculate gain (number of currently uncovered residues covered)
        # gain(r) = sum(1 for x in range(r, N, m) if counts[x] == 0)
        # Optimization: 
        # We can just pick the best r.
        # Since m is small for first few, this is fast.
        # For large m, range is small.
        for r in range(m):
            gain = 0
            # Iterate
            # This loop is O(N/m * m) = O(N) for each m? No.
            # Total cost of greedy init: sum_{m} (m * N/m) = sum_{m} N = 47 * N.
            # 47 * 52000 ~ 2.4M operations. Very fast.
            for x in range(r, N, m):
                if counts[x] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_r = r
        
        current_r[m] = best_r
        # Update counts and uncovered
        for x in range(best_r, N, m):
            counts[x] += 1
            if counts[x] == 1:
                uncovered -= 1
                
    best_uncovered = uncovered
    best_config = {m: r for m, r in current_r.items()}
    
    # 2. Local Search / Simulated Annealing
    # Temperature schedule
    T = 1.0
    cooling = 0.99
    
    while time.time() < deadline:
        if T < 0.1:
            break # Or reset?
            
        m = random.choice(divisors)
        r_old = current_r[m]
        r_new = random.randint(0, m - 1)
        if r_new == r_old:
            continue
            
        # Calculate delta
        # A: x in range(r_old, N, m) where counts[x] == 1
        # B: x in range(r_new, N, m) where counts[x] == 0
        # Note: ranges are disjoint.
        
        # We can combine the iteration?
        # No, just two loops.
        delta = 0
        
        # Loop 1: Remove old
        for x in range(r_old, N, m):
            if counts[x] == 1:
                delta += 1 # Becomes uncovered
                
        # Loop 2: Add new
        for x in range(r_new, N, m):
            if counts[x] == 0:
                delta -= 1 # Becomes covered
                
        # Acceptance criterion
        accept = False
        if delta < 0:
            accept = True
        elif delta == 0:
            accept = random.random() < 0.5
        else:
            # Simulated Annealing
            # Prob = exp(-delta / T)
            # delta is positive
            accept = random.random() < pow(2.718281828, -delta / T)
            
        if accept:
            # Apply changes
            for x in range(r_old, N, m):
                counts[x] -= 1
                if counts[x] == 0:
                    uncovered += 1
            for x in range(r_new, N, m):
                counts[x] += 1
                if counts[x] == 1:
                    uncovered -= 1
                    
            current_r[m] = r_new
            
            if uncovered < best_uncovered:
                best_uncovered = uncovered
                best_config = {m: r for m, r in current_r.items()}
                
        T *= cooling
        
    # Convert best_config to list of [m, r]
    result = []
    for m, r in best_config.items():
        result.append([m, r])
        
    return result
