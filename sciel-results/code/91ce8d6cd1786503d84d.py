import random
import math
import time

def construct():
    # Target interval [128, 256]
    target_min = 128
    target_max = 256
    n = 128
    
    # We will use Simulated Annealing to find a good subset A of [0, 128]
    # such that every integer in [128, 256] is a sum of two elements of A.
    # Objective: Minimize the maximum representation multiplicity.
    
    # Initial set size. 
    # Lower bound on size: If max multiplicity is M, total pairs >= 129.
    # k(k+1)/2 >= 129 => k ~ 16.
    # But we need to cover the range. 
    # Let's start with k=24 to be safe, and maybe allow size to vary?
    # Simpler: Fix size K and optimize membership.
    # Or allow adding/removing.
    # Let's fix K=24 for stability, or try a few K.
    
    best_A = None
    best_peak = float('inf')
    
    # Time limit 60s. We'll run for 55s.
    start_time = time.time()
    time_limit = 55.0
    
    # Try different sizes K
    for K in [20, 22, 24, 26]:
        if time.time() - start_time > time_limit:
            break
            
        # Initialize random set of size K
        # Ensure 0 and 128 are not strictly required, but helps.
        # Just random.
        A = set(random.sample(range(n + 1), K))
        
        # Precompute counts for current A
        # counts[s] for s in 128..256
        counts = [0] * (target_max + 1)
        
        def update_counts_add(x):
            # Add x to A. Update counts.
            # Pairs (x, a) for a in A, a != x
            for a in A:
                if a == x: continue
                s = x + a
                if target_min <= s <= target_max:
                    counts[s] += 1
            s = 2 * x
            if target_min <= s <= target_max:
                counts[s] += 1
            A.add(x)
            
        def update_counts_remove(x):
            # Remove x from A. Update counts.
            for a in A:
                if a == x: continue
                s = x + a
                if target_min <= s <= target_max:
                    counts[s] -= 1
            s = 2 * x
            if target_min <= s <= target_max:
                counts[s] -= 1
            A.remove(x)
            
        # Initialize counts for the random set
        # We need to be careful with the order if we use the update functions.
        # Let's just calculate from scratch initially.
        A_list = list(A)
        A_set = set(A_list)
        for s in range(target_min, target_max + 1):
            c = 0
            for a in A_list:
                b = s - a
                if b >= a and b in A_set:
                    c += 1
            counts[s] = c
            
        def get_cost():
            # Cost = Penalty for uncovered + Penalty for high peak
            # We want to minimize max_count, but first ensure min_count >= 1.
            # If any count is 0, it's invalid.
            # Let's use a large penalty for 0s.
            zero_count = 0
            max_c = 0
            for s in range(target_min, target_max + 1):
                if counts[s] == 0:
                    zero_count += 1
                elif counts[s] > max_c:
                    max_c = counts[s]
            
            # We prioritize covering all sums.
            # If zero_count > 0, cost is huge.
            # Otherwise, cost is max_c.
            if zero_count > 0:
                return zero_count * 1000 + max_c
            else:
                return max_c
        
        current_cost = get_cost()
        best_cost = current_cost
        best_A_local = set(A)
        
        # Simulated Annealing
        T = 100.0
        T_min = 1.0
        cooling_rate = 0.99999
        
        while time.time() - start_time < time_limit:
            # Pick a random move
            # Option 1: Flip an existing element (remove)
            # Option 2: Add a new element (must remove one to keep size K constant? 
            #             Or allow size to vary? 
            #             Keeping size constant is simpler and often optimal for fixed K.)
            # Let's stick to fixed size K.
            # Move: Swap an element in A with an element not in A.
            
            if len(A) == 0:
                break
                
            x_remove = random.choice(list(A))
            # Pick x_add from [0, 128] \ A
            # To be efficient, pick a random candidate and check if it's in A.
            # Or maintain a list of available elements.
            # Simple: try random candidate until not in A.
            while True:
                x_add = random.randint(0, n)
                if x_add not in A:
                    break
            
            if x_remove == x_add:
                continue
                
            # Perform move
            update_counts_remove(x_remove)
            update_counts_add(x_add)
            
            new_cost = get_cost()
            
            # Acceptance criterion
            delta = new_cost - current_cost
            if delta < 0 or (T > T_min and random.random() < math.exp(-delta / T)):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_A_local = set(A)
            else:
                # Reject, revert
                update_counts_remove(x_add)
                update_counts_add(x_remove)
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
                
        # After SA, check if we found a valid solution
        if best_cost < 1000: # Valid solution (no zeros)
            peak = best_cost
            if peak < best_peak:
                best_peak = peak
                best_A = list(best_A_local)
                
    if best_A is None:
        # Fallback: Return a known valid set, e.g., [64, ..., 128]
        # This has peak ~33.
        return list(range(64, 129))
        
    return best_A
