"""Model-and-optimize approach to pack distinct multiples of fixed divisors."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd, exp
from functools import reduce

P = {
    'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39],
    'maximum': 1000000
}

def construct():
    divisors = P['divisors']
    n = len(divisors)
    MAX_V = P['maximum']
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Precompute LCM of all divisors
    def lcm(a, b):
        return a * b // gcd(a, b)
    LCM = reduce(lcm, divisors)
    
    # Initial solution: use the divisors themselves
    naive = divisors[:]
    best_val = max(naive) - min(naive)
    best_assign = naive[:]
    
    def check_window(W, L, divisors, deadline):
        """Check if window [L, L+W] has a valid distinct assignment."""
        if L < 1 or L + W > MAX_V:
            return None
        
        # For each divisor, compute the list of offsets (relative to L) where multiples fall
        candidate_offsets = []
        for d in divisors:
            p = L % d
            first_offset = (d - p) % d
            if first_offset > W:
                return None  # No multiple in window
            offsets = list(range(first_offset, W + 1, d))
            candidate_offsets.append(offsets)
        
        # Convert to sets for O(1) membership testing
        candidate_offsets_set = [set(offsets) for offsets in candidate_offsets]
        
        # Use backtracking with MRV and forward checking
        assignment = [None] * n
        used_positions = set()
        
        # Order divisors by number of available positions (MRV)
        order = sorted(range(n), key=lambda i: len(candidate_offsets[i]))
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return True
            
            i = order[idx]
            # Available positions for divisor i (not used)
            avail = [j for j in candidate_offsets[i] if j not in used_positions]
            
            if not avail:
                return False
            
            # Sort by how many other unassigned divisors also want this position
            # We use the set for faster lookup
            def conflict_count(pos):
                count = 0
                # Only consider divisors that are not yet assigned (indices > idx in order)
                for k in range(idx + 1, n):
                    if pos in candidate_offsets_set[k]:
                        count += 1
                return count
            
            avail.sort(key=conflict_count)
            
            for j in avail:
                used_positions.add(j)
                assignment[i] = L + j
                if backtrack(idx + 1):
                    return True
                used_positions.discard(j)
                assignment[i] = None
            
            return False
        
        if backtrack(0):
            return assignment[:]
        return None
    
    # Systematic search over W from theoretical minimum
    for W in range(9, 100):
        if time.time() > deadline:
            break
        
        max_L = min(LCM, MAX_V - W)
        
        # Try all L values from 1 to max_L
        found = False
        for L in range(1, max_L + 1):
            if time.time() > deadline:
                break
            
            result = check_window(W, L, divisors, deadline)
            if result is not None:
                span = max(result) - min(result)
                if span < best_val:
                    best_val = span
                    best_assign = result[:]
                    found = True
                    break
        
        if found and best_val == W:
            break
    
    # Local search to polish using Simulated Annealing
    if best_assign is not None:
        current_sol = best_assign[:]
        current_cost = max(current_sol) - min(current_sol)
        best_sol = current_sol[:]
        best_cost = current_cost
        
        start_temp = 50.0
        end_temp = 0.1
        cooling_rate = 0.99
        temp = start_temp
        
        # Run SA until deadline or temperature drops
        while temp > end_temp and time.time() < deadline:
            # Pick a random index to change
            idx = random.randint(0, n - 1)
            d = divisors[idx]
            old_val = current_sol[idx]
            
            # Generate a new value: random multiple of d near old_val
            # Step size: random between -20 and 20 multiples
            delta = random.randint(-20, 20)
            new_val = old_val + delta * d
            
            # Ensure bounds
            if new_val < 1 or new_val > MAX_V:
                continue
                
            # Check distinctness
            if any(current_sol[j] == new_val for j in range(n) if j != idx):
                continue
                
            # Calculate new cost
            new_sol = current_sol[:]
            new_sol[idx] = new_val
            new_cost = max(new_sol) - min(new_sol)
            
            delta_cost = new_cost - current_cost
            
            # Accept if better, or with probability exp(-delta_cost / temp)
            if delta_cost < 0 or random.random() < exp(-delta_cost / temp):
                current_sol = new_sol
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_sol = current_sol[:]
            
            temp *= cooling_rate
        
        if best_cost < best_val:
            best_val = best_cost
            best_assign = best_sol
    
    # Final validation
    if best_assign is None:
        best_assign = divisors[:]
    
    result = []
    for i, d in enumerate(divisors):
        v = int(best_assign[i])
        result.append(v)
    
    # Verify all constraints
    for i, d in enumerate(divisors):
        assert isinstance(result[i], int) and not isinstance(result[i], bool)
        assert 1 <= result[i] <= MAX_V
        assert result[i] % d == 0
    assert len(set(result)) == n
    
    return result

# EVOLVE-BLOCK-END
