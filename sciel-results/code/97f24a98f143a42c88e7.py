"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def solve_instance(values, time_budget):
        n = len(values)
        if n == 0:
            return []
        
        # Build conflict graph
        # Edge (i, j) means values[i] + values[j] is in the set
        # We need to find a max independent set in the "sum graph"
        # Actually, the constraint is: for all a, b in S, a+b not in S
        # This is not a standard independent set problem directly.
        
        # Let's use a greedy + local search approach
        # Start with a good initial solution
        # Then iteratively add/remove elements to improve
        
        # Initial solution: odd numbers (sum-free)
        S = set(x for x in values if x % 2 != 0)
        
        # Also try: numbers > n/2 (if all positive, but we have negatives)
        # Better: try multiple initializations and pick the best
        
        best_S = set()
        best_size = 0
        
        # Try odd numbers
        S_odd = set(x for x in values if x % 2 != 0)
        if is_sum_free(S_odd):
            best_S = S_odd
            best_size = len(S_odd)
        
        # Try positive numbers
        S_pos = set(x for x in values if x > 0)
        if is_sum_free(S_pos):
            if len(S_pos) > best_size:
                best_S = S_pos
                best_size = len(S_pos)
        
        # Try negative numbers
        S_neg = set(x for x in values if x < 0)
        if is_sum_free(S_neg):
            if len(S_neg) > best_size:
                best_S = S_neg
                best_size = len(S_neg)
        
        # Try large magnitude numbers
        S_large = set(x for x in values if abs(x) > n // 2)
        if is_sum_free(S_large):
            if len(S_large) > best_size:
                best_S = S_large
                best_size = len(S_large)
        
        # Local search: try to add elements from values not in best_S
        # and remove elements that cause conflicts
        remaining = set(values) - best_S
        current_S = set(best_S)
        current_size = len(current_S)
        
        # Greedy addition
        changed = True
        while changed and time.time() < start_time + time_budget:
            changed = False
            # Try adding each remaining element
            for x in list(remaining):
                # Check if adding x keeps sum-free property
                # For all a in current_S: a+x not in current_S
                # For all a in current_S: x+a not in current_S (same)
                # For a=b=x: 2x not in current_S
                # For a in current_S, b=x: a+x not in current_S
                # For a=x, b in current_S: x+a not in current_S (same)
                conflict = False
                for a in current_S:
                    if a + x in current_S:
                        conflict = True
                        break
                if not conflict and 2*x not in current_S:
                    current_S.add(x)
                    remaining.discard(x)
                    current_size += 1
                    changed = True
                    if current_size > best_size:
                        best_S = set(current_S)
                        best_size = current_size
        
        # Local search: try swapping
        # For each x in current_S and y in remaining, try swapping
        for _ in range(50):  # limit iterations
            if time.time() > start_time + time_budget:
                break
            improved = False
            for x in list(current_S):
                for y in list(remaining):
                    new_S = set(current_S)
                    new_S.discard(x)
                    new_S.add(y)
                    if is_sum_free(new_S) and len(new_S) > len(current_S):
                        current_S = new_S
                        remaining.discard(y)
                        remaining.add(x)
                        current_size = len(current_S)
                        if current_size > best_size:
                            best_S = set(current_S)
                            best_size = current_size
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        return list(best_S)

    results = []
    time_per_instance = (DEADLINE - time.time()) / 3.0
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > DEADLINE:
            # Return best effort
            results.append([])
            continue
        values = instance(size, seed)
        time_budget = min(time_per_instance, DEADLINE - time.time())
        if time_budget < 1:
            time_budget = 0.5
        subset = solve_instance(values, time_budget)
        results.append(subset)
    
    return results
# EVOLVE-BLOCK-END
