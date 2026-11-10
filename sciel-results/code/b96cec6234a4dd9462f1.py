"""Optimized construction for 'no element divides sum of 3 distinct others'."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    t_start = time.time()
    deadline = t_start + 55.0  # safety margin before 60s limit
    
    # Precompute for each candidate x in [1,192], which subsets of 3 distinct
    # numbers from [1,192] have a sum divisible by x.
    # We'll use a different approach: greedy with backtracking / local search.
    
    # Strategy: 
    # 1. Start with a good initial set (larger numbers tend to be less likely to divide sums)
    # 2. Try to add more elements via local search
    # 3. Use randomized restarts if time permits
    
    def is_valid_set(S):
        """Check if set S satisfies the constraint."""
        S_list = list(S)
        for i, a in enumerate(S_list):
            # Check if a divides sum of any 3 distinct OTHER elements
            others = [S_list[j] for j in range(len(S_list)) if j != i]
            for t in combinations(others, 3):
                if sum(t) % a == 0:
                    return False
        return True
    
    def can_add(S, x):
        """Check if x can be added to set S without violating constraints."""
        S_list = list(S)
        # Check: does x divide sum of any 3 distinct others in S?
        for t in combinations(S_list, 3):
            if sum(t) % x == 0:
                return False
        # Check: for each a in S, does a divide sum of x + any 2 distinct others in S?
        for i, a in enumerate(S_list):
            others = [S_list[j] for j in range(len(S_list)) if j != i]
            for pair in combinations(others, 2):
                if (x + pair[0] + pair[1]) % a == 0:
                    return False
        return True
    
    def greedy_construct(order):
        """Greedy construction following a given order of candidates."""
        selected = []
        for x in order:
            if time.time() > deadline:
                break
            if can_add(selected, x):
                selected.append(x)
        return selected
    
    # Initial greedy: try descending order (larger numbers are safer)
    best = greedy_construct(range(192, 0, -1))
    
    # Try ascending order
    if time.time() < deadline:
        alt = greedy_construct(range(1, 193))
        if len(alt) > len(best):
            best = alt
    
    # Local search: try to swap elements to increase size
    if time.time() < deadline:
        import random
        random.seed(42)
        
        # Try multiple random orderings
        for attempt in range(50):
            if time.time() > deadline:
                break
            order = list(range(1, 193))
            random.shuffle(order)
            # Bias towards larger numbers
            order.sort(key=lambda x: x * 0.7 + random.random() * 192)
            result = greedy_construct(order)
            if len(result) > len(best):
                best = result
    
    # Try to extend the best set by removing one element and adding two
    if time.time() < deadline:
        best_set = set(best)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(best)):
                if time.time() > deadline:
                    break
                # Try removing best[i] and adding two new elements
                S_without = [x for j, x in enumerate(best) if j != i]
                removed = best[i]
                
                # Find candidates that could be added
                candidates = [x for x in range(1, 193) if x not in S_without and x != removed]
                
                # Try adding one candidate first
                for c1 in candidates:
                    if time.time() > deadline:
                        break
                    if can_add(S_without, c1):
                        S_temp = S_without + [c1]
                        # Try adding a second candidate
                        for c2 in candidates:
                            if c2 == c1:
                                continue
                            if time.time() > deadline:
                                break
                            if can_add(S_temp, c2):
                                new_set = S_temp + [c2]
                                if len(new_set) > len(best):
                                    best = new_set
                                    improved = True
                                    break
                        if improved:
                            break
                if improved:
                    break
    
    # Final verification
    if not is_valid_set(best):
        # Fallback to simple greedy
        best = []
        for x in range(192, 0, -1):
            if can_add(best, x):
                best.append(x)
    
    return best
# EVOLVE-BLOCK-END
