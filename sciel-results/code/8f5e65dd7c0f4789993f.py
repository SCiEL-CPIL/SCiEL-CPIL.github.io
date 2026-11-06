import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 256
    weights = [Fraction(0)] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = Fraction(1, i)
        
    # Suffix sums for pruning (exact Fraction)
    suffix = [Fraction(0)] * (N + 2)
    for i in range(N, 0, -1):
        suffix[i] = suffix[i + 1] + weights[i]
        
    # Precompute conflicts for each x
    # conflicts[x] = list of (a, b) such that a < b < x and a, b, x form a 3-AP
    # 2b = a + x => b = (a+x)/2
    conflicts = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        for a in range(1, x):
            s = a + x
            if s % 2 == 0:
                b = s // 2
                if 1 <= b < x:
                    conflicts[x].append((a, b))
                    
    # Helper to check validity with bitmask
    def is_valid_add(x, mask):
        for a, b in conflicts[x]:
            if ((mask >> (a - 1)) & 1) and ((mask >> (b - 1)) & 1):
                return False
        return True

    # Greedy Initialization
    def greedy_init():
        mask = 0
        score = Fraction(0)
        for i in range(1, N + 1):
            if is_valid_add(i, mask):
                mask |= (1 << (i - 1))
                score += weights[i]
        return mask, score

    best_mask, best_score = greedy_init()
    
    # Local Search (Hill Climbing) to improve initial bound
    # We try to flip bits. 
    # Since N=256, checking all 256 bits is fast.
    # We iterate until no improvement is found.
    improved = True
    while improved:
        improved = False
        for i in range(1, N + 1):
            bit = 1 << (i - 1)
            if best_mask & bit:
                # Try removing i
                new_mask = best_mask ^ bit
                # Removing always valid, but we check if it allows adding others? 
                # Simple hill climb: just remove if it doesn't decrease score? 
                # No, removing decreases score. We only remove if it allows adding larger weight elements?
                # That's complex. Let's stick to simple add-only or swap.
                # Actually, a 3-AP free set is not necessarily maximal.
                # Let's try a "repair" phase:
                pass
            
            else:
                # Try adding i
                if is_valid_add(i, best_mask):
                    # If we can add i, we should, because weights are positive.
                    # But wait, adding i might block future additions?
                    # In a simple local search on the final set, if we can add i, we do it.
                    # This finds a maximal set.
                    new_mask = best_mask | bit
                    new_score = best_score + weights[i]
                    # Since we are adding positive weight, new_score > best_score always.
                    # So we just keep adding until no more can be added.
                    # This is just completing the greedy to a maximal set.
                    # The greedy above already does this.
                    # So simple "add if valid" is equivalent to greedy if done in order.
                    # To improve, we need to try *different* orders or swaps.
                    
                    # Let's try a random restart with different order?
                    # Or swap: remove j, add i?
                    pass

    # A more effective local search: Randomized Greedy with Restart
    import random
    rng = random.Random(42)
    
    def randomized_greedy():
        mask = 0
        score = Fraction(0)
        # Shuffle indices? No, we must respect the structure.
        # But we can choose which valid element to add next.
        # Standard greedy adds smallest valid.
        # Let's try adding largest valid first?
        # Weights are 1/i, so smallest i has largest weight.
        # Greedy adding smallest i is optimal for maximizing weight if no conflicts.
        # With conflicts, it's a heuristic.
        
        # Let's try a few different strategies
        # Strategy 1: Standard Greedy (smallest first)
        # Strategy 2: Greedy with random tie-breaking? No ties in weights.
        # Strategy 3: Greedy from largest to smallest? (Worst weight first, bad)
        
        # Let's do a simple local search:
        # Start with greedy set.
        # Try to swap: remove x, add y?
        # Or remove x, y and add z, w?
        
        # Given time constraints, the B&B with bitmask is the main driver.
        # The initial bound from simple greedy is likely good enough.
        pass

    # B&B Search
    def dfs(idx, current_sum, mask):
        nonlocal best_score, best_mask
        
        # Time check
        if time.time() > deadline:
            return
            
        # Pruning
        if current_sum + suffix[idx] <= best_score:
            return
            
        if idx > N:
            if current_sum > best_score:
                best_score = current_sum
                best_mask = mask
            return
            
        x = idx
        
        # Option 1: Include x
        if is_valid_add(x, mask):
            new_mask = mask | (1 << (x - 1))
            dfs(idx + 1, current_sum + weights[x], new_mask)
            
        # Option 2: Exclude x
        dfs(idx + 1, current_sum, mask)

    dfs(1, Fraction(0), 0)
    
    # Convert best_mask to list
    result = []
    for i in range(1, N + 1):
        if (best_mask >> (i - 1)) & 1:
            result.append(i)
            
    return result
