import random
import time

def construct():
    deadline = time.time() + 50
    best_set = set(range(66, 101))
    best_size = len(best_set)
    
    def is_valid(s):
        # Check if any subset sums to 200
        # Since max element is 100, only subsets of size 2, 3, 4 can sum to 200?
        # Min sum of 5: 66+67+68+69+70 = 340 > 200.
        # So only check subsets of size 2, 3, 4.
        # Actually, for general sets, we should check all.
        # But for sets of large numbers, small subsets.
        # For robustness, use a DP or bitset if N is small, but N=100, target=200.
        # Bitset DP is fast.
        reachable = 1 # bit 0 set
        for x in s:
            reachable |= (reachable << x)
        return (reachable >> 200) & 1 == 0

    # Initial check
    if not is_valid(best_set):
        # Fallback to greedy if my hardcoded set is invalid (it shouldn't be)
        selected, reachable = [], {0}
        for x in range(100, 0, -1):
            new = {s + x for s in reachable if s + x <= 200}
            if 200 not in new:
                selected.append(x)
                reachable |= new
        return selected

    # Local search
    candidates = list(range(1, 66))
    random.shuffle(candidates)
    
    for x in candidates:
        if time.time() > deadline:
            break
        # Try adding x
        new_set = best_set | {x}
        if is_valid(new_set):
            best_set = new_set
            best_size += 1
        else:
            # Try to repair by removing one element from best_set
            # Find conflicts involving x
            # Simple approach: try removing each y in best_set and see if valid
            # This is O(N * CostValidity). CostValidity is fast.
            for y in list(best_set):
                if time.time() > deadline:
                    break
                candidate_set = (best_set - {y}) | {x}
                if is_valid(candidate_set):
                    best_set = candidate_set
                    # Size remains same, but composition changed, might allow future additions
                    break
    
    return list(best_set)
