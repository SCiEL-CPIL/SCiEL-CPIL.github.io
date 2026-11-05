"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    n = 256
    # Precompute reciprocal weights as Fractions for exact arithmetic
    weights = [Fraction(1, i) for i in range(n + 1)]
    
    # Precompute 3-AP constraints: for each d, list of (a, b, c) such that b-a = c-b = d
    # Actually, for bitset logic, it's better to precompute for each pair (i, j) the forbidden k.
    # If we have i and j, k = 2*j - i.
    # Let's use a bitmask representation for the set.
    # mask is an integer where bit i is 1 if i+1 is in the set.
    
    def is_valid(mask):
        # Check if mask has any 3-AP
        # For each d from 1 to n//3
        # We can iterate through all pairs (i, j) in the set and check if k=2j-i is in the set.
        # This is O(N^2) per check which is too slow for many iterations.
        # Better: Precompute for each element x, the list of (a, b) such that x = 2b - a.
        # Or simply: for each d, check if there exists a such that a, a+d, a+2d are all in set.
        # Bitset trick: 
        # Let S be the set.
        # A 3-AP exists if (S & (S >> d) & (S >> 2d)) != 0 for some d.
        # This is O(N) per d, so O(N^2) total. N=256, so 256^2 = 65536 ops. Very fast.
        for d in range(1, n // 3 + 1):
            if (mask & (mask >> d) & (mask >> 2 * d)) != 0:
                return False
        return True

    def get_score(mask):
        s = Fraction(0)
        for i in range(1, n + 1):
            if (mask >> i) & 1:
                s += weights[i]
        return s

    def count_bits(mask):
        return bin(mask).count("1")

    best_mask = 0
    best_score = Fraction(0)
    
    # Multi-start greedy
    # We will run several random greedy constructions.
    # In each, we pick elements in a random order, but prioritize high weight (low index) if valid.
    # Actually, a simple random permutation greedy is a good baseline.
    # A better one: sort by weight (1, 2, 3...) but with random tie-breaking or slight randomization?
    # No, weights are distinct. 
    # Let's try: 
    # 1. Standard greedy (1..n)
    # 2. Reverse greedy (n..1) - likely bad for reciprocal sum
    # 3. Random permutations
    
    def greedy_construct(order):
        mask = 0
        for x in order:
            new_mask = mask | (1 << x)
            if is_valid(new_mask):
                mask = new_mask
        return mask

    # Start with the original greedy (1..256)
    m = greedy_construct(range(1, n + 1))
    s = get_score(m)
    if s > best_score:
        best_score = s
        best_mask = m
        
    # Try a few random permutations
    # We have ~50s budget. is_valid is ~microseconds. get_score is ~microseconds.
    # Local search will be the main consumer.
    
    # Let's do a local search from the best greedy found so far, and from a few random ones.
    
    def local_search(mask):
        current_mask = mask
        current_score = get_score(current_mask)
        improved = True
        while improved and (time.time() < deadline - 5):
            improved = False
            # 1. Try adding an element
            for i in range(1, n + 1):
                if not (current_mask >> i) & 1:
                    new_mask = current_mask | (1 << i)
                    if is_valid(new_mask):
                        new_score = current_score + weights[i]
                        if new_score > current_score:
                            current_mask = new_mask
                            current_score = new_score
                            improved = True
                            break # Restart loop with new state
            if improved:
                continue
                
            # 2. Try swapping: remove j, add i (where i > j to ensure weight gain? No, i can be anything)
            # To gain weight, we generally want to add small i and remove large j.
            # But removing a small i might allow adding multiple larger ones? 
            # Let's stick to 1-for-1 swaps first.
            # We want to maximize sum. 
            # For each i not in set, for each j in set:
            #   if weight[i] > weight[j] and valid(remove j, add i): swap.
            # This is O(N^2) checks.
            
            # Optimization: Only consider i where weight[i] > weight[j] is possible?
            # Since weight[i] = 1/i, weight[i] > weight[j] iff i < j.
            # So we want to replace a large j with a small i.
            
            # Let's iterate i from 1 to n (small to large, high weight)
            # If i is not in set, try to find a j in set such that j > i and swap is valid.
            # If multiple j work, pick the one that minimizes loss? No, we just swap.
            # Actually, if we swap j for i, score change is weight[i] - weight[j] > 0.
            # So any valid swap with i < j improves score.
            
            for i in range(1, n + 1):
                if (current_mask >> i) & 1:
                    continue
                # Try to find a j > i in current_mask such that swap is valid
                # We can break early if we find one? 
                # If we swap, we restart. So we just need *a* valid swap.
                # To maximize improvement, we should pick the j that allows the swap?
                # Any j > i works for score increase. 
                # But some j might be "critical" for validity of other elements?
                # Let's just pick the first valid j > i.
                # To be more thorough, we could pick the j that results in the "best" future? 
                # Simple hill climbing: take the first valid swap.
                
                # Iterate j from n down to i+1? Or i+1 to n?
                # Larger j have lower weight, so removing them is "cheaper" in terms of weight lost?
                # No, we gain weight[i] and lose weight[j]. Net gain is weight[i] - weight[j].
                # Since i < j, net gain is positive.
                # Larger j means smaller weight[j], so larger net gain.
                # So we should prefer removing the largest j possible?
                # Let's iterate j from n down to i+1.
                
                found_swap = False
                for j in range(n, i, -1):
                    if (current_mask >> j) & 1:
                        new_mask = (current_mask & ~(1 << j)) | (1 << i)
                        if is_valid(new_mask):
                            current_mask = new_mask
                            current_score = current_score + weights[i] - weights[j]
                            improved = True
                            found_swap = True
                            break
                if found_swap:
                    break # Restart while loop
            
            # 3. Try 2-for-1: Remove two elements j1, j2, add one element i.
            # This is O(N^3) which is 16M ops. Might be too slow if done naively in Python loops.
            # But we can prune.
            # We only care if weight[i] > weight[j1] + weight[j2].
            # Since weights are 1/k, this is rare for large k.
            # Given time constraints, let's skip 2-for-1 for now unless 1-for-1 stalls.
            
            if not improved:
                # Try 2-for-1 if we have time
                if time.time() < deadline - 10:
                    # This is a heavy operation.
                    # We'll do a limited version.
                    pass
        
        return current_mask, current_score

    # Run local search on the initial greedy
    m, s = local_search(m)
    if s > best_score:
        best_score = s
        best_mask = m

    # Try other starting points
    # Randomized greedy: shuffle 1..n, then greedy.
    # Since weights favor small numbers, a random shuffle might pick large numbers early, blocking small ones.
    # A better random start: 
    # 1. Pick a random subset of small numbers?
    # 2. Or just run greedy on a sorted list with random perturbations?
    
    # Let's run a few "randomized greedy" starts where we pick the next element from the top 10 candidates randomly?
    # Or simpler: just run greedy on random permutations.
    # How many can we afford?
    # Greedy is O(N^2 * N) = 16M ops? No, is_valid is O(N^2) inside the loop of N. So N^3 = 16M.
    # 16M ops in Python might take ~1s. 
    # So we can do maybe 10-20 random starts.
    
    for _ in range(10):
        if time.time() > deadline - 10:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        m = greedy_construct(order)
        s = get_score(m)
        # Run local search on this start
        m, s = local_search(m)
        if s > best_score:
            best_score = s
            best_mask = m

    # Convert best_mask to list
    result = [i for i in range(1, n + 1) if (best_mask >> i) & 1]
    
    # Final validation
    if not is_valid(best_mask):
        # Fallback to simple greedy if something went wrong
        result = []
        for x in range(1, n + 1):
            if all(2*b-a!=x for a in result for b in result if a<b):
                result.append(x)
    
    return result
# EVOLVE-BLOCK-END
