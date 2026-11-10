import random
import time

def construct():
    deadline = time.time() + 55
    
    def is_valid(s):
        # s is a set or list of ints
        reachable = 1
        for x in s:
            reachable |= (reachable << x)
        return (reachable >> 200) & 1 == 0

    def greedy_construct():
        selected = []
        reachable = 1
        for x in range(100, 0, -1):
            new_reachable = reachable | (reachable << x)
            if not ((new_reachable >> 200) & 1):
                selected.append(x)
                reachable = new_reachable
        return set(selected)

    # Initialize with greedy construction
    best_set = greedy_construct()
    best_size = len(best_set)
    
    # Local Search Improvement
    # We try to add elements not in best_set, or swap elements
    # Elements to consider for adding: 1..100 not in best_set
    # Elements to consider for swapping: any element in best_set
    
    # To make it faster, we can pre-compute or just iterate.
    # Since N=100, the set of candidates is small.
    
    # We will do a few passes.
    # In each pass, we try to add a random candidate.
    # If it fails, we try to swap a random element in best_set with a random candidate.
    
    # Optimization: Instead of checking all swaps for every failed add,
    # we can just do a random walk or hill climbing.
    
    # Let's stick to the current structure but with better init.
    # The current code's local search is:
    # for x in candidates (shuffled 1..65):
    #   try add x
    #   else try swap y for x
    
    # This is biased towards small numbers.
    # A better local search:
    # 1. Identify "blocked" numbers: numbers x such that S U {x} is invalid.
    # 2. For a blocked x, we need to remove some y from S such that S \ {y} U {x} is valid.
    #    Note: If S U {x} is invalid, there is a subset A of S U {x} summing to 200.
    #    If x not in A, then A is subset of S, so S was invalid. Contradiction.
    #    So x must be in A.
    #    So A = {x} U B, where B is subset of S, sum(B) = 200 - x.
    #    To make S U {x} valid, we must remove at least one element from B.
    #    So we can try removing y in B.
    
    # This is a targeted swap!
    # If adding x fails, we know the "conflict" subsets.
    # We can find one conflict subset B (sum 200-x) and try removing elements from B.
    
    # Let's implement this targeted swap.
    
    def get_conflict_subset(s, x):
        # Returns a subset B of s such that sum(B) = 200 - x
        # Uses bitset DP to track predecessors? 
        # Or just backtrack?
        # Since s is small (~70), we can use a simple DP to find one such subset.
        target = 200 - x
        if target < 0: return None
        # DP: reachable sums using elements of s
        # We need to reconstruct one subset.
        # dp[i] = bitmask of elements used? No, too big.
        # We can just use the standard "subset sum" backtracking on the small set s.
        # Or, since we have the bitset `reachable` from is_valid, we can't easily backtrack.
        # Let's just do a quick check: 
        # We know 200-x is reachable in s.
        # We can find one subset by iterating and updating a parent array?
        # Simpler: Just pick a random y from s and check if (s - {y}) U {x} is valid.
        # This is what the current code does (iterating y).
        # To speed it up, we can prioritize y that are "large" or "likely in conflict".
        
        # For now, let's keep the random swap but ensure we use the greedy init.
        pass

    # The main improvement is the initialization.
    # The current code starts with {66..100} (size 35).
    # Greedy 100->1 will likely produce size ~70+.
    
    # Let's also add a few random restarts of the greedy process if time allows?
    # Or just one good greedy + local search.
    
    # We'll modify the loop to use the greedy result.
    
    # The current code's local search logic:
    # candidates = list(range(1, 66))
    # This misses numbers > 66 that might not be in the set?
    # Greedy might miss some large numbers? Unlikely, it goes 100 down.
    # But it might miss some small numbers.
    # So candidates should be all 1..100 not in best_set.
    
    for pass_num in range(50): # Increase passes
        if time.time() > deadline:
            break
        
        candidates = [x for x in range(1, 101) if x not in best_set]
        if not candidates:
            break
        random.shuffle(candidates)
        
        improved = False
        for x in candidates:
            if time.time() > deadline:
                break
            
            new_set = best_set | {x}
            if is_valid(new_set):
                best_set = new_set
                best_size += 1
                improved = True
            else:
                # Try to swap
                # To speed up, shuffle best_set and try a few swaps
                ys = list(best_set)
                random.shuffle(ys)
                # Try at most 10 swaps per failed add to save time?
                # Or all? 100 * 100 = 10k checks. Each check is fast.
                # 10k * 50 passes = 500k checks. Should be fine in 55s.
                for y in ys:
                    if time.time() > deadline:
                        break
                    candidate_set = (best_set - {y}) | {x}
                    if is_valid(candidate_set):
                        best_set = candidate_set
                        # best_size unchanged
                        improved = True
                        break
        
        if not improved:
            break
    
    return list(best_set)
