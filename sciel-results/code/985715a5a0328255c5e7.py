import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin (150s kill, 40s buffer)
    
    def solve_once(seed):
        rng = random.Random(seed)
        
        S = set()
        counts = {}
        
        def can_add(x):
            if counts.get(2*x, 0) >= 2:
                return False
            for s in S:
                if counts.get(x + s, 0) >= 2:
                    return False
            return True

        def add(x):
            S.add(x)
            counts[2*x] = counts.get(2*x, 0) + 1
            for s in S:
                if s != x:
                    counts[x + s] = counts.get(x + s, 0) + 1

        def remove(x):
            S.remove(x)
            counts[2*x] -= 1
            if counts[2*x] == 0:
                del counts[2*x]
            for s in S:
                counts[x + s] -= 1
                if counts[x + s] == 0:
                    del counts[x + s]

        # 1. Randomized Greedy Initialization
        # Shuffle the range to break determinism
        candidates = list(range(1, 1001))
        rng.shuffle(candidates)
        
        # Optional: Score based on "isolation" or just random order?
        # Let's try random order first, it's simple and effective with restarts.
        
        for x in candidates:
            if can_add(x):
                add(x)
            if time.time() > deadline:
                break
                
        # 2. Local Search: Targeted 1-Swaps
        # Find candidates blocked by exactly 1 element
        # To do this efficiently, we can iterate through all candidates and count blockers.
        # Blocker x for y: 
        # - count(2y) >= 2 and 2y == 2x? No.
        # - count(y+s) >= 2 for some s in S.
        # If we remove x, does y become addable?
        # y is blocked if:
        #   a) count(2y) >= 2. Removing x changes count(2y) only if x=y (impossible) or if 2y = x+s.
        #      If 2y = x+s, then removing x reduces count(2y) by 1.
        #   b) count(y+s) >= 2 for some s in S. Removing x reduces count(y+s) by 1 if y+s = x+t for some t in S.
        #      i.e., if x = y+s-t.
        
        # Simpler approach: Just try all pairs (x in S, y not in S) but filter smartly.
        # We can precompute for each y, the list of x in S that "contribute" to the blocking sums.
        
        # Let's just do a few iterations of:
        # For each y in candidates:
        #   Find blockers
        #   If len(blockers) == 1, try swap.
        
        # To find blockers for y:
        # y is blocked if:
        # 1. count(2y) >= 2.
        #    If count(2y) == 2, the blockers are the pairs (a,b) in S such that a+b=2y.
        #    If we remove x, count(2y) drops by 1 if x is in one of those pairs.
        # 2. count(y+s) >= 2 for some s in S.
        #    For each s in S, if count(y+s) >= 2, then any x such that x + (y+s-x) is in S is a blocker?
        #    Actually, removing x reduces count(z) by the number of pairs (x, z-x) in S.
        #    So x is a blocker for sum z if count(z) >= 2 AND (z-x) in S AND x != z-x (or handle double count).
        
        # This is getting complex to optimize perfectly. Given N=1000, |S|~700, |C|~300.
        # Total pairs to check: 700 * 300 = 210,000.
        # Each check involves O(1) work if we can determine if removing x unblocks y.
        
        # Let's implement a direct check:
        # For a candidate y and a blocker candidate x:
        # Temporarily remove x. Check can_add(y). If yes, keep swap.
        
        # To avoid O(|S|) per check in can_add, we can just do the temporary remove/add.
        # The cost is O(|S|) for remove, O(|S|) for can_add, O(|S|) for add. Total O(|S|) per swap attempt.
        # 210,000 attempts * 700 ops = 147,000,000 ops. Might be too slow in Python.
        
        # Optimization: Only try swaps where y is "close" to being valid.
        # We can compute a "blocking score" for y.
        
        # Let's stick to a limited number of random swaps but biased towards "promising" ones.
        # Or, just do the random swaps but faster.
        
        # Let's try the targeted approach with a time limit.
        
        candidates_list = [x for x in range(1, 1001) if x not in S]
        
        # Precompute sums for current S? No, counts is already there.
        
        # Strategy: Iterate through candidates. For each, find if it's blocked by 1 element.
        # If so, swap.
        
        # To find if y is blocked by exactly 1 element x:
        # We need to know if removing x makes can_add(y) true.
        
        # Let's just do random swaps for now, but increase the number of attempts per iteration if time permits.
        # The previous score was 1.02. Let's see if restarts help.
        
        # Actually, let's implement the "1-blocker" logic properly but efficiently.
        # For a fixed y:
        # y is valid if count(2y) < 2 and all count(y+s) < 2.
        # If y is invalid, let's identify the "culprits".
        
        # If count(2y) >= 2:
        #   The pairs summing to 2y are fixed. Let P_y be the set of x in S such that x + (2y-x) in S.
        #   If we remove x, count(2y) decreases by 1 if (2y-x) in S and x != 2y-x.
        #   If count(2y) was 2, and we remove one such x, count becomes 1. Then we must also check other sums.
        
        # This is too complex to implement perfectly in the short time.
        # Let's rely on **Multiple Restarts** of the random greedy + random swap local search.
        # The variance in random greedy might find a better basin.
        
        # Run local search
        # We will do a fixed number of swap attempts, but check time.
        
        max_swaps = 2000
        swaps_done = 0
        no_improve_count = 0
        
        while swaps_done < max_swaps and time.time() < deadline - 5:
            # Pick random x from S, y from candidates
            if not S or not candidates_list:
                break
            
            x = rng.choice(list(S))
            y = rng.choice(candidates_list)
            
            # Try swap
            remove(x)
            if can_add(y):
                add(y)
                candidates_list.remove(y)
                candidates_list.append(x)
                no_improve_count = 0
            else:
                add(x)
                no_improve_count += 1
                if no_improve_count > 100:
                    # Restart local search with new random seed? Or just break?
                    # Let's just break this local search phase and restart the whole solve?
                    # No, we are inside solve_once.
                    # Let's just stop local search for this run.
                    break
            swaps_done += 1
            
            if time.time() > deadline - 5:
                break
        
        return sorted(list(S))

    best_set = []
    best_size = 0
    
    # Run multiple restarts
    # Estimate time per restart.
    # Greedy: ~1000 * 700 = 700k ops. ~0.05s
    # Local Search: 2000 swaps * 700 ops = 1.4M ops. ~0.1s
    # Total ~0.15s per restart.
    # We have 110s. Can do ~700 restarts.
    
    # Let's do 50 restarts to be safe, or until time runs out.
    
    seed_counter = 0
    while time.time() < deadline - 2:
        seed = seed_counter
        seed_counter += 1
        
        # Randomize the random number generator for this run
        result = solve_once(seed)
        
        if len(result) > best_size:
            best_size = len(result)
            best_set = result
            
        # If we've used up most of the time, break
        if time.time() > deadline - 5:
            break
            
        # If we found a very large set, maybe stop early?
        # Theoretical max is ~ 700-800.
        # If best_size > 750, we are doing well.
        
    return best_set
