import time
import random
import numpy as np

def construct():
    start = time.time()
    deadline = start + 55.0
    
    # Helper to check if adding x to set S (represented as bitset of sums) is valid
    # We maintain the set of reachable sums.
    
    def make_bitset():
        return np.zeros(201, dtype=bool)

    def add_to_bitset(bs, x):
        new_sums = np.zeros(201, dtype=bool)
        # Optimization: use numpy shifts if possible, but loop is fine for 200
        # Actually, bs is bool array.
        # new_sums[i+x] = bs[i]
        # We can do: new_sums[x:201] = bs[:201-x]
        new_sums[x:201] = bs[:201-x]
        if new_sums[200]:
            return None
        return bs | new_sums

    def greedy(order):
        selected = []
        bs = make_bitset()
        bs[0] = True
        for x in order:
            new_bs = add_to_bitset(bs, x)
            if new_bs is not None:
                selected.append(x)
                bs = new_bs
        return selected

    # Initialize with greedy largest-first
    best = greedy(range(100, 0, -1))
    best_size = len(best)
    
    # Local Search: Conflict-Driven
    # We represent the set as a list and a bitset of sums.
    # We also maintain a set of "blocked" elements.
    
    def get_blocked_elements(selected, bs):
        blocked = {}
        sel_set = set(selected)
        for x in range(1, 101):
            if x in sel_set:
                continue
            new_bs = add_to_bitset(bs, x)
            if new_bs is None:
                # x is blocked. Find why.
                # We need to find pairs (a,b) in selected such that a+b+x=200
                # or a+b+c+x=200 etc.
                # Since we only track sums, we can find which sums in bs
                # combined with x make 200.
                # Sums s in bs such that s+x=200 => s=200-x.
                # If 200-x is in bs, then x is blocked by some subset summing to 200-x.
                # To find the specific elements, we might need to backtrack.
                # For simplicity, we just know x is blocked.
                blocked[x] = True
        return blocked

    # Better Local Search:
    # 1. Start with best.
    # 2. Try to add any element.
    # 3. If blocked, try to remove one element from the set to unblock it.
    #    To do this efficiently, we need to know which elements are "critical".
    
    # Let's use a simpler, more robust local search:
    # Random restarts of greedy + hill climbing.
    
    rng = random.Random(42)
    for trial in range(500):
        if time.time() > deadline - 5.0:
            break
        
        # Random order
        order = list(range(1, 101))
        rng.shuffle(order)
        candidate = greedy(order)
        cand_size = len(candidate)
        
        if cand_size > best_size:
            best = candidate
            best_size = cand_size
            continue
            
        # Hill climbing on candidate
        # Try to swap: remove 1, add 1 or 2
        # Or just try to add if possible (greedy did that)
        # So we need to try to REMOVE an element to allow ADDING two.
        
        sel_set = set(candidate)
        bs = make_bitset()
        bs[0] = True
        for x in candidate:
            bs = add_to_bitset(bs, x)
            
        improved = True
        while improved:
            improved = False
            # Try to remove each element in candidate
            for remove_x in list(candidate):
                if time.time() > deadline - 5.0:
                    break
                
                # Remove remove_x
                remaining = [x for x in candidate if x != remove_x]
                remaining_set = set(remaining)
                
                # Recalculate bitset for remaining
                bs_rem = make_bitset()
                bs_rem[0] = True
                valid = True
                for x in remaining:
                    new_bs_rem = add_to_bitset(bs_rem, x)
                    if new_bs_rem is None:
                        valid = False
                        break
                    bs_rem = new_bs_rem
                if not valid:
                    continue
                
                # Try to add all missing elements to remaining
                new_candidate = list(remaining)
                new_set = set(remaining)
                added_any = False
                for x in range(1, 101):
                    if x not in new_set:
                        new_bs_rem = add_to_bitset(bs_rem, x)
                        if new_bs_rem is not None:
                            new_candidate.append(x)
                            new_set.add(x)
                            bs_rem = new_bs_rem
                            added_any = True
                            
                if len(new_candidate) > len(candidate):
                    candidate = new_candidate
                    cand_size = len(candidate)
                    improved = True
                    if cand_size > best_size:
                        best = candidate
                        best_size = cand_size
                    break # Restart the while loop with new candidate

    return best
