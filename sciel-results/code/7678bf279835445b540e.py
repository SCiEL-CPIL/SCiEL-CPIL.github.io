import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    max_val = 256
    max_size = 60
    
    def get_subset_sums(chosen):
        """
        Returns list of bitsets S where S[k] has bit i set if sum i is achievable with cardinality k.
        """
        n = len(chosen)
        S = [0] * (n + 1)
        S[0] = 1
        for x in chosen:
            # Update from high to low to avoid using x multiple times
            for k in range(n, 0, -1):
                S[k] |= (S[k-1] << x)
        return S

    def is_valid(S):
        """
        Checks if S[k] are pairwise disjoint for all k.
        """
        n = len(S) - 1
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                if (S[i] & S[j]) != 0:
                    return False
        return True

    def greedy_construct(order):
        """
        Greedily adds numbers in the given order.
        """
        chosen = []
        S = [1]  # S[0] = 1
        # We need to update S dynamically.
        # To be efficient, we can just recompute S from scratch for each candidate?
        # No, we can update it.
        # But S size grows.
        # Let's just recompute for simplicity in the check, or maintain it.
        # Maintaining it:
        # S is list of ints.
        
        for x in order:
            if time.time() > deadline:
                break
            
            # Try adding x
            # New S
            new_S = S[:]
            current_len = len(chosen)
            
            # We need to extend new_S if necessary?
            # S has length current_len + 1.
            # new_S needs length current_len + 2.
            new_S.append(0)
            
            for k in range(current_len + 1, 0, -1):
                new_S[k] = S[k] | (S[k-1] << x)
                
            if is_valid(new_S):
                S = new_S
                chosen.append(x)
                
        return chosen

    # Strategy 1: Greedy Largest
    # This tends to keep sums sparse and disjoint
    order_largest = list(range(max_val, 0, -1))
    set_largest = greedy_construct(order_largest)
    
    # Strategy 2: Greedy Smallest (current behavior, for comparison or fallback)
    # We might skip this if Largest is better, but let's keep it if time allows?
    # No, let's focus on improving the best one.
    
    best_set = set_largest
    
    # Strategy 3: Local Search / Repair
    # Try to add more numbers or swap to improve size
    chosen = best_set
    S = get_subset_sums(chosen)
    unused = set(range(1, max_val + 1)) - set(chosen)
    
    # Convert unused to list for iteration
    unused_list = list(unused)
    random.shuffle(unused_list)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try pure additions
        # Sort unused by some heuristic? Maybe random is fine.
        for y in unused_list:
            if time.time() > deadline:
                break
            
            # Check if adding y is valid
            # We can quickly check by computing new_S
            new_S = S[:]
            current_len = len(chosen)
            new_S.append(0)
            
            for k in range(current_len + 1, 0, -1):
                new_S[k] = S[k] | (S[k-1] << y)
                
            if is_valid(new_S):
                # Accept
                chosen.append(y)
                S = new_S
                unused_list.remove(y)
                improved = True
                break # Restart loop with new state
        
        if improved:
            continue
            
        # 2. Try swaps if no pure addition found
        # Pick a random y from unused
        if not unused_list:
            break
            
        # Try a few random swaps
        swaps_tried = 0
        max_swaps = 100 # Limit per iteration to avoid stalling
        
        while swaps_tried < max_swaps and time.time() < deadline:
            swaps_tried += 1
            y = random.choice(unused_list)
            # Pick a random x from chosen to remove
            if not chosen:
                break
            x = random.choice(chosen)
            
            # Form new set
            # Remove x, add y
            # This requires recomputing S from scratch or careful update.
            # Recomputing is safer and O(N^2). N is small (~20-50).
            new_chosen = [z for z in chosen if z != x]
            new_chosen.append(y)
            
            new_S = get_subset_sums(new_chosen)
            
            if is_valid(new_S):
                # Accept swap
                chosen = new_chosen
                S = new_S
                unused_list.remove(y)
                unused_list.append(x)
                improved = True
                break

    return [int(x) for x in chosen]
