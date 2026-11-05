import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    def time_left():
        return deadline - time.time()
    
    def greedy_build(order=None):
        """Build a B2[2] set using greedy with given order (or default ascending)."""
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
        
        if order is None:
            order = list(range(1, 1001))
        
        for x in order:
            if can_add(x):
                add(x)
            if time.time() > deadline:
                break
        
        return S, counts
    
    def local_improve(S, counts, max_time=None):
        """Improve S via swaps and additions."""
        if max_time is None:
            max_time = deadline - time.time()
        
        candidates = [x for x in range(1, 1001) if x not in S]
        S_list = list(S)
        
        def can_add_to(S, counts, x):
            if counts.get(2*x, 0) >= 2:
                return False
            for s in S:
                if s != x and counts.get(x + s, 0) >= 2:
                    return False
            return True
        
        def do_add(S, counts, x):
            S.add(x)
            counts[2*x] = counts.get(2*x, 0) + 1
            for s in S:
                if s != x:
                    counts[x + s] = counts.get(x + s, 0) + 1
        
        def do_remove(S, counts, x):
            S.remove(x)
            counts[2*x] -= 1
            if counts[2*x] == 0:
                del counts[2*x]
            for s in S:
                counts[x + s] -= 1
                if counts[x + s] == 0:
                    del counts[x + s]
        
        while time_left() > 1.0:
            # Try to add a random candidate
            if candidates:
                idx = random.randint(0, len(candidates) - 1)
                y = candidates[idx]
                if can_add_to(S, counts, y):
                    do_add(S, counts, y)
                    candidates.remove(y)
                    continue
            
            # Try a swap: remove x from S, add y from candidates
            if S and candidates:
                x = random.choice(list(S))
                y = random.choice(candidates)
                
                # Check if swapping x->y is valid
                # After removing x, we need to check if y can be added
                # The counts after removing x: for each sum s+x, decrement
                # Then check if y can be added to S\{x}
                
                valid = True
                if counts.get(2*y, 0) >= 2:
                    valid = False
                else:
                    for s in S:
                        if s == x:
                            continue
                        # After removing x, the count of s+y is unchanged (since x != s)
                        # But we need to check if count(s+y) would be >= 2 after adding y
                        # Currently counts includes x's contributions
                        # For s != x, counts[s+y] is the same as after removing x
                        if counts.get(s + y, 0) >= 2:
                            valid = False
                            break
                        # Also check: if s+y == 2*x or involves x in some way
                        # Actually we just need: for all s in S\{x}, new_count(s+y) < 2
                        # new_count(s+y) = old_count(s+y) - [x+y==s+y ? 1 : 0] + 1
                        # = old_count(s+y) + 1 (since x != s, x+y != s+y unless x==s which we skip)
                        # Wait, that's not right. Let me think again.
                        # After removing x: counts' = counts - contributions from x
                        # After adding y: counts'' = counts' + contributions from y
                        # We need all counts'' values <= 2
                        # For a sum t:
                        #   counts''[t] = counts[t] - contrib_x(t) + contrib_y(t)
                        #   where contrib_x(t) = number of pairs (a,b) in S with a+b=t involving x
                        #   and contrib_y(t) = number of pairs (a,b) in S\{x} with a+b=t involving y
                        # This is complex. Let's just simulate.
                        pass
                
                if valid:
                    # More careful check: simulate the swap
                    # Remove x, check if y can be added
                    do_remove(S, counts, x)
                    if can_add_to(S, counts, y):
                        do_add(S, counts, y)
                        candidates.remove(y)
                        candidates.append(x)
                    else:
                        # Revert
                        do_add(S, counts, x)
                # If not valid, just continue
        
        return S, counts
    
    # Multiple restarts with different greedy orders
    best_S = None
    best_counts = None
    best_size = 0
    
    # First try deterministic ascending
    S, counts = greedy_build()
    S, counts = local_improve(S, counts)
    if len(S) > best_size:
        best_S, best_counts, best_size = S, counts, len(S)
    
    # Try several randomized restarts
    num_restarts = 5
    for r in range(num_restarts):
        if time_left() < 10.0:
            break
        
        # Create a randomized order with some structure
        # Use a mix: first half ascending, second half shuffled
        order = list(range(1, 1001))
        random.shuffle(order)
        # Or try: start with small numbers, then shuffle the rest
        # Let's try different strategies
        
        S, counts = greedy_build(order)
        if len(S) > best_size:
            S, counts = local_improve(S, counts)
            if len(S) > best_size:
                best_S, best_counts, best_size = S, counts, len(S)
    
    # More local search on the best solution found
    if best_S is not None and time_left() > 5.0:
        best_S, best_counts = local_improve(best_S, best_counts)
    
    return sorted(list(best_S))
