import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # explicit guard, safety margin under 90s
    target = 200
    mask_limit = (1 << (target + 1)) - 1  # keep bits 0..200

    def greedy(order):
        selected = []
        reachable = 1  # bitset: bit s set iff sum s reachable
        for x in order:
            new_reach = (reachable << x) & mask_limit
            if (new_reach >> target) & 1:
                continue  # adding x would make target reachable
            reachable |= new_reach
            selected.append(x)
        return selected

    # Guaranteed-valid contiguous block [66,100]: 2-sums<=199, 3-sums>=201
    candidates = [list(range(66, 101))]
    # Greedy in several orderings
    candidates.append(greedy(list(range(100, 0, -1))))
    candidates.append(greedy(list(range(1, 101))))

    best = max(candidates, key=len)

    # Extra orderings within budget
    for order in (
        list(range(100, 0, -2)) + list(range(99, 0, -2)),
        list(range(1, 101, 2)) + list(range(100, 0, -2)),
    ):
        if time.time() > deadline:
            break
        cand = greedy(order)
        if len(cand) > len(best):
            best = cand

    # ZDD / DP approach: find max cardinality subset with no subset summing to 200
    # State: frozenset of reachable sums (as a bitmask integer)
    # We process elements from 100 down to 1, deciding include/exclude
    # DP over (element_index, reachable_sumset) -> max additional elements
    # But reachable_sumset space is huge, so we use memoization with pruning
    
    if time.time() < deadline - 5:
        try:
            # Process elements in descending order (larger elements first)
            # This tends to create more distinct reachable sum sets early
            elements = list(range(100, 0, -1))
            n = len(elements)
            
            # Memo: (idx, reachable_mask) -> (max_count, chosen_elements_tuple)
            # But storing chosen elements in memo is expensive; instead store count and reconstruct
            memo = {}
            
            def dp(idx, reachable_mask):
                """Return max number of elements we can include from elements[idx:] 
                   given current reachable sums, without ever reaching target."""
                if time.time() > deadline - 1:
                    return 0, ()
                
                if idx == n:
                    return 0, ()
                
                key = (idx, reachable_mask)
                if key in memo:
                    return memo[key]
                
                x = elements[idx]
                best_count = 0
                best_choice = ()
                
                # Option 1: exclude x
                count_ex, choice_ex = dp(idx + 1, reachable_mask)
                if count_ex > best_count:
                    best_count = count_ex
                    best_choice = choice_ex
                
                # Option 2: include x (if it doesn't make target reachable)
                new_reach = (reachable_mask << x) & mask_limit
                if not (new_reach >> target) & 1:
                    count_in, choice_in = dp(idx + 1, reachable_mask | new_reach)
                    if count_in + 1 > best_count:
                        best_count = count_in + 1
                        best_choice = (x,) + choice_in
                
                memo[key] = (best_count, best_choice)
                return best_count, best_choice
            
            count, chosen = dp(0, 1)  # start with only sum 0 reachable
            if chosen:
                cand = list(chosen)
                if len(cand) > len(best):
                    best = cand
                    
        except (RecursionError, MemoryError):
            pass

    # Local search: try to add elements not in best
    if time.time() < deadline - 2:
        current = set(best)
        current_reachable = 1
        for x in sorted(current):
            current_reachable |= (current_reachable << x) & mask_limit
        
        # Try adding elements one by one in various orders
        for order in (list(range(1, 101)), list(range(100, 0, -1)), 
                      list(range(1, 101, 2)) + list(range(100, 0, -2))):
            if time.time() > deadline - 1:
                break
            improved = True
            while improved:
                improved = False
                for x in order:
                    if x in current:
                        continue
                    new_reach = (current_reachable << x) & mask_limit
                    if not (new_reach >> target) & 1:
                        current.add(x)
                        current_reachable |= new_reach
                        improved = True
        
        if len(current) > len(best):
            best = list(current)

    return best
# EVOLVE-BLOCK-END
