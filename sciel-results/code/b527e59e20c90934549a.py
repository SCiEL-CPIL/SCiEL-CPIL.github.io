"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Returns a nonempty subset of [1, 100] with no subset summing to 200.
    Maximizes cardinality.
    """
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    def is_valid(selected_set):
        """Check if no subset of selected_set sums to 200 using bitset DP."""
        # Use a boolean array for reachable sums up to 200
        reachable = [False] * 201
        reachable[0] = True
        for x in selected_set:
            # Update in reverse to avoid using x multiple times
            for s in range(200 - x, -1, -1):
                if reachable[s]:
                    reachable[s + x] = True
        return not reachable[200]
    
    # Strategy 1: Threshold set {66..100} - guaranteed valid, 35 elements
    # Proof: min 3-element sum = 66+67+68=201>200, no 2-element pair sums to 200
    # (would need a+b=200 with a,b in [66,100], a!=b; but 100+100 not allowed,
    #  99+101 out of range, etc.)
    threshold_set = list(range(66, 101))
    
    # Strategy 2: Greedy from largest to smallest
    greedy_selected = []
    reachable_greedy = {0}
    
    for x in range(100, 0, -1):
        if time.time() > deadline:
            break
        target = 200 - x
        if target in reachable_greedy:
            continue
        new_sums = {s + x for s in reachable_greedy if s + x < 200}
        greedy_selected.append(x)
        reachable_greedy |= new_sums
    
    # Strategy 3: Start with threshold set, try to add smaller elements
    extended = list(threshold_set)
    # Compute reachable sums for the threshold set
    reach_ext = [False] * 201
    reach_ext[0] = True
    for x in extended:
        for s in range(200 - x, -1, -1):
            if reach_ext[s]:
                reach_ext[s + x] = True
    
    # Try adding elements from 65 down to 1
    for x in range(65, 0, -1):
        if time.time() > deadline:
            break
        # Check if adding x creates sum 200
        if not reach_ext[200 - x]:
            # Safe to add
            extended.append(x)
            # Update reachable sums
            for s in range(200 - x, -1, -1):
                if reach_ext[s]:
                    reach_ext[s + x] = True
    
    # Strategy 4: Greedy from smallest to largest (different order)
    greedy_small = []
    reachable_small = {0}
    for x in range(1, 101):
        if time.time() > deadline:
            break
        target = 200 - x
        if target in reachable_small:
            continue
        new_sums = {s + x for s in reachable_small if s + x < 200}
        greedy_small.append(x)
        reachable_small |= new_sums
    
    # Pick the best valid set
    candidates = [
        ("threshold", threshold_set),
        ("greedy_high", greedy_selected),
        ("extended", extended),
        ("greedy_low", greedy_small),
    ]
    
    best = None
    best_size = 0
    for name, cand in candidates:
        if not cand:
            continue
        if is_valid(cand):
            if len(cand) > best_size:
                best_size = len(cand)
                best = cand
    
    if best is None:
        return [100]
    
    return best
# EVOLVE-BLOCK-END
