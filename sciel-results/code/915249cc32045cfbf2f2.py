import time
import random

def _greedy(order):
    selected, reachable = [], {0}
    for x in order:
        # Check if adding x creates a sum of 200
        # This is equivalent to checking if 200-x is in reachable
        # Note: reachable contains sums of subsets of 'selected'
        # If 200-x is in reachable, then some subset sums to 200-x, adding x makes 200.
        # Also need to ensure x itself doesn't make 200 with 0? 200-0=200. If x=200, but x<=100.
        # So just checking 200-x is sufficient.
        if (200 - x) not in reachable:
            selected.append(x)
            # Update reachable
            # New sums are s + x for s in reachable
            # We only care about sums <= 200
            new_sums = {s + x for s in reachable if s + x <= 200}
            reachable |= new_sums
    return selected

def _local_search(selected, reachable):
    """
    Iteratively try to add excluded elements.
    Returns improved selected list and reachable set.
    """
    selected_set = set(selected)
    excluded = [x for x in range(1, 101) if x not in selected_set]
    
    # We need to be careful with time, but this is fast.
    # Randomize the order of excluded to escape local minima caused by ordering
    random.shuffle(excluded)
    
    improved = True
    while improved:
        improved = False
        # Try to add each excluded element
        # We must iterate over a snapshot or handle list modification carefully
        # Since adding an element changes 'reachable', it might affect safety of others
        # We iterate through the list. If we add one, we don't re-check previous ones in this pass,
        # but the outer while loop will catch them in the next pass.
        for x in excluded:
            if x in selected_set:
                continue
            if (200 - x) not in reachable:
                # Safe to add
                selected.append(x)
                selected_set.add(x)
                new_sums = {s + x for s in reachable if s + x <= 200}
                reachable |= new_sums
                improved = True
        if improved:
            # Re-shuffle excluded for next pass?
            # The list 'excluded' still contains elements already added.
            # We check 'if x in selected_set' so it's fine.
            # Re-shuffling might help find different local maxima.
            random.shuffle(excluded)
            
    return selected, reachable

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    best = []
    best_len = 0
    candidates = [] # Store (len, selected, reachable) for top candidates
    
    # 1. Deterministic orders
    orders = [
        list(range(100, 0, -1)),
        list(range(1, 101)),
    ]
    
    # 2. Random orders
    random.seed(42)
    for _ in range(2000):
        if time.time() > deadline:
            break
        order = list(range(1, 101))
        random.shuffle(order)
        orders.append(order)
        
    for order in orders:
        if time.time() > deadline:
            break
        selected = _greedy(order)
        if len(selected) > best_len:
            best_len = len(selected)
            best = selected
            
        # Keep track of top 5 for local search
        if len(candidates) < 5:
            # Reconstruct reachable for storage? Or just store selected and rebuild later.
            # Rebuilding is fast.
            candidates.append((len(selected), selected))
            candidates.sort(key=lambda x: x[0], reverse=True)
            if len(candidates) > 5:
                candidates.pop()
        else:
            if len(selected) > candidates[-1][0]:
                candidates[-1] = (len(selected), selected)
                candidates.sort(key=lambda x: x[0], reverse=True)

    # 3. Local search on top candidates
    for _, sel in candidates:
        if time.time() > deadline:
            break
        # Rebuild reachable
        reachable = {0}
        for x in sel:
            new_sums = {s + x for s in reachable if s + x <= 200}
            reachable |= new_sums
        
        improved_sel, improved_reach = _local_search(list(sel), reachable)
        
        if len(improved_sel) > best_len:
            best_len = len(improved_sel)
            best = improved_sel

    return best if best else [100]
