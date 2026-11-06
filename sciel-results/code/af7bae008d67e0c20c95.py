"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

TARGET = 200
N = 100
MASK = (1 << (TARGET + 1)) - 1

def _update_reachable(bits, x):
    """Update reachable sums bitset after adding element x."""
    return bits | ((bits << x) & MASK)

def _is_feasible(bits, x):
    """Check if adding x keeps no subset summing to TARGET."""
    if x > TARGET:
        return True
    return (bits & (1 << (TARGET - x))) == 0

def _compute_reachable(selected):
    """Compute reachable sums bitset for a given selected list."""
    bits = 1
    for x in selected:
        bits = _update_reachable(bits, x)
    return bits

def _greedy_desc():
    """Greedy: add elements from largest to smallest if feasible."""
    selected = []
    bits = 1
    for x in range(N, 0, -1):
        if _is_feasible(bits, x):
            selected.append(x)
            bits = _update_reachable(bits, x)
    return selected

def _greedy_asc():
    """Greedy: add elements from smallest to largest if feasible."""
    selected = []
    bits = 1
    for x in range(1, N + 1):
        if _is_feasible(bits, x):
            selected.append(x)
            bits = _update_reachable(bits, x)
    return selected

def _greedy_random(seed=0):
    """Greedy with random ordering."""
    rng = random.Random(seed)
    order = list(range(1, N + 1))
    rng.shuffle(order)
    selected = []
    bits = 1
    for x in order:
        if _is_feasible(bits, x):
            selected.append(x)
            bits = _update_reachable(bits, x)
    return selected

def _greedy_weighted(seed=0):
    """Greedy with weighted random ordering (favor smaller elements)."""
    rng = random.Random(seed)
    order = list(range(1, N + 1))
    # Weight inversely proportional to value to favor smaller elements
    weights = [1.0 / x for x in order]
    total_w = sum(weights)
    remaining = list(range(1, N + 1))
    remaining_weights = [1.0 / x for x in remaining]
    selected = []
    bits = 1
    while remaining:
        # Pick randomly weighted
        r = rng.random() * sum(remaining_weights)
        cum = 0
        idx = 0
        for i, w in enumerate(remaining_weights):
            cum += w
            if r <= cum:
                idx = i
                break
        x = remaining[idx]
        remaining.pop(idx)
        remaining_weights.pop(idx)
        if _is_feasible(bits, x):
            selected.append(x)
            bits = _update_reachable(bits, x)
    return selected

def _local_search_add(selected, bits, deadline):
    """Try to add any missing element that is feasible."""
    selected_set = set(selected)
    improved = True
    while improved:
        if time.time() >= deadline:
            break
        improved = False
        for x in range(1, N + 1):
            if x not in selected_set and _is_feasible(bits, x):
                selected.append(x)
                selected_set.add(x)
                bits = _update_reachable(bits, x)
                improved = True
                break
    return selected, bits

def _local_search_swap(selected, bits, deadline):
    """Try swaps: remove one element, add another to increase size."""
    improved = True
    while improved:
        if time.time() >= deadline:
            break
        improved = False
        for i in range(len(selected)):
            if time.time() >= deadline:
                break
            x = selected[i]
            # Remove x
            new_selected = selected[:i] + selected[i+1:]
            new_bits = _compute_reachable(new_selected)
            new_set = set(new_selected)
            # Try to add a smaller element
            for z in range(1, x):
                if z not in new_set and _is_feasible(new_bits, z):
                    # This swap keeps size same but may allow further additions
                    # Only do it if it enables more additions
                    new_selected.append(z)
                    new_set.add(z)
                    new_bits = _update_reachable(new_bits, z)
                    # Now try to add more
                    new_selected, new_bits = _local_search_add(new_selected, new_bits, deadline)
                    if len(new_selected) > len(selected):
                        selected = new_selected
                        bits = new_bits
                        improved = True
                        break
            if improved:
                break
    return selected, bits

def _simulated_annealing(initial, deadline):
    """Simulated annealing to improve the solution."""
    rng = random.Random(42)
    selected = list(initial)
    bits = _compute_reachable(selected)
    best = list(selected)
    best_bits = bits
    best_size = len(selected)
    
    # Temperature schedule
    t_start = time.time()
    elapsed = 0
    T = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        if not selected:
            break
        
        # Random move: swap one element
        i = rng.randint(0, len(selected) - 1)
        x = selected[i]
        z = rng.randint(1, N)
        if z == x:
            continue
        
        # Remove x, add z
        new_selected = selected[:i] + [z] + selected[i+1:]
        # Check if z is already in the set (minus x)
        if z in selected and z != x:
            continue
        
        new_bits = _compute_reachable(new_selected)
        
        # Check validity
        if (new_bits >> TARGET) & 1:
            continue
        
        new_size = len(new_selected)
        old_size = len(selected)
        
        # Accept if better or with probability based on temperature
        if new_size >= old_size:
            selected = new_selected
            bits = new_bits
            if new_size > best_size:
                best = list(selected)
                best_bits = bits
                best_size = new_size
        else:
            # Worse size, accept with probability exp(-delta/T)
            delta = old_size - new_size
            prob = 2.718281828 ** (-delta / T)
            if rng.random() < prob:
                selected = new_selected
                bits = new_bits
        
        T *= cooling
        if T < 0.01:
            T = 0.01
    
    return best, best_bits

def _construct_optimal_candidate():
    """Try to construct a high-cardinality solution using smart heuristics."""
    # Start with all elements and try to remove minimum to avoid sum 200
    # Alternative: build up greedily
    
    # Strategy: try to include elements 1-100 but remove conflicting ones
    # A key observation: if we have elements a, b, c, d with a+b+c+d=200, we can't have all four
    
    # Let's try a different approach: include all elements > 100 (none), 
    # and for 1-100, use a more careful selection
    
    # Try: include all odd numbers (50 elements) - no two odd numbers sum to even 200
    # But four odd numbers can sum to 200! e.g., 1+3+5+191... but 191>100
    # Actually with elements up to 100, can four odds sum to 200?
    # 1+3+5+191 no. Max four odds: 99+97+95+93=384 > 200
    # Min: 1+3+5+7=16 < 200
    # So yes, some combinations of 4 odds can sum to 200
    
    # Let's just use the search approach
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Try multiple strategies
    candidates = []
    
    # Strategy 1: Greedy largest first
    sol1 = _greedy_desc()
    candidates.append(sol1)
    
    # Strategy 2: Greedy smallest first
    if time.time() < deadline:
        sol2 = _greedy_asc()
        candidates.append(sol2)
    
    # Strategy 3: Random orders
    if time.time() < deadline:
        for seed in range(50):
            if time.time() >= deadline:
                break
            sol = _greedy_random(seed)
            candidates.append(sol)
    
    # Strategy 4: Weighted random (favor smaller)
    if time.time() < deadline:
        for seed in range(50):
            if time.time() >= deadline:
                break
            sol = _greedy_weighted(seed)
            candidates.append(sol)
    
    # Find best initial candidate
    best = max(candidates, key=len)
    best_bits = _compute_reachable(best)
    
    # Local search: add elements
    if time.time() < deadline:
        best, best_bits = _local_search_add(best, best_bits, deadline)
    
    # Local search: swaps
    if time.time() < deadline:
        best, best_bits = _local_search_swap(best, best_bits, deadline)
    
    # Simulated annealing
    if time.time() < deadline:
        sa_result, sa_bits = _simulated_annealing(best, deadline)
        if len(sa_result) > len(best):
            best = sa_result
            best_bits = sa_bits
    
    # Final local search
    if time.time() < deadline:
        best, best_bits = _local_search_add(best, best_bits, deadline)
        best, best_bits = _local_search_swap(best, best_bits, deadline)
    
    # Verify
    if not best:
        return [1]
    
    # Final verification
    bits = 1
    for x in best:
        bits = _update_reachable(bits, x)
    if (bits >> TARGET) & 1:
        # Fallback to greedy desc
        return _greedy_desc()
    
    return best
# EVOLVE-BLOCK-END
