"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import random
import time

def _dist_sq(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _creates_isosceles(p, chosen):
    """Check if adding p to chosen creates any isosceles triangle."""
    n = len(chosen)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = chosen[i], chosen[j]
            d_pa = _dist_sq(p, a)
            d_pb = _dist_sq(p, b)
            d_ab = _dist_sq(a, b)
            if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                return True
    return False

def _is_maximal(chosen):
    """Check if the set is maximal (no point can be added)."""
    chosen_set = set(chosen)
    for p in itertools.product(range(N), repeat=2):
        if p not in chosen_set:
            if not _creates_isosceles(p, chosen):
                return False
    return True

def _greedy_construct():
    """Greedy maximal construction in row-major order."""
    chosen = []
    for p in itertools.product(range(N), repeat=2):
        if not _creates_isosceles(p, chosen):
            chosen.append(p)
    return chosen

def _local_search(chosen, max_iters=5000, time_limit=280):
    """Local search to minimize the size of a maximal no-isosceles set."""
    start_time = time.time()
    chosen = list(chosen)
    chosen_set = set(chosen)
    
    # Precompute: for each point not in chosen, which pairs in chosen block it
    # Actually, let's just do simple local search: try removing points and see if still maximal
    
    best = list(chosen)
    best_size = len(chosen)
    
    for iteration in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Try removing a random point and check if the set is still maximal
        if len(chosen) <= 1:
            break
        
        idx = random.randint(0, len(chosen) - 1)
        p_remove = chosen[idx]
        new_chosen = [p for i, p in enumerate(chosen) if i != idx]
        new_set = set(new_chosen)
        
        # Check if new_chosen is still no-isosceles (it is, since we removed a point)
        # Check if it's still maximal
        is_max = True
        for q in itertools.product(range(N), repeat=2):
            if q not in new_set:
                if not _creates_isosceles(q, new_chosen):
                    is_max = False
                    break
        
        if is_max:
            # We successfully removed a point
            chosen = new_chosen
            chosen_set = new_set
            if len(chosen) < best_size:
                best = list(chosen)
                best_size = len(chosen)
        else:
            # Try adding back and maybe swapping
            pass
    
    return best

def _multi_start_local_search(time_limit=280):
    """Multiple restarts with local search."""
    start_time = time.time()
    best = None
    best_size = float('inf')
    
    # First, get a greedy solution
    greedy = _greedy_construct()
    best = list(greedy)
    best_size = len(greedy)
    
    # Try local search from greedy
    improved = _local_search(greedy, max_iters=3000, time_limit=time_limit - (time.time() - start_time))
    if len(improved) < best_size:
        best = improved
        best_size = len(improved)
    
    # Try different random orderings for greedy
    while time.time() - start_time < time_limit:
        random.shuffle_order = list(itertools.product(range(N), repeat=2))
        random.shuffle(random.shuffle_order)
        
        chosen = []
        for p in random.shuffle_order:
            if not _creates_isosceles(p, chosen):
                chosen.append(p)
        
        # Check if maximal
        if not _is_maximal(chosen):
            # Try to complete it greedily
            for p in itertools.product(range(N), repeat=2):
                if p not in set(chosen):
                    if not _creates_isosceles(p, chosen):
                        chosen.append(p)
        
        if len(chosen) < best_size:
            # Local search from this starting point
            remaining_time = time_limit - (time.time() - start_time)
            if remaining_time > 5:
                improved = _local_search(chosen, max_iters=1000, time_limit=remaining_time - 1)
                if len(improved) < best_size:
                    best = improved
                    best_size = len(improved)
            else:
                best = list(chosen)
                best_size = len(chosen)
    
    return best

def construct():
    """Construct a minimum maximal no-isosceles set."""
    start_time = time.time()
    time_limit = 280  # Leave safety margin before 300s limit
    
    # Use multi-start local search
    result = _multi_start_local_search(time_limit=time_limit)
    
    # Final verification
    assert all(not _creates_isosceles(p, [x for x in result if x != p]) for p in result), "Set is not no-isosceles"
    assert _is_maximal(result), "Set is not maximal"
    
    return result
# EVOLVE-BLOCK-END
