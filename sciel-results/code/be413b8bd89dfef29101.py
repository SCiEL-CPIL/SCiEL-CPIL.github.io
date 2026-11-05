"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import random
import itertools

def _dist_sq(p, q):
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2

def _creates_isosceles(p, chosen):
    """Check if adding p to chosen creates any isosceles triangle."""
    n = len(chosen)
    for i in range(n):
        a = chosen[i]
        d_pa = _dist_sq(p, a)
        for j in range(i + 1, n):
            b = chosen[j]
            d_pb = _dist_sq(p, b)
            d_ab = _dist_sq(a, b)
            if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                return True
    return False

def _complete_to_maximal(base_set, all_points, rng):
    """
    Given a valid (no-isosceles) set base_set, add points to make it maximal.
    Uses a heuristic to pick points that block the most others.
    """
    chosen = list(base_set)
    chosen_set = set(chosen)
    remaining = [p for p in all_points if p not in chosen_set]
    
    while remaining:
        # Find valid candidates
        valid = []
        for p in remaining:
            if not _creates_isosceles(p, chosen):
                valid.append(p)
        
        if not valid:
            break
        
        # Heuristic: pick point that blocks most other valid points
        # To speed up, if valid is large, sample
        if len(valid) > 30:
            sample = rng.sample(valid, 30)
        else:
            sample = valid
            
        best_p = None
        best_block_count = -1
        
        for p in sample:
            block_count = 0
            for q in valid:
                if q == p:
                    continue
                # Check if q is blocked by adding p
                # q is blocked if _creates_isosceles(q, chosen + [p]) is True
                if _creates_isosceles(q, chosen + [p]):
                    block_count += 1
            
            if block_count > best_block_count:
                best_block_count = block_count
                best_p = p
                
        if best_p is None:
            best_p = rng.choice(valid)
            
        chosen.append(best_p)
        chosen_set.add(best_p)
        # Remove best_p from remaining
        remaining.remove(best_p)
        
        # Optional: Prune remaining points that are now blocked?
        # This is complex because a point q might be blocked by p, 
        # but we don't know if q would have been added later.
        # However, if q forms an isosceles triangle with p and some a in chosen,
        # then q can NEVER be added to chosen U {p}.
        # So we can remove such q from remaining immediately.
        new_remaining = []
        for q in remaining:
            if _creates_isosceles(q, chosen):
                # q is now invalid
                continue
            new_remaining.append(q)
        remaining = new_remaining

    return chosen

def _fast_greedy(all_points, rng):
    """Generate an initial maximal set."""
    return _complete_to_maximal([], all_points, rng)

def _local_search(initial_set, all_points, rng, max_iters=100):
    """
    Local search: try removing a point and re-completing.
    """
    current_set = list(initial_set)
    current_size = len(current_set)
    
    for _ in range(max_iters):
        improved = False
        # Shuffle order to avoid local minima bias
        points_to_try = list(current_set)
        rng.shuffle(points_to_try)
        
        for p in points_to_try:
            # Remove p
            base_set = [x for x in current_set if x != p]
            
            # Complete
            new_set = _complete_to_maximal(base_set, all_points, rng)
            
            if len(new_set) < current_size:
                current_set = new_set
                current_size = len(current_set)
                improved = True
                break # Restart the loop with the new set
        
        if not improved:
            break
            
    return current_set

def construct():
    all_points = list(itertools.product(range(N), repeat=2))
    
    best_set = None
    best_size = float('inf')
    
    # Multiple restarts
    for seed in range(50):
        rng = random.Random(seed)
        # 1. Initial greedy
        initial = _fast_greedy(all_points, rng)
        
        # 2. Local search
        refined = _local_search(initial, all_points, rng)
        
        if len(refined) < best_size:
            best_size = len(refined)
            best_set = refined
            
    return best_set
# EVOLVE-BLOCK-END
