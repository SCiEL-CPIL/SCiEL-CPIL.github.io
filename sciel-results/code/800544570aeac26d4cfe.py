"""Combined approach: bitmask precomputation + multi-strategy local search for minimum maximal no-isosceles set."""
N = 16
import random
import time
import numpy as np

# EVOLVE-BLOCK-START
TOTAL = N * N

def _point_idx(p):
    return p[0] * N + p[1]

def _idx_point(i):
    return (i // N, i % N)

# Precompute isosceles conflict masks as bitmasks for fast operations
# iso_mask[i][j] = integer bitmask of all k such that {i, j, k} is isosceles
def _precompute():
    pts = [_idx_point(i) for i in range(TOTAL)]
    # dist_sq[i][j] = squared distance
    dist_sq = np.zeros((TOTAL, TOTAL), dtype=np.int32)
    for i in range(TOTAL):
        a = pts[i]
        for j in range(TOTAL):
            b = pts[j]
            dist_sq[i, j] = (a[0] - b[0])**2 + (a[1] - b[1])**2
    
    # iso_mask[i][j] = bitmask of k where {i,j,k} is isosceles
    iso_mask = np.zeros((TOTAL, TOTAL), dtype=object)
    for i in range(TOTAL):
        for j in range(TOTAL):
            if i == j:
                continue
            da = dist_sq[i, j]
            mask = 0
            for k in range(TOTAL):
                if k == i or k == j:
                    continue
                db = dist_sq[i, k]
                dc = dist_sq[j, k]
                if da == db or da == dc or db == dc:
                    mask |= (1 << k)
            iso_mask[i, j] = mask
    return iso_mask, dist_sq

_iso_mask, _dist_sq = _precompute()

def _compute_blocked_bitmask(chosen):
    """Given a list of chosen point indices, compute blocked as an integer bitmask."""
    if not chosen:
        return 0
    blocked = 0
    n = len(chosen)
    for i in range(n):
        for j in range(i + 1, n):
            blocked |= _iso_mask[chosen[i], chosen[j]]
    for p in chosen:
        blocked |= (1 << p)
    return blocked

def _is_maximal_bitmask(chosen, blocked):
    """Check if no unchosen point can be added."""
    if not chosen:
        return True
    cmask = 0
    for p in chosen:
        cmask |= (1 << p)
    # All points mask
    all_mask = (1 << TOTAL) - 1
    unchosen = all_mask & ~cmask
    # If any unchosen point is not blocked, not maximal
    return (unchosen & ~blocked) == 0

def _greedy_bitmask(order):
    """Greedy construction using bitmask operations. Returns (chosen_list, blocked_int)."""
    chosen = []
    blocked = 0
    for p in order:
        if (blocked >> p) & 1:
            continue
        # Add p
        for q in chosen:
            blocked |= _iso_mask[p, q]
        blocked |= (1 << p)
        chosen.append(p)
    return chosen, blocked

def _shrink_inclusion_minimal(chosen, blocked):
    """Repeatedly remove any point whose removal keeps the set maximal and no-isosceles."""
    changed = True
    while changed:
        changed = False
        n = len(chosen)
        for idx in range(n):
            p = chosen[idx]
            new_chosen = chosen[:idx] + chosen[idx+1:]
            if not new_chosen:
                new_blocked = 0
            else:
                new_blocked = _compute_blocked_bitmask(new_chosen)
            if _is_maximal_bitmask(new_chosen, new_blocked):
                chosen = new_chosen
                blocked = new_blocked
                changed = True
                break
    return chosen, blocked

def _perturb_repair_bitmask(chosen, blocked, n_remove, rng):
    """Remove n_remove random points, then greedily add back to restore maximality."""
    if len(chosen) <= n_remove:
        return None
    to_remove = set(rng.sample(chosen, n_remove))
    new_chosen = [p for p in chosen if p not in to_remove]
    new_blocked = _compute_blocked_bitmask(new_chosen)
    
    # Find unblocked, unchosen points
    cmask = 0
    for p in new_chosen:
        cmask |= (1 << p)
    all_mask = (1 << TOTAL) - 1
    unchosen = all_mask & ~cmask
    unblocked = unchosen & ~new_blocked
    
    if unblocked:
        # Get candidate indices
        candidates = []
        for p in range(TOTAL):
            if (unblocked >> p) & 1:
                candidates.append(p)
        rng.shuffle(candidates)
        for p in candidates:
            if (new_blocked >> p) & 1:
                continue
            # Add p
            for q in new_chosen:
                new_blocked |= _iso_mask[p, q]
            new_blocked |= (1 << p)
            new_chosen.append(p)
    
    if not _is_maximal_bitmask(new_chosen, new_blocked):
        return None
    return new_chosen, new_blocked

def _swap_move_bitmask(chosen, blocked, rng):
    """Try a swap: remove one point, add another, shrink, check size."""
    if len(chosen) < 2:
        return None
    
    idx = rng.randint(0, len(chosen) - 1)
    p_remove = chosen[idx]
    new_chosen = chosen[:idx] + chosen[idx+1:]
    new_blocked = _compute_blocked_bitmask(new_chosen)
    
    cmask = 0
    for p in new_chosen:
        cmask |= (1 << p)
    all_mask = (1 << TOTAL) - 1
    unchosen = all_mask & ~cmask
    unblocked = unchosen & ~new_blocked
    
    if not unblocked:
        if _is_maximal_bitmask(new_chosen, new_blocked):
            new_chosen, new_blocked = _shrink_inclusion_minimal(new_chosen, new_blocked)
            if len(new_chosen) < len(chosen):
                return new_chosen, new_blocked, True
        return None
    
    candidates = []
    for p in range(TOTAL):
        if (unblocked >> p) & 1:
            candidates.append(p)
    rng.shuffle(candidates)
    
    for p_add in candidates:
        if (new_blocked >> p_add) & 1:
            continue
        trial_chosen = new_chosen + [p_add]
        trial_blocked = new_blocked
        for q in new_chosen:
            trial_blocked |= _iso_mask[p_add, q]
        trial_blocked |= (1 << p_add)
        
        if _is_maximal_bitmask(trial_chosen, trial_blocked):
            trial_chosen, trial_blocked = _shrink_inclusion_minimal(trial_chosen, trial_blocked)
            if len(trial_chosen) < len(chosen):
                return trial_chosen, trial_blocked, True
    
    return None

def _constrained_greedy(rng, n_starts=30):
    """Greedy where we pick the point that blocks the most others."""
    all_points = list(range(TOTAL))
    best = None
    best_len = float('inf')
    for _ in range(n_starts):
        if time.time() > _DEADLINE:
            break
        # Randomize initial seed
        seed_points = rng.sample(all_points, k=min(3, len(all_points)))
        chosen = []
        for p in seed_points:
            if not _creates_isosceles_bitmask(p, chosen):
                chosen.append(p)
        # Now greedily add points, preferring those that are "most constrained"
        available = [p for p in all_points if p not in set(chosen)]
        while available:
            if time.time() > _DEADLINE:
                break
            # Score each available point by how many it would block
            best_p = None
            best_score = -1
            sample = available if len(available) <= 200 else rng.sample(available, 200)
            for p in sample:
                if _creates_isosceles_bitmask(p, chosen):
                    continue
                # Count how many other available points this would block
                blocks = 0
                for q in sample:
                    if q == p:
                        continue
                    if _creates_isosceles_bitmask(q, chosen + [p]):
                        blocks += 1
                if blocks > best_score:
                    best_score = blocks
                    best_p = p
            if best_p is None:
                break
            chosen.append(best_p)
            available.remove(best_p)
        if _is_maximal_bitmask(chosen, _compute_blocked_bitmask(chosen)) and len(chosen) < best_len:
            best = chosen
            best_len = len(chosen)
    return best

def _creates_isosceles_bitmask(p, chosen):
    """Check if adding p to chosen creates any isosceles triangle with two existing points."""
    n = len(chosen)
    for i in range(n):
        a = chosen[i]
        d_pa = _dist_sq[p, a]
        for j in range(i + 1, n):
            b = chosen[j]
            d_pb = _dist_sq[p, b]
            d_ab = _dist_sq[a, b]
            if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                return True
    return False

_DEADLINE = time.time() + 320  # safety margin before 330s kill

def construct():
    all_idx = list(range(TOTAL))
    best = None
    best_size = TOTAL + 1
    start = time.time()
    deadline = start + 320  # safety margin before 330s kill

    rng = random.Random(42)
    
    # First try a deterministic ordering (row-major) as baseline
    order = all_idx[:]
    chosen, blocked = _greedy_bitmask(order)
    if _is_maximal_bitmask(chosen, blocked):
        # Shrink
        chosen, blocked = _shrink_inclusion_minimal(chosen, blocked)
        if len(chosen) < best_size:
            best_size = len(chosen)
            best = chosen[:]

    # Try several different deterministic orderings
    # Diagonal ordering
    diag_order = []
    for d in range(2 * N - 1):
        for x in range(N):
            y = d - x
            if 0 <= y < N:
                diag_order.append(x * N + y)
    chosen, blocked = _greedy_bitmask(diag_order)
    if _is_maximal_bitmask(chosen, blocked):
        chosen, blocked = _shrink_inclusion_minimal(chosen, blocked)
        if len(chosen) < best_size:
            best_size = len(chosen)
            best = chosen[:]

    # Reverse row-major
    rev_order = all_idx[::-1]
    chosen, blocked = _greedy_bitmask(rev_order)
    if _is_maximal_bitmask(chosen, blocked):
        chosen, blocked = _shrink_inclusion_minimal(chosen, blocked)
        if len(chosen) < best_size:
            best_size = len(chosen)
            best = chosen[:]

    # Constrained-first greedy
    result = _constrained_greedy(rng, n_starts=20)
    if result is not None and len(result) < best_size:
        best_size = len(result)
        best = result[:]

    # Iterated local search
    current_chosen = best[:] if best else None
    current_blocked = _compute_blocked_bitmask(best) if best else None
    
    for attempt in range(100000):
        if time.time() > deadline:
            break
        
        if current_chosen is None:
            # Start fresh with random greedy
            order = all_idx[:]
            rng.shuffle(order)
            chosen, blocked = _greedy_bitmask(order)
            if not _is_maximal_bitmask(chosen, blocked):
                continue
            chosen, blocked = _shrink_inclusion_minimal(chosen, blocked)
            if len(chosen) < best_size:
                best_size = len(chosen)
                best = chosen[:]
            current_chosen = chosen[:]
            current_blocked = blocked
            continue
        
        # Alternate between perturb_repair and swap_move
        if rng.random() < 0.5:
            # Perturb and repair from current
            n_remove = rng.randint(1, max(1, len(current_chosen) // 3))
            result = _perturb_repair_bitmask(current_chosen, current_blocked, n_remove, rng)
            
            if result is not None:
                new_chosen, new_blocked = result
                # Shrink the new solution
                new_chosen, new_blocked = _shrink_inclusion_minimal(new_chosen, new_blocked)
                
                if len(new_chosen) < best_size:
                    best_size = len(new_chosen)
                    best = new_chosen[:]
                    current_chosen = new_chosen[:]
                    current_blocked = new_blocked
                elif len(new_chosen) < len(current_chosen):
                    # Accept if it improves over current (even if not global best)
                    current_chosen = new_chosen[:]
                    current_blocked = new_blocked
                
                if best_size <= 10:
                    break
            else:
                # Failed to repair; occasionally restart
                if rng.random() < 0.15:
                    current_chosen = None
                    current_blocked = None
        else:
            # Try swap move
            result = _swap_move_bitmask(current_chosen, current_blocked, rng)
            if result is not None:
                new_chosen, new_blocked, improved = result
                if improved:
                    if len(new_chosen) < best_size:
                        best_size = len(new_chosen)
                        best = new_chosen[:]
                    current_chosen = new_chosen[:]
                    current_blocked = new_blocked
                    if best_size <= 10:
                        break
            else:
                # Occasionally restart if stuck
                if rng.random() < 0.05:
                    current_chosen = None
                    current_blocked = None

    if best is None:
        best = all_idx[:]
    return [_idx_point(i) for i in best]

# EVOLVE-BLOCK-END
