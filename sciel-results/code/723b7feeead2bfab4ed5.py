"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
import time
import random
import itertools

# EVOLVE-BLOCK-START
def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _is_iso(a, b, c):
    d1 = _d2(a, b)
    d2 = _d2(a, c)
    d3 = _d2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3

def _precompute_conflicts(all_points):
    """Precompute for each point p, the set of pairs (i,j) such that (p, pts[i], pts[j]) is isosceles.
    Returns a list where conflicts[p] is a set of frozenset({i,j})."""
    n = len(all_points)
    conflicts = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            a = all_points[i]
            b = all_points[j]
            for k in range(n):
                if k != i and k != j:
                    c = all_points[k]
                    if _is_iso(a, b, c):
                        conflicts[k].add(frozenset((i, j)))
    return conflicts

def _can_add(p_idx, chosen_set, conflicts):
    """Check if point p_idx can be added to chosen_set without creating isosceles triangle."""
    for pair in conflicts[p_idx]:
        if pair.issubset(chosen_set):
            return False
    return True

def _is_maximal(chosen_set, all_points, conflicts):
    """Check if chosen_set is maximal."""
    for p_idx in range(len(all_points)):
        if p_idx in chosen_set:
            continue
        if _can_add(p_idx, chosen_set, conflicts):
            return False
    return True

def _make_maximal(chosen_set, all_points, conflicts):
    """Add points to chosen_set until it's maximal."""
    changed = True
    while changed:
        changed = False
        for p_idx in range(len(all_points)):
            if p_idx not in chosen_set:
                if _can_add(p_idx, chosen_set, conflicts):
                    chosen_set.add(p_idx)
                    changed = True
    return chosen_set

def _prune(chosen_set, all_points, conflicts, time_limit=None, start_time=None):
    """Remove redundant points from a maximal set."""
    changed = True
    while changed:
        if start_time is not None and time_limit is not None:
            if time.time() - start_time > time_limit:
                break
        changed = False
        for p_idx in list(chosen_set):
            test_set = chosen_set - {p_idx}
            if _is_maximal(test_set, all_points, conflicts):
                chosen_set.discard(p_idx)
                changed = True
                break
    return chosen_set

def _greedy_coverage(all_points, conflicts, rng=None, k=5):
    """Greedy: pick point that blocks most other addable points."""
    n = len(all_points)
    chosen_set = set()
    remaining = set(range(n))
    
    while remaining:
        # Find addable points
        addable = [p for p in remaining if _can_add(p, chosen_set, conflicts)]
        if not addable:
            break
        
        if len(addable) == 1:
            p_idx = addable[0]
        else:
            # Score by how many other addable points it would block
            # A point q is blocked by p if there exists a in chosen_set such that (p, q, a) is isosceles
            # i.e., if there's a pair involving q and some a in chosen_set that conflicts with p
            # Actually: q is blocked by p iff _can_add(q, chosen_set | {p}, conflicts) is False
            # which means there exists a pair in (chosen_set | {p}) that conflicts with q
            # Since q is addable w.r.t. chosen_set, the only new conflicts come from pairs involving p
            # So q is blocked iff there exists a in chosen_set such that (q, p, a) is isosceles
            # i.e., frozenset((p_idx, a)) is in conflicts[q]
            
            best_p = addable[0]
            best_score = -1
            for p_idx in addable:
                score = 0
                for q_idx in addable:
                    if q_idx == p_idx:
                        continue
                    # Check if q would be blocked by p
                    blocked = False
                    for a_idx in chosen_set:
                        if frozenset((p_idx, a_idx)) in conflicts[q_idx]:
                            blocked = True
                            break
                    if blocked:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_p = p_idx
            
            p_idx = best_p
        
        chosen_set.add(p_idx)
        remaining.discard(p_idx)
    
    return chosen_set

def _greedy_random(all_points, conflicts, rng, k=3):
    """Randomized greedy: pick from top-k by coverage score."""
    n = len(all_points)
    chosen_set = set()
    remaining = set(range(n))
    
    while remaining:
        addable = [p for p in remaining if _can_add(p, chosen_set, conflicts)]
        if not addable:
            break
        
        if len(addable) == 1:
            p_idx = addable[0]
        else:
            scored = []
            for p_idx in addable:
                score = 0
                for q_idx in addable:
                    if q_idx == p_idx:
                        continue
                    blocked = False
                    for a_idx in chosen_set:
                        if frozenset((p_idx, a_idx)) in conflicts[q_idx]:
                            blocked = True
                            break
                    if blocked:
                        score += 1
                scored.append((score, p_idx))
            scored.sort(key=lambda x: -x[0])
            kk = min(k, len(scored))
            p_idx = scored[rng.randint(0, kk-1)][1]
        
        chosen_set.add(p_idx)
        remaining.discard(p_idx)
    
    return chosen_set

def _is_valid_no_iso(chosen_points):
    """Verify no three points form isosceles triangle."""
    for i in range(len(chosen_points)):
        for j in range(i+1, len(chosen_points)):
            for k in range(j+1, len(chosen_points)):
                if _is_iso(chosen_points[i], chosen_points[j], chosen_points[k]):
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 280
    
    all_points = list(itertools.product(range(N), repeat=2))
    n = len(all_points)
    
    # Precompute conflicts
    conflicts = _precompute_conflicts(all_points)
    
    best_set = None
    best_len = float('inf')
    
    # Strategy 1: Greedy with coverage
    chosen = _greedy_coverage(all_points, conflicts)
    chosen = _make_maximal(chosen, all_points, conflicts)
    chosen = _prune(chosen, all_points, conflicts, time_limit=30, start_time=start_time)
    if len(chosen) < best_len:
        best_set = set(chosen)
        best_len = len(chosen)
    
    # Strategy 2: Multiple randomized greedy attempts
    rng = random.Random(42)
    for attempt in range(30):
        if time.time() > deadline - 60:
            break
        
        chosen = _greedy_random(all_points, conflicts, rng, k=3)
        chosen = _make_maximal(chosen, all_points, conflicts)
        chosen = _prune(chosen, all_points, conflicts, time_limit=20, start_time=start_time)
        if len(chosen) < best_len:
            best_set = set(chosen)
            best_len = len(chosen)
    
    # Strategy 3: Different deterministic orderings
    for order_name in ['row', 'diag', 'parity']:
        if time.time() > deadline - 30:
            break
        
        if order_name == 'row':
            order = list(range(n))
        elif order_name == 'diag':
            order = sorted(range(n), key=lambda i: (all_points[i][0]+all_points[i][1], all_points[i][0]))
        else:
            order = sorted(range(n), key=lambda i: (all_points[i][0]+all_points[i][1]) % 2)
        
        chosen = set()
        for p_idx in order:
            if _can_add(p_idx, chosen, conflicts):
                chosen.add(p_idx)
        
        chosen = _make_maximal(chosen, all_points, conflicts)
        chosen = _prune(chosen, all_points, conflicts, time_limit=10, start_time=start_time)
        if len(chosen) < best_len:
            best_set = set(chosen)
            best_len = len(chosen)
    
    # Final verification
    if best_set is not None:
        best_points = [all_points[i] for i in best_set]
        if not _is_valid_no_iso(best_points):
            # Fallback to simple greedy
            chosen = set()
            for p_idx in range(n):
                if _can_add(p_idx, chosen, conflicts):
                    chosen.add(p_idx)
            best_set = chosen
            best_points = [all_points[i] for i in best_set]
        
        # Ensure maximal
        best_set = _make_maximal(best_set, all_points, conflicts)
        best_points = [all_points[i] for i in best_set]
    
    return best_points
# EVOLVE-BLOCK-END
