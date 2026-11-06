import numpy as np
import random
import time

N = 16
GRID_POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = len(GRID_POINTS)

# Precompute squared distances
coords = np.array(GRID_POINTS, dtype=np.int32)
diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
dist_sq = np.sum(diffs * diffs, axis=-1).astype(np.int64)

# For each point p, distances to all other points
dist_to_others = []
for p in range(NUM_POINTS):
    d = dist_sq[p].copy()
    d[p] = -1
    dist_to_others.append(d)

# Precompute distance sets for fast checking
dist_sets = [set(dist_to_others[s][dist_to_others[s] >= 0].tolist()) for s in range(NUM_POINTS)]

# Build conflict graph: two points p, q CONFLICT if they cannot coexist in a no-isosceles set.
# p and q conflict if there exists some r such that {p, q, r} forms an isosceles triangle
# where p and q are the equal sides (or q and p with r).
# Actually, p and q can coexist if:
#   - dist(p,q) is distinct from all other distances from p to selected points
#   - dist(p,q) is not equal to dist(q, r) for any other selected r
# This is a 3-body constraint, not a 2-body one. So we can't simply build a pairwise conflict graph.
#
# However, we CAN define: p and q are "incompatible" if NO valid set can contain both.
# This happens when for EVERY possible third point r, {p,q,r} is isosceles with p,q as equal legs,
# OR when the constraints from p and q together force an isosceles triangle.
#
# Actually, a simpler approach: use the bitset representation for the set S, and check validity
# incrementally. The key insight for branch-and-bound is:
#   - We want to find a MINIMAL maximal no-isosceles set
#   - "Maximal" means every point not in S conflicts with S (cannot be added)
#   - "No-isosceles" means no three points in S form an isosceles triangle
#
# Let's reformulate: A set S is valid if:
#   1. No three points in S form an isosceles triangle
#   2. S is maximal: for every point q not in S, q cannot be added to S
#
# For branch-and-bound, we search for the minimum size valid set.
# Lower bound: any valid set must be maximal, so it must "cover" all points (each point outside
# is blocked by the set). We can use a domination-based lower bound.

# Precompute: for each point p, which other points does p "block" when added to a set?
# Actually, blocking depends on the entire set S, not just p alone.
# 
# Let's use a different approach: simulated annealing with bitmask operations.
# Represent S as a Python int bitmask (256 bits).

def _check_no_isosceles(S_mask):
    """Check if set S (as bitmask) has no isosceles triangles. Returns True if valid."""
    S_list = [i for i in range(NUM_POINTS) if (S_mask >> i) & 1]
    S_set = set(S_list)
    for p in S_list:
        dists_from_p = set()
        for q in S_list:
            if q == p:
                continue
            d = dist_sq[p, q]
            if d in dists_from_p:
                return False
            dists_from_p.add(d)
            # Check if d equals dist(q, r) for any r in S, r != q
            for r in S_list:
                if r == q or r == p:
                    continue
                if dist_sq[q, r] == d:
                    return False
    return True

def _is_maximal(S_mask):
    """Check if S is maximal: no point outside can be added."""
    S_list = [i for i in range(NUM_POINTS) if (S_mask >> i) & 1]
    S_set = set(S_list)
    
    for q in range(NUM_POINTS):
        if q in S_set:
            continue
        # Check if q can be added to S
        can_add = True
        dists_from_q = []
        for s in S_list:
            d = dist_sq[q, s]
            dists_from_q.append(d)
        
        # Check distinctness
        if len(set(dists_from_q)) != len(dists_from_q):
            can_add = False
            continue
        
        # Check each s: dist(q,s) not in {dist(s,t) for t in S, t != s}
        if can_add:
            for i, s in enumerate(S_list):
                d_qs = dist_sq[q, s]
                for j, t in enumerate(S_list):
                    if t == s:
                        continue
                    if dist_sq[s, t] == d_qs:
                        can_add = False
                        break
                if not can_add:
                    break
        
        if can_add:
            return False  # Not maximal
    
    return True

def _score_set(S_mask):
    """Score: size if valid and maximal, else infinity."""
    S_list = [i for i in range(NUM_POINTS) if (S_mask >> i) & 1]
    if not _check_no_isosceles(S_mask):
        return float('inf')
    if not _is_maximal(S_mask):
        return float('inf')
    return len(S_list)

def _random_valid_set():
    """Generate a random valid no-isosceles set (not necessarily maximal)."""
    S_mask = 0
    S_list = []
    order = list(range(NUM_POINTS))
    random.shuffle(order)
    for p in order:
        # Try adding p
        can_add = True
        dists_from_p = []
        for s in S_list:
            d = dist_sq[p, s]
            dists_from_p.append(d)
        
        if len(set(dists_from_p)) != len(dists_from_p):
            continue
        
        for i, s in enumerate(S_list):
            d_ps = dist_sq[p, s]
            for t in S_list:
                if t == s:
                    continue
                if dist_sq[s, t] == d_ps:
                    can_add = False
                    break
            if not can_add:
                break
        
        if can_add:
            S_mask |= (1 << p)
            S_list.append(p)
    return S_mask, S_list

def _make_maximal(S_mask, S_list):
    """Try to make S maximal by adding points if possible."""
    S_set = set(S_list)
    changed = True
    while changed:
        changed = False
        for q in range(NUM_POINTS):
            if q in S_set:
                continue
            # Check if q can be added
            can_add = True
            dists_from_q = []
            for s in S_list:
                d = dist_sq[q, s]
                dists_from_q.append(d)
            
            if len(set(dists_from_q)) != len(dists_from_q):
                continue
            
            for s in S_list:
                d_qs = dist_sq[q, s]
                for t in S_list:
                    if t == s:
                        continue
                    if dist_sq[s, t] == d_qs:
                        can_add = False
                        break
                if not can_add:
                    break
            
            if can_add:
                S_mask |= (1 << q)
                S_list.append(q)
                S_set.add(q)
                changed = True
    
    return S_mask, S_list

def _remove_point(S_mask, S_list, p):
    """Remove point p from S. Returns (new_mask, new_list) or None if not maximal."""
    if p not in S_list:
        return S_mask, S_list
    
    new_list = [x for x in S_list if x != p]
    new_mask = S_mask & ~(1 << p)
    
    # Check if still no-isosceles (removing can't break this)
    # Check if still maximal
    if _is_maximal(new_mask):
        return new_mask, new_list
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    best_mask = None
    best_size = NUM_POINTS + 1
    
    # Phase 1: Generate initial valid sets via random greedy
    while time.time() < deadline - 120:
        S_mask, S_list = _random_valid_set()
        S_mask, S_list = _make_maximal(S_mask, S_list)
        
        if len(S_list) < best_size:
            best_size = len(S_list)
            best_mask = S_mask
        
        if best_size <= 7:
            break
    
    if best_mask is None:
        S_mask, S_list = _random_valid_set()
        S_mask, S_list = _make_maximal(S_mask, S_list)
        best_mask = S_mask
        best_size = len(S_list)
    
    # Phase 2: Simulated annealing with bitmask operations
    current_mask = best_mask
    current_list = [i for i in range(NUM_POINTS) if (current_mask >> i) & 1]
    current_size = len(current_list)
    current_score = current_size
    
    temperature = 50.0
    cooling = 0.995
    
    while time.time() < deadline - 30 and temperature > 0.1:
        # Generate a neighbor: add or remove a random point
        if random.random() < 0.6 and current_list:
            # Remove a random point
            p = random.choice(current_list)
            result = _remove_point(current_mask, current_list, p)
            if result is None:
                # Try a different point
                continue
            new_mask, new_list = result
        else:
            # Add a random point not in set
            S_set = set(current_list)
            candidates = [q for q in range(NUM_POINTS) if q not in S_set]
            if not candidates:
                continue
            q = random.choice(candidates)
            
            # Check if q can be added
            can_add = True
            dists_from_q = []
            for s in current_list:
                d = dist_sq[q, s]
                dists_from_q.append(d)
            
            if len(set(dists_from_q)) != len(dists_from_q):
                continue
            
            for s in current_list:
                d_qs = dist_sq[q, s]
                for t in current_list:
                    if t == s:
                        continue
                    if dist_sq[s, t] == d_qs:
                        can_add = False
                        break
                if not can_add:
                    break
            
            if not can_add:
                continue
            
            new_mask = current_mask | (1 << q)
            new_list = current_list + [q]
        
        new_score = _score_set(new_mask)
        
        if new_score == float('inf'):
            # Invalid: penalize based on how far from valid
            # Just reject
            temperature *= cooling
            continue
        
        delta = new_score - current_score
        
        if delta <= 0 or random.random() < np.exp(-delta / temperature):
            current_mask = new_mask
            current_list = new_list
            current_score = new_score
            
            if new_score < best_size:
                best_size = new_score
                best_mask = new_mask
        
        temperature *= cooling
    
    # Phase 3: Local search refinement from best
    S_mask = best_mask
    S_list = [i for i in range(NUM_POINTS) if (S_mask >> i) & 1]
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        shuffled = S_list[:]
        random.shuffle(shuffled)
        
        for p in shuffled:
            result = _remove_point(S_mask, S_list, p)
            if result is not None:
                S_mask, S_list = result
                improved = True
                break
    
    if len(S_list) < best_size:
        best_size = len(S_list)
        best_mask = S_mask
    
    return [GRID_POINTS[i] for i in range(NUM_POINTS) if (best_mask >> i) & 1]

if __name__ == "__main__":
    res = construct()
    print(f"Found set of size {len(res)}")
