"""Maximum cardinality subset of [1,512] with pairwise lcm <= 512.

Strategy:
1. Build a conflict graph where an edge exists between a,b if lcm(a,b) > 512.
2. The problem is Maximum Independent Set on this graph.
3. Use a branch-and-bound solver with:
   - A strong initial lower bound from the divisor set of 360 (size 24) plus greedy fill.
   - Upper bounds from greedy coloring on the remaining candidate set.
   - Vertex ordering by degree for efficient search.
4. Local search (1-swap, 2-swap) to further expand the solution.
"""
import time
from math import gcd

LIMIT = 512
DEADLINE = 87.0  # safety margin before 90s kill


def _lcm(a, b):
    return a * b // gcd(a, b)


def _validate(s):
    s = list(set(s))
    if not s:
        return False
    for x in s:
        if x < 1 or x > LIMIT:
            return False
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            if _lcm(s[i], s[j]) > LIMIT:
                return False
    return True


def _build_conflict():
    """conflict[a] is a bitmask of vertices b (1..512) such that lcm(a,b) > 512."""
    n = LIMIT
    conflict = [0] * (n + 1)
    for a in range(1, n + 1):
        mask = 0
        for b in range(a + 1, n + 1):
            if _lcm(a, b) > LIMIT:
                mask |= (1 << b)
                conflict[b] |= (1 << a)
        conflict[a] = mask
    return conflict


def _greedy_color_bound(cand_mask, conflict):
    """Greedy coloring of the candidate set to get an upper bound on independent set size.
    Returns (bound, ordered_vertices) where ordered_vertices is sorted by color ascending."""
    uncolored = cand_mask
    color = 0
    colors = {}
    while uncolored:
        color += 1
        avail = uncolored
        while avail:
            # Pick vertex with highest bit for determinism
            v_bit = 1 << (avail.bit_length() - 1)
            v = v_bit.bit_length() - 1
            colors[v] = color
            # Remove v and its neighbors (conflicts) from avail
            avail &= ~(v_bit | (conflict[v] & uncolored))
            uncolored &= ~v_bit
    ordered = sorted(colors.keys(), key=lambda v: (colors[v], v))
    return color, ordered


def _initial_solution(conflict):
    """Get a good initial solution: divisors of 360 plus greedy fill."""
    # Divisors of 360: 1,2,3,4,5,6,8,9,10,12,15,18,20,24,30,36,40,45,60,72,90,120,180,360
    base = set()
    for d in [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24, 30, 36, 40, 45, 60, 72, 90, 120, 180, 360]:
        if d <= LIMIT:
            base.add(d)
    
    # Greedy fill: add any vertex compatible with all in base
    improved = True
    while improved:
        improved = False
        for v in range(1, LIMIT + 1):
            if v in base:
                continue
            # Check v compatible with all in base
            ok = True
            for u in base:
                if v != u and (conflict[v] >> u) & 1:
                    ok = False
                    break
            if ok:
                base.add(v)
                improved = True
    return base


def _max_independent_set(conflict, deadline):
    """Branch-and-bound for maximum independent set."""
    n = LIMIT
    full_mask = (1 << (n + 1)) - 2  # bits 1..n
    
    # Initial solution
    best_set = _initial_solution(conflict)
    best_size = len(best_set)
    
    # Order vertices by degree (number of conflicts) ascending for better pruning
    deg = [bin(conflict[i]).count("1") for i in range(n + 1)]
    order = list(range(1, n + 1))
    order.sort(key=lambda v: deg[v])
    
    # Precompute prefix masks
    prefix = [0] * (n + 1)
    for k in range(1, n + 1):
        prefix[k] = prefix[k - 1] | (1 << order[k - 1])
    
    aborted = [False]
    
    def branch(cand_mask, current_mask, depth):
        nonlocal best_size, best_set, aborted
        if aborted[0] or time.time() > deadline:
            aborted[0] = True
            return
        
        if cand_mask == 0:
            if depth > best_size:
                best_size = depth
                best_set = {i for i in range(1, n + 1) if (current_mask >> i) & 1}
            return
        
        # Upper bound via greedy coloring
        bound, ordered = _greedy_color_bound(cand_mask, conflict)
        if depth + bound <= best_size:
            return
        
        # Process vertices in reverse order (highest color first)
        for v in reversed(ordered):
            if aborted[0] or time.time() > deadline:
                aborted[0] = True
                return
            if not (cand_mask >> v) & 1:
                continue
            
            # Branch 1: include v
            if depth + 1 > best_size:
                new_cand = cand_mask & ~(conflict[v] | (1 << v))
                branch(new_cand, current_mask | (1 << v), depth + 1)
            
            # Branch 2: exclude v
            cand_mask &= ~(1 << v)
            
            if cand_mask == 0:
                break
            
            # Recheck bound
            cb, _ = _greedy_color_bound(cand_mask, conflict)
            if depth + cb <= best_size:
                break
    
    branch(full_mask, 0, 0)
    return best_set


def _local_search(conflict, s, deadline):
    """Local search: try 1-swap and 2-swap improvements."""
    s = set(s)
    start = time.time()
    
    def compatible_with_all(v, base_set):
        for u in base_set:
            if v != u and (conflict[v] >> u) & 1:
                return False
        return True
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try pure addition
        for y in range(1, LIMIT + 1):
            if y in s:
                continue
            if compatible_with_all(y, s):
                s.add(y)
                improved = True
                break
        
        if improved:
            continue
        
        # Try 1-swap: remove x, add y, then try to add z
        found = False
        for x in list(s):
            if time.time() > deadline:
                break
            base = s - {x}
            for y in range(1, LIMIT + 1):
                if y in base:
                    continue
                if not compatible_with_all(y, base):
                    continue
                s_trial = base | {y}
                # Try to add another
                for z in range(1, LIMIT + 1):
                    if z in s_trial:
                        continue
                    if compatible_with_all(z, s_trial):
                        s = s_trial | {z}
                        improved = True
                        found = True
                        break
                if found:
                    break
            if found:
                break
    
    return s


def construct():
    start = time.time()
    conflict = _build_conflict()
    
    # Phase 1: Branch and bound
    bb_deadline = start + DEADLINE - 5.0
    if time.time() < bb_deadline:
        best = _max_independent_set(conflict, bb_deadline)
    else:
        best = _initial_solution(conflict)
    
    # Phase 2: Local search
    ls_deadline = start + DEADLINE - 1.0
    if time.time() < ls_deadline and best:
        best = _local_search(conflict, best, ls_deadline)
    
    # Validate
    if not _validate(best):
        # Fallback: greedy
        best = set()
        for v in range(1, LIMIT + 1):
            if all(_lcm(u, v) <= LIMIT for u in best):
                best.add(v)
    
    if not best:
        best = {1}
    
    return list(best)
