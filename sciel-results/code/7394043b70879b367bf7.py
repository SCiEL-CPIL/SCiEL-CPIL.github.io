"""Maximum cardinality subset of [1,512] with pairwise lcm <= 512.

Strategy:
1. Build a conflict graph where an edge exists between a,b if lcm(a,b) > 512.
2. The problem is Maximum Independent Set on this graph.
3. Use a branch-and-bound solver with:
   - A strong initial lower bound from greedy + divisor set.
   - Upper bounds from greedy coloring on the remaining candidate set.
   - Vertex ordering by degree for efficient search.
   - Improved branching: process vertices in a smart order.
4. Local search (1-swap, 2-swap, multi-swap) to further expand the solution.
5. Additional structural exploitation: numbers that divide 512 are compatible with
   many others; use this to build a large core.
"""
import time
from math import gcd
from itertools import combinations

LIMIT = 512
DEADLINE = 85.0  # safety margin before 90s kill


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
    """Get a good initial solution using multiple strategies and take the best."""
    candidates = []
    
    # Strategy 1: Divisors of 360 plus greedy fill
    base1 = set()
    for d in [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24, 30, 36, 40, 45, 60, 72, 90, 120, 180, 360]:
        if d <= LIMIT:
            base1.add(d)
    base1 = _greedy_fill(base1, conflict)
    candidates.append(base1)
    
    # Strategy 2: Divisors of 240 plus greedy fill
    base2 = set()
    for d in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 30, 40, 48, 60, 80, 120, 240]:
        if d <= LIMIT:
            base2.add(d)
    base2 = _greedy_fill(base2, conflict)
    candidates.append(base2)
    
    # Strategy 3: Pure greedy by degree (prefer low-degree vertices)
    deg = [bin(conflict[i]).count("1") for i in range(LIMIT + 1)]
    order = list(range(1, LIMIT + 1))
    order.sort(key=lambda v: (deg[v], v))
    base3 = set()
    for v in order:
        if all(not ((conflict[v] >> u) & 1) for u in base3):
            base3.add(v)
    base3 = _greedy_fill(base3, conflict)
    candidates.append(base3)
    
    # Strategy 4: Numbers with many small prime factors (highly composite-like)
    # Prioritize numbers that divide many numbers in [1,512]
    base4 = set()
    # Add all powers of 2 up to 256, powers of 3 up to 243, etc.
    for v in range(1, LIMIT + 1):
        # Count how many numbers in [1,512] are multiples of v
        multiples = sum(1 for k in range(1, LIMIT + 1) if k % v == 0)
        if multiples >= 10:  # v divides at least 10 numbers
            base4.add(v)
    base4 = _greedy_fill(base4, conflict)
    candidates.append(base4)
    
    # Strategy 5: Start with small numbers
    base5 = set(range(1, 20))
    base5 = _greedy_fill(base5, conflict)
    candidates.append(base5)
    
    # Take the best
    best = max(candidates, key=len)
    return best


def _greedy_fill(base, conflict):
    """Greedily add compatible vertices to base."""
    improved = True
    while improved:
        improved = False
        for v in range(1, LIMIT + 1):
            if v in base:
                continue
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
    """Branch-and-bound for maximum independent set with improved heuristics."""
    n = LIMIT
    full_mask = (1 << (n + 1)) - 2  # bits 1..n
    
    # Initial solution
    best_set = _initial_solution(conflict)
    best_size = len(best_set)
    
    # Precompute degrees
    deg = [bin(conflict[i]).count("1") for i in range(n + 1)]
    
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
        
        # If we can potentially improve, try to extend greedily first
        # Quick greedy extension from current state
        if depth + min(bound, 20) > best_size:
            # Try to add vertices greedily
            trial_mask = current_mask
            trial_depth = depth
            for v in ordered:
                if not (cand_mask >> v) & 1:
                    continue
                # Check if v is compatible with all in current_mask
                if trial_mask & conflict[v]:
                    continue
                trial_mask |= (1 << v)
                trial_depth += 1
                if trial_depth > best_size:
                    best_size = trial_depth
                    best_set = {i for i in range(1, n + 1) if (trial_mask >> i) & 1}
        
        # Process vertices in reverse order (highest color first for tighter pruning)
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
    """Local search: try 1-swap, 2-swap, and pure additions."""
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
        
        # Try 1-swap: remove x, add y
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
                # Check if adding y allows adding more
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
        
        if found:
            continue
        
        # Try 2-swap: remove x1, x2, add y1, y2
        s_list = list(s)
        for i in range(len(s_list)):
            for j in range(i + 1, len(s_list)):
                if time.time() > deadline:
                    break
                x1, x2 = s_list[i], s_list[j]
                base = s - {x1, x2}
                for y1 in range(1, LIMIT + 1):
                    if y1 in base:
                        continue
                    if not compatible_with_all(y1, base):
                        continue
                    for y2 in range(1, LIMIT + 1):
                        if y2 in base or y2 == y1:
                            continue
                        if not compatible_with_all(y2, base | {y1}):
                            continue
                        # Check net gain
                        if len(base | {y1, y2}) > len(s):
                            s = base | {y1, y2}
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    
    return s


def _iterative_deepening(conflict, best_size, deadline):
    """Try to find a solution of size best_size+1, best_size+2, etc."""
    n = LIMIT
    full_mask = (1 << (n + 1)) - 2
    
    aborted = [False]
    
    def dfs(cand_mask, current_mask, depth, target):
        nonlocal aborted
        if aborted[0] or time.time() > deadline:
            aborted[0] = True
            return None
        
        if depth == target:
            return {i for i in range(1, n + 1) if (current_mask >> i) & 1}
        
        if cand_mask == 0:
            return None
        
        # Upper bound
        bound, ordered = _greedy_color_bound(cand_mask, conflict)
        if depth + bound < target:
            return None
        
        for v in reversed(ordered):
            if aborted[0] or time.time() > deadline:
                aborted[0] = True
                return None
            if not (cand_mask >> v) & 1:
                continue
            
            new_cand = cand_mask & ~(conflict[v] | (1 << v))
            result = dfs(new_cand, current_mask | (1 << v), depth + 1, target)
            if result is not None:
                return result
        
        return None
    
    # Try increasing targets
    for target in range(best_size + 1, best_size + 10):
        if time.time() > deadline:
            break
        result = dfs(full_mask, 0, 0, target)
        if result is not None:
            return result
        if aborted[0]:
            break
    
    return None


def construct():
    start = time.time()
    conflict = _build_conflict()
    
    # Phase 1: Get initial solution
    best = _initial_solution(conflict)
    best_size = len(best)
    
    # Phase 2: Branch and bound
    bb_deadline = start + DEADLINE - 10.0
    if time.time() < bb_deadline:
        bb_result = _max_independent_set(conflict, bb_deadline)
        if len(bb_result) > best_size:
            best = bb_result
            best_size = len(best)
    
    # Phase 3: Iterative deepening to push the boundary
    id_deadline = start + DEADLINE - 5.0
    if time.time() < id_deadline:
        id_result = _iterative_deepening(conflict, best_size, id_deadline)
        if id_result is not None and len(id_result) > best_size:
            best = id_result
            best_size = len(best)
    
    # Phase 4: Local search
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
