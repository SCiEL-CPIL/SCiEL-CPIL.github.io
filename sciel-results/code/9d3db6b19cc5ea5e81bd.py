"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
from math import gcd, isqrt
from collections import defaultdict

EVOLVE_DEADLINE = 85.0  # safety margin before 90s kill

def _divisors_up_to(n, limit=512):
    """All divisors of n that are <= limit."""
    res = []
    for i in range(1, isqrt(n) + 1):
        if n % i == 0:
            res.append(i)
            if i != n // i and n // i <= limit:
                res.append(n // i)
            elif i != n // i and i <= limit:
                pass
    return res

def _lcm(a, b):
    return a * b // gcd(a, b)

def _validate(s):
    """Check that s is a non-empty subset of [1,512] with all pairwise lcm <= 512."""
    s = set(s)
    if not s:
        return False
    for x in s:
        if x < 1 or x > 512:
            return False
    lst = list(s)
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if _lcm(lst[i], lst[j]) > 512:
                return False
    return True

def _divisor_set_candidate():
    """Find the largest divisor set of any M in [1,512] (divisors must be <= 512)."""
    best = set()
    for M in range(1, 513):
        d = _divisors_up_to(M, 512)
        if len(d) > len(best):
            best = set(d)
    return best

def _greedy_by_order(initial_set, max_time=None):
    """Try to add numbers 1..512 in increasing order, checking lcm with all current members."""
    s = set(initial_set)
    start = time.time()
    for v in range(1, 513):
        if max_time is not None and time.time() - start > max_time:
            break
        if v in s:
            continue
        ok = True
        for u in s:
            if _lcm(u, v) > 512:
                ok = False
                break
        if ok:
            s.add(v)
    return s

def _greedy_by_compatibility():
    """Greedy: pick numbers that are 'compatible' with the most remaining candidates."""
    candidates = list(range(1, 513))
    s = set()
    remaining = set(candidates)
    # Precompute compatibility graph
    compat = {}
    for i in candidates:
        compat[i] = set()
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            if _lcm(candidates[i], candidates[j]) <= 512:
                compat[candidates[i]].add(candidates[j])
                compat[candidates[j]].add(candidates[i])
    
    while remaining:
        # Pick the element with max degree in the induced subgraph on (remaining + s)
        # Actually, simpler: pick the element that can coexist with the most other remaining elements
        # and also with all elements in s.
        best_v = None
        best_score = -1
        for v in list(remaining):
            # Check v is compatible with all in s
            ok = True
            for u in s:
                if v not in compat.get(u, set()) and _lcm(u, v) > 512:
                    ok = False
                    break
            if not ok:
                continue
            # Count how many remaining elements are compatible with v
            cnt = sum(1 for w in remaining if w != v and _lcm(v, w) <= 512)
            if cnt > best_score:
                best_score = cnt
                best_v = v
        if best_v is None:
            break
        s.add(best_v)
        remaining.discard(best_v)
    return s

def _try_augment(s, max_time=None):
    """Try to add one more number to s that is compatible with all members."""
    s = set(s)
    start = time.time()
    for v in range(1, 513):
        if max_time is not None and time.time() - start > max_time:
            break
        if v in s:
            continue
        ok = True
        for u in s:
            if _lcm(u, v) > 512:
                ok = False
                break
        if ok:
            s.add(v)
            return s
    return s

def _build_conflict_graph():
    """Build conflict graph as bitmasks. conflict[i] has bit j set if lcm(i+1, j+1) > 512."""
    N = 512
    conflict = [0] * N
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            g = gcd(a, b)
            if (a * b) // g > 512:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    return conflict

def _greedy_fill(conflict, initial_set=None, max_time=None):
    """Greedy fill: add vertices in order of decreasing degree in the compatibility graph."""
    N = 512
    if initial_set is None:
        current = set()
    else:
        current = set(initial_set)
    
    # Compute compatibility degrees
    # compatible_mask[i] = bits of j that are compatible with i
    compat_mask = [0] * N
    for i in range(N):
        compat_mask[i] = ((1 << N) - 1) & ~(conflict[i] | (1 << i))
    
    start = time.time()
    candidates = list(range(N))
    
    # Sort by degree in compatibility graph (higher degree first)
    candidates.sort(key=lambda i: bin(compat_mask[i]).count('1'), reverse=True)
    
    for i in candidates:
        if max_time is not None and time.time() - start > max_time:
            break
        if (i + 1) in current:
            continue
        # Check if i is compatible with all in current
        ok = True
        for v in current:
            idx = v - 1
            if (conflict[i] >> idx) & 1:
                ok = False
                break
        if ok:
            current.add(i + 1)
    
    return current

def _local_search(conflict, initial_set, max_time=None):
    """Local search: try 1-for-1 and 1-for-2 swaps to improve the set."""
    N = 512
    current = set(initial_set)
    start = time.time()
    
    # Precompute compatibility masks
    compat_mask = [0] * N
    for i in range(N):
        compat_mask[i] = ((1 << N) - 1) & ~(conflict[i] | (1 << i))
    
    improved = True
    while improved and (max_time is None or time.time() - start < max_time):
        improved = False
        
        # Try 1-for-1 swaps
        for x in list(current):
            idx_x = x - 1
            # Try removing x and adding y
            for y in range(1, N + 1):
                if y in current:
                    continue
                idx_y = y - 1
                # Check if y is compatible with all in current - {x}
                ok = True
                for v in current:
                    if v == x:
                        continue
                    idx_v = v - 1
                    if (conflict[idx_y] >> idx_v) & 1:
                        ok = False
                        break
                if ok:
                    # This swap is valid and keeps size same (1-for-1)
                    # But we want to increase size, so 1-for-1 doesn't help directly
                    # unless it enables further additions. Skip for now.
                    pass
        
        # Try adding a new element (if possible)
        for y in range(1, N + 1):
            if y in current:
                continue
            idx_y = y - 1
            ok = True
            for v in current:
                idx_v = v - 1
                if (conflict[idx_y] >> idx_v) & 1:
                    ok = False
                    break
            if ok:
                current.add(y)
                improved = True
                break
    
    return current

def _branch_and_bound_mis(conflict, deadline):
    """
    Branch and bound for Maximum Independent Set on the conflict graph.
    An independent set in the conflict graph = a set with all pairwise lcm <= 512.
    """
    N = 512
    vertices = list(range(N))  # indices 0..511 representing values 1..512
    
    # Precompute compatibility masks
    compat_mask = [0] * N
    for i in range(N):
        compat_mask[i] = ((1 << N) - 1) & ~(conflict[i] | (1 << i))
    
    best_set = set()
    best_size = 0
    
    # Greedy coloring bound for upper bound on independent set size
    # In the conflict graph, we want max independent set.
    # Equivalent to max clique in the compatibility graph.
    # For max clique, greedy coloring of the induced subgraph gives an upper bound.
    
    def greedy_color_bound(candidates_mask):
        """
        Greedy coloring of the compatibility graph induced by candidates_mask.
        Returns the number of colors, which is an upper bound on the max clique size
        (and thus on the max independent set size in the conflict graph).
        """
        if candidates_mask == 0:
            return 0
        
        candidates = []
        for i in range(N):
            if (candidates_mask >> i) & 1:
                candidates.append(i)
        
        if not candidates:
            return 0
        
        # Greedy coloring of the compatibility graph
        color = {}
        next_color = 0
        for v in candidates:
            used_colors = set()
            for w in candidates:
                if w == v or w not in color:
                    continue
                # Check if v and w are compatible
                if (compat_mask[v] >> w) & 1:
                    used_colors.add(color[w])
            
            c = 0
            while c in used_colors:
                c += 1
            color[v] = c
            next_color = max(next_color, c + 1)
        
        return next_color
    
    # Branch and bound search
    def search(candidates_mask, current_set):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
        
        # Upper bound
        ub = greedy_color_bound(candidates_mask)
        current_size = len(current_set)
        
        if current_size + ub <= best_size:
            return
        
        if candidates_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)
            return
        
        # Pick a vertex to branch on
        # Heuristic: pick vertex with highest degree in the conflict subgraph
        # (i.e., incompatible with the most candidates)
        best_v = -1
        best_degree = -1
        for i in range(N):
            if not (candidates_mask >> i) & 1:
                continue
            deg = bin(conflict[i] & candidates_mask).count('1')
            if deg > best_degree:
                best_degree = deg
                best_v = i
        
        if best_v == -1:
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)
            return
        
        v_val = best_v + 1
        
        # Branch 1: Include v
        new_set = current_set | {v_val}
        # New candidates: those compatible with v
        new_candidates = candidates_mask & compat_mask[best_v]
        search(new_candidates, new_set)
        
        if time.time() > deadline:
            return
        
        # Branch 2: Exclude v
        new_candidates = candidates_mask & ~(1 << best_v)
        search(new_candidates, current_set)
    
    # Start with all candidates
    full_mask = (1 << N) - 1
    search(full_mask, set())
    
    return best_set

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Build conflict graph once
    conflict = _build_conflict_graph()
    
    # Strategy 1: Divisor set of a highly composite number
    div_candidate = _divisor_set_candidate()
    
    # Strategy 2: Greedy in increasing order starting from div_candidate
    greedy1 = _greedy_by_order(div_candidate)
    
    # Strategy 3: Greedy by compatibility (start empty)
    greedy2 = _greedy_by_compatibility()
    
    # Strategy 4: Greedy fill using conflict graph
    greedy3 = _greedy_fill(conflict, initial_set=None, max_time=EVOLVE_DEADLINE - 20)
    
    # Strategy 5: Greedy fill starting from divisor candidate
    greedy4 = _greedy_fill(conflict, initial_set=div_candidate, max_time=EVOLVE_DEADLINE - 20)
    
    # Strategy 6: Local search on the best so far
    best = max([div_candidate, greedy1, greedy2, greedy3, greedy4], key=len)
    
    # Try local search to improve
    remaining_time = EVOLVE_DEADLINE - (time.time() - start_time)
    if remaining_time > 10:
        improved = _local_search(conflict, best, max_time=remaining_time - 5)
        if len(improved) > len(best):
            best = improved
    
    # Strategy 7: Branch and bound with time limit
    remaining_time = EVOLVE_DEADLINE - (time.time() - start_time)
    if remaining_time > 15:
        bb_result = _branch_and_bound_mis(conflict, deadline=start_time + remaining_time - 2)
        if len(bb_result) > len(best):
            best = bb_result
    
    # Final augmentation attempts
    remaining_time = EVOLVE_DEADLINE - (time.time() - start_time)
    if remaining_time > 2:
        best = _try_augment(best, max_time=remaining_time - 1)
    
    # Validation
    if not _validate(best):
        # Fallback to known valid set
        best = set(range(1, 21))
    
    return list(best)
# EVOLVE-BLOCK-END
