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

def _max_independent_set_branch_and_bound(compat, deadline):
    """
    Branch and bound for maximum independent set.
    compat: dict mapping vertex -> set of compatible vertices
    Returns the best independent set found before deadline.
    """
    n = 512
    vertices = list(range(1, 513))
    
    # Use bitmasks for efficient set operations
    # Map each vertex to an index
    vertex_to_idx = {v: i for i, v in enumerate(vertices)}
    idx_to_vertex = {i: v for i, v in enumerate(vertices)}
    
    # Build adjacency as bitmasks (compatible = edge in independent set context)
    # Actually, for independent set in compatibility graph:
    # Two vertices are "incompatible" if lcm > 512
    # We want max set where all pairs are compatible
    # So we build a graph where edges = compatible, and find max clique
    # Or build incompatibility graph and find max independent set
    
    # Let's build incompatibility bitmask: bit j set in mask[i] means i and j are INCOMPATIBLE
    incompat_mask = [0] * n
    for i in range(n):
        vi = vertices[i]
        for j in range(i + 1, n):
            vj = vertices[j]
            if _lcm(vi, vj) > 512:
                incompat_mask[i] |= (1 << j)
                incompat_mask[j] |= (1 << i)
    
    best_set = set()
    best_size = 0
    best_mask = 0
    
    # Upper bound: use greedy coloring on the remaining candidates
    def greedy_color_bound(candidates_mask):
        """
        Greedy coloring to get an upper bound on the size of a max independent set
        in the subgraph induced by candidates_mask.
        In the incompatibility graph, an independent set is a set with no edges.
        We color the incompatibility graph: vertices of same color form an independent set.
        The number of colors is an upper bound on the clique number... 
        Wait, I need to think about this more carefully.
        
        Actually for maximum independent set in graph G:
        - If we color G, the max independent set size <= number of colors (each color class is independent, but we want ONE independent set)
        - No wait: in a proper coloring, each color class is an independent set. The largest independent set has size at most the maximum color class size, but that's not a good bound.
        
        Better: use the complement. In the complement graph G', an independent set in G is a clique in G'.
        So max independent set in G = max clique in G'.
        For max clique, greedy coloring gives an upper bound: if we can color the subgraph with k colors, then the max clique is at most k.
        
        Let me just use a simple bound: the number of candidates is an upper bound, but we can do better.
        Use: for each vertex, 1 + max clique in its neighborhood in G' (which is non-neighborhood in G).
        
        Simpler approach: just use the count of candidates as a weak bound, and rely on the branch and bound structure.
        """
        # Simple bound: count of candidates
        return bin(candidates_mask).count('1')
    
    def bound_with_coloring(candidates_mask):
        """
        Greedy coloring bound for max independent set.
        In the incompatibility graph, we want max independent set.
        Equivalent to max clique in compatibility graph.
        For max clique: greedy coloring of the subgraph gives upper bound.
        
        Let me implement greedy coloring on the compatibility subgraph:
        - Vertices: those in candidates_mask
        - Edges: compatibility
        - Color greedily: assign smallest possible color
        - Max clique <= number of colors
        """
        if candidates_mask == 0:
            return 0
        
        # Get list of candidate indices
        candidates = []
        for i in range(n):
            if (candidates_mask >> i) & 1:
                candidates.append(i)
        
        if not candidates:
            return 0
        
        # Greedy coloring of the compatibility graph induced by candidates
        # color[v] = color assigned to vertex v
        color = {}
        next_color = 0
        for v in candidates:
            # Find colors used by compatible neighbors of v that are already colored
            used_colors = set()
            mask_v = incompat_mask[v]
            for w in candidates:
                if w == v:
                    continue
                if w not in color:
                    continue
                # Check if v and w are compatible (i.e., NOT incompatible)
                if not (mask_v >> w) & 1:
                    # They are compatible, so w's color is forbidden
                    used_colors.add(color[w])
            
            # Assign smallest available color
            c = 0
            while c in used_colors:
                c += 1
            color[v] = c
            next_color = max(next_color, c + 1)
        
        return next_color
    
    # Branch and bound
    def search(candidates_mask, current_set):
        nonlocal best_set, best_size, best_mask
        
        if time.time() > deadline:
            return
        
        # Upper bound
        ub = bound_with_coloring(candidates_mask)
        current_size = len(current_set)
        
        if current_size + ub <= best_size:
            return
        
        if candidates_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)
                best_mask = 0
                for v in current_set:
                    best_mask |= (1 << vertex_to_idx[v])
            return
        
        # Pick a vertex to branch on
        # Heuristic: pick vertex with highest degree in the incompatibility subgraph
        # (i.e., incompatible with the most candidates)
        best_v = -1
        best_degree = -1
        for i in range(n):
            if not (candidates_mask >> i) & 1:
                continue
            # Count incompatible neighbors in candidates
            deg = bin(incompat_mask[i] & candidates_mask).count('1')
            if deg > best_degree:
                best_degree = deg
                best_v = i
        
        if best_v == -1:
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)
            return
        
        v_vertex = vertices[best_v]
        
        # Branch 1: Include v
        new_set = current_set + [v_vertex]
        # New candidates: those compatible with v (i.e., not incompatible with v)
        # Incompatibility mask for v: bits set where incompatible
        # Compatible with v: ~(incompat_mask[best_v]) & candidates_mask, excluding v itself
        new_candidates = (candidates_mask & ~(incompat_mask[best_v] | (1 << best_v)))
        search(new_candidates, new_set)
        
        if time.time() > deadline:
            return
        
        # Branch 2: Exclude v
        new_candidates = candidates_mask & ~(1 << best_v)
        search(new_candidates, current_set)
    
    # Start with all candidates
    full_mask = (1 << n) - 1
    search(full_mask, [])
    
    return best_set

def _greedy_seeded_max_independent_set(compat, deadline):
    """
    Start with a greedy solution, then try to improve using local search / branch and bound
    on the remaining vertices.
    """
    n = 512
    vertices = list(range(1, 513))
    vertex_to_idx = {v: i for i, v in enumerate(vertices)}
    idx_to_vertex = {i: v for i, v in enumerate(vertices)}
    
    # Build incompatibility bitmask
    incompat_mask = [0] * n
    for i in range(n):
        vi = vertices[i]
        for j in range(i + 1, n):
            vj = vertices[j]
            if _lcm(vi, vj) > 512:
                incompat_mask[i] |= (1 << j)
                incompat_mask[j] |= (1 << i)
    
    # Start with greedy solution
    s = _greedy_by_compatibility()
    if not s:
        s = {1}
    
    best_set = set(s)
    best_size = len(s)
    
    # Build mask of current set
    current_mask = 0
    for v in s:
        idx = vertex_to_idx[v]
        current_mask |= (1 << idx)
    
    # Vertices not in current set
    remaining_mask = ((1 << n) - 1) & ~current_mask
    
    # Try to add vertices one by one (greedy improvement)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if not (remaining_mask >> i) & 1:
                continue
            v = vertices[i]
            # Check if v is compatible with all in current set
            ok = True
            for j in range(n):
                if (current_mask >> j) & 1:
                    if (incompat_mask[i] >> j) & 1:
                        ok = False
                        break
            if ok:
                best_set.add(v)
                current_mask |= (1 << i)
                remaining_mask &= ~(1 << i)
                best_size += 1
                improved = True
                break
    
    # Now try branch and bound on remaining to see if we can do better
    # But B&B on 512 vertices is too expensive. Let's try a limited B&B.
    
    # Alternative: try to remove some vertices and add others
    # For now, return the greedy improved set
    return best_set

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Strategy 1: Divisor set of a highly composite number
    div_candidate = _divisor_set_candidate()
    
    # Strategy 2: Greedy in increasing order starting from div_candidate
    greedy1 = _greedy_by_order(div_candidate)
    
    # Strategy 3: Greedy by compatibility (start empty)
    greedy2 = _greedy_by_compatibility()
    
    # Strategy 4: Improved greedy with local search
    greedy3 = _greedy_seeded_max_independent_set(None, EVOLVE_DEADLINE - 10)
    
    best = max([div_candidate, greedy1, greedy2, greedy3], key=len)
    
    # Strategy 5: Try divisor sets for all M and take best
    for M in range(1, 513):
        if time.time() - start_time > EVOLVE_DEADLINE - 10:
            break
        d = _divisors_up_to(M, 512)
        # Try to greedily augment
        aug = _greedy_by_order(set(d))
        if len(aug) > len(best):
            best = aug
    
    # Final augmentation attempts
    remaining_time = EVOLVE_DEADLINE - (time.time() - start_time)
    if remaining_time > 2:
        best = _try_augment(best, max_time=remaining_time - 1)
    
    # Try a more sophisticated local search: try swapping elements
    remaining_time = EVOLVE_DEADLINE - (time.time() - start_time)
    if remaining_time > 5:
        # Try to improve by removing one element and adding up to two
        best_set = set(best)
        improved = True
        while improved and time.time() - start_time < EVOLVE_DEADLINE - 2:
            improved = False
            for x in list(best_set):
                # Try removing x
                trial = best_set - {x}
                # Try adding elements not in trial
                for v in range(1, 513):
                    if v in trial:
                        continue
                    # Check if v is compatible with all in trial
                    ok = True
                    for u in trial:
                        if _lcm(u, v) > 512:
                            ok = False
                            break
                    if ok:
                        trial.add(v)
                        # Try to add one more
                        for w in range(1, 513):
                            if w in trial or w == v:
                                continue
                            ok2 = True
                            for u in trial:
                                if _lcm(u, w) > 512:
                                    ok2 = False
                                    break
                            if ok2:
                                trial.add(w)
                                break
                        if len(trial) > len(best_set):
                            best_set = trial
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    
    # Validation
    if not _validate(best_set):
        # Fallback to known valid set
        best_set = set(range(1, 21))
    
    return list(best_set)
# EVOLVE-BLOCK-END
