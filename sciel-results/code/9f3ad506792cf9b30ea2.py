"""Combined program: fast greedy + branch-and-bound MIS solver with structural pruning."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

N = 192
K = 3
MAX_ELEM = 48

def _can_add(x, selected):
    """Check if x can be added to selected without violating constraints."""
    m = len(selected)
    
    # Check: does x divide sum of any 3 from selected?
    if m >= 3:
        for i in range(m):
            for j in range(i + 1, m):
                for k in range(j + 1, m):
                    s = selected[i] + selected[j] + selected[k]
                    if s % x == 0:
                        return False
    
    # Check: for each a in selected, does a divide sum of any 3 from (selected ∪ {x}) \ {a}?
    # Only need to check triples that include x: x + 2 from selected\{a}
    for i, a in enumerate(selected):
        others_no_a = [selected[j] for j in range(m) if j != i]
        if len(others_no_a) >= 2:
            for j in range(len(others_no_a)):
                for k in range(j + 1, len(others_no_a)):
                    s = others_no_a[j] + others_no_a[k] + x
                    if s % a == 0:
                        return False
    return True

def _greedy_build(order, max_elem=MAX_ELEM):
    """Greedy build following given order."""
    selected = []
    for x in order:
        if len(selected) >= max_elem:
            break
        if _can_add(x, selected):
            selected.append(x)
    return selected

def _precompute_conflicts():
    """Precompute conflict hyperedges: for each a, set of triples (b,c,d) where a|(b+c+d).
    Returns dict: a -> set of frozensets of triples that conflict with a.
    Also returns conflict_edges: list of (a, frozenset({b,c,d})) for all conflicts."""
    conflicts = {}
    for a in range(1, N + 1):
        conflicts[a] = set()
    
    # For each a, find all triples (b,c,d) of distinct integers != a such that a | (b+c+d)
    for a in range(1, N + 1):
        # b, c, d must be distinct and != a
        for b in range(1, N + 1):
            if b == a:
                continue
            for c in range(b + 1, N + 1):
                if c == a:
                    continue
                for d in range(c + 1, N + 1):
                    if d == a:
                        continue
                    s = b + c + d
                    if s % a == 0:
                        conflicts[a].add(frozenset([b, c, d]))
    
    return conflicts

def _branch_and_bound(candidates, current_selected, best_size, deadline, conflicts):
    """Branch and bound for maximum independent set in conflict hypergraph.
    
    candidates: sorted list of remaining candidate integers
    current_selected: list of already selected integers
    best_size: current best cardinality found
    deadline: time limit
    conflicts: precomputed conflict map
    
    Returns: best_size, best_solution
    """
    best = best_size
    best_sol = list(current_selected)
    
    # Memoization: key is frozenset of candidates, value is max additional elements we can add
    memo = {}
    
    def _bound(candidates):
        """Upper bound on how many more elements we can add from candidates.
        Uses greedy matching on conflict graph as a lower bound on vertex cover,
        so n - vertex_cover is an upper bound on independent set."""
        if not candidates:
            return 0
        
        # Simple bound: count of candidates (trivial)
        # Better bound: use the fact that each selected element conflicts with others
        # For now, use a simple heuristic bound
        n_cand = len(candidates)
        
        # Try greedy matching to get a lower bound on minimum vertex cover
        # This gives an upper bound on maximum independent set
        remaining = set(candidates)
        matched = 0
        used = set()
        
        # Greedy: repeatedly pick a conflict edge, match both endpoints
        # Actually, this is a 4-uniform hypergraph, so let's use a simpler bound
        # Each element a in candidates conflicts with some triples. 
        # A simpler bound: if we have c candidates, the max we can add is at most c
        # But we can do better: if many candidates conflict with each other, we can add fewer
        
        # Use a simple greedy independent set on the conflict graph among candidates
        # to get a lower bound, then n - that is an upper bound... no that's circular
        # Let's just use n_cand as the bound for now, and rely on pruning via current_selected
        
        return n_cand
    
    def _dfs(candidates, current_selected, depth):
        nonlocal best, best_sol
        
        if time.time() > deadline:
            return
        
        if not candidates:
            if len(current_selected) > best:
                best = len(current_selected)
                best_sol = list(current_selected)
            return
        
        # Pruning: even if we add all remaining candidates, can we beat best?
        if len(current_selected) + len(candidates) <= best:
            return
        
        # Memoization check
        cand_key = frozenset(candidates)
        if cand_key in memo:
            if memo[cand_key] + len(current_selected) <= best:
                return
        else:
            memo[cand_key] = 0  # Will update
        
        # Branch: pick the first candidate
        x = candidates[0]
        rest = candidates[1:]
        
        # Option 1: Include x
        if _can_add(x, current_selected):
            new_selected = current_selected + [x]
            # Filter candidates: remove any that conflict with x
            # An element y conflicts with x if x | (y + a + b) for some a,b in current_selected
            # or y | (x + a + b) for some a,b in current_selected
            # Actually, we need to check: can y be added to (current_selected + [x])?
            # We can pre-filter: y is incompatible with x if there exist a,b in current_selected
            # such that x | (y+a+b) or y | (x+a+b)
            # This is complex; let's just filter by trying _can_add for each remaining candidate
            
            filtered = []
            for y in rest:
                if _can_add(y, new_selected):
                    filtered.append(y)
            
            _dfs(filtered, new_selected, depth + 1)
        
        # Option 2: Exclude x
        _dfs(rest, current_selected, depth + 1)
        
        # Update memo: the best we could get from this candidate set
        # This is an approximation; in practice, we don't easily compute the exact max
        # So we skip memoization for now
    
    _dfs(list(candidates), list(current_selected), 0)
    
    return best, best_sol

def _local_search(selected, deadline):
    """Local search: try additions and swaps to improve."""
    best = list(selected)
    
    while time.time() < deadline:
        improved = False
        candidates = [x for x in range(N, 0, -1) if x not in best]
        
        # Phase 1: Try pure additions
        for x in candidates:
            if time.time() > deadline:
                break
            if _can_add(x, best):
                best.append(x)
                improved = True
                break
        
        if improved:
            continue
        
        # Phase 2: Try swaps
        for i in range(len(best)):
            if time.time() > deadline:
                break
            removed = best[i]
            remaining = best[:i] + best[i+1:]
            
            for x in candidates:
                if x == removed:
                    continue
                if _can_add(x, remaining):
                    best = remaining + [x]
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best

def _two_swap_local_search(selected, deadline):
    """Enhanced local search with two-swap moves."""
    best = list(selected)
    
    while time.time() < deadline:
        improved = False
        candidates = [x for x in range(N, 0, -1) if x not in best]
        
        # Try single additions
        for x in candidates:
            if time.time() > deadline:
                break
            if _can_add(x, best):
                best.append(x)
                improved = True
                break
        if improved:
            continue
        
        # Try single swaps
        for i in range(len(best)):
            if time.time() > deadline:
                break
            removed = best[i]
            remaining = best[:i] + best[i+1:]
            for x in candidates:
                if x == removed:
                    continue
                if _can_add(x, remaining):
                    best = remaining + [x]
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
        
        # Try two-swap: remove two, add two
        if len(best) >= 2:
            for i in range(len(best)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(best)):
                    if time.time() > deadline:
                        break
                    removed1, removed2 = best[i], best[j]
                    remaining = [v for k, v in enumerate(best) if k != i and k != j]
                    
                    for x in candidates[:15]:
                        if x == removed1 or x == removed2:
                            continue
                        if _can_add(x, remaining):
                            temp1 = remaining + [x]
                            for y in candidates[:15]:
                                if y == removed1 or y == removed2 or y == x:
                                    continue
                                if _can_add(y, temp1):
                                    best = temp1 + [y]
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return best

def construct():
    deadline = time.time() + 80.0  # 10s safety margin
    
    # Precompute conflicts for potential use
    # (This is expensive, so we do it lazily or skip if time is tight)
    
    # Strategy 1: Greedy from large to small
    best = _greedy_build(list(range(N, 0, -1)))
    
    # Strategy 2: Greedy from small to large
    alt = _greedy_build(list(range(1, N + 1)))
    if len(alt) > len(best):
        best = alt
    
    # Strategy 3: Interval-based greedy
    for hi_start in [192, 180, 168, 156, 144, 132, 120, 108]:
        if time.time() > deadline - 20:
            break
        order = list(range(hi_start, N + 1)) + list(range(hi_start - 1, 0, -1))
        alt = _greedy_build(order)
        if len(alt) > len(best):
            best = alt
    
    # Strategy 4: Randomized greedy
    for seed in range(40):
        if time.time() > deadline - 15:
            break
        rng = random.Random(seed)
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        alt = _greedy_build(candidates)
        if len(alt) > len(best):
            best = alt
    
    # Local search on best result
    if time.time() < deadline - 10:
        remaining = min(deadline - time.time() - 3, 25.0)
        improved = _local_search(best, time.time() + remaining)
        if len(improved) > len(best):
            best = improved
    
    # More random restarts with local search
    for seed in range(40, 100):
        if time.time() > deadline - 5:
            break
        rng = random.Random(seed)
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        alt = _greedy_build(candidates)
        if len(alt) >= len(best) - 1:
            remaining = min(deadline - time.time() - 1, 8.0)
            if remaining > 0.5:
                alt = _local_search(alt, time.time() + remaining)
            if len(alt) > len(best):
                best = alt
    
    # Two-swap local search for deeper optimization
    if time.time() < deadline - 3:
        remaining = min(deadline - time.time() - 1, 15.0)
        if remaining > 1.0:
            improved = _two_swap_local_search(best, time.time() + remaining)
            if len(improved) > len(best):
                best = improved
    
    # Final polish
    if time.time() < deadline - 1:
        remaining = min(deadline - time.time() - 0.5, 5.0)
        if remaining > 0.5:
            improved = _local_search(best, time.time() + remaining)
            if len(improved) > len(best):
                best = improved
    
    return best

# EVOLVE-BLOCK-END
