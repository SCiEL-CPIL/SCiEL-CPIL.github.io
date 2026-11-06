"""Small inclusion-maximal Sidon sets in [1,256]"""
import time
import random

N = 256

# EVOLVE-BLOCK-START

def _is_sidon(selected):
    """Check if selected is a Sidon set (all a+b with a<=b distinct)."""
    s = set()
    for i, a in enumerate(selected):
        for b in selected[i:]:
            total = a + b
            if total in s:
                return False
            s.add(total)
    return True

def _compute_sums(selected):
    """Compute the set of all a+b sums for a Sidon set."""
    sums = set()
    for i, a in enumerate(selected):
        for b in selected[i:]:
            sums.add(a + b)
    return sums

def _is_addable(x, selected, sums):
    """Check if x can be added to the Sidon set."""
    for a in selected:
        if (x + a) in sums:
            return False
    if (2 * x) in sums:
        return False
    return True

def _get_addable_candidates(selected, sums):
    """Get all elements in [1, N] that can be added to the current Sidon set."""
    sel_set = set(selected)
    candidates = []
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        if _is_addable(x, selected, sums):
            candidates.append(x)
    return candidates

def _blocking_score_fast(x, selected, sums, candidates):
    """
    Fast estimate of how many candidates are blocked by adding x.
    Uses the fact that new sums are {2x} U {x+a | a in selected}.
    A candidate y is blocked if:
    - 2y in new_sums, or
    - y+a in new_sums for some a in selected U {x}
    """
    new_sums = set(sums)
    new_sums.add(2 * x)
    for a in selected:
        new_sums.add(x + a)
    
    blocked = 0
    for y in candidates:
        if y == x:
            continue
        if 2 * y in new_sums:
            blocked += 1
            continue
        conflict = False
        for a in selected:
            if y + a in new_sums:
                conflict = True
                break
        if not conflict and y + x in new_sums:
            conflict = True
        if conflict:
            blocked += 1
    return blocked

def _greedy_build(deadline, rng, initial_sel=None, initial_sums=None):
    """
    Build a maximal Sidon set using greedy blocking heuristic.
    Optionally start from an initial state.
    """
    if initial_sel is not None:
        selected = list(initial_sel)
        sums = set(initial_sums)
    else:
        selected = []
        sums = set()
    
    while time.time() < deadline:
        candidates = _get_addable_candidates(selected, sums)
        if not candidates:
            break
        
        if len(candidates) == 1:
            x = candidates[0]
        else:
            scores = []
            for x in candidates:
                score = _blocking_score_fast(x, selected, sums, candidates)
                scores.append((score, x))
            scores.sort(key=lambda item: -item[0])
            top_k = min(3, len(scores))
            idx = rng.randint(0, top_k - 1)
            x = scores[idx][1]
        
        selected.append(x)
        for a in selected:
            if a != x:
                sums.add(x + a)
        sums.add(2 * x)
    
    return selected, sums

def _is_maximal(selected, sums):
    """Check if the Sidon set is inclusion-maximal."""
    sel_set = set(selected)
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        if _is_addable(x, selected, sums):
            return False
    return True

def _local_search_reduce(selected, sums, deadline, rng):
    """
    Try to reduce size by removing elements and re-maximalizing.
    """
    best_sel = list(selected)
    best_sums = set(sums)
    best_size = len(best_sel)
    
    # Try removing each element one at a time
    for y in list(best_sel):
        if time.time() > deadline:
            break
        if len(best_sel) <= 2:
            break
        
        # Remove y
        candidate = [x for x in best_sel if x != y]
        cand_sums = _compute_sums(candidate)
        
        # Re-maximalize
        cand_sel, cand_sums = _greedy_build(deadline, rng, candidate, cand_sums)
        
        if len(cand_sel) < best_size:
            if _is_sidon(cand_sel) and _is_maximal(cand_sel, cand_sums):
                best_sel = cand_sel
                best_sums = cand_sums
                best_size = len(best_sel)
    
    return best_sel, best_sums

def _backtrack_search(selected, sums, deadline, rng, max_depth=3):
    """
    Backtracking search: try alternative choices at recent decisions.
    Returns a potentially smaller maximal Sidon set.
    """
    if max_depth <= 0 or time.time() > deadline:
        return selected, sums
    
    best_sel = list(selected)
    best_sums = set(sums)
    best_size = len(best_sel)
    
    # Try undoing the last few additions and choosing differently
    for depth in range(min(max_depth, len(selected))):
        if time.time() > deadline:
            break
        
        # Take the first len(selected) - depth elements
        base_sel = selected[:-depth] if depth > 0 else list(selected)
        base_sums = _compute_sums(base_sel)
        
        # Try up to 3 alternative completions from this base
        for _ in range(3):
            if time.time() > deadline:
                break
            cand_sel, cand_sums = _greedy_build(deadline, rng, base_sel, base_sums)
            if len(cand_sel) < best_size:
                if _is_sidon(cand_sel) and _is_maximal(cand_sel, cand_sums):
                    best_sel = cand_sel
                    best_sums = cand_sums
                    best_size = len(best_sel)
    
    return best_sel, best_sums

def construct():
    start_time = time.time()
    deadline = start_time + 25.0
    
    best_selected = None
    best_size = N + 1
    
    rng = random.Random(42)
    
    # Run multiple trials of the greedy blocking search
    attempts = 0
    while time.time() < deadline and attempts < 200:
        attempts += 1
        
        sel, sums = _greedy_build(deadline, rng)
        
        if not sel:
            continue
        
        # Verify
        if _is_sidon(sel) and _is_maximal(sel, sums):
            if len(sel) < best_size:
                best_size = len(sel)
                best_selected = list(sel)
                
                # Try local search reduction if we have time
                if time.time() < deadline - 5.0 and len(sel) > 5:
                    reduced_sel, reduced_sums = _local_search_reduce(sel, sums, deadline - 3.0, rng)
                    if len(reduced_sel) < best_size:
                        best_size = len(reduced_sel)
                        best_selected = reduced_sel
                
                # Try backtracking search if we have time
                if time.time() < deadline - 3.0 and len(sel) > 5:
                    bt_sel, bt_sums = _backtrack_search(sel, sums, deadline - 2.0, rng, max_depth=2)
                    if len(bt_sel) < best_size:
                        best_size = len(bt_sel)
                        best_selected = bt_sel
    
    if best_selected is None:
        # Fallback: simple greedy
        selected = []
        sums = set()
        for x in range(1, N + 1):
            if _is_addable(x, selected, sums):
                selected.append(x)
                for a in selected:
                    if a != x:
                        sums.add(x + a)
                sums.add(2 * x)
        best_selected = selected
        best_size = len(selected)
    
    return sorted(best_selected)

# EVOLVE-BLOCK-END
