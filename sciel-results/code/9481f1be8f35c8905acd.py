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

def _greedy_build(order, deadline):
    """Build a Sidon set by processing elements in the given order.
    Returns (selected, sums) where sums is the set of all a+b sums."""
    selected = []
    sums = set()
    for x in order:
        if time.time() > deadline:
            break
        # Check if x can be added: all x+a (a in selected, a!=x) and 2x must be new
        ok = True
        for a in selected:
            if (x + a) in sums:
                ok = False
                break
        if ok and (2 * x) in sums:
            ok = False
        if ok:
            selected.append(x)
            for a in selected:
                if a != x:
                    sums.add(x + a)
            sums.add(2 * x)
    return selected, sums

def _is_addable(x, selected, sums):
    """Check if x can be added to the Sidon set."""
    for a in selected:
        if (x + a) in sums:
            return False
    if (2 * x) in sums:
        return False
    return True

def _maximalize(selected, sums, deadline):
    """Make the set maximal by adding elements that don't break Sidon.
    Returns (new_selected, new_sums)."""
    new_selected = list(selected)
    new_sums = set(sums)
    sel_set = set(new_selected)
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Find an addable element
        for x in range(1, N + 1):
            if x in sel_set:
                continue
            if _is_addable(x, new_selected, new_sums):
                new_selected.append(x)
                sel_set.add(x)
                for a in new_selected:
                    if a != x:
                        new_sums.add(x + a)
                new_sums.add(2 * x)
                changed = True
                break
    
    return new_selected, new_sums

def _is_maximal(selected, sums):
    """Check if the Sidon set is inclusion-maximal."""
    sel_set = set(selected)
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        if _is_addable(x, selected, sums):
            return False
    return True

def _local_search_reduce(selected, sums, deadline):
    """Try to reduce the size of a maximal Sidon set via swap operations.
    For each element y in selected, try to remove it and re-maximalize.
    If the resulting maximal set is smaller, keep it.
    Returns (new_selected, new_sums)."""
    best_sel = list(selected)
    best_sums = set(sums)
    best_size = len(best_sel)
    
    # Try removing each element and re-maximalizing
    for y in list(best_sel):
        if time.time() > deadline:
            break
        if len(best_sel) <= 1:
            break
        
        # Remove y
        candidate = [x for x in best_sel if x != y]
        cand_sums = _compute_sums(candidate)
        
        # Re-maximalize
        cand_sel, cand_sums = _maximalize(candidate, cand_sums, deadline)
        
        # Check if it's actually smaller
        if len(cand_sel) < best_size:
            # Verify it's valid and maximal
            if _is_sidon(cand_sel) and _is_maximal(cand_sel, cand_sums):
                best_sel = cand_sel
                best_sums = cand_sums
                best_size = len(best_sel)
    
    return best_sel, best_sums

def construct():
    start_time = time.time()
    deadline = start_time + 25.0  # Safety margin before 60s kill
    
    # Strategy: try many random orderings, keep the best (smallest) maximal set
    best_selected = None
    best_size = N + 1
    
    # First, try the standard ascending greedy as baseline
    selected, sums = _greedy_build(range(1, N + 1), deadline)
    if selected:
        # Make it maximal
        selected, sums = _maximalize(selected, sums, deadline)
        if _is_sidon(selected) and _is_maximal(selected, sums):
            # Try to reduce via local search
            selected, sums = _local_search_reduce(selected, sums, deadline)
            if _is_sidon(selected) and _is_maximal(selected, sums):
                best_selected = selected
                best_size = len(selected)
    
    # Now try random orderings
    rng = random.Random(42)
    attempts = 0
    while time.time() < deadline and attempts < 5000:
        attempts += 1
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        selected, sums = _greedy_build(order, deadline)
        if not selected:
            continue
        
        # Make it maximal
        selected, sums = _maximalize(selected, sums, deadline)
        
        if len(selected) >= best_size:
            continue
        
        # Verify
        if _is_sidon(selected) and _is_maximal(selected, sums):
            # Try to reduce via local search
            selected, sums = _local_search_reduce(selected, sums, deadline)
            if _is_sidon(selected) and _is_maximal(selected, sums) and len(selected) < best_size:
                best_selected = selected
                best_size = len(selected)
    
    if best_selected is None:
        # Fallback: return the ascending greedy result
        selected, sums = _greedy_build(range(1, N + 1), deadline)
        selected, sums = _maximalize(selected, sums, deadline)
        best_selected = selected
    
    return sorted(best_selected)

# EVOLVE-BLOCK-END
