"""
Optimized: maximize 3-APs in a 4-AP-free subset of [1,256].
Uses precomputed structures and bitmask for speed.
"""

import time
import random

N = 256

def _precompute():
    """Precompute 4-AP conflicts and 3-AP contributions."""
    # For each element x, list of 4-APs containing x
    elem_4aps = {x: [] for x in range(1, N + 1)}
    all_4aps = []
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= N:
                ap = (a, b, c, e)
                all_4aps.append(ap)
                for x in ap:
                    elem_4aps[x].append(ap)
    
    # For each element x, list of 3-APs containing x
    elem_3aps = {x: [] for x in range(1, N + 1)}
    all_3aps = []
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 2 + 1):
            b = a + d
            c = a + 2 * d
            if c <= N:
                ap = (a, b, c)
                all_3aps.append(ap)
                for x in ap:
                    elem_3aps[x].append(ap)
    
    return all_4aps, all_3aps, elem_4aps, elem_3aps

_ALL_4APS, _ALL_3APS, _ELEM_4APS, _ELEM_3APS = _precompute()

def _count_3aps(selected):
    """Count 3-APs in selected set using precomputed list."""
    count = 0
    for (a, b, c) in _ALL_3APS:
        if a in selected and b in selected and c in selected:
            count += 1
    return count

def _has_4ap(selected):
    """Check if selected contains any 4-AP."""
    for (a, b, c, e) in _ALL_4APS:
        if a in selected and b in selected and c in selected and e in selected:
            return True
    return False

def _greedy_construct():
    """
    Greedy: start with all elements, iteratively remove the element 
    that breaks the most 4-APs while losing the fewest 3-APs.
    """
    selected = set(range(1, N + 1))
    
    # Find all 4-APs present
    def find_bad_aps(sel):
        bad = []
        for (a, b, c, e) in _ALL_4APS:
            if a in sel and b in sel and c in sel and e in sel:
                bad.append((a, b, c, e))
        return bad
    
    changed = True
    while changed:
        changed = False
        bad_aps = find_bad_aps(selected)
        if not bad_aps:
            break
        
        # For each element in bad 4-APs, compute (num_4aps_broken, num_3aps_lost)
        elem_stats = {}
        for ap in bad_aps:
            for x in ap:
                if x not in elem_stats:
                    # Count how many 4-APs x is in (within bad_aps)
                    n4 = sum(1 for ap2 in bad_aps if x in ap2)
                    # Count how many 3-APs x participates in within selected
                    n3 = 0
                    for ap3 in _ELEM_3APS[x]:
                        if all(y in selected for y in ap3):
                            n3 += 1
                    elem_stats[x] = (n4, n3)
        
        # Remove element with best ratio: most 4-APs broken per 3-AP lost
        # Prefer breaking more 4-APs with fewer 3-APs lost
        best_elem = None
        best_score = None
        for x, (n4, n3) in elem_stats.items():
            # Score: n4 / (n3 + 1) higher is better
            score = n4 / (n3 + 1)
            if best_score is None or score > best_score:
                best_score = score
                best_elem = x
        
        if best_elem is not None:
            selected.remove(best_elem)
            changed = True
    
    return selected

def _local_search(selected, deadline):
    """Local search with add, remove, and swap moves."""
    best_count = _count_3aps(selected)
    best_selected = set(selected)
    
    # Helper: check if adding x to sel creates a 4-AP
    def add_creates_4ap(x, sel):
        for ap in _ELEM_4APS[x]:
            others = [y for y in ap if y != x]
            if all(y in sel for y in others):
                return True
        return False
    
    # Helper: count 3-APs gained by adding x to sel
    def add_gain_3ap(x, sel):
        gain = 0
        for ap in _ELEM_3APS[x]:
            others = [y for y in ap if y != x]
            if all(y in sel for y in others):
                gain += 1
        return gain
    
    # Helper: count 3-APs lost by removing x from sel
    def remove_loss_3ap(x, sel):
        loss = 0
        for ap in _ELEM_3APS[x]:
            others = [y for y in ap if y != x]
            if all(y in sel for y in others):
                loss += 1
        return loss
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Try adding elements
        for x in range(1, N + 1):
            if x in selected:
                continue
            if add_creates_4ap(x, selected):
                continue
            gain = add_gain_3ap(x, selected)
            if gain > 0:
                selected.add(x)
                best_count += gain
                best_selected = set(selected)
                improved = True
                break
        
        if not improved:
            # Phase 2: Try swaps (remove x, add y)
            swap_improved = False
            for x in sorted(selected, key=lambda z: remove_loss_3ap(z, selected)):
                loss_x = remove_loss_3ap(x, selected)
                temp_sel = set(selected)
                temp_sel.discard(x)
                
                for y in range(1, N + 1):
                    if y in temp_sel:
                        continue
                    if add_creates_4ap(y, temp_sel):
                        continue
                    gain_y = add_gain_3ap(y, temp_sel)
                    net = gain_y - loss_x
                    if net > 0:
                        temp_sel.add(y)
                        selected = temp_sel
                        best_count += net
                        best_selected = set(selected)
                        swap_improved = True
                        break
                
                if swap_improved:
                    break
            
            if not swap_improved:
                # Phase 3: Random perturbation
                if len(selected) > 10 and random.random() < 0.3:
                    # Remove 2 random elements
                    to_remove = random.sample(list(selected), min(2, len(selected)))
                    for x in to_remove:
                        selected.discard(x)
                    best_count = _count_3aps(selected)
                    best_selected = set(selected)
    
    return best_selected

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s safety margin
    
    # Greedy construction
    selected = _greedy_construct()
    
    if time.time() > deadline:
        return sorted(selected)
    
    # Local search
    selected = _local_search(selected, deadline)
    
    return sorted(selected)
