"""Optimized program: smart greedy with incremental maximality tracking and multi-strategy search."""
import random
import time

N = 256

def _compute_sums(sel):
    sums = set()
    for i in range(len(sel)):
        a = sel[i]
        for j in range(i, len(sel)):
            sums.add(a + sel[j])
    return sums

def _is_maximal_fast(sel, sums):
    """
    Check if sel is inclusion-maximal.
    Returns True if no missing element can be added.
    """
    sel_set = set(sel)
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # Check if x can be added: 2*x not in sums AND x+a not in sums for all a in sel
        if 2 * x in sums:
            continue
        blocked = False
        for a in sel:
            if (x + a) in sums:
                blocked = True
                break
        if not blocked:
            return False
    return True

def _greedy_construct(order):
    """
    Greedily select elements from order to form a Sidon set.
    Returns the selected list and the set of sums.
    """
    selected = []
    sums = set()
    for x in order:
        if 2 * x in sums:
            continue
        can_add = True
        for a in selected:
            if (x + a) in sums:
                can_add = False
                break
        if can_add:
            selected.append(x)
            sums.add(2 * x)
            for a in selected:
                sums.add(x + a)
    return selected, sums

def _greedy_with_blocking_priority(rng, base_order):
    """
    Greedy that prioritizes elements that block the most missing elements.
    """
    selected = []
    sums = set()
    sel_set = set()
    remaining = set(base_order)
    
    # Precompute which elements each candidate would block
    while remaining:
        best_x = None
        best_block_count = -1
        
        # Sample a subset for speed
        candidates = list(remaining)
        if len(candidates) > 50:
            candidates = rng.sample(candidates, 50)
        
        for x in candidates:
            # Check if x can be added
            if 2 * x in sums:
                continue
            can_add = True
            for a in selected:
                if (x + a) in sums:
                    can_add = False
                    break
            if not can_add:
                continue
            
            # Count how many missing elements this would block
            new_sums = sums | {2 * x} | {x + a for a in selected}
            block_count = 0
            for y in remaining:
                if y == x:
                    continue
                if y in sel_set:
                    continue
                # Check if y is blocked by new_sums
                if 2 * y in new_sums:
                    block_count += 1
                    continue
                y_blocked = False
                for a in selected + [x]:
                    if (y + a) in new_sums:
                        y_blocked = True
                        break
                if y_blocked:
                    block_count += 1
            
            if block_count > best_block_count:
                best_block_count = block_count
                best_x = x
        
        if best_x is None:
            break
        
        # Add best_x
        selected.append(best_x)
        sel_set.add(best_x)
        remaining.remove(best_x)
        sums.add(2 * best_x)
        for a in selected:
            sums.add(best_x + a)
    
    return selected, sums

def _try_remove(sel, sums):
    """
    Try to remove one element from sel while maintaining maximality.
    Returns (new_sel, new_sums) if successful, else (None, None).
    """
    for i in range(len(sel)):
        x_to_remove = sel[i]
        new_sel = sel[:i] + sel[i+1:]
        if not new_sel:
            break
        new_sums = _compute_sums(new_sel)
        if _is_maximal_fast(new_sel, new_sums):
            return new_sel, new_sums
    return None, None

def construct():
    deadline = time.time() + 25  # Safety margin for 30s limit
    
    best_set = None
    best_size = float('inf')
    
    # Strategy 1: Ascending greedy
    order_asc = list(range(1, N + 1))
    sel, sums = _greedy_construct(order_asc)
    if _is_maximal_fast(sel, sums):
        best_set = sel
        best_size = len(sel)
    
    # Strategy 2: Descending greedy
    if time.time() < deadline:
        order_desc = list(range(N, 0, -1))
        sel, sums = _greedy_construct(order_desc)
        if _is_maximal_fast(sel, sums):
            if len(sel) < best_size:
                best_size = len(sel)
                best_set = sel
    
    # Strategy 3: Randomized greedy with blocking priority
    rng = random.Random(42)
    trials = 0
    while time.time() < deadline - 5:
        # Try different random orders
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        # Use blocking-priority greedy
        sel, sums = _greedy_with_blocking_priority(rng, order)
        
        if _is_maximal_fast(sel, sums):
            if len(sel) < best_size:
                best_size = len(sel)
                best_set = sel
                
                # Local search: try removing elements
                if time.time() < deadline - 2:
                    improved = True
                    while improved:
                        improved = False
                        new_sel, new_sums = _try_remove(sel, sums)
                        if new_sel is not None:
                            sel = new_sel
                            sums = new_sums
                            improved = True
                            best_size = len(sel)
                            best_set = sel
                
                if best_size <= 8:
                    break
        
        trials += 1
    
    # Strategy 4: More aggressive random search with different seeds
    if time.time() < deadline - 3:
        for seed in range(100):
            if time.time() > deadline - 1:
                break
            rng2 = random.Random(seed)
            order = list(range(1, N + 1))
            rng2.shuffle(order)
            
            sel, sums = _greedy_with_blocking_priority(rng2, order)
            
            if _is_maximal_fast(sel, sums):
                if len(sel) < best_size:
                    best_size = len(sel)
                    best_set = sel
                    
                    if time.time() < deadline - 1:
                        improved = True
                        while improved:
                            improved = False
                            new_sel, new_sums = _try_remove(sel, sums)
                            if new_sel is not None:
                                sel = new_sel
                                sums = new_sums
                                improved = True
                                best_size = len(sel)
                                best_set = sel
                    
                    if best_size <= 7:
                        break
    
    if best_set is None:
        return [1]
        
    return best_set
