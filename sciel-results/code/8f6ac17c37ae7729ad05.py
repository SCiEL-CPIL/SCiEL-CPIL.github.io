"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys
from collections import defaultdict

N = 256
DEADLINE = 55.0  # safety margin before 60s kill

# EVOLVE-BLOCK-START

def _compute_sums(selected):
    """Compute all pairwise sums a+b for a<=b in selected as a set."""
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        sums.add(2 * a)
        for j in range(i + 1, n):
            sums.add(a + selected[j])
    return sums

def _is_sidon(selected):
    """Fast Sidon check."""
    if len(selected) == 0:
        return True
    sums = _compute_sums(selected)
    total = len(selected) * (len(selected) + 1) // 2
    return len(sums) == total

def _is_maximal(selected, n=N):
    """Check if selected is inclusion-maximal Sidon in [1,n]."""
    selected_set = set(selected)
    sums = _compute_sums(selected)
    for x in range(1, n + 1):
        if x in selected_set:
            continue
        if 2 * x in sums:
            continue
        conflict = False
        for a in selected:
            if x + a in sums:
                conflict = True
                break
        if not conflict:
            return False
    return True

def _can_add(selected, sums, selected_set, x):
    """Check if x can be added to selected (Sidon-compatible)."""
    if x in selected_set:
        return False
    if 2 * x in sums:
        return False
    for a in selected:
        if x + a in sums:
            return False
    return True

def _get_addable(selected, sums, selected_set, n=N):
    """Return list of elements that can be added to selected."""
    addable = []
    for x in range(1, n + 1):
        if x in selected_set:
            continue
        if _can_add(selected, sums, selected_set, x):
            addable.append(x)
    return addable

def _forced_additions(core, rng, deadline, max_iter=200):
    """
    Given a Sidon core, greedily add elements to make it maximal.
    At each step, pick the element that blocks the most currently-addable elements.
    """
    selected = list(core)
    sums = _compute_sums(selected)
    selected_set = set(selected)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        addable = _get_addable(selected, sums, selected_set)
        if not addable:
            break
        
        # Find the element (among all non-selected) that, when added, blocks the most addable
        best_candidate = None
        best_blocked = -1
        best_new_sums = None
        best_new_set = None
        best_new_sset = None
        
        candidates = [x for x in range(1, N + 1) if x not in selected_set]
        
        for cand in candidates:
            if not _can_add(selected, sums, selected_set, cand):
                continue
            
            new_selected = selected + [cand]
            new_sset = selected_set | {cand}
            new_sums = sums | set(cand + a for a in selected) | {2 * cand}
            
            # Count how many addable elements become non-addable
            blocked = 0
            for y in addable:
                if y == cand:
                    continue
                if not _can_add(new_selected, new_sums, new_sset, y):
                    blocked += 1
            
            if blocked > best_blocked:
                best_blocked = blocked
                best_candidate = cand
                best_new_sums = new_sums
                best_new_set = new_selected
                best_new_sset = new_sset
        
        if best_candidate is None:
            break
        
        selected = best_new_set
        selected_set = best_new_sset
        sums = best_new_sums
    
    return selected

def _local_search_reduce(selected, deadline, max_passes=20):
    """
    Try to reduce size while maintaining Sidon + maximal.
    Phase 1: Try removals.
    Phase 2: Try swaps (remove one, add one different).
    """
    current = list(selected)
    improved = True
    pass_count = 0
    
    while improved and time.time() < deadline and pass_count < max_passes:
        improved = False
        pass_count += 1
        
        # Phase 1: Try removing each element
        for x in list(current):
            if time.time() > deadline:
                return current
            new_list = [y for y in current if y != x]
            if len(new_list) == 0:
                continue
            if _is_sidon(new_list) and _is_maximal(new_list):
                current = new_list
                improved = True
                break
        
        if improved:
            continue
        
        # Phase 2: Try swaps - remove x, add y
        current_set = set(current)
        for x in list(current):
            if time.time() > deadline:
                return current
            without_x = [y for y in current if y != x]
            if len(without_x) == 0:
                continue
            without_x_set = set(without_x)
            sums_x = _compute_sums(without_x)
            
            for y in range(1, N + 1):
                if y in without_x_set:
                    continue
                if not _can_add(without_x, sums_x, without_x_set, y):
                    continue
                
                candidate = without_x + [y]
                if _is_sidon(candidate) and _is_maximal(candidate):
                    current = candidate
                    improved = True
                    break
            
            if improved:
                break
    
    return current

def _construct_spread_core(seed_val, target_size):
    """Construct a well-spread initial core of given target size."""
    rng = random.Random(seed_val)
    core = []
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    
    sums = set()
    core_set = set()
    
    for x in candidates:
        if len(core) >= target_size:
            break
        if _can_add(core, sums, core_set, x):
            core.append(x)
            core_set.add(x)
            sums = sums | set(x + a for a in core[:-1]) | {2 * x}
    
    return core

def _greedy_maximal_from_order(order, deadline):
    """Greedy: process elements in order, add if Sidon-compatible. Then force maximality."""
    selected = []
    sums = set()
    selected_set = set()
    
    for x in order:
        if time.time() > deadline:
            break
        if x in selected_set:
            continue
        if _can_add(selected, sums, selected_set, x):
            selected.append(x)
            selected_set.add(x)
            sums = sums | set(x + a for a in selected[:-1]) | {2 * x}
    
    # Now force maximality
    if not _is_maximal(selected):
        selected = _forced_additions(selected, random.Random(0), deadline)
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_result = None
    best_size = float('inf')
    
    def try_result(candidate):
        nonlocal best_result, best_size
        if candidate is None or len(candidate) == 0:
            return
        if len(candidate) >= best_size:
            return
        if _is_sidon(candidate) and _is_maximal(candidate):
            best_result = list(candidate)
            best_size = len(candidate)
    
    # Strategy 1: Diverse small cores + forced additions + local search reduce
    rng_master = random.Random(12345)
    
    # Try various core sizes
    core_sizes = [3, 4, 5, 6, 7, 8, 10, 12]
    attempts = 0
    max_attempts = 300
    
    while attempts < max_attempts and time.time() < deadline - 5:
        attempts += 1
        target_size = core_sizes[attempts % len(core_sizes)]
        seed_val = rng_master.randint(0, 10**9)
        
        core = _construct_spread_core(seed_val, target_size)
        if len(core) < 2:
            continue
        
        # Force maximality
        result = _forced_additions(core, rng_master, deadline)
        
        if not _is_sidon(result):
            continue
        
        # Local search to reduce
        if time.time() < deadline - 3:
            result = _local_search_reduce(result, deadline, max_passes=15)
        
        try_result(result)
        
        # If we found something very small, intensify
        if best_size <= 12 and time.time() < deadline - 5:
            # Try more cores of similar size
            for extra in range(3):
                if time.time() > deadline - 3:
                    break
                seed_val2 = rng_master.randint(0, 10**9)
                core2 = _construct_spread_core(seed_val2, max(3, target_size - 1))
                if len(core2) < 2:
                    continue
                result2 = _forced_additions(core2, rng_master, deadline)
                if _is_sidon(result2) and time.time() < deadline - 2:
                    result2 = _local_search_reduce(result2, deadline, max_passes=10)
                try_result(result2)
    
    # Strategy 2: Greedy from various orderings
    orderings = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        list(range(2, N + 1, 2)) + list(range(1, N + 1, 2)),
        list(range(1, N + 1, 2)) + list(range(2, N + 1, 2)),
    ]
    
    # Add some random orderings
    for _ in range(5):
        if time.time() > deadline - 5:
            break
        order = list(range(1, N + 1))
        rng_master.shuffle(order)
        orderings.append(order)
    
    for order in orderings:
        if time.time() > deadline - 5:
            break
        result = _greedy_maximal_from_order(order, deadline)
        if result and _is_sidon(result) and time.time() < deadline - 3:
            result = _local_search_reduce(result, deadline, max_passes=10)
        try_result(result)
    
    # Strategy 3: If we have a good result, try aggressive reduction
    if best_result is not None and best_size > 8 and time.time() < deadline - 5:
        for _ in range(10):
            if time.time() > deadline - 3:
                break
            # Perturb: remove a random element, try to rebuild
            perturbed = list(best_result)
            if len(perturbed) > 2:
                idx = rng_master.randint(0, len(perturbed) - 1)
                perturbed.pop(idx)
            if _is_sidon(perturbed) and time.time() < deadline - 2:
                perturbed = _local_search_reduce(perturbed, deadline, max_passes=15)
            try_result(perturbed)
    
    if best_result is None:
        # Fallback: simple ascending greedy
        best_result = _greedy_maximal_from_order(list(range(1, N + 1)), deadline)
    
    return best_result

# EVOLVE-BLOCK-END
