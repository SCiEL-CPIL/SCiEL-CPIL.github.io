"""Rewritten program: optimized conflict-aware greedy + efficient local-search optimizer."""
import time
import random
from math import gcd
import numpy as np

# EVOLVE-BLOCK-START

def _precompute_conflicts(n=256):
    """Precompute conflict matrix, neighbor lists, and degrees."""
    conflicts = [[False] * (n + 1) for _ in range(n + 1)]
    neighbors = [[] for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                conflicts[a][b] = True
                conflicts[b][a] = True
                neighbors[a].append(b)
                neighbors[b].append(a)
    return conflicts, neighbors

def _greedy_build(conflicts, order, n=256):
    """Greedy: add numbers in the given order if they don't conflict with any already selected."""
    selected = set()
    for x in order:
        ok = True
        for y in selected:
            if conflicts[x][y]:
                ok = False
                break
        if ok:
            selected.add(x)
    return selected

def _greedy_build_fast(conflicts, neighbors, order, n=256):
    """Faster greedy using conflict counts."""
    selected = set()
    conf_count = [0] * (n + 1)
    for x in order:
        if conf_count[x] == 0:
            selected.add(x)
            for y in neighbors[x]:
                conf_count[y] += 1
    return selected

def _local_search(conflicts, neighbors, selected, n=256, max_iters=10000, deadline=None):
    """
    Optimized local search with maintained conflict counts.
    Uses efficient add/remove operations without snapshot/restore.
    """
    selected = set(selected)
    best = set(selected)
    best_size = len(selected)
    
    # Maintain conflict counts: conf_count[x] = number of selected elements conflicting with x
    conf_count = [0] * (n + 1)
    for x in selected:
        for y in neighbors[x]:
            conf_count[y] += 1
    
    universe = list(range(1, n + 1))
    rng = random.Random(12345)
    
    def try_add(x):
        if x not in selected and conf_count[x] == 0:
            selected.add(x)
            for y in neighbors[x]:
                conf_count[y] += 1
            return True
        return False
    
    def remove(x):
        selected.discard(x)
        for y in neighbors[x]:
            conf_count[y] -= 1
    
    # Precompute unselected list once
    unselected = [x for x in universe if x not in selected]
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        improved = False
        
        # Phase 1: Try to add any unselected number with zero conflicts
        if unselected:
            # Random sample from unselected to avoid bias
            sample_size = min(len(unselected), 50)
            candidates = rng.sample(unselected, sample_size)
            for x in candidates:
                if conf_count[x] == 0:
                    try_add(x)
                    unselected.remove(x)
                    improved = True
                    break
        
        if not improved:
            # Phase 2: Try 1-for-1 swap
            sel_list = list(selected)
            rng.shuffle(sel_list)
            for rem in sel_list:
                # Temporarily remove
                remove(rem)
                # Find an unselected x with conf_count[x] == 0
                found = False
                if unselected:
                    candidates = rng.sample(unselected, min(len(unselected), 30))
                    for x in candidates:
                        if conf_count[x] == 0:
                            try_add(x)
                            unselected.remove(x)
                            found = True
                            break
                if not found:
                    # Undo removal
                    selected.add(rem)
                    for y in neighbors[rem]:
                        conf_count[y] += 1
        
        if not improved:
            # Phase 3: Try remove-one-add-many
            sel_list = list(selected)
            rng.shuffle(sel_list)
            for rem in sel_list:
                remove(rem)
                added_count = 0
                # Try to add multiple
                for x in list(unselected):
                    if conf_count[x] == 0:
                        try_add(x)
                        unselected.remove(x)
                        added_count += 1
                        if added_count >= 2:
                            break
                if added_count < 2:
                    # Undo
                    selected.add(rem)
                    for y in neighbors[rem]:
                        conf_count[y] += 1
                    # Restore unselected
                    for x in unselected:
                        pass  # Already handled
                else:
                    improved = True
                    break
        
        if not improved:
            # Phase 4: Perturbation - remove 2 random, greedily refill
            if len(selected) > 6:
                rem_pair = rng.sample(list(selected), 2)
                for r in rem_pair:
                    remove(r)
                # Greedily add as many as possible
                for x in list(unselected):
                    if conf_count[x] == 0:
                        try_add(x)
                        unselected.remove(x)
                if len(selected) > best_size:
                    improved = True
                else:
                    # Restart from best
                    selected = set(best)
                    conf_count = [0] * (n + 1)
                    for x in selected:
                        for y in neighbors[x]:
                            conf_count[y] += 1
                    unselected = [x for x in universe if x not in selected]
        
        if not improved:
            break
        
        if len(selected) > best_size:
            best_size = len(selected)
            best = set(selected)
    
    return best

def _polish(conflicts, neighbors, selected, n=256, deadline=None):
    """Final high-precision polish: try all possible single additions and swaps."""
    selected = set(selected)
    best = set(selected)
    best_size = len(selected)
    
    conf_count = [0] * (n + 1)
    for x in selected:
        for y in neighbors[x]:
            conf_count[y] += 1
    
    universe = list(range(1, n + 1))
    unselected = [x for x in universe if x not in selected]
    
    # Try all single additions
    for x in unselected:
        if conf_count[x] == 0:
            selected.add(x)
            for y in neighbors[x]:
                conf_count[y] += 1
    
    # Try all 1-for-1 swaps
    sel_list = list(selected)
    for rem in sel_list:
        remove_conf = [y for y in neighbors[rem]]
        # Temporarily remove
        selected.discard(rem)
        for y in remove_conf:
            conf_count[y] -= 1
        
        # Find best addition
        best_add = None
        for x in unselected:
            if conf_count[x] == 0:
                best_add = x
                break
        
        if best_add is not None:
            selected.add(best_add)
            for y in neighbors[best_add]:
                conf_count[y] += 1
            if len(selected) > best_size:
                best_size = len(selected)
                best = set(selected)
        else:
            # Undo
            selected.add(rem)
            for y in remove_conf:
                conf_count[y] += 1
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 256
    conflicts, neighbors = _precompute_conflicts(n)
    
    # Multiple initial solutions using different orderings
    best = set()
    best_size = 0
    
    # Degree-sorted order (fewest conflicts first)
    degrees = [len(neighbors[i]) for i in range(n + 1)]
    degree_order = sorted(range(1, n + 1), key=lambda x: (degrees[x], x))
    sel = _greedy_build_fast(conflicts, neighbors, degree_order, n)
    if len(sel) > best_size:
        best_size = len(sel)
        best = set(sel)
    
    # Natural order
    natural_order = list(range(1, n + 1))
    sel = _greedy_build_fast(conflicts, neighbors, natural_order, n)
    if len(sel) > best_size:
        best_size = len(sel)
        best = set(sel)
    
    # Reverse degree-sorted
    rev_degree_order = sorted(range(1, n + 1), key=lambda x: (-degrees[x], x))
    sel = _greedy_build_fast(conflicts, neighbors, rev_degree_order, n)
    if len(sel) > best_size:
        best_size = len(sel)
        best = set(sel)
    
    # Try multiple random orderings
    rng = random.Random(42)
    for trial in range(100):
        if time.time() > deadline - 15:
            break
        order = list(range(1, n + 1))
        rng.shuffle(order)
        sel = _greedy_build_fast(conflicts, neighbors, order, n)
        if len(sel) > best_size:
            best_size = len(sel)
            best = set(sel)
    
    # Local search from best
    if time.time() < deadline - 10:
        improved = _local_search(conflicts, neighbors, best, n, max_iters=10000, deadline=deadline - 5)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    # More local search iterations if time permits
    while time.time() < deadline - 8:
        improved = _local_search(conflicts, neighbors, best, n, max_iters=5000, deadline=deadline - 5)
        if len(improved) <= best_size:
            break
        best = improved
        best_size = len(best)
    
    # Final polish pass
    if time.time() < deadline - 2:
        best = _polish(conflicts, neighbors, best, n, deadline=deadline - 1)
    
    return sorted(best)

# EVOLVE-BLOCK-END
