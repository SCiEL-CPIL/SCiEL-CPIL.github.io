"""Rewritten program: structural change to a conflict-aware greedy + local-search optimizer."""
import time
import random
from math import gcd
import heapq

# EVOLVE-BLOCK-START

def _precompute_conflicts(n=256):
    """Precompute conflict matrix and neighbor lists."""
    conflicts = [[False] * (n + 1) for _ in range(n + 1)]
    neighbors = [[] for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                conflicts[a][b] = True
                conflicts[b][a] = True
                neighbors[a].append(b)
                neighbors[b].append(a)
    # Compute degrees for ordering
    degrees = [len(neighbors[i]) for i in range(n + 1)]
    return conflicts, neighbors, degrees

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

def _degree_sorted_order(n=256, neighbors=None, reverse=False):
    """Return elements sorted by degree (fewest conflicts first by default)."""
    if neighbors is None:
        return list(range(1, n + 1))
    degrees = [len(neighbors[i]) for i in range(n + 1)]
    return sorted(range(1, n + 1), key=lambda x: (degrees[x], x), reverse=reverse)

def _local_search(conflicts, neighbors, selected, n=256, max_iters=5000, deadline=None):
    """
    Optimized local search with maintained conflict counts using neighbor lists.
    """
    selected = set(selected)
    best = set(selected)
    best_size = len(selected)
    
    # Maintain conflict counts: conf_count[x] = number of selected elements conflicting with x
    conf_count = [0] * (n + 1)
    for x in selected:
        for y in neighbors[x]:
            conf_count[y] += 1
    
    # Precompute the full universe
    universe = list(range(1, n + 1))
    
    def try_add(x):
        """Add x if it has no conflicts with selected. Returns True if added."""
        if conf_count[x] == 0:
            selected.add(x)
            for y in neighbors[x]:
                conf_count[y] += 1
            return True
        return False
    
    def remove(x):
        """Remove x from selected."""
        selected.discard(x)
        for y in neighbors[x]:
            conf_count[y] -= 1
    
    def snapshot_conf_count():
        return list(conf_count)
    
    def restore_conf_count(saved):
        for i in range(1, n + 1):
            conf_count[i] = saved[i]
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        improved = False
        
        # Phase 1: Try to add any unselected number with zero conflicts
        # Use a fixed random permutation to avoid reshuffling every time
        unselected_perm = list(universe)
        random.shuffle(unselected_perm)
        for x in unselected_perm:
            if x not in selected and conf_count[x] == 0:
                try_add(x)
                improved = True
                break
        
        if not improved:
            # Phase 2: Try 1-for-1 swap (remove one, add one different)
            sel_list = list(selected)
            random.shuffle(sel_list)
            for rem in sel_list:
                saved_cc = snapshot_conf_count()
                saved_sel = set(selected)
                remove(rem)
                # Find an unselected x with conf_count[x] == 0 after removal
                found = False
                for x in unselected_perm:
                    if x not in selected and conf_count[x] == 0:
                        try_add(x)
                        found = True
                        break
                if found:
                    improved = True
                    break
                else:
                    # Undo
                    restore_conf_count(saved_cc)
                    selected = saved_sel
        
        if not improved:
            # Phase 3: Try remove-one-add-many (net gain > 0)
            sel_list = list(selected)
            random.shuffle(sel_list)
            for rem in sel_list:
                saved_cc = snapshot_conf_count()
                saved_sel = set(selected)
                remove(rem)
                # Greedily add as many as possible
                added_count = 0
                for x in unselected_perm:
                    if x not in selected and conf_count[x] == 0:
                        try_add(x)
                        added_count += 1
                if added_count > 1:  # net gain
                    improved = True
                    break
                else:
                    # Undo
                    restore_conf_count(saved_cc)
                    selected = saved_sel
        
        if not improved:
            # Phase 4: Try remove-two-add-many (net gain > 0)
            if len(selected) > 4:
                sel_list = list(selected)
                random.shuffle(sel_list)
                for i in range(min(len(sel_list), 20)):
                    rem1 = sel_list[i]
                    for j in range(i + 1, min(len(sel_list), i + 10)):
                        rem2 = sel_list[j]
                        saved_cc = snapshot_conf_count()
                        saved_sel = set(selected)
                        remove(rem1)
                        remove(rem2)
                        # Greedily add as many as possible
                        added_count = 0
                        for x in unselected_perm:
                            if x not in selected and conf_count[x] == 0:
                                try_add(x)
                                added_count += 1
                        if added_count > 2:  # net gain
                            improved = True
                            break
                        else:
                            # Undo
                            restore_conf_count(saved_cc)
                            selected = saved_sel
                    if improved:
                        break
        
        if not improved:
            # Phase 5: Perturbation - remove 2 random, greedily refill
            if len(selected) > 4:
                rem_pair = random.sample(list(selected), 2)
                for r in rem_pair:
                    remove(r)
                # Greedily add as many as possible
                for x in unselected_perm:
                    if x not in selected and conf_count[x] == 0:
                        try_add(x)
                if len(selected) > best_size:
                    improved = True
                else:
                    # Restart from best
                    selected = set(best)
                    conf_count = [0] * (n + 1)
                    for x in selected:
                        for y in neighbors[x]:
                            conf_count[y] += 1
        
        if not improved:
            break
        
        if len(selected) > best_size:
            best_size = len(selected)
            best = set(selected)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 256
    conflicts, neighbors, degrees = _precompute_conflicts(n)
    
    # Initial greedy with degree-sorted order (fewest conflicts first)
    degree_order = _degree_sorted_order(n, neighbors, reverse=False)
    selected = _greedy_build(conflicts, degree_order, n)
    
    # Try multiple orderings for greedy
    best = set(selected)
    best_size = len(selected)
    
    rng = random.Random(42)
    
    # Try natural order
    natural_order = list(range(1, n + 1))
    sel = _greedy_build(conflicts, natural_order, n)
    if len(sel) > best_size:
        best_size = len(sel)
        best = sel
    
    # Try reverse degree-sorted
    rev_degree_order = _degree_sorted_order(n, neighbors, reverse=True)
    sel = _greedy_build(conflicts, rev_degree_order, n)
    if len(sel) > best_size:
        best_size = len(sel)
        best = sel
    
    # Try multiple random orderings
    for trial in range(80):
        if time.time() > deadline - 10:
            break
        
        # Random order
        order = list(range(1, n + 1))
        rng.shuffle(order)
        sel = _greedy_build(conflicts, order, n)
        if len(sel) > best_size:
            best_size = len(sel)
            best = sel
    
    # Local search from best
    if time.time() < deadline - 5:
        improved = _local_search(conflicts, neighbors, best, n, max_iters=5000, deadline=deadline - 3)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    # More local search iterations if time permits
    while time.time() < deadline - 5:
        improved = _local_search(conflicts, neighbors, best, n, max_iters=3000, deadline=deadline - 3)
        if len(improved) <= best_size:
            break
        best = improved
        best_size = len(best)
    
    return sorted(best)

# EVOLVE-BLOCK-END
