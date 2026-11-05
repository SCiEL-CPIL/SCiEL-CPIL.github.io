"""Rewritten program: structural change to a conflict-aware greedy + local-search optimizer."""
import time
import random
from math import gcd
from functools import lru_cache

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

def _build_bitsets(neighbors, n=256):
    """Build bitset representations of conflict neighborhoods."""
    # bitset[i] is a Python int where bit j is set if i conflicts with j
    bitsets = [0] * (n + 1)
    for i in range(1, n + 1):
        mask = 0
        for j in neighbors[i]:
            mask |= (1 << j)
        bitsets[i] = mask
    return bitsets

def _branch_and_bound(bitsets, n=256, deadline=None, initial_solution=None):
    """
    Exact maximum independent set via branch-and-bound with bitset upper bound.
    Uses greedy coloring heuristic for tighter bounds.
    """
    best_size = 0
    best_set = set()
    
    if initial_solution:
        best_size = len(initial_solution)
        best_set = set(initial_solution)
    
    # Build adjacency as bitsets
    # adj[i] has bit j set if i and j conflict
    
    # Greedy coloring for upper bound estimation
    def greedy_color_bound(candidates_mask, n):
        """
        Upper bound on MIS size using greedy coloring.
        candidates_mask: bitmask of candidate vertices.
        Returns upper bound on independent set size within candidates.
        """
        # Simple greedy coloring: repeatedly pick a vertex, remove it and its neighbors
        # Count how many "layers" we need
        remaining = candidates_mask
        color_count = 0
        while remaining:
            # Pick the vertex with lowest degree in remaining (or just first set bit)
            # For speed, just pick first available
            v_bit = remaining & (-remaining)
            v = v_bit.bit_length() - 1
            remaining ^= v_bit
            # Remove neighbors of v from remaining
            remaining &= ~bitsets[v]
            color_count += 1
        return color_count
    
    # More efficient: use a proper coloring-based bound
    def coloring_bound(candidates_mask, n):
        """
        Use greedy coloring to get an upper bound.
        The number of colors in a proper coloring of the subgraph induced by
        candidates is an upper bound on the MIS size.
        """
        if candidates_mask == 0:
            return 0
        
        # Greedy coloring: assign colors greedily
        # We'll do a simple sequential coloring
        uncolored = candidates_mask
        color_classes = []
        
        while uncolored:
            # Start a new color class
            color_class = 0
            remaining = uncolored
            
            while remaining:
                # Pick first available vertex
                v_bit = remaining & (-remaining)
                v = v_bit.bit_length() - 1
                
                # Can we add v to current color class?
                # v must not conflict with any vertex already in color_class
                if (color_class & bitsets[v]) == 0:
                    color_class |= v_bit
                    remaining ^= v_bit
            
            color_classes.append(color_class)
            # Remove all vertices in this color class from uncolored
            uncolored &= ~color_class
        
        return len(color_classes)
    
    # Branch and bound
    def bnb(candidates_mask, current_set, current_size, depth=0):
        nonlocal best_size, best_set
        
        if deadline and time.time() > deadline:
            return
        
        if candidates_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)
            return
        
        # Upper bound using coloring
        ub = coloring_bound(candidates_mask, n)
        if current_size + ub <= best_size:
            return
        
        # Pick a vertex to branch on
        # Heuristic: pick vertex with highest degree in candidates
        # For speed, just pick first set bit
        v_bit = candidates_mask & (-candidates_mask)
        v = v_bit.bit_length() - 1
        
        # Branch 1: Include v
        new_candidates = candidates_mask & ~bitsets[v] & ~v_bit
        new_set = current_set | {v}
        bnb(new_candidates, new_set, current_size + 1, depth + 1)
        
        if deadline and time.time() > deadline:
            return
        
        # Branch 2: Exclude v
        new_candidates = candidates_mask & ~v_bit
        bnb(new_candidates, current_set, current_size, depth + 1)
    
    # Start with all vertices as candidates
    all_mask = 0
    for i in range(1, n + 1):
        all_mask |= (1 << i)
    
    bnb(all_mask, set(), 0)
    
    return best_set

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
            if y != x:
                conf_count[y] += 1
    
    def try_add(x):
        """Add x if it has no conflicts with selected. Returns True if added."""
        if conf_count[x] == 0:
            selected.add(x)
            for y in neighbors[x]:
                if y != x:
                    conf_count[y] += 1
            return True
        return False
    
    def remove(x):
        """Remove x from selected."""
        selected.discard(x)
        for y in neighbors[x]:
            if y != x:
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
        unselected = [x for x in range(1, n + 1) if x not in selected]
        random.shuffle(unselected)
        for x in unselected:
            if conf_count[x] == 0:
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
                unselected = [x for x in range(1, n + 1) if x not in selected]
                random.shuffle(unselected)
                found = False
                for x in unselected:
                    if conf_count[x] == 0:
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
                unselected = [x for x in range(1, n + 1) if x not in selected]
                random.shuffle(unselected)
                added_count = 0
                for x in unselected:
                    if conf_count[x] == 0:
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
                        unselected = [x for x in range(1, n + 1) if x not in selected]
                        random.shuffle(unselected)
                        added_count = 0
                        for x in unselected:
                            if conf_count[x] == 0:
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
                unselected = [x for x in range(1, n + 1) if x not in selected]
                random.shuffle(unselected)
                for x in unselected:
                    if conf_count[x] == 0:
                        try_add(x)
                if len(selected) > best_size:
                    improved = True
                else:
                    # Restart from best
                    selected = set(best)
                    conf_count = [0] * (n + 1)
                    for x in selected:
                        for y in neighbors[x]:
                            if y != x:
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
    conflicts, neighbors = _precompute_conflicts(n)
    bitsets = _build_bitsets(neighbors, n)
    
    # Initial greedy with natural order
    order = list(range(1, n + 1))
    selected = _greedy_build(conflicts, order, n)
    
    # Try multiple random orderings for greedy
    best = set(selected)
    best_size = len(selected)
    
    rng = random.Random(42)
    
    for trial in range(100):
        if time.time() > deadline - 20:
            break
        
        # Random order
        order = list(range(1, n + 1))
        rng.shuffle(order)
        sel = _greedy_build(conflicts, order, n)
        if len(sel) > best_size:
            best_size = len(sel)
            best = sel
    
    # Use branch-and-bound to try to find exact or near-exact solution
    if time.time() < deadline - 15:
        bnb_result = _branch_and_bound(bitsets, n, deadline=deadline - 10, initial_solution=best)
        if len(bnb_result) > best_size:
            best = bnb_result
            best_size = len(best)
    
    # Local search from best
    if time.time() < deadline - 5:
        improved = _local_search(conflicts, neighbors, best, n, max_iters=3000, deadline=deadline - 3)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    # More local search iterations if time permits
    while time.time() < deadline - 5:
        improved = _local_search(conflicts, neighbors, best, n, max_iters=2000, deadline=deadline - 3)
        if len(improved) <= best_size:
            break
        best = improved
        best_size = len(best)
    
    return sorted(best)

# EVOLVE-BLOCK-END
