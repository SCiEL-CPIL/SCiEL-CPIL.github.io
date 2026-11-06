"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def _is_perfect_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

def _build_conflict_graph():
    """Build adjacency: conflict[i] = set of j where i+j is a perfect square."""
    n = 256
    conflict = [set() for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            s = i + j
            if _is_perfect_square(s):
                conflict[i].add(j)
                if j != i:
                    conflict[j].add(i)
    return conflict

def _greedy_construct(conflict, order):
    """Greedy: add elements in given order if no conflict with already-selected."""
    selected = set()
    for v in order:
        if not (selected & conflict[v]):
            selected.add(v)
    return selected

def _local_search(conflict, selected, max_iters=50000, deadline=None):
    """Local search: try single additions, swaps, and removals to maximize size."""
    selected = set(selected)
    best_size = len(selected)
    
    # Precompute available (not selected)
    all_nodes = set(range(1, 257))
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        # Try adding an unselected node
        added = False
        for v in sorted(all_nodes - selected):
            if not (selected & conflict[v]):
                selected.add(v)
                added = True
                break
        
        if added:
            if len(selected) > best_size:
                best_size = len(selected)
            continue
        
        # Try swapping: remove one selected, add two unselected
        improved = False
        sel_list = list(selected)
        random.shuffle(sel_list)
        for rem in sel_list:
            if deadline and time.time() > deadline:
                break
            # After removing rem, which unselected nodes can be added?
            candidate_rems = all_nodes - selected
            addable = []
            for v in candidate_rems:
                if not ((selected - {rem}) & conflict[v]):
                    addable.append(v)
            if len(addable) >= 2:
                # Try adding two
                for i in range(len(addable)):
                    for j in range(i + 1, len(addable)):
                        a, b = addable[i], addable[j]
                        if a not in conflict[b]:
                            selected.discard(rem)
                            selected.add(a)
                            selected.add(b)
                            improved = True
                            break
                    if improved:
                        break
            if improved:
                break
        
        if improved:
            if len(selected) > best_size:
                best_size = len(selected)
            continue
        
        # Try single swap: remove one, add one (same size, but might enable future improvements)
        # Actually skip single swaps for now, focus on growth
        break
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s out of 90s budget
    
    conflict = _build_conflict_graph()
    
    # Start with the seed (mod 3 class)
    best = set(range(1, 257, 3))
    best_size = len(best)
    
    # Try multiple random orderings for greedy
    for trial in range(200):
        if time.time() > deadline:
            break
        order = list(range(1, 257))
        random.shuffle(order)
        candidate = _greedy_construct(conflict, order)
        if len(candidate) > best_size:
            best = candidate
            best_size = len(candidate)
    
    # Local search from best
    if time.time() < deadline - 5:
        improved = _local_search(conflict, best, max_iters=100000, deadline=deadline - 2)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    # Try local search from a few random greedy solutions
    for trial in range(50):
        if time.time() > deadline - 5:
            break
        order = list(range(1, 257))
        random.shuffle(order)
        candidate = _greedy_construct(conflict, order)
        if len(candidate) >= best_size - 2:
            improved = _local_search(conflict, candidate, max_iters=20000, deadline=deadline - 2)
            if len(improved) > best_size:
                best = improved
                best_size = len(best)
    
    return sorted(best)
# EVOLVE-BLOCK-END
