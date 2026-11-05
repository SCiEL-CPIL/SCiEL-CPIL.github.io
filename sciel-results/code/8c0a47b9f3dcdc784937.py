"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _conflict_matrix(n=256):
    """Return a boolean numpy array C where C[i,j] is True if (i+1)*(j+1) % ((i+1)+(j+1)) == 0."""
    C = np.zeros((n, n), dtype=bool)
    for i in range(n):
        a = i + 1
        for j in range(i + 1, n):
            b = j + 1
            if (a * b) % (a + b) == 0:
                C[i, j] = True
                C[j, i] = True
    return C

def _greedy_build(C, order, selected=None):
    """Greedily build an independent set following the given order."""
    if selected is None:
        selected = set()
    else:
        selected = set(selected)
    n = C.shape[0]
    for idx in order:
        if idx in selected:
            continue
        # Check if idx conflicts with any already selected
        conflict = False
        for s in selected:
            if C[idx, s]:
                conflict = True
                break
        if not conflict:
            selected.add(idx)
    return selected

def _local_search(C, selected, max_swaps=500, rng=None):
    """Try to improve the independent set by single swaps (remove one, add two) or single adds."""
    if rng is None:
        rng = random
    selected = set(selected)
    n = C.shape[0]
    improved = True
    swap_count = 0
    while improved and swap_count < max_swaps:
        improved = False
        # Try single adds: add a vertex not in selected that has no conflicts
        for v in range(n):
            if v in selected:
                continue
            conflict = False
            for s in selected:
                if C[v, s]:
                    conflict = True
                    break
            if not conflict:
                selected.add(v)
                improved = True
                swap_count += 1
                break
        if improved:
            continue
        # Try swaps: remove one, add two
        # For efficiency, try removing a low-degree vertex and adding two high-value vertices
        for v in list(selected):
            # Find candidates that conflict with v (so removing v might help)
            # Actually, we want to remove v and add two vertices that were previously blocked
            # Simple approach: remove v, then try to add two vertices
            selected.discard(v)
            added = []
            for w in range(n):
                if w in selected or w == v:
                    continue
                conflict = False
                for s in selected:
                    if C[w, s]:
                        conflict = True
                        break
                if not conflict:
                    added.append(w)
                    if len(added) == 2:
                        break
            if len(added) == 2:
                selected.update(added)
                improved = True
                swap_count += 1
                break
            else:
                # Revert
                selected.add(v)
    return selected

def construct():
    deadline = time.time() + 85  # 5s safety margin
    C = _conflict_matrix(256)
    
    best_set = set(range(1, 257, 2))  # Start with odd numbers
    best_size = len(best_set)
    
    rng = random.Random(42)
    
    # Multi-start greedy with different orderings
    start_times = [
        list(range(256)),                    # Natural order
        list(range(255, -1, -1)),            # Reverse order
        sorted(range(256), key=lambda i: C[i].sum()),  # Low degree first
        sorted(range(256), key=lambda i: C[i].sum(), reverse=True),  # High degree first
    ]
    
    for _ in range(5):
        start_times.append(rng.sample(range(256), 256))
    
    for order in start_times:
        if time.time() > deadline:
            break
        s = _greedy_build(C, order)
        s = _local_search(C, s, max_swaps=200, rng=rng)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
    
    # Additional local search passes on the best set
    if time.time() < deadline:
        best_set = _local_search(C, best_set, max_swaps=500, rng=rng)
        best_size = len(best_set)
    
    # More randomized restarts if time permits
    while time.time() < deadline - 5:
        order = rng.sample(range(256), 256)
        s = _greedy_build(C, order)
        s = _local_search(C, s, max_swaps=100, rng=rng)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            # Further polish
            best_set = _local_search(C, best_set, max_swaps=200, rng=rng)
            best_size = len(best_set)
    
    # Convert to 1-indexed list
    result = sorted(v + 1 for v in best_set)
    return result
# EVOLVE-BLOCK-END
