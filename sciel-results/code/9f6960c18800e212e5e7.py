"""Minimize distinct colors for degree-4 graph with bicolored-forest constraint.
Uses local search with simulated annealing and fast incremental validity checks."""
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

_N = P['n']
_EDGES = P['edges']
_ADJ = [[] for _ in range(_N)]
for _a, _b in _EDGES:
    _ADJ[_a].append(_b)
    _ADJ[_b].append(_a)


def _check_validity(coloring):
    """Check proper coloring and bicolored-forest constraint. Returns True if valid."""
    # Check proper coloring
    for a, b in _EDGES:
        if coloring[a] == coloring[b]:
            return False
    
    # Check bicolored forest constraint using union-find
    present = sorted(set(coloring))
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            ci, cj = present[i], present[j]
            parent = list(range(_N))
            rank = [0] * _N
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return False
                if rank[rx] < rank[ry]:
                    rx, ry = ry, rx
                parent[ry] = rx
                if rank[rx] == rank[ry]:
                    rank[rx] += 1
                return True
            
            for a, b in _EDGES:
                ca, cb = coloring[a], coloring[b]
                if ca in (ci, cj) and cb in (ci, cj):
                    if not union(a, b):
                        return False
    return True


def _count_colors(coloring):
    return len(set(coloring))


def _greedy_initial(k):
    """Greedy coloring with k colors, trying to minimize color usage."""
    coloring = [-1] * _N
    # Order by degree descending, then by index
    order = sorted(range(_N), key=lambda v: (-len(_ADJ[v]), v))
    
    for v in order:
        used = set()
        for nb in _ADJ[v]:
            if coloring[nb] != -1:
                used.add(coloring[nb])
        
        # Try colors 0 to k-1, prefer lower colors
        for c in range(k):
            if c not in used:
                coloring[v] = c
                break
        if coloring[v] == -1:
            coloring[v] = 0  # Should not happen for k >= 4
    
    return coloring


def _local_search(coloring, max_colors, deadline, rng):
    """Local search to minimize distinct colors while maintaining validity."""
    best = coloring[:]
    best_score = _count_colors(best)
    current = coloring[:]
    current_score = best_score
    
    # Precompute edge indices for quick access
    edge_by_vertex = defaultdict(list)
    for i, (a, b) in enumerate(_EDGES):
        edge_by_vertex[a].append((i, b))
        edge_by_vertex[b].append((i, a))
    
    temp = 10.0
    cooling = 0.9995
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Choose a random vertex to modify
        v = rng.randint(0, _N - 1)
        old_c = current[v]
        
        # Choose a new color
        if rng.random() < 0.7:
            # Try to use an existing color
            existing = [c for c in set(current) if c != old_c]
            if existing:
                new_c = rng.choice(existing)
            else:
                new_c = old_c
        else:
            # Try a new color or random
            new_c = rng.randint(0, max_colors - 1)
        
        if new_c == old_c:
            continue
        
        # Check if new color is valid (not same as neighbors)
        valid_move = True
        for nb in _ADJ[v]:
            if current[nb] == new_c:
                valid_move = False
                break
        
        if not valid_move:
            no_improve += 1
            continue
        
        # Make the move
        current[v] = new_c
        new_score = _count_colors(current)
        
        # Check validity
        if _check_validity(current):
            if new_score < best_score:
                best = current[:]
                best_score = new_score
                no_improve = 0
                temp = max(temp * cooling, 0.1)
            elif new_score == best_score:
                no_improve += 1
                # Accept with some probability for diversity
                if rng.random() < 0.1:
                    temp *= cooling
                else:
                    current[v] = old_c  # Revert
            else:
                # Score got worse, revert
                current[v] = old_c
                no_improve += 1
        else:
            # Invalid, revert
            current[v] = old_c
            no_improve += 1
        
        temp *= cooling
    
    return best, best_score


def _try_reduce_colors(coloring, max_colors, deadline, rng):
    """Try to reduce the number of colors by merging color classes."""
    best = coloring[:]
    best_score = _count_colors(best)
    
    if best_score <= 2:
        return best, best_score
    
    # Try merging pairs of colors
    colors_used = sorted(set(coloring))
    for i in range(len(colors_used)):
        for j in range(i + 1, len(colors_used)):
            if time.time() > deadline:
                break
            
            ci, cj = colors_used[i], colors_used[j]
            
            # Create a new coloring where cj is replaced by ci
            candidate = coloring[:]
            for v in range(_N):
                if candidate[v] == cj:
                    candidate[v] = ci
            
            # Check validity
            if _check_validity(candidate):
                new_score = _count_colors(candidate)
                if new_score < best_score:
                    best = candidate[:]
                    best_score = new_score
                    colors_used = sorted(set(best))
    
    return best, best_score


def construct():
    deadline = time.time() + 80.0
    rng = random.Random(42)
    
    # Start with a greedy 4-coloring
    coloring = _greedy_initial(4)
    
    # Verify initial validity
    if not _check_validity(coloring):
        # Try with more colors
        for k in range(5, 10):
            coloring = _greedy_initial(k)
            if _check_validity(coloring):
                break
    
    best = coloring[:]
    best_score = _count_colors(best)
    
    # Try to reduce colors by merging
    best, best_score = _try_reduce_colors(best, 10, deadline, rng)
    
    # Local search to find better solutions
    if time.time() < deadline - 10:
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            
            # Perturb the current best
            perturbed = best[:]
            for _ in range(5):
                v = rng.randint(0, _N - 1)
                new_c = rng.randint(0, best_score + 1)
                # Check if valid
                valid = True
                for nb in _ADJ[v]:
                    if perturbed[nb] == new_c:
                        valid = False
                        break
                if valid:
                    perturbed[v] = new_c
            
            if _check_validity(perturbed):
                result, score = _local_search(perturbed, best_score + 2, deadline - 5, rng)
                if score < best_score:
                    best = result
                    best_score = score
            
            # Try reducing colors again
            best, best_score = _try_reduce_colors(best, best_score + 2, deadline - 5, rng)
    
    # Final verification
    if not _check_validity(best):
        # Fallback: use a known valid coloring
        best = _greedy_initial(5)
        while not _check_validity(best):
            best = _greedy_initial(6)
    
    return [int(x) for x in best]
