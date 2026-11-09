import time
import random
import sys

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

# Build adjacency structure once
_ADJ = [set() for _ in range(P['n'])]
_EDGE_SET = set()
for u, v in P['edges']:
    _ADJ[u].add(v)
    _ADJ[v].add(u)
    _EDGE_SET.add((min(u, v), max(u, v)))

def _is_edge(u, v):
    return (min(u, v), max(u, v)) in _EDGE_SET

def _greedy_path(start, time_limit):
    """Greedy path construction using Warnsdorff-like heuristic."""
    path = [start]
    seen = {start}
    while True:
        candidates = _ADJ[path[-1]] - seen
        if not candidates:
            break
        # Choose candidate with fewest onward options (Warnsdorff)
        v = min(candidates, key=lambda w: (len(_ADJ[w] - seen), w))
        path.append(v)
        seen.add(v)
    return path

def _extend_both_ends(path, time_limit):
    """Try to extend a path from both ends greedily."""
    path = list(path)
    seen = set(path)
    changed = True
    while changed:
        changed = False
        # Try extending front
        if path:
            front = path[0]
            candidates = _ADJ[front] - seen
            if candidates:
                v = min(candidates, key=lambda w: (len(_ADJ[w] - seen), w))
                path.insert(0, v)
                seen.add(v)
                changed = True
        # Try extending back
        if path:
            back = path[-1]
            candidates = _ADJ[back] - seen
            if candidates:
                v = min(candidates, key=lambda w: (len(_ADJ[w] - seen), w))
                path.append(v)
                seen.add(v)
                changed = True
    return path

def _backtrack_path(start, max_depth, time_limit, deadline):
    """
    Depth-limited backtracking search for a long simple path starting at 'start'.
    Uses Warnsdorff ordering to prune efficiently.
    Returns the longest path found within max_depth and time budget.
    """
    best_path = [start]
    
    def dfs(path, seen, depth):
        nonlocal best_path
        if time.time() > deadline:
            return
        if len(path) > len(best_path):
            best_path = list(path)
        if depth >= max_depth:
            return
        
        current = path[-1]
        candidates = _ADJ[current] - seen
        if not candidates:
            return
        
        # Order by Warnsdorff: fewest onward options first
        candidates = sorted(candidates, key=lambda w: (len(_ADJ[w] - seen), w))
        
        for v in candidates:
            if time.time() > deadline:
                return
            path.append(v)
            seen.add(v)
            dfs(path, seen, depth + 1)
            path.pop()
            seen.discard(v)
    
    dfs([start], {start}, 0)
    return best_path

def construct():
    deadline = time.time() + 75  # Safety margin before 90s limit
    best_path = []
    
    # Phase 1: Greedy from every start vertex
    for start in range(P['n']):
        if time.time() > deadline:
            break
        path = _greedy_path(start, deadline)
        path = _extend_both_ends(path, deadline)
        if len(path) > len(best_path):
            best_path = path
    
    # Phase 2: Backtracking search from promising starts
    # Use the best greedy path's endpoints and high-degree vertices as seeds
    seed_vertices = set()
    if best_path:
        seed_vertices.add(best_path[0])
        seed_vertices.add(best_path[-1])
    
    # Add high-degree vertices
    for v in range(P['n']):
        if len(_ADJ[v]) >= 4:
            seed_vertices.add(v)
    
    # Sort seeds by degree descending for better initial exploration
    seed_vertices = sorted(seed_vertices, key=lambda v: -len(_ADJ[v]))
    
    for start in seed_vertices:
        if time.time() > deadline:
            break
        # Try increasing depth limits
        for max_depth in [15, 25, 40, 60]:
            if time.time() > deadline:
                break
            path = _backtrack_path(start, max_depth, deadline, deadline)
            if len(path) > len(best_path):
                best_path = path
                # Early exit if we found a Hamiltonian path
                if len(best_path) == P['n']:
                    return best_path
    
    # Phase 3: Randomized restarts with greedy + backtracking hybrid
    rng = random.Random(748)
    attempts = 0
    while time.time() < deadline and attempts < 200:
        attempts += 1
        start = rng.randint(0, P['n'] - 1)
        path = _greedy_path(start, deadline)
        path = _extend_both_ends(path, deadline)
        
        # If greedy path is short, try backtracking from its endpoint
        if len(path) < len(best_path) + 5:
            for endpoint in [path[0], path[-1]]:
                if time.time() > deadline:
                    break
                bp = _backtrack_path(endpoint, 50, deadline, deadline)
                if len(bp) > len(best_path):
                    best_path = bp
        
        if len(path) > len(best_path):
            best_path = path
    
    # Final: try to extend best_path from both ends one more time
    if best_path:
        best_path = _extend_both_ends(best_path, deadline)
    
    return best_path
# EVOLVE-BLOCK-END
