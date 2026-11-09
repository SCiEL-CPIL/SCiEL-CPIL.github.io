"""
Longest simple path in pinned G(100, 0.018) graph.
Strategy:
1. Identify connected components.
2. For the largest component, use multiple heuristics:
   - Greedy Warnsdorff from every start
   - Random walks with different seeds
   - Backtracking extension
   - IDDFS with time limit
3. Return the longest valid path found.
"""
import time
import random
import sys
from collections import deque

# Pinned graph parameters
P = {
    'n': 100,
    'edges': [
        [0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18],
        [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13],
        [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18],
        [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76],
        [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52],
        [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55],
        [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70],
        [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42],
        [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48],
        [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78],
        [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80],
        [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81],
        [69, 90], [70, 92], [74, 85], [75, 82]
    ]
}

N = P['n']
ADJ = [set() for _ in range(N)]
for u, v in P['edges']:
    ADJ[u].add(v)
    ADJ[v].add(u)

# Precompute degrees
DEG = [len(ADJ[i]) for i in range(N)]

def find_components():
    visited = set()
    comps = []
    for s in range(N):
        if s not in visited:
            comp = set()
            stack = [s]
            while stack:
                u = stack.pop()
                if u in visited:
                    continue
                visited.add(u)
                comp.add(u)
                for v in ADJ[u]:
                    if v not in visited:
                        stack.append(v)
            comps.append(comp)
    return comps

def greedy_warnsdorff(start, comp_set):
    """Greedy path using Warnsdorff's rule (prefer vertices with fewer unvisited neighbors)."""
    seen = {start}
    path = [start]
    while True:
        last = path[-1]
        cands = [v for v in ADJ[last] if v in comp_set and v not in seen]
        if not cands:
            break
        # Warnsdorff: pick vertex with fewest unvisited neighbors
        cands.sort(key=lambda v: (len([w for w in ADJ[v] if w in comp_set and w not in seen]), v))
        path.append(cands[0])
        seen.add(cands[0])
    return path

def extend_both_ends(path, comp_set):
    """Try to extend path from both ends using Warnsdorff."""
    path = list(path)
    seen = set(path)
    improved = True
    while improved:
        improved = False
        # Extend front
        while True:
            cands = [v for v in ADJ[path[0]] if v in comp_set and v not in seen]
            if not cands:
                break
            cands.sort(key=lambda v: (len([w for w in ADJ[v] if w in comp_set and w not in seen]), v))
            path.insert(0, cands[0])
            seen.add(cands[0])
            improved = True
        # Extend back
        while True:
            cands = [v for v in ADJ[path[-1]] if v in comp_set and v not in seen]
            if not cands:
                break
            cands.sort(key=lambda v: (len([w for w in ADJ[v] if w in comp_set and w not in seen]), v))
            path.append(cands[0])
            seen.add(cands[0])
            improved = True
    return path

def random_walk_path(start, comp_set, rng):
    """Random walk to build a path, then extend greedily."""
    path = [start]
    seen = {start}
    for _ in range(200):
        cur = path[-1]
        candidates = [nb for nb in ADJ[cur] if nb in comp_set and nb not in seen]
        if not candidates:
            break
        v = rng.choice(candidates)
        path.append(v)
        seen.add(v)
    return extend_both_ends(path, comp_set)

def backtracking_extend(path, comp_set, max_depth=50, deadline=None):
    """Try to extend path using limited backtracking."""
    path = list(path)
    seen = set(path)
    
    def dfs(cur, depth):
        if deadline is not None and time.time() > deadline:
            return False
        if depth >= max_depth:
            return False
        candidates = [nb for nb in ADJ[cur] if nb in comp_set and nb not in seen]
        if not candidates:
            return False
        # Sort by degree heuristic
        candidates.sort(key=lambda w: (len([x for x in ADJ[w] if x in comp_set and x not in seen]), w))
        for v in candidates:
            path.append(v)
            seen.add(v)
            if dfs(v, depth + 1):
                return True
            path.pop()
            seen.remove(v)
        return False
    
    # Try extending from the end
    if path:
        dfs(path[-1], 0)
    
    # Try extending from the start
    if path:
        start = path[0]
        candidates = [nb for nb in ADJ[start] if nb in comp_set and nb not in seen]
        for v in candidates:
            path.insert(0, v)
            seen.add(v)
            # Continue extending from the new start
            def dfs_start(cur, depth):
                if deadline is not None and time.time() > deadline:
                    return False
                if depth >= max_depth:
                    return False
                cands = [nb for nb in ADJ[cur] if nb in comp_set and nb not in seen]
                if not cands:
                    return False
                cands.sort(key=lambda w: (len([x for x in ADJ[w] if x in comp_set and x not in seen]), w))
                for nv in cands:
                    path.insert(0, nv)
                    seen.add(nv)
                    if dfs_start(nv, depth + 1):
                        return True
                    path.pop(0)
                    seen.remove(nv)
                return False
            dfs_start(v, 0)
            # Revert
            if path and path[0] == v:
                path.pop(0)
                seen.remove(v)
    
    return path

def iddfs_longest_path(comp, deadline):
    """
    Iterative deepening DFS to find longest simple path in component.
    Uses Warnsdorff ordering and time-limited search.
    """
    comp_set = set(comp)
    comp_list = sorted(comp)
    target_len = len(comp)
    best_path = []
    
    # Start with greedy to get a good initial path
    for start in comp_list:
        p = greedy_warnsdorff(start, comp_set)
        p = extend_both_ends(p, comp_set)
        if len(p) > len(best_path):
            best_path = p
    
    if len(best_path) >= target_len:
        return best_path
    
    # Now try to improve with DFS
    # Use the best greedy path as a starting point and try to insert missing vertices
    missing = comp_set - set(best_path)
    
    # Try DFS from each end of best_path, trying to incorporate missing vertices
    def dfs_extend(path, seen, remaining_budget):
        """DFS to extend path, trying to cover more vertices."""
        nonlocal best_path
        if time.time() > deadline:
            return False
        
        if len(path) > len(best_path):
            best_path = list(path)
            if len(path) >= target_len:
                return True
        
        last = path[-1]
        cands = [v for v in ADJ[last] if v in comp_set and v not in seen]
        if not cands:
            return False
        
        # Warnsdorff ordering
        cands.sort(key=lambda v: (len([w for w in ADJ[v] if w in comp_set and w not in seen]), v))
        
        for v in cands:
            path.append(v)
            seen.add(v)
            if dfs_extend(path, seen, remaining_budget):
                return True
            path.pop()
            seen.remove(v)
            if time.time() > deadline:
                return False
        return False
    
    # Try extending from both ends of the best greedy path
    for start in [best_path[0], best_path[-1]]:
        if time.time() > deadline:
            break
        seen = set(best_path)
        # Try adding missing vertices one by one via DFS
        dfs_extend(list(best_path), seen, deadline - time.time())
    
    # Try full DFS from various starts with time limit
    for start in comp_list:
        if time.time() > deadline:
            break
        if len(best_path) >= target_len:
            break
        seen = {start}
        path = [start]
        dfs_extend(path, seen, deadline - time.time())
    
    return best_path

def construct():
    t0 = time.time()
    deadline = t0 + 85  # Safety margin: return well before 90s limit
    
    # Find components
    comps = find_components()
    comps.sort(key=len, reverse=True)
    
    best_path = []
    
    for comp in comps:
        if time.time() > deadline:
            break
        comp_list = sorted(comp)
        comp_set = set(comp)
        
        # Strategy 1: Greedy from every start in the component
        for start in comp_list:
            if time.time() > deadline - 5:
                break
            p = greedy_warnsdorff(start, comp_set)
            p = extend_both_ends(p, comp_set)
            if len(p) > len(best_path):
                best_path = p
        
        # Strategy 2: Random walks from various starts
        rng = random.Random(42)
        for start in comp_list:
            if time.time() > deadline - 10:
                break
            for _ in range(20):
                if time.time() > deadline - 10:
                    break
                path = random_walk_path(start, comp_set, rng)
                if len(path) > len(best_path):
                    best_path = path
        
        # Strategy 3: Backtracking extension on the best path found so far
        if best_path and time.time() < deadline - 15:
            extended = backtracking_extend(best_path, comp_set, max_depth=30, deadline=deadline - 5)
            if len(extended) > len(best_path):
                best_path = extended
        
        # If we haven't found a full path, try IDDFS
        if len(best_path) < len(comp) and time.time() < deadline - 20:
            improved = iddfs_longest_path(comp_list, deadline - 5)
            if len(improved) > len(best_path):
                best_path = improved
    
    # Strategy 4: Try more random walks with different seeds
    for seed in range(100):
        if time.time() > deadline - 5:
            break
        rng = random.Random(seed)
        for comp in comps:
            if len(comp) <= len(best_path):
                break
            comp_set = set(comp)
            comp_list = sorted(comp)
            for _ in range(5):
                if time.time() > deadline - 5:
                    break
                start = rng.choice(comp_list)
                path = random_walk_path(start, comp_set, rng)
                if len(path) > len(best_path):
                    best_path = path
    
    # Final attempt: extend best path more aggressively
    if best_path and time.time() < deadline - 2:
        comp_set = set(best_path) | set(ADJ[best_path[0]]) | set(ADJ[best_path[-1]])
        extended = extend_both_ends(best_path, comp_set)
        if len(extended) > len(best_path):
            best_path = extended
    
    if not best_path:
        best_path = [0]
    
    # Verify validity
    for i in range(len(best_path) - 1):
        if best_path[i+1] not in ADJ[best_path[i]]:
            return [0]
    if len(set(best_path)) != len(best_path):
        seen = set()
        fixed = []
        for v in best_path:
            if v not in seen:
                fixed.append(v)
                seen.add(v)
        return fixed
    
    return best_path
