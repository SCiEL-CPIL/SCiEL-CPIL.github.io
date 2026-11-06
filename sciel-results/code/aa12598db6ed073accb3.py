"""Long-path search on the pinned G(100,1.8/100) graph (seed 748)."""
import time
import random

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
ADJ = [set() for _ in range(N)]
for u, v in P['edges']:
    ADJ[u].add(v)
    ADJ[v].add(u)
DEADLINE = time.time() + 75.0  # safety margin under the 90s wall-clock kill


def _components():
    seen = set()
    comps = []
    for s in range(N):
        if s in seen:
            continue
        stack = [s]
        seen.add(s)
        comp = []
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in ADJ[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps


def _backtrack_search(start, best_len, rng):
    """
    Time-budgeted randomized backtracking from a given start vertex.
    Uses Warnsdorff heuristic with randomization to explore diverse paths.
    Returns the best path found during this search.
    """
    path = [start]
    visited = {start}
    best = [start]
    best_len_local = 1
    nodes_explored = [0]
    
    def _sort_candidates(cur, visited_set):
        """Sort candidates by Warnsdorff criterion with some randomness."""
        cands = [v for v in ADJ[cur] if v not in visited_set]
        if not cands:
            return []
        # Warnsdorff: prefer vertices with fewer onward options
        # Add small random perturbation to break ties and explore diversity
        def score(v):
            onward = len(ADJ[v] - visited_set)
            # Mix of Warnsdorff score and random noise
            return (onward, rng.random())
        cands.sort(key=score)
        return cands
    
    def _dfs(cur, depth, max_depth):
        nonlocal best, best_len_local
        nodes_explored[0] += 1
        
        # Check time periodically
        if nodes_explored[0] % 1024 == 0 and time.time() >= DEADLINE:
            return
        
        if len(path) > best_len_local:
            best_len_local = len(path)
            best = list(path)
        
        if depth >= max_depth:
            return
        
        cands = _sort_candidates(cur, visited)
        for nxt in cands:
            path.append(nxt)
            visited.add(nxt)
            _dfs(nxt, depth + 1, max_depth)
            path.pop()
            visited.discard(nxt)
    
    # Adaptive depth: start shallow, go deeper if time permits
    base_depth = 15
    max_depth = min(base_depth, N - 1)
    
    _dfs(start, 0, max_depth)
    
    return best


def _greedy_path(start, mode=0):
    """Quick greedy path for initial seed."""
    p = [start]
    s = {start}
    while True:
        cur = p[-1]
        cand = ADJ[cur] - s
        if not cand:
            break
        if mode == 0:
            nxt = min(cand, key=lambda w: (len(ADJ[w] - s), w))
        else:
            nxt = min(cand, key=lambda w: (-len(ADJ[w]), w))
        p.append(nxt)
        s.add(nxt)
    return p


def _try_insert(path, budget=30):
    """Insert unused vertices into path where possible."""
    p = list(path)
    used = set(p)
    unused = [v for v in range(N) if v not in used]
    improved = True
    steps = 0
    while improved and steps < budget and time.time() < DEADLINE:
        improved = False
        steps += 1
        for i in range(len(p) - 1):
            a, b = p[i], p[i + 1]
            for v in unused:
                if v in ADJ[a] and v in ADJ[b] and v not in used:
                    p.insert(i + 1, v)
                    used.add(v)
                    unused.remove(v)
                    improved = True
                    break
            if improved:
                break
    return p


def construct():
    t0 = time.time()
    rng = random.Random(748)
    comps = _components()
    
    if not comps:
        return [0]
    
    # Focus on the largest component
    main_comp = comps[0]
    main_set = set(main_comp)
    
    # Get all vertices in the main component sorted by degree
    vertices = list(main_comp)
    vertices.sort(key=lambda v: -len(ADJ[v]))
    
    best_path = []
    best_len = 0
    
    # Phase 1: Quick greedy passes to get a good initial lower bound
    for start in vertices[:20]:
        if time.time() >= DEADLINE:
            break
        for mode in (0, 1):
            p = _greedy_path(start, mode)
            if len(p) > best_len:
                best_len = len(p)
                best_path = list(p)
    
    # Phase 2: Multiple randomized backtracking searches from different starts
    # Use more sophisticated search with adaptive depth
    search_starts = vertices[:30]  # Try top 30 degree vertices
    
    for i, start in enumerate(search_starts):
        if time.time() >= DEADLINE:
            break
        
        # Adaptive: use more time for earlier searches
        remaining = DEADLINE - time.time()
        if remaining < 5:
            break
        
        # Run backtracking search
        result = _backtrack_search(start, best_len, rng)
        if len(result) > best_len:
            best_len = len(result)
            best_path = list(result)
        
        # Try to improve by insertion
        if time.time() < DEADLINE:
            improved = _try_insert(best_path, budget=20)
            if len(improved) > best_len:
                best_len = len(improved)
                best_path = list(improved)
    
    # Phase 3: If we have time, do a few more targeted searches
    # from vertices not yet tried as starts, focusing on those that
    # are endpoints of the current best path
    if time.time() < DEADLINE - 10:
        # Try extending from both ends of best_path
        if best_path and len(best_path) < N:
            for end in [best_path[0], best_path[-1]]:
                if time.time() >= DEADLINE:
                    break
                result = _backtrack_search(end, best_len, rng)
                if len(result) > best_len:
                    best_len = len(result)
                    best_path = list(result)
                
                improved = _try_insert(best_path, budget=15)
                if len(improved) > best_len:
                    best_len = len(improved)
                    best_path = list(improved)
    
    # Final insertion pass
    if time.time() < DEADLINE - 2:
        final = _try_insert(best_path, budget=50)
        if len(final) > best_len:
            best_path = list(final)
    
    if not best_path:
        best_path = [vertices[0]]
    
    return best_path
