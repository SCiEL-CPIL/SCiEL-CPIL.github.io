"""Find longest simple path in the pinned graph.

Combines Iterative Deepening DFS (IDDFS) with Warnsdorff heuristic for deep search
and a robust refinement pipeline (extend both ends, insert unvisited vertices)
to maximize the simple path length in the pinned graph.
"""
import time
import random

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
ADJ = [set() for _ in range(N)]
for u, v in P['edges']:
    ADJ[u].add(v)
    ADJ[v].add(u)

def find_components():
    """Find all connected components."""
    visited = [False] * N
    comps = []
    for i in range(N):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nb in ADJ[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
            comps.append(comp)
    return comps

def greedy_path_from(start, adj, seen=None):
    """Greedy path starting from start (Warnsdorff: fewest remaining options)."""
    if seen is None:
        seen = set()
    path = [start]
    seen.add(start)
    while True:
        cur = path[-1]
        candidates = [nb for nb in adj[cur] if nb not in seen]
        if not candidates:
            break
        v = min(candidates, key=lambda w: (len([x for x in adj[w] if x not in seen]), w))
        path.append(v)
        seen.add(v)
    return path

def extend_both_ends(path, adj):
    """Try to extend path from both ends (Warnsdorff)."""
    path = list(path)
    seen = set(path)
    changed = True
    while changed:
        changed = False
        if path:
            cur = path[0]
            candidates = [nb for nb in adj[cur] if nb not in seen]
            if candidates:
                v = min(candidates, key=lambda w: (len([x for x in adj[w] if x not in seen]), w))
                path.insert(0, v)
                seen.add(v)
                changed = True
        if path:
            cur = path[-1]
            candidates = [nb for nb in adj[cur] if nb not in seen]
            if candidates:
                v = min(candidates, key=lambda w: (len([x for x in adj[w] if x not in seen]), w))
                path.append(v)
                seen.add(v)
                changed = True
    return path

def try_insert(path, adj):
    """Try to insert unvisited vertices between consecutive path nodes."""
    path = list(path)
    seen = set(path)
    unvisited = set(range(N)) - seen
    improved = True
    while improved and unvisited:
        improved = False
        for w in list(unvisited):
            for i in range(len(path) - 1):
                u = path[i]
                v = path[i + 1]
                if w in adj[u] and w in adj[v]:
                    path.insert(i + 1, w)
                    seen.add(w)
                    unvisited.remove(w)
                    improved = True
                    break
            if improved:
                break
    return path

def refine(path, adj):
    """Polish a candidate path with the refinement pipeline."""
    path = extend_both_ends(path, adj)
    path = try_insert(path, adj)
    path = extend_both_ends(path, adj)
    return path

def iddfs_search(start, adj, deadline, initial_depth=5, max_depth=None):
    """
    Iterative Deepening DFS with Warnsdorff heuristic.
    Returns the best path found within the time budget.
    """
    if max_depth is None:
        max_depth = N - 1
    
    best_path = [start]
    best_len = 1
    found_deeper = False

    # We will iterate depth from initial_depth up to max_depth
    # But we stop early if we find a path of length N (Hamiltonian)
    # Or if time runs out.
    
    # Optimization: If we already have a very long path, we can start deeper
    # But for safety and simplicity, we start from a small depth and increase.
    # However, pure IDDFS from 1 is slow. We start from a reasonable base.
    
    # Let's try to jump to the current best length if we have one, 
    # but since this is called per-start, we might not have a global best yet.
    # We'll just iterate.
    
    current_depth = initial_depth
    
    while current_depth <= max_depth:
        if time.time() > deadline:
            break
            
        # DFS with depth limit 'current_depth'
        # We want to find a path of length 'current_depth'
        # Actually, IDDFS usually checks for a solution at exactly depth D.
        # Here we want the LONGEST path. 
        # So we run DFS up to depth D, and keep track of the longest path seen.
        # But that's just DFS with a limit.
        # True IDDFS for longest path: 
        # 1. Try to find a path of length D.
        # 2. If found, best_len = D, current_depth = D + 1.
        # 3. If not found, we might still have found a path of length D-1.
        
        # A better strategy for "longest path" with IDDFS:
        # Run DFS with a limit. If we find a path of length == limit, we increase limit.
        # If we exhaust the search space for a limit without finding a path of that length,
        # we know no path of that length exists starting from 'start' (with this heuristic/pruning).
        
        path = [start]
        seen = {start}
        
        # We need a flag to indicate if we found a path of length 'current_depth'
        found_at_depth = False
        
        def dfs(cur, depth):
            nonlocal found_at_depth
            if time.time() > deadline:
                return
            if depth == current_depth:
                # We found a path of length 'current_depth'
                nonlocal best_path, best_len
                best_path = list(path)
                best_len = current_depth
                found_at_depth = True
                return
            
            candidates = [nb for nb in adj[cur] if nb not in seen]
            if not candidates:
                return
            
            # Warnsdorff heuristic: sort by fewest unvisited neighbors
            candidates.sort(key=lambda w: (len([x for x in adj[w] if x not in seen]), w))
            
            for v in candidates:
                path.append(v)
                seen.add(v)
                dfs(v, depth + 1)
                path.pop()
                seen.remove(v)
                
                if found_at_depth or time.time() > deadline:
                    return

        dfs(start, 1)
        
        if found_at_depth:
            current_depth += 1
            # If we found a Hamiltonian path, we are done
            if best_len == N:
                break
        else:
            # If we didn't find a path of length 'current_depth', 
            # it means the max path length from this start is < current_depth.
            # We can stop increasing depth for this start.
            break

    return best_path

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    comps = find_components()
    comps.sort(key=len, reverse=True)
    if not comps:
        return [0]

    largest_comp = comps[0]
    if not largest_comp:
        return [0]

    best_path = list(largest_comp[:1])
    best_len = len(best_path)

    # Phase 1: Greedy initialization
    # Run greedy from all nodes to get a decent starting point and identify promising starts
    promising_starts = []
    for start in largest_comp:
        if time.time() > deadline - 30:
            break
        p = greedy_path_from(start, ADJ)
        p = refine(p, ADJ)
        if len(p) > best_len:
            best_path = p
            best_len = len(p)
        # Track starts that yielded long paths
        if len(p) > best_len * 0.8: # Heuristic: keep starts that are "good"
            promising_starts.append(start)

    # If we have a very long path already, we might skip some IDDFS
    if best_len == N:
        return best_path

    # Phase 2: IDDFS from promising starts
    # We prioritize starts that are high degree or were good in greedy
    # Sort promising starts by degree descending to find "central" paths first
    unique_promising = list(set(promising_starts))
    # Add top degree nodes from the largest component if not already present
    deg_sorted = sorted(largest_comp, key=lambda v: len(ADJ[v]), reverse=True)
    for v in deg_sorted[:10]:
        if v not in unique_promising:
            unique_promising.append(v)
            
    # Limit the number of IDDFS runs to ensure we finish in time
    # Each IDDFS run can be expensive. 
    # We have ~50s left. Let's allow ~10-20 runs.
    max_iddfs_runs = 15
    
    for start in unique_promising[:max_iddfs_runs]:
        if time.time() > deadline - 10:
            break
            
        # Start IDDFS from a depth close to current best to save time
        # If we have a path of length L, we try to find L+1, L+2...
        # But IDDFS is more effective if we start lower. 
        # However, if we know a path of length 80 exists, starting IDDFS at 1 is wasteful.
        # Let's start at max(5, best_len - 5) to catch nearby improvements.
        # Actually, if we start at best_len, and we find a path of that length, we go to best_len+1.
        # If we don't, we stop. This is efficient.
        
        start_depth = max(5, best_len - 2)
        
        p = iddfs_search(start, ADJ, deadline - 10, initial_depth=start_depth)
        p = refine(p, ADJ)
        
        if len(p) > best_len:
            best_path = p
            best_len = len(p)
            
            if best_len == N:
                return best_path

    # Phase 3: Final Refinement
    if best_path and time.time() < deadline - 1:
        extended = refine(best_path, ADJ)
        if len(extended) > best_len:
            best_path = extended
            best_len = len(best_path)

    # Validate
    if not best_path:
        best_path = [0]
    
    # Final sanity check
    for i in range(len(best_path) - 1):
        if best_path[i + 1] not in ADJ[best_path[i]]:
            # Fallback to a valid greedy path
            for s in range(N):
                p = greedy_path_from(s, ADJ)
                if p:
                    best_path = p
                    break
            break

    return best_path
