"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_clique(vertices, adj):
    for u in vertices:
        for v in vertices:
            if u != v and v not in adj[u]:
                return False
    return True

def _is_independent(vertices, adj):
    for u in vertices:
        for v in vertices:
            if u != v and v in adj[u]:
                return False
    return True

def _greedy_partition(n, adj):
    parts = []
    for v in range(n):
        best_part = None
        best_gain = -1
        for i, part in enumerate(parts):
            trial = list(part) + [v]
            if _is_clique(trial, adj) or _is_independent(trial, adj):
                gain = len(trial)
                if gain > best_gain:
                    best_gain = gain
                    best_part = i
        if best_part is not None:
            parts[best_part].append(v)
        else:
            parts.append([v])
    return parts

def _try_extend(parts, unassigned, adj, n):
    """Try to extend the current partition by assigning one vertex from unassigned."""
    if not unassigned:
        return True
    v = unassigned[0]
    rest = unassigned[1:]
    # Try adding v to each existing part
    for i, part in enumerate(parts):
        trial = list(part) + [v]
        if _is_clique(trial, adj) or _is_independent(trial, adj):
            parts[i] = trial
            if _try_extend(parts, rest, adj, n):
                return True
            parts[i] = list(part)  # backtrack
    # Try creating a new part with v
    parts.append([v])
    if _try_extend(parts, rest, adj, n):
        return True
    parts.pop()
    return False

def _exact_search(n, adj, deadline):
    """Find minimum number of parts via backtracking with deadline."""
    start_time = time.time()
    best_parts = None
    best_count = n + 1
    
    # First get a greedy solution as upper bound
    greedy = _greedy_partition(n, adj)
    best_count = len(greedy)
    best_parts = greedy
    
    # Try to find a solution with fewer parts
    for target in range(1, best_count):
        if time.time() > deadline:
            break
        # Use backtracking to find a partition with exactly 'target' parts
        found = _search_with_target(n, adj, target, time.time, deadline)
        if found is not None:
            best_parts = found
            best_count = target
            break
    
    return best_parts

def _search_with_target(n, adj, target, time_func, deadline):
    """Find a cochromatic partition with exactly 'target' parts."""
    # We'll use backtracking: assign vertices to parts
    # Parts are mutable lists, we track which vertices are assigned
    parts = [[] for _ in range(target)]
    assigned = [False] * n
    
    def backtrack(v_idx):
        if time_func() > deadline:
            return False
        if v_idx == n:
            # Check all parts are non-empty
            return all(len(p) > 0 for p in parts)
        
        # Try adding vertex v_idx to each part
        for i in range(target):
            trial = parts[i] + [v_idx]
            if _is_clique(trial, adj) or _is_independent(trial, adj):
                parts[i] = trial
                if backtrack(v_idx + 1):
                    return True
                parts[i] = list(trial[:-1])  # backtrack
        
        return False
    
    if backtrack(0):
        return [list(p) for p in parts if p]
    return None

def construct():
    n = P['n']
    adj = _build_adj(n, P['edges'])
    
    # Deadline: 85 seconds (5s safety margin from 90s limit)
    deadline = time.time() + 85.0
    
    # Try exact search first
    result = _exact_search(n, adj, deadline)
    
    if result is None:
        # Fallback to greedy
        result = _greedy_partition(n, adj)
    
    # Ensure proper types: list of lists of Python ints
    final_result = []
    for part in result:
        final_result.append([int(v) for v in part])
    
    return final_result
# EVOLVE-BLOCK-END
