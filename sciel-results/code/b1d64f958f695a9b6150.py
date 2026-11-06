"""Rewritten program: bipartite-matching-guided DFS for longest path in pinned G(100, 0.018)."""
import random
from collections import deque

# Rebuild the pinned graph exactly as in the seed (same seed, same sequence).
def _build_graph():
    n = 100
    rng = random.Random(748)
    edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < 1.8 / 100:
                edges.add((u, v))
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return n, adj

N, ADJ = _build_graph()

# ---------- Bipartite matching relaxation (Hopcroft-Karp style) ----------
# We model a "path skeleton" as a bipartite graph between two copies of vertices:
# left = positions 0,2,4,... (even-indexed in the path), right = positions 1,3,5,...
# An edge (u,v) in the original graph can serve as a path edge if u is on the left
# and v on the right, or vice versa. The maximum matching in this bipartite graph
# gives an upper bound on the number of edges in any simple path, hence a bound
# on the number of vertices.
#
# For guidance: vertices that are "tight" (matched in many maximum matchings, or
# have low slack) are more likely to be internal path vertices. We use this to
# prioritize DFS branching.

def _max_matching_size(adj, n):
    """
    Compute the size of the maximum matching in the bipartite graph where
    left side = vertices 0..n-1, right side = vertices 0..n-1,
    and (u,v) is an edge if u != v and v in adj[u].
    This is essentially the maximum matching in the original graph treated as bipartite
    with both sides being all vertices (a standard trick for path problems).
    """
    # We'll use a simple DFS-based augmenting path algorithm (Kuhn's algorithm).
    # Left nodes: 0..n-1, Right nodes: 0..n-1.
    match_r = [-1] * n  # match_r[v] = u means right node v is matched to left node u
    result = 0
    for u in range(n):
        visited = [False] * n
        def dfs(u):
            for v in adj[u]:
                if v == u or not visited[v]:
                    visited[v] = True
                    if match_r[v] == -1 or dfs(match_r[v]):
                        match_r[v] = u
                        return True
            return False
        if dfs(u):
            result += 1
    return result

# Compute matching-based upper bound on path length (in vertices)
MATCHING_SIZE = _max_matching_size(ADJ, N)
# Path with k vertices has k-1 edges, so max vertices <= MATCHING_SIZE + 1
PATH_UPPER_BOUND = MATCHING_SIZE + 1

# ---------- Slack computation for guidance ----------
# For each vertex, compute how many neighbors it has. Vertices with fewer neighbors
# are more "constrained" and should be explored earlier to avoid trapping.
# We also compute a "degree-based priority" for DFS.

def _compute_priority():
    """
    Compute a priority score for each vertex. Lower priority = explore earlier.
    We use a combination of:
    - Low degree vertices first (they're more likely to be endpoints)
    - Vertices in the largest connected component first
    """
    # Find largest connected component
    visited = [False] * N
    components = []
    for i in range(N):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in ADJ[u]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            components.append(comp)
    
    largest_comp = max(components, key=len)
    in_largest = set(largest_comp)
    
    priority = {}
    for u in range(N):
        deg = len(ADJ[u])
        # Lower degree = higher priority (explore first)
        # Vertices not in largest component get very low priority
        if u not in in_largest:
            priority[u] = 10000 + deg
        else:
            priority[u] = deg
    return priority, in_largest

PRIORITY, LARGEST_COMP = _compute_priority()
LARGEST_COMP_SET = set(LARGEST_COMP)

# ---------- DFS with backtracking and pruning ----------
def _dfs_longest_path(start, adj, n, priority, comp_set, time_limit=55.0):
    """
    Depth-first search for the longest simple path starting from 'start'.
    Uses pruning based on remaining reachable vertices and matching bound.
    """
    import time
    start_time = time.time()
    
    best_path = [start]
    best_len = 1
    
    def dfs(path, visited):
        nonlocal best_path, best_len
        if time.time() - start_time > time_limit:
            return
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
        
        u = path[-1]
        
        # Pruning: if we've already found a path of length PATH_UPPER_BOUND, stop
        if best_len >= PATH_UPPER_BOUND:
            return
        
        # Get unvisited neighbors, sorted by priority (lower priority first)
        unvisited_neighbors = [v for v in adj[u] if not visited[v]]
        if not unvisited_neighbors:
            return
        
        # Sort by priority: explore lower-priority (more constrained) vertices first
        unvisited_neighbors.sort(key=lambda v: priority[v])
        
        for v in unvisited_neighbors:
            # Additional pruning: if adding v would not beat best, and we can't
            # possibly reach more vertices, skip.
            # Simple bound: remaining vertices = n - len(path)
            if len(path) + (n - len(path)) <= best_len:
                break
            
            path.append(v)
            visited[v] = True
            dfs(path, visited)
            path.pop()
            visited[v] = False
            
            if time.time() - start_time > time_limit:
                return
    
    visited = {start}
    dfs([start], visited)
    return best_path

# ---------- Main construct function ----------
def construct():
    """
    Find the longest simple path in the pinned graph.
    Returns an ordered list of distinct vertices forming a simple path.
    """
    import time
    overall_start = time.time()
    time_limit = 55.0  # Leave 5 seconds margin for safety
    
    # If the entire graph is one component, the longest path could be up to N
    # Try DFS from multiple starting points, prioritized by vertex priority.
    
    # Start from vertices in the largest component, sorted by priority
    start_candidates = [v for v in LARGEST_COMP if v in LARGEST_COMP_SET]
    start_candidates.sort(key=lambda v: PRIORITY[v])
    
    best_path = []
    best_len = 0
    
    for start in start_candidates:
        if time.time() - overall_start > time_limit:
            break
        
        path = _dfs_longest_path(start, ADJ, N, PRIORITY, LARGEST_COMP_SET, 
                                  time_limit=time.time() - overall_start)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
        
        # If we've hit the upper bound, stop
        if best_len >= PATH_UPPER_BOUND:
            break
    
    # Fallback: if no path found (shouldn't happen), return a single vertex
    if not best_path:
        best_path = [start_candidates[0]] if start_candidates else [0]
    
    return best_path
