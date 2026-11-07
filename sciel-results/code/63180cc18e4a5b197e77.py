"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 15
DEADLINE = 80.0  # safety margin before 90s kill

def _degrees(adj):
    """Return degree list from adjacency matrix (list of lists of 0/1)."""
    return [sum(row) for row in adj]

def _is_valid(adj):
    """
    Check if the graph represented by adj (15x15 adjacency matrix) is valid.
    Valid means: no simple 3-edge path s-u-v-t (s != t) has deg(s) == deg(t).
    """
    deg = _degrees(adj)
    n = N
    # For every edge (u, v), and every neighbor s of u (s != v), and every neighbor t of v (t != u),
    # if s != t, then we have a path s-u-v-t. We need deg(s) != deg(t).
    # But note: the path is s-u-v-t, so s is neighbor of u, t is neighbor of v.
    # s != t is required. Also s != v and t != u are implied by simple path (no repeated vertices).
    # Actually, in a simple path s-u-v-t, we need s,u,v,t all distinct.
    # So s is in N(u), t is in N(v), s != v, t != u, s != t.
    
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v] == 0:
                continue
            # For edge (u, v), consider all s in N(u) \ {v} and t in N(v) \ {u}
            for s in range(n):
                if s == v or adj[u][s] == 0:
                    continue
                for t in range(n):
                    if t == u or s == t or adj[v][t] == 0:
                        continue
                    if deg[s] == deg[t]:
                        return False
    return True

def _try_add_edge(adj, u, v):
    """Try to add edge (u,v) to adj. Return True if valid after addition."""
    if u == v or adj[u][v]:
        return False
    # Quick check: adding edge (u,v) creates paths s-u-v-t and s-v-u-t
    deg = _degrees(adj)
    adj[u][v] = 1
    adj[v][u] = 1
    valid = _is_valid(adj)
    if not valid:
        adj[u][v] = 0
        adj[v][u] = 0
    return valid

def _construct_greedy_with_restarts():
    """
    Greedy edge addition with multiple random restarts and orderings.
    """
    best_edges = []
    best_count = 0
    start_time = time.time()
    
    for restart in range(50):
        if time.time() - start_time > DEADLINE:
            break
        
        # Random permutation of vertices for ordering
        order = list(range(N))
        random.shuffle(order)
        
        # Build adjacency matrix
        adj = [[0] * N for _ in range(N)]
        
        # Try to add all possible edges in random order
        edges_to_try = []
        for i in range(N):
            for j in range(i + 1, N):
                edges_to_try.append((i, j))
        random.shuffle(edges_to_try)
        
        for (u, v) in edges_to_try:
            if _try_add_edge(adj, u, v):
                pass  # edge added
        
        # Count edges
        count = sum(sum(row) for row in adj) // 2
        if count > best_count:
            best_count = count
            best_edges = [[u, v] for u in range(N) for v in range(u + 1, N) if adj[u][v]]
    
    return best_edges

def _construct_structural():
    """
    Try a structural approach: partition vertices into two groups and
    carefully add edges.
    """
    start_time = time.time()
    best_edges = []
    best_count = 0
    
    # Try different partitions
    for split in range(1, N):
        if time.time() - start_time > DEADLINE:
            break
        
        A = list(range(split))
        B = list(range(split, N))
        
        adj = [[0] * N for _ in range(N)]
        
        # Start with no edges, try to add all possible edges greedily
        edges_to_try = []
        for i in range(N):
            for j in range(i + 1, N):
                edges_to_try.append((i, j))
        random.shuffle(edges_to_try)
        
        for (u, v) in edges_to_try:
            if _try_add_edge(adj, u, v):
                pass
        
        count = sum(sum(row) for row in adj) // 2
        if count > best_count:
            best_count = count
            best_edges = [[u, v] for u in range(N) for v in range(u + 1, N) if adj[u][v]]
    
    return best_edges

def construct():
    """
    Return a list of edges [u, v] forming a valid graph on vertices 0..14.
    """
    start_time = time.time()
    random.seed(42)
    
    # Try greedy with restarts
    edges1 = _construct_greedy_with_restarts()
    
    # Try structural approach
    if time.time() - start_time < DEADLINE:
        edges2 = _construct_structural()
        if len(edges2) > len(edges1):
            edges1 = edges2
    
    # Final validation
    if edges1:
        adj = [[0] * N for _ in range(N)]
        for (u, v) in edges1:
            adj[u][v] = 1
            adj[v][u] = 1
        if not _is_valid(adj):
            # Fallback to star
            edges1 = [[0, v] for v in range(1, N)]
    
    # Ensure proper types
    result = []
    for e in edges1:
        result.append([int(e[0]), int(e[1])])
    
    return result

# EVOLVE-BLOCK-END
