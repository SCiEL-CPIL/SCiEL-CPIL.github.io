"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 15
DEADLINE = 85.0  # safety margin before 90s kill

def _check_validity(adj):
    """Check if graph with adjacency matrix adj is valid.
    No simple 3-edge path s-u-v-t (s!=t) should have deg(s) == deg(t).
    """
    n = N
    deg = [0] * n
    for i in range(n):
        for j in range(n):
            if adj[i][j]:
                deg[i] += 1
    
    # For every edge (u,v), check all s in N(u)\{v} and t in N(v)\{u} with s!=t
    for u in range(n):
        for v in range(n):
            if u == v or not adj[u][v]:
                continue
            for s in range(n):
                if s == u or s == v or not adj[s][u]:
                    continue
                for t in range(n):
                    if t == u or t == v or t == s or not adj[v][t]:
                        continue
                    if deg[s] == deg[t]:
                        return False
    return True

def _edge_count(adj):
    count = 0
    for i in range(N):
        for j in range(i+1, N):
            if adj[i][j]:
                count += 1
    return count

def _bipartite_init(a):
    """Create complete bipartite graph K_{a, N-a}."""
    adj = [[False]*N for _ in range(N)]
    for i in range(a):
        for j in range(a, N):
            adj[i][j] = True
            adj[j][i] = True
    return adj

def _greedy_fill(adj, deadline):
    """Try to add edges greedily while maintaining validity."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            for j in range(i+1, N):
                if adj[i][j]:
                    continue
                adj[i][j] = True
                adj[j][i] = True
                if _check_validity(adj):
                    improved = True
                else:
                    adj[i][j] = False
                    adj[j][i] = False
    return adj

def _local_search(adj, deadline):
    """Local search: try removing one edge and re-greedy-filling."""
    best_adj = [row[:] for row in adj]
    best_count = _edge_count(adj)
    
    while time.time() < deadline:
        # Try removing each edge and re-filling
        found = False
        for i in range(N):
            for j in range(i+1, N):
                if not adj[i][j]:
                    continue
                # Remove edge
                adj[i][j] = False
                adj[j][i] = False
                # Re-greedy fill
                adj2 = [row[:] for row in adj]
                adj2 = _greedy_fill(adj2, deadline)
                c2 = _edge_count(adj2)
                if c2 > best_count:
                    best_adj = [row[:] for row in adj2]
                    best_count = c2
                    found = True
                # Restore
                adj[i][j] = True
                adj[j][i] = True
        if not found:
            break
    
    return best_adj, best_count

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_adj = None
    best_count = 0
    
    # Start from various bipartite bases
    for a in range(1, N):
        if time.time() > deadline:
            break
        adj = _bipartite_init(a)
        adj = _greedy_fill(adj, deadline)
        count = _edge_count(adj)
        
        if count > best_count:
            best_count = count
            best_adj = [row[:] for row in adj]
    
    # Local search from best
    if best_adj is not None and time.time() < deadline:
        best_adj, best_count = _local_search(best_adj, deadline)
    
    # Try multiple random restarts if time permits
    while time.time() < deadline - 5:
        a = random.randint(1, N-1)
        adj = _bipartite_init(a)
        adj = _greedy_fill(adj, deadline)
        count = _edge_count(adj)
        if count > best_count:
            best_count = count
            best_adj = [row[:] for row in adj]
        
        # Local search
        if time.time() < deadline - 10:
            adj2, count2 = _local_search(adj, deadline)
            if count2 > best_count:
                best_count = count2
                best_adj = adj2
    
    # Convert to edge list
    edges = []
    for i in range(N):
        for j in range(i+1, N):
            if best_adj[i][j]:
                edges.append([int(i), int(j)])
    
    return edges
# EVOLVE-BLOCK-END
