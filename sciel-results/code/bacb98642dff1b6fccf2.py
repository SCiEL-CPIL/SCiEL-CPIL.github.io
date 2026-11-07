"""Improved program: greedy + local search for dense odd-order graphs avoiding equal-degree path endpoints."""
import time
import random
import itertools

P = {'n': 15}

def _degrees(adj, n):
    """Compute degree of each vertex from adjacency matrix."""
    deg = [0] * n
    for i in range(n):
        for j in range(n):
            if adj[i][j]:
                deg[i] += 1
    return deg

def _is_valid(adj, n):
    """Check if no simple 3-edge path has endpoints of equal degree."""
    deg = _degrees(adj, n)
    # For every pair of vertices s, t with s != t, check if they're at distance 3
    # s-u-v-t: adj[s][u] and adj[u][v] and adj[v][t], s != t, u != t, s != v
    # Actually, a simple path s-u-v-t means s,u,v,t are all distinct
    # So we need: adj[s][u] and adj[u][v] and adj[v][t], with s,u,v,t all distinct
    # And deg[s] == deg[t] must NOT happen
    
    # Build list of all simple 3-edge paths
    for s in range(n):
        for u in range(n):
            if s == u or not adj[s][u]:
                continue
            for v in range(n):
                if v == s or v == u or not adj[u][v]:
                    continue
                for t in range(n):
                    if t == s or t == u or t == v:
                        continue
                    if adj[v][t]:
                        if deg[s] == deg[t]:
                            return False
    return True

def _edge_count(adj, n):
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                count += 1
    return count

def _try_add_edge(adj, n, u, v):
    """Try to add edge (u,v). Return True if valid after adding."""
    if adj[u][v]:
        return True
    adj[u][v] = 1
    adj[v][u] = 1
    if _is_valid(adj, n):
        return True
    else:
        adj[u][v] = 0
        adj[v][u] = 0
        return False

def _remove_edge(adj, n, u, v):
    adj[u][v] = 0
    adj[v][u] = 0

def _construct_greedy(adj, n, order):
    """Greedy: try to add edges in given order."""
    for u, v in order:
        _try_add_edge(adj, n, u, v)
    return adj

def _local_search(adj, n, max_iter=5000, deadline=None):
    """Local search: try swapping edges to increase density."""
    best_adj = [row[:] for row in adj]
    best_count = _edge_count(best_adj, n)
    
    for iteration in range(max_iter):
        if deadline and time.time() > deadline:
            break
        improved = False
        # Try removing one edge and adding another
        # First, collect all edges
        edges = []
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    edges.append((i, j))
                else:
                    non_edges.append((i, j))
        
        if not non_edges:
            break
        
        # Try random swaps: remove a random edge, add a random non-edge
        # But this might decrease count. Better: try to add a non-edge, if invalid, try removing some edge to make it valid
        # Simpler: try all pairs of (remove edge, add non-edge) that keep same count or increase
        # For efficiency, try random samples
        
        for _ in range(50):
            if deadline and time.time() > deadline:
                break
            if not edges or not non_edges:
                break
            ru, rv = random.choice(edges)
            au, av = random.choice(non_edges)
            # Remove (ru,rv), add (au,av)
            adj[ru][rv] = 0
            adj[rv][ru] = 0
            adj[au][av] = 1
            adj[av][au] = 1
            if _is_valid(adj, n):
                # Check if this leads to improvement via subsequent additions
                new_adj = [row[:] for row in adj]
                # Try to greedily add more edges
                for eu, ev in non_edges:
                    if (eu, ev) == (au, av):
                        continue
                    _try_add_edge(new_adj, n, eu, ev)
                new_count = _edge_count(new_adj, n)
                if new_count > best_count:
                    best_adj = [row[:] for row in new_adj]
                    best_count = new_count
                    adj = [row[:] for row in new_adj]
                    improved = True
                    break
                else:
                    # Revert
                    adj[ru][rv] = 1
                    adj[rv][ru] = 1
                    adj[au][av] = 0
                    adj[av][au] = 0
            else:
                # Revert
                adj[ru][rv] = 1
                adj[rv][ru] = 1
                adj[au][av] = 0
                adj[av][au] = 0
        
        if improved:
            # Recompute edges and non_edges
            edges = []
            non_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j]:
                        edges.append((i, j))
                    else:
                        non_edges.append((i, j))
    
    return best_adj, best_count

def construct():
    n = P['n']
    deadline = time.time() + 80  # Safety margin
    
    # Start with empty graph
    best_adj = [[0]*n for _ in range(n)]
    best_count = 0
    
    # Generate all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Multiple restarts with different random orderings
    random.seed(42)
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Random order
        order = all_edges[:]
        random.shuffle(order)
        
        adj = [[0]*n for _ in range(n)]
        for u, v in order:
            _try_add_edge(adj, n, u, v)
        
        count = _edge_count(adj, n)
        
        if count > best_count:
            best_adj = [row[:] for row in adj]
            best_count = count
        
        # Local search
        if time.time() > deadline - 10:
            break
        ls_adj, ls_count = _local_search(adj, n, max_iter=2000, deadline=deadline - 5)
        if ls_count > best_count:
            best_adj = [row[:] for row in ls_adj]
            best_count = ls_count
    
    # Build edge list from best_adj
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                edges.append([int(i), int(j)])
    
    return edges
