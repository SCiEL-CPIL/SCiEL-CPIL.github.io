import random
import time
import sys

def _build_adjacency(edges, n):
    """Build adjacency matrix from edge list."""
    adj = [[0]*n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = 1
        adj[v][u] = 1
    return adj

def _compute_degrees(adj, n):
    """Compute degree array from adjacency matrix."""
    return [sum(adj[i]) for i in range(n)]

def _find_triangles(adj, n):
    """Find all triangles in the graph. Returns list of (u, v, w) tuples."""
    triangles = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                for k in range(j+1, n):
                    if adj[i][k] and adj[j][k]:
                        triangles.append((i, j, k))
    return triangles

def _score_graph(edges, n):
    """Compute the maximum degree sum over all triangles."""
    adj = _build_adjacency(edges, n)
    deg = _compute_degrees(adj, n)
    triangles = _find_triangles(adj, n)
    if not triangles:
        return 0
    max_sum = 0
    for (u, v, w) in triangles:
        s = deg[u] + deg[v] + deg[w]
        if s > max_sum:
            max_sum = s
    return max_sum

def _baseline_edges():
    """K_{8,8} plus one edge inside part A."""
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
    edges.append([0, 1])
    return edges

def _local_search(edges, n, time_budget):
    """
    Local search via edge swaps to minimize max triangle degree sum.
    Returns (best_edges, best_score).
    """
    start = time.time()
    deadline = start + time_budget
    
    adj = _build_adjacency(edges, n)
    deg = _compute_degrees(adj, n)
    triangles = _find_triangles(adj, n)
    current_score = max(deg[u] + deg[v] + deg[w] for (u, v, w) in triangles) if triangles else 0
    
    best_edges = [list(e) for e in edges]
    best_score = current_score
    
    # Precompute edge set for fast lookup
    edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
    
    # Generate all possible edge swaps (remove one edge, add another)
    # We'll do this in a structured way: for each existing edge, try swapping with a non-existing edge
    
    random.seed(42)
    
    while time.time() < deadline - 2:
        # Pick a random existing edge to remove
        idx = random.randint(0, len(edges) - 1)
        u, v = edges[idx]
        
        # Find a non-existing edge to add
        # Try a few random candidates
        improved = False
        for attempt in range(50):
            if time.time() > deadline - 2:
                break
            ru, rv = random.randint(0, n-1), random.randint(0, n-1)
            if ru == rv:
                continue
            if ru > rv:
                ru, rv = rv, ru
            if (ru, rv) in edge_set:
                continue
            # Also make sure we're not removing the same edge
            if ru == u and rv == v:
                continue
            if (ru, rv) == (min(u,v), max(u,v)):
                continue
            
            # Apply swap
            # Remove (u, v)
            adj[u][v] = 0
            adj[v][u] = 0
            deg[u] -= 1
            deg[v] -= 1
            edge_set.discard((min(u,v), max(u,v)))
            
            # Add (ru, rv)
            adj[ru][rv] = 1
            adj[rv][ru] = 1
            deg[ru] += 1
            deg[rv] += 1
            edge_set.add((ru, rv))
            
            # Recompute score
            triangles = _find_triangles(adj, n)
            if triangles:
                new_score = max(deg[i] + deg[j] + deg[k] for (i, j, k) in triangles)
            else:
                new_score = 0
            
            if new_score < current_score:
                # Accept
                edges[idx] = [ru, rv]
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_edges = [list(e) for e in edges]
                improved = True
                break
            else:
                # Revert
                adj[ru][rv] = 0
                adj[rv][ru] = 0
                deg[ru] -= 1
                deg[rv] -= 1
                edge_set.discard((ru, rv))
                
                adj[u][v] = 1
                adj[v][u] = 1
                deg[u] += 1
                deg[v] += 1
                edge_set.add((min(u,v), max(u,v)))
        
        if not improved:
            # Try a few more random restarts of local search
            continue
    
    return best_edges, best_score

def construct():
    start_time = time.time()
    # Use 140s of the 150s budget, leaving 10s margin
    deadline = start_time + 140
    
    n = 16
    target_edges = 65
    
    # Start with baseline
    edges = _baseline_edges()
    current_score = _score_graph(edges, n)
    
    best_edges = [list(e) for e in edges]
    best_score = current_score
    
    # Run local search with multiple random seeds
    for seed in range(100):
        if time.time() > deadline - 10:
            break
        
        random.seed(seed)
        
        # Start from current best or slightly perturbed
        if seed == 0:
            trial_edges = [list(e) for e in best_edges]
        else:
            # Perturb the best edges slightly
            trial_edges = [list(e) for e in best_edges]
            # Remove 1-3 random edges and add new ones
            for _ in range(random.randint(1, 3)):
                idx = random.randint(0, len(trial_edges) - 1)
                old_u, old_v = trial_edges[idx]
                # Find a new edge to replace it
                for _ in range(20):
                    nu, nv = random.randint(0, n-1), random.randint(0, n-1)
                    if nu == nv:
                        continue
                    if nu > nv:
                        nu, nv = nv, nu
                    # Check if already exists
                    exists = False
                    for eu, ev in trial_edges:
                        if (min(eu,ev), max(eu,ev)) == (nu, nv):
                            exists = True
                            break
                    if not exists:
                        trial_edges[idx] = [nu, nv]
                        break
        
        # Run local search on this trial
        result_edges, result_score = _local_search(trial_edges, n, time_budget=min(15, deadline - time.time() - 5))
        
        if result_score < best_score:
            best_score = result_score
            best_edges = result_edges
    
    # Final check
    if time.time() < deadline - 2:
        final_score = _score_graph(best_edges, n)
        if final_score < best_score:
            best_score = final_score
    
    return {'n': n, 'edges': best_edges}
