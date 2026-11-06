"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _has_k33(adj):
    """Check if graph with adjacency matrix adj (n x n boolean) contains K3,3."""
    n = adj.shape[0]
    # For each triple of vertices, check if they share >= 3 common neighbors outside the triple
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                # Common neighbors of a, b, c
                common = adj[a] & adj[b] & adj[c]
                # Exclude a, b, c themselves (they're not neighbors of themselves in simple graph)
                common[a] = False
                common[b] = False
                common[c] = False
                if np.sum(common) >= 3:
                    return True
    return False

def _count_k33_violations(adj):
    """Count how many triples have >= 3 common neighbors (K3,3 witness count)."""
    n = adj.shape[0]
    count = 0
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                common = adj[a] & adj[b] & adj[c]
                common[a] = False
                common[b] = False
                common[c] = False
                if np.sum(common) >= 3:
                    count += 1
    return count

def _build_edges_from_adj(adj):
    """Convert adjacency matrix to edge list."""
    n = adj.shape[0]
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j]:
                edges.append([i, j])
    return edges

def _try_add_edge(adj, u, v):
    """Try to add edge (u,v). Returns True if graph remains K3,3-free."""
    if adj[u, v]:
        return True
    adj[u, v] = True
    adj[v, u] = True
    if _has_k33(adj):
        adj[u, v] = False
        adj[v, u] = False
        return False
    return True

def _greedy_fill(adj, deadline):
    """Greedily add edges in random order until no more can be added or deadline."""
    n = adj.shape[0]
    remaining = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i, j]]
    random.shuffle(remaining)
    for u, v in remaining:
        if time.time() > deadline:
            break
        _try_add_edge(adj, u, v)
    return adj

def _local_search(adj, deadline, max_iter=5000):
    """Local search: try to add edges, if stuck try removing one and re-adding."""
    n = adj.shape[0]
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        # Try to add any missing edge
        remaining = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i, j]]
        if not remaining:
            break
        random.shuffle(remaining)
        added = False
        for u, v in remaining:
            if time.time() > deadline:
                break
            if _try_add_edge(adj, u, v):
                added = True
                break
        if not added:
            # Stuck: try removing a random existing edge to unlock
            existing = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i, j]]
            if not existing:
                break
            random.shuffle(existing)
            for u, v in existing[:5]:  # try a few removals
                if time.time() > deadline:
                    break
                adj[u, v] = False
                adj[v, u] = False
                # Now try to add more edges
                added2 = False
                remaining2 = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i, j]]
                random.shuffle(remaining2)
                for u2, v2 in remaining2:
                    if time.time() > deadline:
                        break
                    if _try_add_edge(adj, u2, v2):
                        added2 = True
                        break
                if not added2:
                    # Restore the removed edge
                    adj[u, v] = True
                    adj[v, u] = True
    return adj

def construct():
    n = 20
    deadline = time.time() + 110  # safety margin
    
    best_edges = None
    best_count = 0
    
    # Start with seed: K2,18
    adj_seed = np.zeros((n, n), dtype=bool)
    for a in range(2):
        for b in range(2, n):
            adj_seed[a, b] = True
            adj_seed[b, a] = True
    seed_edges = _build_edges_from_adj(adj_seed)
    best_count = len(seed_edges)
    best_edges = seed_edges
    
    # Try random initial graphs and greedy fill
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Random initial graph with moderate density
        p = random.uniform(0.2, 0.5)
        adj = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj[i, j] = True
                    adj[j, i] = True
        
        # Check if K3,3-free; if not, remove edges until it is
        while _has_k33(adj):
            # Find a violating triple and remove an edge
            for a in range(n):
                for b in range(a+1, n):
                    for c in range(b+1, n):
                        common = adj[a] & adj[b] & adj[c]
                        common[a] = False
                        common[b] = False
                        common[c] = False
                        if np.sum(common) >= 3:
                            # Remove a random edge from the K3,3
                            common_vertices = [v for v in range(n) if common[v]]
                            # Pick random pair from {a,b,c} x common_vertices
                            side1 = random.choice([a, b, c])
                            side2 = random.choice(common_vertices)
                            adj[side1, side2] = False
                            adj[side2, side1] = False
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break
        
        # Greedy fill
        adj = _greedy_fill(adj, deadline)
        
        # Local search
        adj = _local_search(adj, deadline, max_iter=200)
        
        edges = _build_edges_from_adj(adj)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
    
    # Final greedy pass on best
    if best_edges is not None and time.time() < deadline:
        adj_best = np.zeros((n, n), dtype=bool)
        for u, v in best_edges:
            adj_best[u, v] = True
            adj_best[v, u] = True
        adj_best = _greedy_fill(adj_best, deadline)
        adj_best = _local_search(adj_best, deadline, max_iter=100)
        final_edges = _build_edges_from_adj(adj_best)
        if len(final_edges) > best_count:
            best_edges = final_edges
    
    return best_edges
# EVOLVE-BLOCK-END
