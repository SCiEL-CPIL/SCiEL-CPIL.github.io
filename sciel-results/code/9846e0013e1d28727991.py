"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import deque

P = {'n': 16}
START_TIME = time.time()
TIME_LIMIT = 85  # Safety margin before 90s kill

def bfs_distances(adj, start):
    """Compute shortest distances from start node to all others."""
    n = len(adj)
    dist = [-1] * n
    dist[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        du = dist[u]
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = du + 1
                q.append(v)
    return dist

def compute_diameter(adj):
    """Compute the diameter of the graph. Returns -1 if disconnected."""
    n = len(adj)
    max_dist = 0
    for i in range(n):
        dists = bfs_distances(adj, i)
        if -1 in dists:
            return -1
        max_dist = max(max_dist, max(dists))
    return max_dist

def is_diameter_two(adj):
    """Check if graph is connected and has diameter exactly 2."""
    return compute_diameter(adj) == 2

def edge_is_critical(adj, u, v):
    """
    Check if edge (u, v) is critical.
    An edge is critical if deleting it either disconnects the graph 
    or increases the diameter above 2.
    """
    # Create a copy of the adjacency list without edge (u, v)
    # To be efficient, we can modify in place and revert, but for n=16, copying is fine.
    # However, we need to be careful with the representation.
    
    # Let's use a set of edges for quick lookup or just list of lists.
    # Current adj is list of lists.
    
    # Remove edge
    try:
        adj[u].remove(v)
    except ValueError:
        pass
    try:
        adj[v].remove(u)
    except ValueError:
        pass
    
    # Check new diameter
    new_diam = compute_diameter(adj)
    
    # Re-add edge
    adj[u].append(v)
    adj[v].append(u)
    
    # Critical if:
    # 1. Disconnected (new_diam == -1)
    # 2. Diameter > 2 (new_diam > 2)
    return new_diam > 2 or new_diam == -1

def construct():
    n = P['n']
    best_edges = []
    best_score = 0
    
    # Initialize with a random sparse graph or a known structure
    # A star is diameter 2. All edges are critical?
    # In a star K_{1,15}, deleting any leaf edge disconnects the leaf. So all 15 are critical.
    # Can we do better?
    # A cycle C_16 has diameter 8. Not good.
    # A complete bipartite graph K_{8,8} has diameter 2.
    # Is every edge critical in K_{8,8}?
    # If we remove an edge (u,v) from K_{8,8}, the graph is still connected.
    # Diameter of K_{8,8} is 2.
    # If we remove an edge, does diameter become 3?
    # In K_{8,8}, any two nodes in the same partition have distance 2 (via any node in other partition).
    # If we remove one edge (u1, v1), u1 and v1 are still distance 2 via v2 or u2.
    # So diameter remains 2. Thus, edges in K_{8,8} are NOT critical.
    
    # We need a graph where EVERY edge is a bridge for connectivity or a unique shortest path for some pair.
    # Actually, "critical for diameter 2" means:
    # If we remove edge e, either:
    # a) Graph disconnects
    # b) There exists a pair (u,v) such that dist(u,v) > 2.
    
    # Let's try a random search.
    
    def generate_random_graph():
        # Start with a spanning tree to ensure connectivity, then add random edges
        edges = []
        # Random spanning tree
        perm = list(range(n))
        random.shuffle(perm)
        for i in range(1, n):
            u = perm[i]
            v = random.choice(perm[:i])
            edges.append((u, v))
        
        # Add some random edges
        # Probability of adding edge
        p = 0.3
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    if (i,j) not in edges and (j,i) not in edges:
                        edges.append((i,j))
        return edges

    def edges_to_adj(edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def evaluate_graph(edges):
        """
        Returns (is_valid, score)
        score is number of critical edges.
        We only care about graphs with diameter exactly 2.
        """
        adj = edges_to_adj(edges)
        if not is_diameter_two(adj):
            return False, 0
        
        critical_count = 0
        for u, v in edges:
            if edge_is_critical(adj, u, v):
                critical_count += 1
        
        return True, critical_count

    # Local Search
    current_edges = generate_random_graph()
    current_valid, current_score = evaluate_graph(current_edges)
    
    if current_valid and current_score > best_score:
        best_edges = current_edges
        best_score = current_score
    
    max_iter = 5000
    iter_count = 0
    
    while time.time() - START_TIME < TIME_LIMIT and iter_count < max_iter:
        iter_count += 1
        
        # Perturbation: Add or remove a random edge
        new_edges = list(current_edges)
        if random.random() < 0.5 and len(new_edges) < n*(n-1)//2:
            # Add edge
            while True:
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v:
                    continue
                if (u,v) in new_edges or (v,u) in new_edges:
                    continue
                new_edges.append((u,v))
                break
        elif len(new_edges) > n: # Don't go below spanning tree size
            # Remove edge
            idx = random.randint(0, len(new_edges)-1)
            new_edges.pop(idx)
        
        new_valid, new_score = evaluate_graph(new_edges)
        
        if new_valid:
            if new_score > current_score:
                current_edges = new_edges
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_edges = list(new_edges)
            else:
                # Accept worse solutions with some probability to escape local optima
                # Simple hill climbing for now, might get stuck.
                # Let's add a simple restart or random jump if stuck for a while.
                # For now, just reject.
                pass
        else:
            # If invalid, revert (don't update current)
            pass
            
        # Occasionally, restart from a new random graph if stuck
        if iter_count % 500 == 0:
             if random.random() < 0.1:
                 current_edges = generate_random_graph()
                 current_valid, current_score = evaluate_graph(current_edges)
                 if current_valid and current_score > best_score:
                     best_edges = current_edges
                     best_score = current_score

    # Final check and format
    if best_score == 0:
        # Fallback to star if search failed
        return [[0, v] for v in range(1, n)]
    
    # Convert to list of lists of ints
    result = []
    for u, v in best_edges:
        result.append([int(u), int(v)])
        
    return result

# EVOLVE-BLOCK-END
