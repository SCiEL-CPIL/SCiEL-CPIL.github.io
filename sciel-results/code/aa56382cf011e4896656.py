"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _has_c6(adj, n):
    """Check if adjacency matrix contains a 6-cycle using BFS from each edge."""
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            # Find all 4-paths from u to v (length 4, i.e., u-a-b-c-v)
            # That would complete a 6-cycle with edge u-v
            # Actually we need to check if adding edge (u,v) creates a C6
            # But here adj already has the edge, so we check if there's a path of length 5 from u to v
            # that doesn't use edge (u,v) directly
    
    # More efficient: for each pair of vertices, check if they have a common neighbor
    # through a path of length 5
    # C6 exists iff there are vertices u,v such that there's a path u-w1-w2-w3-w4-v 
    # where all edges exist and no vertex repeats, and (u,v) is an edge or not
    
    # Standard approach: for each edge (u,v), check if there's a path of length 5 from u to v
    # that avoids the direct edge. This is expensive.
    
    # Better: use the fact that C6-free means for any two vertices at distance 2,
    # they cannot share a common neighbor at distance 4... this gets complex.
    
    # Let's use a different approach: iterate over all 6-tuples? No, too slow.
    
    # Efficient check: A graph has a C6 iff there exist vertices a,b,c,d,e,f 
    # forming a cycle. We can check by looking at pairs of vertices and their 
    # common neighbors at various distances.
    
    # For a 40-vertex graph, we can do:
    # For each pair (u,v), compute the number of paths of length 2, 3, 4, 5 between them.
    # If there's a path of length 5 and an edge, we have a C6 (if the path is simple).
    
    # Simpler: for each edge (u,v), check if there's a simple path of length 5 from u to v.
    # Use BFS with depth limit.
    
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            # Check if there's a path of length 5 from u to v not using edge (u,v)
            # BFS from u, depth up to 5, avoiding direct edge to v until the end
            visited = [[False]*n for _ in range(6)]  # depth 0..5
            visited[0][u] = True
            frontier = [u]
            for depth in range(1, 6):
                new_frontier = []
                for node in frontier:
                    for nb in adj[node]:
                        if not visited[depth][nb]:
                            if depth == 5 and nb == v:
                                return True
                            if depth < 5:
                                visited[depth][nb] = True
                                new_frontier.append(nb)
                frontier = new_frontier
    return False

def _creates_c6(adj, n, u, v):
    """Check if adding edge (u,v) creates a C6."""
    # Adding edge (u,v) creates a C6 iff there's a path of length 5 from u to v
    # that is simple and doesn't use edge (u,v)
    # BFS from u to v with depth exactly 5, avoiding the direct edge
    
    # We need a simple path of length 5 from u to v
    # This is NP-hard in general, but for n=40 we can do limited BFS
    
    # Use DFS with pruning, depth limit 5
    def dfs(node, depth, path_set, target):
        if depth == 5:
            return node == target
        for nb in adj[node]:
            if nb not in path_set:
                path_set.add(nb)
                if dfs(nb, depth + 1, path_set, target):
                    path_set.remove(nb)
                    return True
                path_set.remove(nb)
        return False
    
    path_set = {u}
    return dfs(u, 0, path_set, v)

def _count_edges(edges):
    return len(edges)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s to leave safety margin
    
    n = 40
    
    # Start with 8 disjoint K5s
    edges = set()
    adj = [set() for _ in range(n)]
    
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u, v = 5 * k + a, 5 * k + b
                edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
    
    # Greedy: try to add edges between different K5 components
    # Order edges by some heuristic
    candidate_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in edges:
                candidate_edges.append((u, v))
    
    random.shuffle(candidate_edges)
    
    # Try adding edges greedily
    for u, v in candidate_edges:
        if time.time() > deadline:
            break
        if not _creates_c6(adj, n, u, v):
            edges.add((u, v))
            adj[u].add(v)
            adj[v].add(u)
    
    # Simulated annealing refinement
    current_edges = len(edges)
    best_edges = edges.copy()
    best_adj = [s.copy() for s in adj]
    best_count = current_edges
    
    temp = 5.0
    cooling = 0.999
    
    while time.time() < deadline and temp > 0.1:
        # Pick a random edge to remove
        if not edges:
            break
        u, v = random.choice(list(edges))
        
        # Remove it
        edges.discard((u, v))
        adj[u].discard(v)
        adj[v].discard(u)
        
        # Try to add a different random edge
        added = False
        for _ in range(100):  # Try up to 100 random edges
            ru = random.randint(0, n - 1)
            rv = random.randint(0, n - 1)
            if ru == rv:
                continue
            if (ru, rv) in edges or (rv, ru) in edges:
                continue
            if not _creates_c6(adj, n, ru, rv):
                edges.add((ru, rv))
                adj[ru].add(rv)
                adj[rv].add(ru)
                added = True
                break
        
        delta = len(edges) - current_edges
        if delta >= 0 or random.random() < np.exp(delta / temp):
            current_edges = len(edges)
            if current_edges > best_count:
                best_count = current_edges
                best_edges = edges.copy()
                best_adj = [s.copy() for s in adj]
        else:
            # Revert
            edges.add((u, v))
            adj[u].add(v)
            adj[v].add(u)
            if added:
                # Need to find and remove the added edge - this is problematic
                # Let's restructure: save state before modification
                pass
        
        temp *= cooling
    
    # Return best found
    return [list(e) for e in best_edges]

# EVOLVE-BLOCK-END
