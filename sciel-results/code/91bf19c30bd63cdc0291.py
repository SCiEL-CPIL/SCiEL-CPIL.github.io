"""Minimize four-cycles in graphs with 31 vertices and 91 edges."""
import time
import random
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin before 150s kill
    
    n = 31
    target_edges = 91
    
    # Precompute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pair_index = {}
    for idx, (u, v) in enumerate(all_pairs):
        pair_index[(u, v)] = idx
        pair_index[(v, u)] = idx
    
    total_pairs = len(all_pairs)
    
    # Build adjacency structure for fast C4 counting
    # We'll use a bitset approach for each vertex's neighbors
    
    def count_c4(adjacency):
        """Count 4-cycles using the formula: sum over pairs of vertices with common neighbors"""
        # For each pair of vertices, count common neighbors, then sum C(common, 2)
        # This counts each 4-cycle exactly once if we're careful
        # Actually, a 4-cycle a-b-c-d-a is counted by:
        # - common neighbors of (a,c) includes b and d -> contributes 1
        # - common neighbors of (b,d) includes a and c -> contributes 1
        # So each 4-cycle is counted twice, divide by 2
        
        c4_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                # Find common neighbors of i and j
                common = 0
                for k in range(n):
                    if k != i and k != j:
                        if adjacency[i][k] and adjacency[j][k]:
                            common += 1
                c4_count += common * (common - 1) // 2
        return c4_count // 2
    
    def build_adjacency(edges):
        """Build adjacency matrix from edge list"""
        adj = [[False] * n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        return adj
    
    # Initial solution: greedy construction
    def greedy_construct():
        """Greedy: add edges one by one, choosing the edge that minimizes incremental C4"""
        edges = set()
        adj = [[False] * n for _ in range(n)]
        deg = [0] * n
        
        # Sort candidate edges by some heuristic
        candidates = list(all_pairs)
        random.shuffle(candidates)
        
        for u, v in candidates:
            if len(edges) >= target_edges:
                break
            # Calculate how many new C4s this edge would create
            # A new edge (u,v) creates C4s with any pair (a,b) where a is neighbor of u, b is neighbor of v,
            # and a-b is an edge, and a!=b, a!=v, b!=u
            new_c4s = 0
            for a in range(n):
                if a == v or not adj[u][a]:
                    continue
                for b in range(n):
                    if b == u or b == a or not adj[v][b]:
                        continue
                    if adj[a][b]:
                        new_c4s += 1
            
            if new_c4s == 0 or len(edges) < target_edges:
                # Add this edge
                edges.add((min(u,v), max(u,v)))
                adj[u][v] = True
                adj[v][u] = True
                deg[u] += 1
                deg[v] += 1
        
        # If we didn't get enough edges, add more randomly
        if len(edges) < target_edges:
            for u, v in all_pairs:
                if len(edges) >= target_edges:
                    break
                key = (min(u,v), max(u,v))
                if key not in edges:
                    edges.add(key)
                    adj[u][v] = True
                    adj[v][u] = True
                    deg[u] += 1
                    deg[v] += 1
        
        return list(edges), adj
    
    # Better greedy: prioritize low-degree vertices
    def greedy_construct_v2():
        """Greedy with degree-based heuristic"""
        edges = set()
        adj = [[False] * n for _ in range(n)]
        deg = [0] * n
        
        # Start with a sparse random graph, then add edges
        # First, create a random spanning tree to ensure connectivity
        random.shuffle(list(range(n)))
        for i in range(1, n):
            u = random.choice(range(i))
            v = i
            key = (min(u,v), max(u,v))
            edges.add(key)
            adj[u][v] = True
            adj[v][u] = True
            deg[u] += 1
            deg[v] += 1
        
        # Now add remaining edges greedily
        remaining = target_edges - len(edges)
        candidates = [(u, v) for u, v in all_pairs if (min(u,v), max(u,v)) not in edges]
        
        # Score each candidate by incremental C4 count
        def incremental_c4(u, v):
            count = 0
            for a in range(n):
                if a == v or not adj[u][a]:
                    continue
                for b in range(n):
                    if b == u or b == a or not adj[v][b]:
                        continue
                    if adj[a][b]:
                        count += 1
            return count
        
        # Add edges in order of lowest incremental C4
        while remaining > 0:
            best_edge = None
            best_score = float('inf')
            # Sample a subset for speed
            sample = random.sample(candidates, min(len(candidates), 200))
            for u, v in sample:
                score = incremental_c4(u, v)
                if score < best_score:
                    best_score = score
                    best_edge = (u, v)
            
            if best_edge is None:
                break
            
            u, v = best_edge
            key = (min(u,v), max(u,v))
            edges.add(key)
            adj[u][v] = True
            adj[v][u] = True
            deg[u] += 1
            deg[v] += 1
            candidates.remove((u, v))
            remaining -= 1
        
        return list(edges), adj
    
    # Simulated annealing to improve
    def simulated_annealing(edges, adj, max_iters=50000):
        """SA to reduce C4 count"""
        edges_set = set(edges)
        current_c4 = count_c4(adj)
        best_edges = list(edges_set)
        best_c4 = current_c4
        
        T = 10.0
        cooling = 0.9999
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Choose a random edge to remove and a random non-edge to add
            edge_to_remove = random.choice(list(edges_set))
            
            # Find a non-edge
            non_edges = [(u, v) for u, v in all_pairs if (min(u,v), max(u,v)) not in edges_set]
            if not non_edges:
                break
            u, v = random.choice(non_edges)
            new_edge_key = (min(u,v), max(u,v))
            
            # Calculate change in C4 count
            # Removing edge (a,b) and adding edge (c,d)
            a, b = edge_to_remove
            c, d = new_edge_key
            
            # C4s that use the removed edge
            removed_c4s = 0
            for x in range(n):
                if x == b or not adj[a][x]:
                    continue
                for y in range(n):
                    if y == a or y == x or not adj[b][y]:
                        continue
                    if adj[x][y]:
                        removed_c4s += 1
            
            # C4s that use the new edge
            added_c4s = 0
            for x in range(n):
                if x == d or not adj[c][x]:
                    continue
                for y in range(n):
                    if y == c or y == x or not adj[d][y]:
                        continue
                    if adj[x][y]:
                        added_c4s += 1
            
            delta = added_c4s - removed_c4s
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                # Accept move
                edges_set.remove(edge_to_remove)
                edges_set.add(new_edge_key)
                adj[a][b] = False
                adj[b][a] = False
                adj[c][d] = True
                adj[d][c] = True
                current_c4 += delta
                
                if current_c4 < best_c4:
                    best_c4 = current_c4
                    best_edges = list(edges_set)
            
            T *= cooling
            if T < 0.01:
                T = 0.01
        
        return best_edges, best_c4
    
    # Start with greedy
    edges, adj = greedy_construct_v2()
    current_c4 = count_c4(adj)
    best_edges = list(edges)
    best_c4 = current_c4
    
    # Run SA with multiple restarts
    restarts = 0
    while time.time() < deadline - 10:
        restarts += 1
        improved_edges, improved_c4 = simulated_annealing(best_edges, build_adjacency(best_edges), max_iters=20000)
        if improved_c4 < best_c4:
            best_c4 = improved_c4
            best_edges = improved_edges
    
    # Final local search
    def local_search(edges):
        """Try all single edge swaps to improve"""
        edges_set = set(edges)
        adj = build_adjacency(edges_set)
        current_c4 = count_c4(adj)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for edge_to_remove in list(edges_set):
                if time.time() > deadline - 5:
                    break
                a, b = edge_to_remove
                # Try adding each non-edge
                for u, v in all_pairs:
                    new_edge_key = (min(u,v), max(u,v))
                    if new_edge_key in edges_set:
                        continue
                    
                    # Calculate delta
                    removed_c4s = 0
                    for x in range(n):
                        if x == b or not adj[a][x]:
                            continue
                        for y in range(n):
                            if y == a or y == x or not adj[b][y]:
                                continue
                            if adj[x][y]:
                                removed_c4s += 1
                    
                    added_c4s = 0
                    for x in range(n):
                        if x == v or not adj[u][x]:
                            continue
                        for y in range(n):
                            if y == u or y == x or not adj[v][y]:
                                continue
                            if adj[x][y]:
                                added_c4s += 1
                    
                    delta = added_c4s - removed_c4s
                    if delta < 0:
                        # Make the move
                        edges_set.remove(edge_to_remove)
                        edges_set.add(new_edge_key)
                        adj[a][b] = False
                        adj[b][a] = False
                        adj[u][v] = True
                        adj[v][u] = True
                        current_c4 += delta
                        improved = True
                        break
        
        return list(edges_set), current_c4
    
    best_edges, best_c4 = local_search(best_edges)
    
    # Convert to required format
    result = [[u, v] for u, v in best_edges]
    result.sort()
    
    # Verify we have exactly 91 edges
    assert len(result) == 91, f"Expected 91 edges, got {len(result)}"
    
    return result
