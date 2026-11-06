"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a graph on 16 vertices with exactly 65 edges that minimizes
    the maximum degree sum over triangles.
    
    Strategy: Use bitmask-based adjacency for fast triangle detection and scoring.
    Start with K_{8,8} + one edge, then use a targeted local search that
    prioritizes reducing the involvement of high-degree vertices in worst triangles.
    """
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    
    n = 16
    target_edges = 65
    
    # Bitmask representation: adj[i] is an integer with bit j set iff edge (i,j) exists
    def init_adj(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges
    
    def count_edges(adj):
        total = 0
        for u in range(n):
            total += bin(adj[u]).count('1')
        return total // 2
    
    def degrees(adj):
        return [bin(adj[u]).count('1') for u in range(n)]
    
    def compute_score(adj):
        """Compute the maximum degree sum over all triangles using bitmasks."""
        deg = degrees(adj)
        max_sum = 0
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    # Common neighbors of u and v (excluding u and v themselves)
                    common = adj[u] & adj[v]
                    # Only consider w > v to avoid double counting
                    for w in range(v + 1, n):
                        if (common >> w) & 1:
                            s = deg[u] + deg[v] + deg[w]
                            if s > max_sum:
                                max_sum = s
        return max_sum
    
    def find_worst_triangles(adj, deg):
        """Find all triangles with the maximum degree sum."""
        max_sum = 0
        triangles = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    for w in range(v + 1, n):
                        if (common >> w) & 1:
                            s = deg[u] + deg[v] + deg[w]
                            if s > max_sum:
                                max_sum = s
                                triangles = [(u, v, w)]
                            elif s == max_sum:
                                triangles.append((u, v, w))
        return max_sum, triangles
    
    # Initial construction: K_{8,8} + one edge within part A
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
    edges.append([0, 1])
    
    adj = init_adj(edges)
    current_score = compute_score(adj)
    best_score = current_score
    best_edges = edges[:]
    
    # Local search with targeted moves
    temperature = 3.0
    cooling_rate = 0.998
    no_improve = 0
    max_no_improve = 5000
    
    iteration = 0
    while time.time() < deadline and no_improve < max_no_improve:
        iteration += 1
        
        # Get current degrees and worst triangles
        deg = degrees(adj)
        _, worst_tris = find_worst_triangles(adj, deg)
        
        if not worst_tris:
            break
        
        # Generate candidate moves:
        # 1. Targeted: remove an edge incident to a high-degree vertex in a worst triangle,
        #    and add an edge between two low-degree vertices
        # 2. Random: swap two random edges
        
        move_type = random.random()
        if move_type < 0.4 and worst_tris:
            # Targeted move: focus on worst triangles
            u, v, w = random.choice(worst_tris)
            # Pick the highest-degree vertex in this triangle
            tri_verts = [u, v, w]
            tri_verts.sort(key=lambda x: deg[x], reverse=True)
            high_v = tri_verts[0]
            
            # Try to remove an edge from high_v to one of the other two
            candidates = []
            for other in tri_verts[1:]:
                if (adj[high_v] >> other) & 1:
                    candidates.append((high_v, other))
            
            if candidates:
                a, b = random.choice(candidates)
                # Find a non-edge to add: prefer edges between low-degree vertices
                # Collect all non-edges
                non_edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if not ((adj[i] >> j) & 1):
                            non_edges.append((i, j))
                
                if non_edges:
                    # Weight by inverse degree sum to prefer low-degree pairs
                    weights = [1.0 / (1 + deg[i] + deg[j]) for i, j in non_edges]
                    total_w = sum(weights)
                    r = random.random() * total_w
                    cumulative = 0
                    for idx, (i, j) in enumerate(non_edges):
                        cumulative += weights[idx]
                        if r <= cumulative:
                            c, d = i, j
                            break
                    else:
                        c, d = non_edges[-1]
                    
                    # Check that adding (c,d) doesn't create the same edge
                    if (adj[c] >> d) & 1:
                        continue
                    # Check that (a,b) is actually an edge
                    if not ((adj[a] >> b) & 1):
                        continue
                    
                    # Apply move
                    adj[a] &= ~(1 << b)
                    adj[b] &= ~(1 << a)
                    adj[c] |= (1 << d)
                    adj[d] |= (1 << c)
                    
                    new_score = compute_score(adj)
                    
                    if new_score < current_score:
                        current_score = new_score
                        if new_score < best_score:
                            best_score = new_score
                            best_edges = adj_to_edges(adj)
                        no_improve = 0
                    elif new_score == current_score:
                        # Accept equal
                        no_improve += 1
                    else:
                        delta = new_score - current_score
                        prob = np.exp(-delta / temperature)
                        if random.random() < prob:
                            no_improve += 1
                        else:
                            # Revert
                            adj[a] |= (1 << b)
                            adj[b] |= (1 << a)
                            adj[c] &= ~(1 << d)
                            adj[d] &= ~(1 << c)
                            no_improve += 1
        else:
            # Random edge swap
            all_edges = adj_to_edges(adj)
            if len(all_edges) < 2:
                break
            
            # Pick two random distinct edges
            idx1 = random.randint(0, len(all_edges) - 1)
            idx2 = random.randint(0, len(all_edges) - 1)
            if idx1 == idx2:
                continue
            a, b = all_edges[idx1]
            c, d = all_edges[idx2]
            
            # Check validity
            if (adj[c] >> d) & 1:
                continue
            if not ((adj[a] >> b) & 1):
                continue
            
            # Apply swap
            adj[a] &= ~(1 << b)
            adj[b] &= ~(1 << a)
            adj[c] |= (1 << d)
            adj[d] |= (1 << c)
            
            new_score = compute_score(adj)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_edges = adj_to_edges(adj)
                no_improve = 0
            elif new_score == current_score:
                no_improve += 1
            else:
                delta = new_score - current_score
                prob = np.exp(-delta / temperature)
                if random.random() < prob:
                    no_improve += 1
                else:
                    # Revert
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
                    adj[c] &= ~(1 << d)
                    adj[d] &= ~(1 << c)
                    no_improve += 1
        
        temperature *= cooling_rate
    
    # Final polish: try more targeted moves if time remains
    if time.time() < deadline - 5:
        polish_deadline = time.time() + min(10, deadline - time.time() - 2)
        temp_polish = 1.0
        for _ in range(5000):
            if time.time() > polish_deadline:
                break
            deg = degrees(adj)
            _, worst_tris = find_worst_triangles(adj, deg)
            if not worst_tris:
                break
            
            u, v, w = random.choice(worst_tris)
            tri_verts = [u, v, w]
            tri_verts.sort(key=lambda x: deg[x], reverse=True)
            high_v = tri_verts[0]
            
            candidates = []
            for other in tri_verts[1:]:
                if (adj[high_v] >> other) & 1:
                    candidates.append((high_v, other))
            
            if not candidates:
                continue
            
            a, b = random.choice(candidates)
            
            non_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not ((adj[i] >> j) & 1):
                        non_edges.append((i, j))
            
            if not non_edges:
                continue
            
            # Prefer edges that don't involve high_v
            safe_non_edges = [(i, j) for i, j in non_edges if i != high_v and j != high_v]
            if safe_non_edges:
                c, d = random.choice(safe_non_edges)
            else:
                c, d = random.choice(non_edges)
            
            if (adj[c] >> d) & 1:
                continue
            if not ((adj[a] >> b) & 1):
                continue
            
            adj[a] &= ~(1 << b)
            adj[b] &= ~(1 << a)
            adj[c] |= (1 << d)
            adj[d] |= (1 << c)
            
            new_score = compute_score(adj)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_edges = adj_to_edges(adj)
            else:
                delta = new_score - current_score
                prob = np.exp(-delta / temp_polish)
                if random.random() < prob:
                    pass
                else:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
                    adj[c] &= ~(1 << d)
                    adj[d] &= ~(1 << c)
            
            temp_polish *= 0.99
    
    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
