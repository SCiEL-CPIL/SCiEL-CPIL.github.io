import random
import time
import numpy as np

def construct():
    deadline = time.time() + 115
    n = 16
    target_edges = 65
    
    def compute_score(adj, deg):
        """Compute the maximum degree sum over all triangles."""
        max_sum = 0
        worst_tri = None
        for u in range(n):
            for v in range(u + 1, n):
                if not adj[u, v]:
                    continue
                for w in range(v + 1, n):
                    if adj[u, w] and adj[v, w]:
                        s = deg[u] + deg[v] + deg[w]
                        if s > max_sum:
                            max_sum = s
                            worst_tri = (u, v, w)
        return max_sum, worst_tri
    
    def get_triangles(adj):
        """Get all triangles in the graph."""
        triangles = []
        for u in range(n):
            for v in range(u + 1, n):
                if not adj[u, v]:
                    continue
                for w in range(v + 1, n):
                    if adj[u, w] and adj[v, w]:
                        triangles.append((u, v, w))
        return triangles
    
    def local_search(adj, deg, score, worst_tri, max_iters, deadline, use_targeted=True):
        """Local search with optional targeted moves on worst triangles."""
        current_adj = adj.copy()
        current_deg = deg.copy()
        current_score = score
        current_worst = worst_tri
        
        T = 2.0
        cooling = 0.998
        
        all_possible_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            # Determine move: targeted or random
            if use_targeted and current_worst is not None:
                u, v, w = current_worst
                # Try to remove an edge from the worst triangle or add an edge that helps
                # Option 1: Remove one edge from the worst triangle
                candidate_removals = [(u, v), (u, w), (v, w)]
                random.shuffle(candidate_removals)
                
                moved = False
                for e1 in candidate_removals:
                    if not current_adj[e1[0], e1[1]]:
                        continue
                    
                    # Find a non-edge to add
                    for _ in range(20):
                        e2 = random.choice(all_possible_edges)
                        if current_adj[e2[0], e2[1]]:
                            continue
                        
                        u1, v1 = e1
                        u2, v2 = e2
                        
                        new_adj = current_adj.copy()
                        new_deg = current_deg.copy()
                        
                        new_adj[u1, v1] = False
                        new_adj[v1, u1] = False
                        new_deg[u1] -= 1
                        new_deg[v1] -= 1
                        
                        new_adj[u2, v2] = True
                        new_adj[v2, u2] = True
                        new_deg[u2] += 1
                        new_deg[v2] += 1
                        
                        new_score, new_worst = compute_score(new_adj, new_deg)
                        delta = new_score - current_score
                        
                        if delta < 0 or random.random() < np.exp(-delta / T):
                            current_adj = new_adj
                            current_deg = new_deg
                            current_score = new_score
                            current_worst = new_worst
                            moved = True
                            break
                    
                    if moved:
                        break
                
                if not moved:
                    # Fallback to random move
                    for _ in range(10):
                        e1 = random.choice(all_possible_edges)
                        if not current_adj[e1[0], e1[1]]:
                            continue
                        e2 = random.choice(all_possible_edges)
                        if current_adj[e2[0], e2[1]]:
                            continue
                        break
                    else:
                        continue
                    
                    u1, v1 = e1
                    u2, v2 = e2
                    
                    new_adj = current_adj.copy()
                    new_deg = current_deg.copy()
                    
                    new_adj[u1, v1] = False
                    new_adj[v1, u1] = False
                    new_deg[u1] -= 1
                    new_deg[v1] -= 1
                    
                    new_adj[u2, v2] = True
                    new_adj[v2, u2] = True
                    new_deg[u2] += 1
                    new_deg[v2] += 1
                    
                    new_score, new_worst = compute_score(new_adj, new_deg)
                    delta = new_score - current_score
                    
                    if delta < 0 or random.random() < np.exp(-delta / T):
                        current_adj = new_adj
                        current_deg = new_deg
                        current_score = new_score
                        current_worst = new_worst
            else:
                # Random move
                for _ in range(10):
                    e1 = random.choice(all_possible_edges)
                    if not current_adj[e1[0], e1[1]]:
                        continue
                    e2 = random.choice(all_possible_edges)
                    if current_adj[e2[0], e2[1]]:
                        continue
                    break
                else:
                    continue
                
                u1, v1 = e1
                u2, v2 = e2
                
                new_adj = current_adj.copy()
                new_deg = current_deg.copy()
                
                new_adj[u1, v1] = False
                new_adj[v1, u1] = False
                new_deg[u1] -= 1
                new_deg[v1] -= 1
                
                new_adj[u2, v2] = True
                new_adj[v2, u2] = True
                new_deg[u2] += 1
                new_deg[v2] += 1
                
                new_score, new_worst = compute_score(new_adj, new_deg)
                delta = new_score - current_score
                
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current_adj = new_adj
                    current_deg = new_deg
                    current_score = new_score
                    current_worst = new_worst
            
            T *= cooling
        
        return current_adj, current_deg, current_score, current_worst
    
    def polish(adj, deg, score, worst_tri, deadline):
        """Final high-precision polish: exhaustive local moves on worst triangles."""
        current_adj = adj.copy()
        current_deg = deg.copy()
        current_score = score
        current_worst = worst_tri
        
        all_possible_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        # Do several passes of targeted improvement
        for pass_num in range(5):
            if time.time() > deadline - 2:
                break
            
            improved = False
            # Try all possible edge swaps that involve the worst triangle vertices
            if current_worst is not None:
                u, v, w = current_worst
                triangle_vertices = [u, v, w]
                
                # For each edge in the worst triangle, try removing it and adding various edges
                for e1 in [(u, v), (u, w), (v, w)]:
                    if not current_adj[e1[0], e1[1]]:
                        continue
                    
                    # Try adding each possible non-edge
                    for e2 in all_possible_edges:
                        if current_adj[e2[0], e2[1]]:
                            continue
                        
                        u1, v1 = e1
                        u2, v2 = e2
                        
                        new_adj = current_adj.copy()
                        new_deg = current_deg.copy()
                        
                        new_adj[u1, v1] = False
                        new_adj[v1, u1] = False
                        new_deg[u1] -= 1
                        new_deg[v1] -= 1
                        
                        new_adj[u2, v2] = True
                        new_adj[v2, u2] = True
                        new_deg[u2] += 1
                        new_deg[v2] += 1
                        
                        new_score, new_worst = compute_score(new_adj, new_deg)
                        
                        if new_score < current_score:
                            current_adj = new_adj
                            current_deg = new_deg
                            current_score = new_score
                            current_worst = new_worst
                            improved = True
                            break
                    
                    if improved:
                        break
                
                # Also try adding edges incident to worst triangle vertices without removing
                if not improved:
                    for vertex in triangle_vertices:
                        for other in range(n):
                            if other == vertex or current_adj[vertex, other]:
                                continue
                            
                            # Find an edge to remove (not incident to create issues)
                            best_remove = None
                            for e1 in all_possible_edges:
                                if not current_adj[e1[0], e1[1]]:
                                    continue
                                # Avoid removing edges that would disconnect or create bad structures
                                # Just try it
                                u1, v1 = e1
                                u2, v2 = vertex, other
                                
                                new_adj = current_adj.copy()
                                new_deg = current_deg.copy()
                                
                                new_adj[u1, v1] = False
                                new_adj[v1, u1] = False
                                new_deg[u1] -= 1
                                new_deg[v1] -= 1
                                
                                new_adj[u2, v2] = True
                                new_adj[v2, u2] = True
                                new_deg[u2] += 1
                                new_deg[v2] += 1
                                
                                new_score, new_worst = compute_score(new_adj, new_deg)
                                
                                if new_score < current_score:
                                    current_adj = new_adj
                                    current_deg = new_deg
                                    current_score = new_score
                                    current_worst = new_worst
                                    improved = True
                                    break
                        
                        if improved:
                            break
            
            if not improved:
                break
        
        return current_adj, current_deg, current_score, current_worst
    
    # Try multiple random initializations and local searches
    best_score = float('inf')
    best_adj = None
    best_deg = None
    best_worst_tri = None
    
    for seed in range(80):
        if time.time() > deadline - 10:
            break
        random.seed(seed)
        
        # Start from K_{8,8} + random edge, or random near-bipartite
        adj = np.zeros((n, n), dtype=bool)
        
        # Create a random bipartite-like structure
        # Partition vertices into two sets
        perm = list(range(n))
        random.shuffle(perm)
        split = random.randint(7, 9)  # More balanced partitions
        A = set(perm[:split])
        B = set(perm[split:])
        
        # Add all edges between A and B
        for u in A:
            for v in B:
                adj[u, v] = True
                adj[v, u] = True
        
        edge_count = len(A) * len(B)
        
        # Add remaining edges within A or B to reach target
        remaining = target_edges - edge_count
        if remaining > 0:
            # Add random edges within parts
            internal_edges = []
            for part in [A, B]:
                part_list = list(part)
                for i in range(len(part_list)):
                    for j in range(i + 1, len(part_list)):
                        u, v = part_list[i], part_list[j]
                        if not adj[u, v]:
                            internal_edges.append((u, v))
            random.shuffle(internal_edges)
            for u, v in internal_edges[:remaining]:
                adj[u, v] = True
                adj[v, u] = True
        elif remaining < 0:
            # Remove some cross edges
            cross_edges = [(u, v) for u in A for v in B]
            random.shuffle(cross_edges)
            for u, v in cross_edges[:abs(remaining)]:
                adj[u, v] = False
                adj[v, u] = False
        
        deg = adj.sum(axis=1).astype(int)
        score, worst_tri = compute_score(adj, deg)
        
        if score < best_score:
            best_score = score
            best_adj = adj.copy()
            best_deg = deg.copy()
            best_worst_tri = worst_tri
        
        # Local search from this starting point
        current_adj, current_deg, current_score, current_worst = local_search(
            adj, deg, score, worst_tri, 3000, deadline, use_targeted=True
        )
        
        if current_score < best_score:
            best_score = current_score
            best_adj = current_adj.copy()
            best_deg = current_deg.copy()
            best_worst_tri = current_worst
    
    # Final polish pass
    if best_adj is not None:
        polished_adj, polished_deg, polished_score, polished_worst = polish(
            best_adj, best_deg, best_score, best_worst_tri, deadline
        )
        
        if polished_score < best_score:
            best_score = polished_score
            best_adj = polished_adj.copy()
            best_deg = polished_deg.copy()
            best_worst_tri = polished_worst
    
    # Convert best_adj to edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                edges.append([u, v])
    
    return {'n': n, 'edges': edges}
