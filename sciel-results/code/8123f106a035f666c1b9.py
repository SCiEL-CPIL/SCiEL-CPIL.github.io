import random
import time
from itertools import combinations
import numpy as np

def construct():
    n = 20
    deadline = time.time() + 110  # 110s safety margin
    
    best_edges = 0
    best_adj = [[0]*n for _ in range(n)]
    
    # Precompute all pairs for efficiency
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def is_valid_add(u, v, adj, neighbors):
        """Check if adding edge (u,v) creates a K3,3 using precomputed neighbors."""
        Nv = neighbors[v]
        Nu = neighbors[u]
        
        for i in range(len(Nv)):
            a1 = Nv[i]
            for j in range(i+1, len(Nv)):
                a2 = Nv[j]
                count = 0
                for b in Nu:
                    if adj[a1][b] and adj[a2][b]:
                        count += 1
                        if count >= 2:
                            return False
        return True
    
    def can_add(u, v, adj, neighbors):
        """Fast check if edge (u,v) can be added without creating K3,3."""
        if adj[u][v]:
            return False
        return is_valid_add(u, v, adj, neighbors)
    
    def add_edge(u, v, adj, neighbors):
        """Add edge (u,v) and update neighbor lists."""
        adj[u][v] = 1
        adj[v][u] = 1
        neighbors[u].append(v)
        neighbors[v].append(u)
    
    def remove_edge(u, v, adj, neighbors):
        """Remove edge (u,v) and update neighbor lists."""
        adj[u][v] = 0
        adj[v][u] = 0
        neighbors[u].remove(v)
        neighbors[v].remove(u)
    
    def count_edges(adj):
        """Count edges in adjacency matrix."""
        return sum(sum(row) for row in adj) // 2
    
    def build_neighbors(adj):
        """Build neighbor lists from adjacency matrix."""
        neighbors = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    neighbors[i].append(j)
                    neighbors[j].append(i)
        return neighbors
    
    def spectral_guided_greedy():
        """Greedy construction with spectral guidance to bias edge selection."""
        adj = [[0]*n for _ in range(n)]
        neighbors = [[] for _ in range(n)]
        edges = []
        
        # Use spectral properties to guide initial edge selection
        # Start with a random sparse graph and use its spectral structure
        # to determine which edges to add first
        
        # Initialize with a random subset of edges (about 40% density)
        initial_density = 0.4
        all_edges = all_pairs[:]
        random.shuffle(all_edges)
        initial_count = int(len(all_edges) * initial_density)
        
        for u, v in all_edges[:initial_count]:
            if can_add(u, v, adj, neighbors):
                add_edge(u, v, adj, neighbors)
                edges.append((u, v))
        
        # Compute spectral properties of current graph
        adj_matrix = np.array(adj, dtype=float)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(adj_matrix)
            # Sort by eigenvalue magnitude (descending)
            idx = np.argsort(-np.abs(eigenvalues))
            eigenvectors = eigenvectors[:, idx]
            eigenvalues = eigenvalues[idx]
            
            # Use the top eigenvector to determine vertex importance
            top_eigvec = np.abs(eigenvectors[:, 0])
            vertex_scores = top_eigvec / (top_eigvec.sum() + 1e-10)
        except:
            vertex_scores = np.ones(n) / n
        
        # Now greedily add remaining edges, prioritizing those connecting
        # high-scoring vertices
        remaining_edges = all_edges[initial_count:]
        random.shuffle(remaining_edges)
        
        # Score each remaining edge by the product of vertex scores
        scored_edges = []
        for u, v in remaining_edges:
            score = vertex_scores[u] * vertex_scores[v]
            scored_edges.append((score, u, v))
        
        # Sort by score descending, with some randomization
        scored_edges.sort(key=lambda x: -x[0])
        
        # Add edges in order of score, but with some randomization
        # to avoid getting stuck in local optima
        for score, u, v in scored_edges:
            if random.random() < 0.3:  # 30% chance to skip for diversity
                continue
            if can_add(u, v, adj, neighbors):
                add_edge(u, v, adj, neighbors)
                edges.append((u, v))
        
        return adj, neighbors, len(edges)
    
    def greedy_random():
        """Randomized greedy construction."""
        adj = [[0]*n for _ in range(n)]
        neighbors = [[] for _ in range(n)]
        edges = []
        
        all_edges = all_pairs[:]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if can_add(u, v, adj, neighbors):
                add_edge(u, v, adj, neighbors)
                edges.append((u, v))
        
        return adj, neighbors, len(edges)
    
    def local_search(adj, neighbors, time_limit):
        """Local search with edge swaps to improve the solution."""
        start_time = time.time()
        improved = True
        current_edges = count_edges(adj)
        
        while improved and time.time() - start_time < time_limit:
            improved = False
            
            degree = [len(neighbors[i]) for i in range(n)]
            high_degree = [i for i in range(n) if degree[i] > n//2]
            
            if not high_degree:
                high_degree = list(range(n))
            
            random.shuffle(high_degree)
            
            for u in high_degree:
                if time.time() - start_time > time_limit:
                    break
                
                neighbors_u = neighbors[u][:]
                random.shuffle(neighbors_u)
                
                for v in neighbors_u:
                    if time.time() - start_time > time_limit:
                        break
                    
                    remove_edge(u, v, adj, neighbors)
                    
                    added = []
                    for w in range(n):
                        if w != u and w != v and not adj[u][w]:
                            if can_add(u, w, adj, neighbors):
                                add_edge(u, w, adj, neighbors)
                                added.append(w)
                                if len(added) >= 2:
                                    break
                    
                    if len(added) >= 2:
                        improved = True
                        break
                    else:
                        for w in added:
                            remove_edge(u, w, adj, neighbors)
                        add_edge(u, v, adj, neighbors)
            
            if not improved and time.time() - start_time < time_limit:
                current_edges_list = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i][j]]
                if len(current_edges_list) >= 2:
                    for _ in range(min(100, len(current_edges_list))):
                        if time.time() - start_time > time_limit:
                            break
                        
                        e1, e2 = random.sample(current_edges_list, 2)
                        u1, v1 = e1
                        u2, v2 = e2
                        
                        remove_edge(u1, v1, adj, neighbors)
                        remove_edge(u2, v2, adj, neighbors)
                        
                        added = []
                        candidates = [(i, j) for i, j in all_pairs if not adj[i][j]]
                        random.shuffle(candidates)
                        
                        for u, v in candidates:
                            if len(added) >= 3:
                                break
                            if can_add(u, v, adj, neighbors):
                                add_edge(u, v, adj, neighbors)
                                added.append((u, v))
                        
                        if len(added) >= 3:
                            improved = True
                            current_edges_list = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i][j]]
                            break
                        else:
                            for u, v in added:
                                remove_edge(u, v, adj, neighbors)
                            add_edge(u1, v1, adj, neighbors)
                            add_edge(u2, v2, adj, neighbors)
        
        return adj, neighbors, count_edges(adj)
    
    # Run multiple restarts with both greedy strategies
    restart_count = 0
    use_spectral = True
    
    while time.time() < deadline:
        if use_spectral:
            adj, neighbors, edge_count = spectral_guided_greedy()
        else:
            adj, neighbors, edge_count = greedy_random()
        
        use_spectral = not use_spectral
        
        if edge_count > best_edges:
            best_edges = edge_count
            best_adj = [row[:] for row in adj]
        
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            local_time = min(3.0, remaining_time * 0.3)
            adj, neighbors, edge_count = local_search(adj, neighbors, local_time)
            
            if edge_count > best_edges:
                best_edges = edge_count
                best_adj = [row[:] for row in adj]
        
        restart_count += 1
        if best_edges >= 85:
            break
    
    # Final polish
    if time.time() < deadline - 2:
        neighbors = build_neighbors(best_adj)
        for u, v in all_pairs:
            if not best_adj[u][v]:
                if can_add(u, v, best_adj, neighbors):
                    add_edge(u, v, best_adj, neighbors)
    
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                result.append([i, j])
    return result
