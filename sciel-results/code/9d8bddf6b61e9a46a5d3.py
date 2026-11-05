"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def construct():
    deadline = time.time() + 140.0
    
    def random_graph(n, p, seed):
        import random
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges
    
    def solve_instance(n, p, seed):
        edges = random_graph(n, p, seed)
        if not edges:
            return []
        
        # Build adjacency matrix
        adj_mat = np.zeros((n, n), dtype=np.int8)
        for u, v in edges:
            adj_mat[u, v] = 1
            adj_mat[v, u] = 1
        
        # Build edge set for tracking
        edge_set = set()
        for u, v in edges:
            edge_set.add((min(u, v), max(u, v)))
        
        remaining_edges = set(edge_set)
        cliques = []
        
        # Spectral clustering to identify dense regions
        # Compute Laplacian
        degree = adj_mat.sum(axis=1)
        L = np.diag(degree) - adj_mat.astype(np.float64)
        
        # Get eigenvalues and eigenvectors
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
        except:
            # Fallback to simple greedy if spectral fails
            return _simple_greedy_solve(n, edges, remaining_edges, deadline)
        
        # Use the smallest non-zero eigenvalue's eigenvector for clustering
        # For dense graphs, the Fiedler vector (2nd smallest eigenvalue) is useful
        # But for finding cliques, we might want to look at the principal components
        # Let's use the first few eigenvectors to get a low-dimensional embedding
        
        # Take the first min(n, 4) eigenvectors
        k = min(n, 4)
        embedding = eigenvectors[:, :k]
        
        # Cluster vertices based on their embedding
        # Simple approach: group vertices that are close in embedding space
        # and also densely connected in the original graph
        
        # For now, let's use a simpler spectral-inspired heuristic:
        # Rank vertices by their centrality (eigenvector centrality)
        # and use that to guide clique extraction
        
        # Compute eigenvector centrality (dominant eigenvector of adjacency)
        try:
            from scipy.sparse.linalg import eigsh
            # Use power iteration or scipy
            # For small n, just use numpy
            w, v = np.linalg.eig(adj_mat.astype(np.float64))
            # Take the eigenvector corresponding to the largest eigenvalue
            max_idx = np.argmax(np.abs(w))
            centrality = np.abs(v[:, max_idx])
        except:
            centrality = degree.astype(np.float64)
        
        # Normalize centrality
        centrality = centrality / (centrality.max() + 1e-10)
        
        # Greedy clique extraction with improved heuristic
        # Maintain remaining adjacency
        rem_adj = adj_mat.copy()
        
        while remaining_edges:
            if time.time() > deadline - 10:
                # Fallback: cover remaining edges individually
                for e in remaining_edges:
                    cliques.append(list(e))
                break
            
            # Find an edge that is "hardest" to cover - prioritize edges
            # where both endpoints have few remaining connections
            # This helps pack edges efficiently
            best_edge = None
            best_score = -1
            
            # Sample a subset of edges for efficiency
            edge_list = list(remaining_edges)
            if len(edge_list) > 100:
                import random
                sample_rng = random.Random(42)
                edge_list = sample_rng.sample(edge_list, 100)
            
            for u, v in edge_list:
                # Score: prefer edges where u and v have few remaining neighbors
                # (they're "isolated" and need to be covered soon)
                deg_u = rem_adj[u].sum()
                deg_v = rem_adj[v].sum()
                # Also consider centrality - high centrality vertices can form bigger cliques
                score = (deg_u + deg_v) * 0.1 + centrality[u] + centrality[v]
                if score > best_score:
                    best_score = score
                    best_edge = (u, v)
            
            if best_edge is None:
                best_edge = next(iter(remaining_edges))
            
            u, v = best_edge
            
            # Find maximum clique containing edge (u, v) in remaining graph
            # Start with {u, v} and greedily add vertices
            clique = [u, v]
            
            # Get common neighbors of u and v in remaining graph
            common = set(np.where(rem_adj[u] == 1)[0].tolist()) & \
                     set(np.where(rem_adj[v] == 1)[0].tolist())
            common.discard(u)
            common.discard(v)
            
            # Sort common neighbors by centrality (descending)
            common_sorted = sorted(common, key=lambda x: centrality[x], reverse=True)
            
            for w in common_sorted:
                # Check if w is connected to all current clique members
                if all(rem_adj[w, x] == 1 for x in clique):
                    clique.append(w)
            
            # Try to extend further: look for vertices connected to all clique members
            # that weren't common neighbors initially but might still work
            all_vertices = set(range(n))
            candidates = all_vertices - set(clique)
            # Only consider vertices connected to at least one clique member
            candidates = [w for w in candidates if any(rem_adj[w, x] == 1 for x in clique)]
            candidates.sort(key=lambda x: centrality[x], reverse=True)
            
            for w in candidates:
                if all(rem_adj[w, x] == 1 for x in clique):
                    clique.append(w)
            
            if len(clique) < 2:
                # Shouldn't happen, but fallback
                clique = [best_edge[0], best_edge[1]]
            
            cliques.append(clique)
            
            # Remove edges in this clique from remaining graph
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    x, y = clique[i], clique[j]
                    e_key = (min(x, y), max(x, y))
                    if e_key in remaining_edges:
                        remaining_edges.remove(e_key)
                        rem_adj[x, y] = 0
                        rem_adj[y, x] = 0
        
        return cliques
    
    def _simple_greedy_solve(n, edges, remaining_edges, deadline):
        """Fallback simple greedy solver with improved edge selection"""
        adj_mat = np.zeros((n, n), dtype=np.int8)
        for u, v in edges:
            adj_mat[u, v] = 1
            adj_mat[v, u] = 1
        
        degree = adj_mat.sum(axis=1)
        cliques = []
        
        while remaining_edges:
            if time.time() > deadline - 10:
                for e in remaining_edges:
                    cliques.append(list(e))
                break
            
            # Pick edge with lowest combined degree (hardest to cover)
            edge_list = list(remaining_edges)
            if len(edge_list) > 100:
                import random
                sample_rng = random.Random(42)
                edge_list = sample_rng.sample(edge_list, 100)
            
            best_edge = min(edge_list, key=lambda e: degree[e[0]] + degree[e[1]])
            u, v = best_edge
            
            # Find clique containing this edge
            clique = [u, v]
            common = set(np.where(adj_mat[u] == 1)[0].tolist()) & \
                     set(np.where(adj_mat[v] == 1)[0].tolist())
            common.discard(u)
            common.discard(v)
            common_sorted = sorted(common, key=lambda x: degree[x], reverse=True)
            
            for w in common_sorted:
                if all(adj_mat[w, x] == 1 for x in clique):
                    clique.append(w)
            
            if len(clique) < 2:
                clique = [u, v]
            
            cliques.append(clique)
            
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    x, y = clique[i], clique[j]
                    e_key = (min(x, y), max(x, y))
                    if e_key in remaining_edges:
                        remaining_edges.remove(e_key)
                        adj_mat[x, y] = 0
                        adj_mat[y, x] = 0
        
        return cliques
    
    result1 = solve_instance(32, 0.6, 601)
    result2 = solve_instance(64, 0.55, 602)
    
    return [result1, result2]
# EVOLVE-BLOCK-END
