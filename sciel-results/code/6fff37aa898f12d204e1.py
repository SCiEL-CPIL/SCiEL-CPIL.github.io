import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    def is_c4_free(adj):
        """Check C4-free: no two distinct vertices share 2+ common neighbors."""
        n = adj.shape[0]
        if n <= 1:
            return True
        A = adj.astype(np.int32)
        A2 = A @ A
        # Off-diagonal entries >= 2 means a C4
        np.fill_diagonal(A2, 0)
        return not np.any(A2 >= 2)

    def paley_graph(q):
        """
        Construct the Paley graph on q vertices.
        Vertices are 0..q-1.
        Edge (u, v) iff u != v and (u - v) % q is a quadratic residue.
        """
        adj = np.zeros((q, q), dtype=np.int32)
        # Compute quadratic residues
        residues = set()
        for i in range(1, q):
            residues.add((i * i) % q)
        
        for u in range(q):
            for v in range(u + 1, q):
                diff = (u - v) % q
                if diff in residues:
                    adj[u, v] = 1
                    adj[v, u] = 1
        return adj

    def get_edges(adj):
        n = adj.shape[0]
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    edges.append([i, j])
        return edges

    # Try n=47 first using Paley graph P_47
    # P_47 is C4-free and 23-regular.
    # Min degree 23 >= 47 - 39 = 8.
    n_target = 47
    q = 47
    if time.time() - start_time < deadline:
        adj = paley_graph(q)
        if is_c4_free(adj):
            min_deg = np.min(adj.sum(axis=1))
            if min_deg >= n_target - 39:
                return {'n': n_target, 'edges': get_edges(adj)}
    
    # Fallback: n=46 using P_47 minus one vertex
    n_target = 46
    if time.time() - start_time < deadline:
        adj_full = paley_graph(47)
        # Remove vertex 0
        adj = adj_full[1:, 1:].copy()
        if is_c4_free(adj):
            min_deg = np.min(adj.sum(axis=1))
            if min_deg >= n_target - 39:
                # Renumber vertices 0..45
                return {'n': n_target, 'edges': get_edges(adj)}

    # Fallback: n=45 using P_47 minus two vertices
    n_target = 45
    if time.time() - start_time < deadline:
        adj_full = paley_graph(47)
        # Remove vertices 0 and 1
        mask = np.ones(47, dtype=bool)
        mask[0] = False
        mask[1] = False
        adj = adj_full[np.ix_(mask, mask)].copy()
        if is_c4_free(adj):
            min_deg = np.min(adj.sum(axis=1))
            if min_deg >= n_target - 39:
                return {'n': n_target, 'edges': get_edges(adj)}
        
    # Fallback: Greedy construction for smaller n
    def greedy_c4_free(n, target_deg):
        adj = np.zeros((n, n), dtype=np.int32)
        degrees = np.zeros(n, dtype=int)
        common = np.zeros((n, n), dtype=np.int32)
        
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        import random
        random.seed(42)
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if np.min(degrees) >= target_deg:
                break
            if degrees[u] >= target_deg and degrees[v] >= target_deg:
                continue
            if common[u, v] <= 1:
                adj[u, v] = 1
                adj[v, u] = 1
                degrees[u] += 1
                degrees[v] += 1
                for w in range(n):
                    if w != u and w != v:
                        if adj[u, w]:
                            common[w, v] += 1
                            common[v, w] += 1
                        if adj[v, w]:
                            common[u, w] += 1
                            common[w, u] += 1
                            
        if np.min(degrees) >= target_deg and is_c4_free(adj.astype(bool)):
            return {'n': n, 'edges': get_edges(adj)}
        return None

    for n in [44, 43, 42]:
        if time.time() - start_time > deadline - 5:
            break
        result = greedy_c4_free(n, n - 39)
        if result:
            return result

    return {'n': 39, 'edges': []}
