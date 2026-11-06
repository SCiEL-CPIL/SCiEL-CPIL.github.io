import random
import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n = 20
    # Precompute all triples of vertices
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triples.append((i, j, k))
    
    # Total possible edges
    total_pairs = n * (n - 1) // 2
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def is_valid(adj):
        """
        adj: list of integers representing adjacency bitmasks.
        Returns True if graph is K3,3-free.
        """
        for i, j, k in triples:
            common = adj[i] & adj[j] & adj[k]
            if common.bit_count() >= 3:
                return False
        return True

    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def get_violating_triple(adj):
        """
        Returns a triple (i, j, k) such that common neighbors >= 3, or None.
        """
        for i, j, k in triples:
            common = adj[i] & adj[j] & adj[k]
            if common.bit_count() >= 3:
                return (i, j, k, common)
        return None

    def repair(adj):
        """
        Removes edges until the graph is valid.
        """
        max_iter = 200
        while max_iter > 0:
            max_iter -= 1
            v = get_violating_triple(adj)
            if v is None:
                break
            i, j, k, common = v
            
            # Get list of nodes in common
            candidates = []
            mask = common
            while mask:
                lsb = mask & -mask
                m = lsb.bit_length() - 1
                candidates.append(m)
                mask ^= lsb
            
            if not candidates:
                break
                
            m = random.choice(candidates)
            # Remove edge (i, m)
            if adj[i] & (1 << m):
                adj[i] &= ~(1 << m)
                adj[m] &= ~(1 << i)
            else:
                # Should not happen, but safety
                break
                
        return adj

    def try_add_edge(adj, u, v):
        """
        Try to add edge (u, v). Returns True if added and valid.
        """
        if u == v:
            return False
        if adj[u] & (1 << v):
            return True  # Already there
        
        # Add edge
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        if is_valid(adj):
            return True
        else:
            # Revert
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            return False

    def greedy_fill(adj):
        """
        Try to add all possible missing edges.
        """
        changed = True
        while changed:
            changed = False
            for u, v in all_pairs:
                if not (adj[u] & (1 << v)):
                    if try_add_edge(adj, u, v):
                        changed = True
        return adj

    def local_search(adj, time_limit):
        """
        Perform local search: try edge swaps.
        """
        current_edges = count_edges(adj)
        best_adj = [x for x in adj]
        best_edges = current_edges
        
        no_improve_count = 0
        max_iterations = 5000
        iter_count = 0
        
        while iter_count < max_iterations:
            if time.time() > time_limit:
                break
            iter_count += 1
            
            # Try to add a random non-edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            if adj[u] & (1 << v):
                continue
            
            # Try adding it
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if is_valid(adj):
                current_edges += 1
                no_improve_count = 0
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = [x for x in adj]
            else:
                # Revert
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                no_improve_count += 1
                
                # If stuck for a while, try a swap: remove a random edge, then try to add
                if no_improve_count > 20:
                    # Pick a random existing edge to remove
                    removed_edge = None
                    for _ in range(10):
                        ru = random.randint(0, n - 1)
                        if adj[ru] == 0:
                            continue
                        neighbors = []
                        mask = adj[ru]
                        while mask:
                            lsb = mask & -mask
                            rv = lsb.bit_length() - 1
                            neighbors.append(rv)
                            mask ^= lsb
                        if neighbors:
                            rv = random.choice(neighbors)
                            # Remove edge (ru, rv)
                            adj[ru] &= ~(1 << rv)
                            adj[rv] &= ~(1 << ru)
                            current_edges -= 1
                            removed_edge = (ru, rv)
                            break
                    
                    if removed_edge:
                        # Now try to add a random non-edge
                        added = False
                        for _ in range(10):
                            au = random.randint(0, n - 1)
                            av = random.randint(0, n - 1)
                            if au == av:
                                continue
                            if adj[au] & (1 << av):
                                continue
                            
                            adj[au] |= (1 << av)
                            adj[av] |= (1 << au)
                            if is_valid(adj):
                                current_edges += 1
                                added = True
                                if current_edges > best_edges:
                                    best_edges = current_edges
                                    best_adj = [x for x in adj]
                                break
                            else:
                                adj[au] &= ~(1 << av)
                                adj[av] &= ~(1 << au)
                        
                        if not added:
                            # Put the removed edge back if we didn't improve
                            ru, rv = removed_edge
                            adj[ru] |= (1 << rv)
                            adj[rv] |= (1 << ru)
                            current_edges += 1
                    
                    no_improve_count = 0
                    
        return best_adj, best_edges

    def spectral_init(seed):
        """
        Initialize graph using spectral heuristic.
        Construct a random symmetric matrix, find top eigenvector, threshold.
        """
        np.random.seed(seed)
        # Random symmetric matrix
        M = np.random.randn(n, n)
        M = (M + M.T) / 2
        
        # Compute eigenvectors
        eigvals, eigvecs = np.linalg.eigh(M)
        
        # Take the eigenvector corresponding to the largest eigenvalue
        idx = np.argmax(eigvals)
        v = eigvecs[:, idx]
        
        # Threshold at the mean to create a dense random-like starting point
        # We want to create a graph where edges are likely between vertices with similar "spectral position"
        # Or maybe just use the sign of the vector?
        # Let's try: if v[i] * v[j] > 0, add edge? That would be a bipartite-like structure if signs are mixed.
        # Or if |v[i] - v[j]| < threshold?
        
        # Let's try a simple heuristic:
        # Compute pairwise products v[i] * v[j]
        # If product > 0, they are in the same "cluster" (sign-wise).
        # K3,3-free graphs often have structure.
        
        # Alternative: Use the eigenvector to define a probability of edge.
        # p_ij = 1 / (1 + exp(-alpha * (v[i] * v[j])))
        
        # Let's just use the sign for now, or a threshold on the product.
        # If we use sign, we get a complete bipartite graph between positive and negative vertices.
        # That is K3,3-free if one side has size < 3 or the other < 3.
        # But we want dense.
        
        # Let's try: edge if v[i] * v[j] > 0.
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if v[i] * v[j] > 0:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        return adj

    def random_init(seed, p=0.5):
        """
        Random graph initialization.
        """
        random.seed(seed)
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def bipartite_init(seed, p=0.8):
        """
        Bipartite initialization.
        """
        random.seed(seed)
        adj = [0] * n
        for i in range(10):
            for j in range(10, 20):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    best_overall_edges = 0
    best_overall_adj = None
    
    # Try multiple random starts
    # We can use different initialization strategies
    for seed in range(200):
        if time.time() > deadline - 5.0:
            break
            
        # Mix initialization strategies
        if seed % 3 == 0:
            adj = spectral_init(seed)
        elif seed % 3 == 1:
            adj = random_init(seed)
        else:
            adj = bipartite_init(seed)
        
        # Repair
        adj = repair(adj)
        
        # Greedy fill
        adj = greedy_fill(adj)
        
        # Local search
        # Allocate remaining time roughly evenly, but cap per iteration
        # Let's do a time slice for local search per seed.
        # We have ~140s total. 200 seeds -> 0.7s per seed roughly.
        # But some seeds might be faster.
        # Let's just run local search until time runs out for this seed, or a small limit.
        
        adj, edges = local_search(adj, deadline - 2.0)
        
        if edges > best_overall_edges:
            best_overall_edges = edges
            best_overall_adj = [x for x in adj]
            
        # If we found a very good solution, we can spend more time polishing it
        # But for now, we just keep the best.

    # Final polish on the best solution
    if best_overall_adj:
        best_overall_adj, best_overall_edges = local_search(best_overall_adj, deadline - 1.0)
        # One more greedy pass
        best_overall_adj = greedy_fill(best_overall_adj)
        best_overall_edges = count_edges(best_overall_adj)

    # Convert to edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_overall_adj[i] & (1 << j):
                edges.append([i, j])
                
    return edges
