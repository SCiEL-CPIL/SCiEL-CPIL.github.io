"""Construct C6-free graph on 40 vertices with maximum edges."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s limit
    
    n = 40
    
    # Precompute all pairs
    all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
    n_pairs = len(all_pairs)
    
    # Use bitmasks for fast adjacency checks
    # adj[i] is an integer where bit j is set if edge (i,j) exists
    
    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2
    
    def has_c6(adj, u, v):
        """
        Check if adding edge (u,v) to graph with adjacency adj creates a C6.
        Optimized with bit twiddling.
        """
        if (adj[u] >> v) & 1:
            return True  # Already an edge
        
        # Get neighbors of u and v excluding each other
        nu_mask = adj[u] & ~(1 << v)
        nv_mask = adj[v] & ~(1 << u)
        
        # Get lists of neighbors for a and d
        a_list = []
        tmp = nu_mask
        while tmp:
            a = (tmp & -tmp).bit_length() - 1
            a_list.append(a)
            tmp &= tmp - 1
            
        d_list = []
        tmp = nv_mask
        while tmp:
            d = (tmp & -tmp).bit_length() - 1
            d_list.append(d)
            tmp &= tmp - 1

        for a in a_list:
            # Precompute base mask for b (neighbors of a excluding u, v, a)
            b_base = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << a)
            for d in d_list:
                if a == d:
                    continue
                # Exclude d from b candidates
                b_mask = b_base & ~(1 << d)
                if b_mask == 0:
                    continue
                
                # Find c: neighbor of d, not in {u,v,a,d}
                c_mask = adj[d] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << d)
                if c_mask == 0:
                    continue
                
                # Check if there exists b in b_mask such that b is connected to any c in c_mask
                b_tmp = b_mask
                while b_tmp:
                    b = (b_tmp & -b_tmp).bit_length() - 1
                    b_tmp &= b_tmp - 1
                    if adj[b] & c_mask:
                        return True
        return False
    
    def try_add_edge(adj, u, v):
        """Try to add edge (u,v). Returns True if added."""
        if not has_c6(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            return True
        return False
    
    def remove_edge(adj, u, v):
        """Remove edge (u,v)."""
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    def add_edge(adj, u, v):
        """Add edge (u,v) without checking."""
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def get_edges(adj):
        """Get list of edges from adjacency masks."""
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges
    
    def local_search(adj, max_iter=2000):
        """
        Local search: try to add any missing edge that doesn't create C6.
        """
        improved = True
        pairs = all_pairs[:]
        while improved:
            improved = False
            random.shuffle(pairs)
            for u, v in pairs:
                if time.time() > deadline:
                    return adj
                if not ((adj[u] >> v) & 1):
                    if try_add_edge(adj, u, v):
                        improved = True
        return adj
    
    random.seed(42)
    best_adj = None
    best_count = 0
    
    # Phase 1: Multiple random restarts with greedy construction
    n_restarts = 100
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        adj = [0] * n
        shuffled = all_pairs[:]
        random.shuffle(shuffled)
        
        # Greedy add
        for u, v in shuffled:
            if time.time() > deadline:
                break
            try_add_edge(adj, u, v)
        
        # Local search to fill in missing edges
        adj = local_search(adj)
        
        current_count = count_edges(adj)
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
        
        if best_count >= 105:
            # Try a few more restarts to see if we can do better
            if restart >= 10:
                break
    
    # Phase 2: Perturbation-based local search from best solution
    if best_adj is not None:
        for perturb_round in range(50):
            if time.time() > deadline:
                break
            
            adj = best_adj[:]
            
            # Remove k random edges to perturb
            k = random.randint(3, 8)
            current_edges = get_edges(adj)
            if len(current_edges) > k:
                removed = random.sample(current_edges, k)
                for u, v in removed:
                    remove_edge(adj, u, v)
            
            # Try to re-add edges greedily
            shuffled = all_pairs[:]
            random.shuffle(shuffled)
            for u, v in shuffled:
                if time.time() > deadline:
                    break
                if not ((adj[u] >> v) & 1):
                    try_add_edge(adj, u, v)
            
            # Local search
            adj = local_search(adj)
            
            current_count = count_edges(adj)
            if current_count > best_count:
                best_count = current_count
                best_adj = adj[:]
    
    return get_edges(best_adj)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    t0 = time.time()
    edges = construct()
    t1 = time.time()
    print(f"Edges: {len(edges)}, Time: {t1-t0:.2f}s")
