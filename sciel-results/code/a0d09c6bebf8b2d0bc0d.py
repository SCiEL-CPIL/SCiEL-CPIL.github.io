"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n = 20
    
    def popcount(x):
        """Fast popcount for Python integers."""
        return bin(x).count('1')
    
    def can_add(adj_bits, u, v):
        """
        Check if adding edge (u,v) to graph with adjacency bitmasks adj_bits
        maintains K2,3-freeness.
        """
        # Condition 1: common neighbors of u and v
        common_uv = popcount(adj_bits[u] & adj_bits[v])
        if common_uv > 2:
            return False
        
        # Condition 2: for each w in N(v), w != u
        nv = adj_bits[v]
        w = nv
        while w:
            low = w & (-w)
            wi = low.bit_length() - 1
            w ^= low
            if wi == u:
                continue
            # N(u) gains v, and v~w, so v becomes common neighbor of u and w
            common_uw = popcount(adj_bits[u] & adj_bits[wi])
            if common_uw > 1:
                return False
        
        # Condition 3: for each w in N(u), w != v
        nu = adj_bits[u]
        w = nu
        while w:
            low = w & (-w)
            wi = low.bit_length() - 1
            w ^= low
            if wi == v:
                continue
            common_vw = popcount(adj_bits[v] & adj_bits[wi])
            if common_vw > 1:
                return False
        
        return True
    
    def count_tight_pairs(adj_bits):
        """
        Count the number of pairs (i,j) with exactly 2 common neighbors.
        """
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                common = popcount(adj_bits[i] & adj_bits[j])
                if common == 2:
                    count += 1
        return count
    
    def greedy_min_tight(adj_bits, edges):
        """
        Greedily add edges choosing the one that minimizes the increase in tight pairs.
        """
        improved = True
        while improved:
            improved = False
            best_edge = None
            best_tight_delta = float('inf')
            best_tight_count = float('inf')
            
            for u in range(n):
                for v in range(u+1, n):
                    if adj_bits[u] & (1 << v):
                        continue  # already an edge
                    
                    if not can_add(adj_bits, u, v):
                        continue
                    
                    # Simulate adding the edge and count tight pairs
                    old_tight = count_tight_pairs(adj_bits)
                    
                    # Try adding
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    
                    new_tight = count_tight_pairs(adj_bits)
                    
                    # Revert
                    adj_bits[u] &= ~(1 << v)
                    adj_bits[v] &= ~(1 << u)
                    
                    delta = new_tight - old_tight
                    if delta < best_tight_delta or (delta == best_tight_delta and new_tight < best_tight_count):
                        best_tight_delta = delta
                        best_tight_count = new_tight
                        best_edge = (u, v)
            
            if best_edge is not None:
                u, v = best_edge
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                edges.append([u, v])
                improved = True
        
        return edges, adj_bits

    def greedy_random(adj_bits, edges, rng):
        """Standard random greedy fill."""
        all_missing = [(i, j) for i in range(n) for j in range(i+1, n) 
                       if not (adj_bits[i] & (1 << j))]
        rng.shuffle(all_missing)
        for u, v in all_missing:
            if can_add(adj_bits, u, v):
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                edges.append([u, v])
        return edges, adj_bits
    
    def local_search_multi_swap(adj_bits, edges, rng, max_iters=20):
        """
        Local search with multi-edge swaps: remove k random edges, then re-greedy fill.
        This allows escaping local optima by rearranging edge structure.
        """
        best_edges = list(edges)
        best_adj = list(adj_bits)
        best_count = len(best_edges)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            cur_adj = list(adj_bits)
            cur_edges = list(edges)
            
            # Remove k random edges (k=1 or 2)
            k = rng.choice([1, 1, 2])
            for _ in range(k):
                if not cur_edges:
                    break
                rem_idx = rng.randint(0, len(cur_edges) - 1)
                u, v = cur_edges[rem_idx]
                cur_adj[u] &= ~(1 << v)
                cur_adj[v] &= ~(1 << u)
                cur_edges.pop(rem_idx)
            
            # Re-greedy fill with min-tight heuristic
            cur_edges, cur_adj = greedy_min_tight(cur_adj, cur_edges)
            
            if len(cur_edges) > best_count:
                best_count = len(cur_edges)
                best_edges = cur_edges
                best_adj = cur_adj
                # Update the working state
                adj_bits = cur_adj
                edges = cur_edges
        
        return edges, adj_bits, best_count

    def iterated_local_search(seed_edges, seed_adj, rng, num_iterations=50):
        """
        Iterated Local Search:
        1. Start from a seed solution
        2. Perturb (remove k random edges)
        3. Local search (greedy fill)
        4. Accept if better, else revert
        """
        best_edges = list(seed_edges)
        best_adj = list(seed_adj)
        best_count = len(best_edges)
        cur_edges = list(seed_edges)
        cur_adj = list(seed_adj)
        
        for it in range(num_iterations):
            if time.time() > deadline - 10:
                break
            
            # Perturb: remove k random edges
            k = rng.randint(1, 3)
            perturbed_edges = list(cur_edges)
            perturbed_adj = list(cur_adj)
            
            for _ in range(k):
                if not perturbed_edges:
                    break
                rem_idx = rng.randint(0, len(perturbed_edges) - 1)
                u, v = perturbed_edges[rem_idx]
                perturbed_adj[u] &= ~(1 << v)
                perturbed_adj[v] &= ~(1 << u)
                perturbed_edges.pop(rem_idx)
            
            # Local search: greedy fill
            new_edges, new_adj = greedy_min_tight(perturbed_adj, perturbed_edges)
            
            # Accept if better
            if len(new_edges) > best_count:
                best_count = len(new_edges)
                best_edges = new_edges
                best_adj = new_adj
                cur_edges = new_edges
                cur_adj = new_adj
            elif len(new_edges) == best_count and rng.random() < 0.3:
                # Accept equal with some probability to diversify
                cur_edges = new_edges
                cur_adj = new_adj
        
        return best_edges, best_adj, best_count

    # Main strategy: Multiple approaches, keep the best
    best_edges_overall = []
    best_count_overall = 0
    
    rng = random.Random(42)
    
    # Approach 1: Pure min-tight greedy from empty
    if time.time() < deadline - 20:
        adj_bits = [0] * n
        edges = []
        edges, adj_bits = greedy_min_tight(adj_bits, edges)
        if len(edges) > best_count_overall:
            best_count_overall = len(edges)
            best_edges_overall = edges
    
    # Approach 2: Random restarts with ILS
    for r in range(20):
        if time.time() > deadline - 15:
            break
        
        # Random initial seed
        adj_bits = [0] * n
        edges = []
        
        # Add a random subset of edges
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        rng.shuffle(all_edges)
        
        num_seed = rng.randint(5, 15)
        for u, v in all_edges[:num_seed]:
            if can_add(adj_bits, u, v):
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                edges.append([u, v])
        
        # Greedy fill
        edges, adj_bits = greedy_min_tight(adj_bits, edges)
        
        # Iterated Local Search
        if time.time() < deadline - 10:
            edges, adj_bits, _ = iterated_local_search(edges, adj_bits, rng, num_iterations=30)
        
        if len(edges) > best_count_overall:
            best_count_overall = len(edges)
            best_edges_overall = edges
    
    # Approach 3: Structured base + ILS
    if time.time() < deadline - 20:
        adj_bits = [0] * n
        edges = []
        
        # Add a 20-cycle
        for i in range(n):
            j = (i + 1) % n
            adj_bits[i] |= (1 << j)
            adj_bits[j] |= (1 << i)
            edges.append([i, j])
        
        # Greedy fill
        edges, adj_bits = greedy_min_tight(adj_bits, edges)
        
        # ILS
        if time.time() < deadline - 10:
            edges, adj_bits, _ = iterated_local_search(edges, adj_bits, rng, num_iterations=30)
        
        if len(edges) > best_count_overall:
            best_count_overall = len(edges)
            best_edges_overall = edges
    
    # Final local search on the best solution
    if time.time() < deadline - 3 and best_edges_overall:
        # Reconstruct adj_bits from best_edges_overall
        adj_bits = [0] * n
        for u, v in best_edges_overall:
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
        
        best_edges_overall, adj_bits, _ = local_search_multi_swap(
            adj_bits, best_edges_overall, rng, max_iters=30
        )
    
    return best_edges_overall
# EVOLVE-BLOCK-END
