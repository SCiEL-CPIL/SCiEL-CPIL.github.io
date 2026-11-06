import random
import time
import numpy as np

def construct():
    """
    Constructs a C6-free graph on 40 vertices with a high number of edges.
    Combines randomized greedy construction, 1-swap local search, and
    a final high-precision polish pass with aggressive edge-addition.
    """
    N = 40
    start_time = time.time()
    deadline = start_time + 110  # 40s safety margin before 150s limit
    
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    all_edges_set = set(all_edges)
    
    best_edges = []
    best_count = 0
    
    def would_create_c6(adj, u, v):
        """
        Check if adding edge (u,v) creates a C6.
        C6: u-a-x-y-b-v where all 6 vertices are distinct.
        For each a in N(u), b in N(v) with a != b, a != v, b != u:
          Check if there exists x in N(a) \ {u,v,b} and y in N(b) \ {u,v,a}
          such that (x,y) is an edge and x != y.
        """
        nu = adj[u] & ~(1 << v)  # a in N(u), a != v
        nv = adj[v] & ~(1 << u)  # b in N(v), b != u
        
        a_mask = nu
        while a_mask:
            a_bit = a_mask & (-a_mask)
            a = a_bit.bit_length() - 1
            a_mask ^= a_bit
            
            b_mask = nv & ~(1 << a)  # b != a
            while b_mask:
                b_bit = b_mask & (-b_mask)
                b = b_bit.bit_length() - 1
                b_mask ^= b_bit
                
                # x in N(a) \ {u,v,b}, y in N(b) \ {u,v,a}
                x_mask = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << b)
                if not x_mask:
                    continue
                y_mask = adj[b] & ~(1 << u) & ~(1 << v) & ~(1 << a)
                if not y_mask:
                    continue
                
                # Check if any edge (x,y) with x in x_mask, y in y_mask, x != y
                xm = x_mask
                while xm:
                    x_bit = xm & (-xm)
                    x = x_bit.bit_length() - 1
                    xm ^= x_bit
                    if adj[x] & (y_mask & ~(1 << x)):
                        return True
        
        return False
    
    def greedy_random(seed):
        """Greedy construction with random edge ordering."""
        rng = random.Random(seed)
        edges_list = list(all_edges)
        rng.shuffle(edges_list)
        
        adj = [0] * N
        edge_set = set()
        
        for u, v in edges_list:
            if not would_create_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add((u, v))
        
        return edge_set, adj
    
    def local_search_1swap(edge_set, adj, seed, max_iters=3000):
        """
        Local search with 1-swap moves: remove one edge, add another.
        Also tries pure additions.
        """
        rng = random.Random(seed)
        current_edges = set(edge_set)
        current_adj = list(adj)
        current_count = len(current_edges)
        
        for it in range(max_iters):
            if time.time() >= deadline:
                break
            
            improved = False
            
            # Try pure additions (fast path)
            for _ in range(20):
                u, v = rng.randint(0, N-1), rng.randint(0, N-1)
                if u == v:
                    continue
                u, v = min(u, v), max(u, v)
                if (u, v) in current_edges:
                    continue
                if not would_create_c6(current_adj, u, v):
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_edges.add((u, v))
                    current_count += 1
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 1-swap: remove one random edge, try to add another
            if current_count > 0:
                edges_list = list(current_edges)
                u1, v1 = rng.choice(edges_list)
                
                # Remove (u1, v1)
                current_adj[u1] &= ~(1 << v1)
                current_adj[v1] &= ~(1 << u1)
                current_edges.discard((u1, v1))
                
                # Try to add a random non-edge
                added = False
                for _ in range(30):
                    u2, v2 = rng.randint(0, N-1), rng.randint(0, N-1)
                    if u2 == v2:
                        continue
                    u2, v2 = min(u2, v2), max(u2, v2)
                    if (u2, v2) in current_edges:
                        continue
                    if not would_create_c6(current_adj, u2, v2):
                        current_adj[u2] |= (1 << v2)
                        current_adj[v2] |= (1 << u2)
                        current_edges.add((u2, v2))
                        current_count += 1
                        added = True
                        break
                
                # If we didn't add anything, restore the removed edge
                if not added:
                    current_adj[u1] |= (1 << v1)
                    current_adj[v1] |= (1 << u1)
                    current_edges.add((u1, v1))
                else:
                    # Net change: removed 1, added 1, count stays same
                    # But we might have freed up space for more edges
                    pass
        
        return current_edges, current_adj
    
    def polish_pass(edge_set, adj):
        """
        Final high-precision polish: systematically try all non-edges
        in multiple random orderings to squeeze out more edges.
        """
        current_edges = set(edge_set)
        current_adj = list(adj)
        current_count = len(current_edges)
        
        # Multiple passes with different random orderings
        for pass_num in range(5):
            if time.time() >= deadline - 5:
                break
            
            non_edges = [(u, v) for (u, v) in all_edges if (u, v) not in current_edges]
            random.shuffle(non_edges)
            
            for u, v in non_edges:
                if time.time() >= deadline - 3:
                    break
                if (u, v) in current_edges:
                    continue
                if not would_create_c6(current_adj, u, v):
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_edges.add((u, v))
                    current_count += 1
        
        return current_edges, current_adj
    
    # Main search loop: multiple greedy restarts with local search and polish
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() >= deadline - 15:
            break
        
        seed = restart * 12345 + 67890
        edge_set, adj = greedy_random(seed)
        current_count = len(edge_set)
        
        # Local search with 1-swaps
        if time.time() < deadline - 10:
            edge_set, adj = local_search_1swap(edge_set, adj, seed, max_iters=2000)
            current_count = len(edge_set)
        
        # Final polish
        if time.time() < deadline - 5:
            edge_set, adj = polish_pass(edge_set, adj)
            current_count = len(edge_set)
        
        if current_count > best_count:
            best_count = current_count
            best_edges = list(edge_set)
            # Print progress (optional, for debugging)
            # print(f"Restart {restart}: {current_count} edges")
    
    # If we still have time, do more polish on the best solution
    if best_edges and time.time() < deadline - 5:
        adj = [0] * N
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Aggressive polish with more passes
        for pass_num in range(10):
            if time.time() >= deadline - 2:
                break
            
            non_edges = [(u, v) for (u, v) in all_edges if (u, v) not in set(best_edges)]
            random.shuffle(non_edges)
            
            improved = False
            for u, v in non_edges:
                if (u, v) in set(best_edges):
                    continue
                if not would_create_c6(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    best_edges.append((u, v))
                    best_count += 1
                    improved = True
            
            if not improved:
                break
    
    # Return in the required format: list of [u, v]
    return [[u, v] for u, v in best_edges]
