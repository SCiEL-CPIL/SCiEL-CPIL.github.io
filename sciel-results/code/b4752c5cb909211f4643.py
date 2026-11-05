"""Rewritten program: bitmask local search to minimize largest book."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    n = 40
    m = 401
    deadline = time.time() + 140  # safety margin

    def build_adj(edge_list):
        adj = [0] * n
        for u, v in edge_list:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def compute_max_book(adj, edge_list):
        max_book = 0
        for u, v in edge_list:
            common = adj[u] & adj[v]
            b = common.bit_count()
            if b > max_book:
                max_book = b
        return max_book

    def random_graph():
        edges = set()
        while len(edges) < m:
            u = random.randrange(n)
            v = random.randrange(n)
            if u != v:
                edges.add((min(u, v), max(u, v)))
        return list(edges)

    def local_search(edge_list, adj, current_max, time_limit):
        edge_set = set(edge_list)
        
        while time.time() < time_limit:
            # Find edges with max book
            max_edges = []
            for u, v in edge_list:
                b = (adj[u] & adj[v]).bit_count()
                if b == current_max:
                    max_edges.append((u, v))
            
            if not max_edges:
                break
                
            random.shuffle(max_edges)
            
            improved = False
            for u, v in max_edges:
                if time.time() >= time_limit:
                    break
                    
                # Try to swap (u, v) with a random non-edge
                best_candidate = None
                best_candidate_max = current_max
                
                # Try multiple random non-edges
                for _ in range(100):
                    if time.time() >= time_limit:
                        break
                    c = random.randrange(n)
                    d = random.randrange(n)
                    if c == d:
                        continue
                    c, d = min(c, d), max(c, d)
                    if (c, d) in edge_set:
                        continue
                    if (c, d) == (u, v):
                        continue
                    
                    # Simulate swap
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    adj[c] |= (1 << d)
                    adj[d] |= (1 << c)
                    
                    # Compute new max book
                    new_max = 0
                    for a, b in edge_list:
                        common = adj[a] & adj[b]
                        bk = common.bit_count()
                        if bk > new_max:
                            new_max = bk
                            if new_max > best_candidate_max:
                                break
                    
                    # Revert simulation
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    adj[c] &= ~(1 << d)
                    adj[d] &= ~(1 << c)
                    
                    if new_max < best_candidate_max:
                        best_candidate_max = new_max
                        best_candidate = (c, d)
                
                if best_candidate is not None:
                    c, d = best_candidate
                    # Perform swap
                    edge_set.discard((u, v))
                    edge_set.add((c, d))
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    adj[c] |= (1 << d)
                    adj[d] |= (1 << c)
                    
                    edge_list = list(edge_set)
                    current_max = best_candidate_max
                    improved = True
                    break # Restart outer loop with new state
            
            if not improved:
                # Stuck. Perturb.
                if not edge_list:
                    break
                u, v = random.choice(edge_list)
                for _ in range(200):
                    c = random.randrange(n)
                    d = random.randrange(n)
                    if c == d:
                        continue
                    c, d = min(c, d), max(c, d)
                    if (c, d) in edge_set or (c, d) == (u, v):
                        continue
                    
                    # Swap
                    edge_set.discard((u, v))
                    edge_set.add((c, d))
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    adj[c] |= (1 << d)
                    adj[d] |= (1 << c)
                    
                    edge_list = list(edge_set)
                    current_max = compute_max_book(adj, edge_list)
                    break
        
        return edge_list, adj, current_max

    best_score = float('inf')
    best_edge_list = None
    
    while time.time() < deadline:
        edge_list = random_graph()
        adj = build_adj(edge_list)
        current_max = compute_max_book(adj, edge_list)
        
        # Local search
        edge_list, adj, current_max = local_search(edge_list, adj, current_max, deadline)
        
        if current_max < best_score:
            best_score = current_max
            best_edge_list = edge_list
            
        if best_score <= 7:
            break

    return [[u, v] for u, v in best_edge_list]
# EVOLVE-BLOCK-END
