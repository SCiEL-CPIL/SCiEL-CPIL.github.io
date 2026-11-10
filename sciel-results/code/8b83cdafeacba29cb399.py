import time
import random
import numpy as np

def construct():
    deadline = time.time() + 110
    n = 40
    
    # 1. Initialize with 8 disjoint K5s (104 edges)
    edges = []
    adj_mask = [0] * n
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u = 5 * k + a
                v = 5 * k + b
                edges.append((u, v))
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
    
    edge_set = set(edges)
    
    # 2. Fast C6 Detection using Bitsets
    def creates_c6(u, v):
        mask_u = adj_mask[u]
        mask_v = adj_mask[v]
        
        # Check if adding (u,v) creates a C6
        # Need path u-w-x-y-z-v with all distinct
        w_mask = mask_u & ~(1 << v)
        while w_mask:
            w_bit = w_mask & -w_mask
            w = w_bit.bit_length() - 1
            w_mask ^= w_bit
            
            x_mask = adj_mask[w] & ~(1 << u) & ~(1 << v) & ~(1 << w)
            while x_mask:
                x_bit = x_mask & -x_mask
                x = x_bit.bit_length() - 1
                x_mask ^= x_bit
                
                y_mask = adj_mask[x] & ~(1 << u) & ~(1 << v) & ~(1 << w) & ~(1 << x)
                while y_mask:
                    y_bit = y_mask & -y_mask
                    y = y_bit.bit_length() - 1
                    y_mask ^= y_bit
                    
                    z_mask = adj_mask[y] & mask_v
                    forbidden = (1 << u) | (1 << v) | (1 << w) | (1 << x) | (1 << y)
                    if z_mask & ~forbidden:
                        return True
        return False
    
    # 3. Helper functions for edge operations
    def add_edge(u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        edge_set.add((u, v))
    
    def remove_edge(u, v):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        edge_set.discard((u, v))
    
    # 4. Structured Local Search
    best_edges = set(edge_set)
    best_score = len(best_edges)
    
    # Precompute all possible edges for systematic exploration
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Phase 1: Greedy addition - try all non-edges systematically
    # Sort by a heuristic: prefer edges that connect vertices with low common neighbor count
    non_edges = [e for e in all_pairs if e not in edge_set]
    
    # Heuristic: prioritize edges where endpoints have few shared 2-step paths
    # This reduces likelihood of creating C6
    def c6_risk(e):
        u, v = e
        # Count potential C6 witnesses
        risk = 0
        mask_u = adj_mask[u]
        mask_v = adj_mask[v]
        w_mask = mask_u & ~(1 << v)
        while w_mask:
            w_bit = w_mask & -w_mask
            w = w_bit.bit_length() - 1
            w_mask ^= w_bit
            x_mask = adj_mask[w] & ~(1 << u) & ~(1 << v) & ~(1 << w)
            while x_mask:
                x_bit = x_mask & -x_mask
                x = x_bit.bit_length() - 1
                x_mask ^= x_bit
                y_mask = adj_mask[x] & ~(1 << u) & ~(1 << v) & ~(1 << w) & ~(1 << x)
                while y_mask:
                    y_bit = y_mask & -y_mask
                    y = y_bit.bit_length() - 1
                    y_mask ^= y_bit
                    z_mask = adj_mask[y] & mask_v
                    forbidden = (1 << u) | (1 << v) | (1 << w) | (1 << x) | (1 << y)
                    if z_mask & ~forbidden:
                        risk += 1
        return risk
    
    # Sort non-edges by C6 risk (low risk first)
    non_edges.sort(key=lambda e: c6_risk(e))
    
    # Try adding low-risk edges first
    for u, v in non_edges:
        if time.time() > deadline - 10:
            break
        if (u, v) in edge_set:
            continue
        if not creates_c6(u, v):
            add_edge(u, v)
            if len(edge_set) > best_score:
                best_score = len(edge_set)
                best_edges = set(edge_set)
    
    # Phase 2: Local search with 1-flip and 2-flip moves
    # 1-flip: try adding a random non-edge
    # 2-flip: remove one edge, try to add two new edges (net +1)
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline - 5:
        improved = False
        
        # Try 1-flip moves: add a single edge
        # Sample from non-edges
        attempts = 0
        while attempts < 200 and time.time() < deadline - 5:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            if (u, v) in edge_set:
                continue
            attempts += 1
            
            if not creates_c6(u, v):
                add_edge(u, v)
                improved = True
                if len(edge_set) > best_score:
                    best_score = len(edge_set)
                    best_edges = set(edge_set)
                break
        
        if improved:
            no_improve_count = 0
            continue
        
        # Try 2-flip moves: remove edge e1, add edges e2 and e3
        # This allows escaping local optima
        if no_improve_count < 10:  # Only try 2-flips occasionally
            # Pick a random edge to remove
            if edge_set:
                e1 = random.choice(list(edge_set))
                u1, v1 = e1
                remove_edge(u1, v1)
                
                # Try to add two new edges
                added = 0
                for _ in range(50):
                    if time.time() > deadline - 5:
                        break
                    u2 = random.randint(0, n - 1)
                    v2 = random.randint(0, n - 1)
                    if u2 == v2:
                        continue
                    if u2 > v2:
                        u2, v2 = v2, u2
                    if (u2, v2) in edge_set:
                        continue
                    if (u2, v2) == e1:
                        continue
                    
                    if not creates_c6(u2, v2):
                        add_edge(u2, v2)
                        added += 1
                        if added == 2:
                            break
                
                # If we added 2 edges, net gain is +1
                if added == 2:
                    if len(edge_set) > best_score:
                        best_score = len(edge_set)
                        best_edges = set(edge_set)
                    improved = True
                else:
                    # Roll back: re-add e1, remove any added edges
                    if added == 1:
                        # Find which edge was added and remove it
                        for u2, v2 in [(u, v) for u in range(n) for v in range(u + 1, n) 
                                       if (u, v) in edge_set and (u, v) != e1]:
                            # This is inefficient, let's track differently
                            pass
                        # Simpler: just re-add e1 and accept the loss
                        add_edge(u1, v1)
                    else:
                        add_edge(u1, v1)
        
        if not improved:
            no_improve_count += 1
        
        # If stuck for too long, try a more aggressive restart
        if no_improve_count > max_no_improve:
            # Restore best state
            # Clear and rebuild from best_edges
            adj_mask = [0] * n
            edge_set.clear()
            for u, v in best_edges:
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                edge_set.add((u, v))
            no_improve_count = 0
    
    return [[a, b] for (a, b) in sorted(best_edges)]
