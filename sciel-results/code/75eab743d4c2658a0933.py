"""Spectral construction + local swaps for C6-free graphs on 40 vertices."""
import numpy as np
import random
import time

def _has_c6(adj_bits, n):
    """Check if the graph with adjacency bitmasks adj_bits contains a C6."""
    # Iterate over paths of length 2: v1 - v2 - v3
    for v1 in range(n):
        # Get neighbors of v1
        n1_mask = adj_bits[v1]
        # Iterate v2 in N(v1)
        v2 = n1_mask
        while v2:
            lsb = v2 & -v2
            v2 ^= lsb
            v2_idx = lsb.bit_length() - 1
            
            # Get neighbors of v2, excluding v1
            n2_mask = adj_bits[v2_idx] & ~lsb & ~(1 << v1)
            
            # Iterate v3 in N(v2) \ {v1}
            v3 = n2_mask
            while v3:
                lsb3 = v3 & -v3
                v3 ^= lsb3
                v3_idx = lsb3.bit_length() - 1
                
                # We have path v1-v2-v3
                # Need v6 in N(v1) \ {v2, v3}
                # v6 must be connected to v5, v5 connected to v4, v4 connected to v3
                # v6 in N(v1)
                # v5 in N(v6)
                # v4 in N(v3) AND N(v5)
                
                # Candidates for v6
                v6_mask = adj_bits[v1] & ~(1 << v2_idx) & ~(1 << v3_idx) & ~(1 << v1)
                
                v6 = v6_mask
                while v6:
                    lsb6 = v6 & -v6
                    v6 ^= lsb6
                    v6_idx = lsb6.bit_length() - 1
                    
                    # Candidates for v5
                    # v5 in N(v6)
                    # v5 != v1, v6
                    # v5 must be connected to some v4 in N(v3)
                    # So v5 must be in N(v6) AND (union of N(w) for w in N(v3))
                    # Simpler: iterate v5 in N(v6)
                    
                    v5_mask = adj_bits[v6_idx] & ~(1 << v1) & ~(1 << v6_idx)
                    
                    v5 = v5_mask
                    while v5:
                        lsb5 = v5 & -v5
                        v5 ^= lsb5
                        v5_idx = lsb5.bit_length() - 1
                        
                        # v5 must be distinct from v2, v3
                        if v5_idx == v2_idx or v5_idx == v3_idx:
                            continue
                            
                        # v4 must be in N(v3) AND N(v5)
                        # v4 must be distinct from v1, v2, v6
                        # v4 must be distinct from v5 (implied by simple graph if we check intersection properly? No, N(v3) and N(v5) can share v5? No, v5 is not in N(v5). v5 is not in N(v3) unless v3-v5 edge exists. If v3-v5 edge exists, v4 could be v3? No v4 distinct from v3.)
                        
                        common = adj_bits[v3_idx] & adj_bits[v5_idx]
                        # Remove v1, v2, v6
                        common &= ~(1 << v1)
                        common &= ~(1 << v2_idx)
                        common &= ~(1 << v6_idx)
                        
                        if common:
                            # Check if any v4 in common makes a distinct set of 6
                            # We need v4 distinct from v1, v2, v3, v5, v6
                            # v4 is in N(v3) so v4 != v3.
                            # v4 is in N(v5) so v4 != v5.
                            # We removed v1, v2, v6.
                            # So we just need to ensure v4 is not v3 or v5 (already true) 
                            # and that all 6 are distinct.
                            # The set is {v1, v2, v3, v4, v5, v6}.
                            # We know v1!=v2, v2!=v3, v3!=v1 (from path construction? 
                            # v2 in N(v1) so v1!=v2.
                            # v3 in N(v2)\{v1} so v3!=v2, v3!=v1.
                            # v6 in N(v1)\{v2,v3} so v6!=v1, v6!=v2, v6!=v3.
                            # v5 in N(v6)\{v1,v6} so v5!=v6, v5!=v1.
                            # We checked v5!=v2, v5!=v3.
                            # v4 in N(v3)\{v1,v2,v6} so v4!=v3, v4!=v1, v4!=v2, v4!=v6.
                            # We need v4!=v5.
                            # Since v4 is in N(v5), v4!=v5 in a simple graph.
                            # So if common is not empty, we have a C6.
                            return True
    return False

def _count_edges(adj_bits, n):
    return sum(bin(x).count('1') for x in adj_bits) // 2

def construct():
    n = 40
    start_time = time.time()
    
    # 1. Initialization
    # Try a random 5-regular graph or similar
    # We will use a random graph with p=0.15 (approx 120 edges) and prune
    
    def random_graph_edges(n, p):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < p:
                    edges.append((u, v))
        return edges
    
    # Start with a slightly denser graph to have room to prune
    # Target ~105 edges. Start with ~115.
    # p = 2 * 115 / (40*39) = 230 / 1560 = 0.147
    
    random.seed(42)
    
    # We'll use a list of edges and a bitmask representation
    # adj_bits[i] is an integer with bit j set if i-j is an edge
    
    adj_bits = [0] * n
    
    # Generate initial edges
    # Let's try to construct a specific graph first to ensure C6-free start
    # A circulant graph C_40(1, 2, 3) might have C6s.
    # Let's just use random and prune.
    
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if random.random() < 0.15:
                edges.append((u, v))
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                
    # 2. Prune C6s
    # Loop until no C6
    # To speed up, we can stop if time is running out, but we need a valid graph.
    # A graph with C6s is invalid.
    
    max_prune_iter = 500
    for _ in range(max_prune_iter):
        if _has_c6(adj_bits, n):
            # Find a C6 to remove an edge from
            # We can just remove a random edge and re-check, but that's slow.
            # Better: find the C6 in _has_c6 and return the edge.
            # For now, let's modify _has_c6 to return an edge in the C6
            
            # To keep it simple, let's just pick a random edge and remove it, 
            # then check again. This is O(E) checks per removal.
            # E is ~100. Checks are fast.
            
            if edges:
                idx = random.randint(0, len(edges)-1)
                u, v = edges.pop(idx)
                adj_bits[u] &= ~(1 << v)
                adj_bits[v] &= ~(1 << u)
            else:
                break
        else:
            break
            
    # 3. Greedy Addition
    # Try to add all possible edges that don't create a C6
    
    # Order candidates by some heuristic? Random is fine.
    candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if not (adj_bits[u] & (1 << v)):
                candidates.append((u, v))
                
    random.shuffle(candidates)
    
    for u, v in candidates:
        if time.time() - start_time > 100:
            break
        # Test addition
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)
        
        if _has_c6(adj_bits, n):
            # Revert
            adj_bits[u] &= ~(1 << v)
            adj_bits[v] &= ~(1 << u)
        else:
            edges.append((u, v))
            
    # 4. Local Swaps
    # Try to swap an existing edge for a non-edge to increase density or maintain?
    # We want to MAXIMIZE edges. So we only accept swaps if they increase edge count?
    # No, swaps keep edge count constant. We need to ADD edges.
    # The greedy phase added all possible edges. 
    # So we are stuck at a local maximum for addition.
    # To escape, we must REMOVE an edge to allow ADDING two edges?
    # Or REMOVE one and ADD one (same count) doesn't help maximize.
    # We need a move that increases edge count.
    # "Remove e1, Add e2" keeps count.
    # "Remove e1, e2; Add e3, e4" increases count by 0? No.
    # We need a sequence of moves that increases the total count.
    # Usually, this is done by:
    # 1. Remove edge e.
    # 2. Add edge f (if possible).
    # This doesn't increase count.
    
    # Actually, if we are at a local maximum where no single edge can be added,
    # we might be able to remove one edge and add TWO edges?
    # Or remove one edge, add one edge, then add another?
    
    # Let's try a "2-edge swap" or "remove and add 2".
    # Move: Remove (u1, v1). Try to add (u2, v2) and (u3, v3).
    # If the new graph is C6-free, we gain 1 edge.
    
    # This is expensive.
    
    # Alternative: The greedy phase might have failed to find a global max due to order.
    # Random restarts?
    
    # Let's just run the greedy phase multiple times with different seeds/initializations
    # and keep the best result.
    
    best_edges = edges
    best_count = len(edges)
    
    # Try a few more initializations
    for seed in [1, 2, 3]:
        random.seed(seed)
        adj_bits_temp = [0] * n
        edges_temp = []
        
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < 0.15:
                    edges_temp.append((u, v))
                    adj_bits_temp[u] |= (1 << v)
                    adj_bits_temp[v] |= (1 << u)
                    
        # Prune
        for _ in range(500):
            if _has_c6(adj_bits_temp, n):
                if edges_temp:
                    idx = random.randint(0, len(edges_temp)-1)
                    u, v = edges_temp.pop(idx)
                    adj_bits_temp[u] &= ~(1 << v)
                    adj_bits_temp[v] &= ~(1 << u)
                else:
                    break
            else:
                break
                
        # Greedy
        candidates_temp = []
        for u in range(n):
            for v in range(u+1, n):
                if not (adj_bits_temp[u] & (1 << v)):
                    candidates_temp.append((u, v))
        random.shuffle(candidates_temp)
        
        for u, v in candidates_temp:
            if time.time() - start_time > 110:
                break
            adj_bits_temp[u] |= (1 << v)
            adj_bits_temp[v] |= (1 << u)
            
            if _has_c6(adj_bits_temp, n):
                adj_bits_temp[u] &= ~(1 << v)
                adj_bits_temp[v] &= ~(1 << u)
            else:
                edges_temp.append((u, v))
                
        if len(edges_temp) > best_count:
            best_count = len(edges_temp)
            best_edges = edges_temp
            adj_bits = adj_bits_temp

    return [[u, v] for u, v in best_edges]
