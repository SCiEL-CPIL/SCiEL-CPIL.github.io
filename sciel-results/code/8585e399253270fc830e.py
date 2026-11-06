import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 40
    best_edges = []
    best_count = 0
    
    # Pre-compute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Fast C6 check using bitsets (Python integers as bitmasks)
    def would_create_c6(u, v, adj_masks):
        # adj_masks[i] is a bitmask of neighbors of i
        # Check if there's a simple path u-a-b-c-d-v of length 5
        # Using iterative deepening with bitmasks for visited
        
        # For each neighbor a of u
        neighbors_u = adj_masks[u]
        if neighbors_u == 0:
            return False
        
        # We need path u-a-b-c-d-v where all are distinct
        # a in N(u), b in N(a)\{u}, c in N(b)\{u,a}, d in N(c)\{u,a,b}, v in N(d)\{u,a,b,c}
        
        # Iterate over possible a values
        a_mask = neighbors_u
        while a_mask:
            a_bit = a_mask & (-a_mask)
            a = a_bit.bit_length() - 1
            a_mask ^= a_bit
            
            # b in N(a) excluding u
            b_mask = adj_masks[a] & ~(1 << u)
            if b_mask == 0:
                continue
            
            while b_mask:
                b_bit = b_mask & (-b_mask)
                b = b_bit.bit_length() - 1
                b_mask ^= b_bit
                
                # c in N(b) excluding u, a
                c_mask = adj_masks[b] & ~(1 << u) & ~(1 << a)
                if c_mask == 0:
                    continue
                
                while c_mask:
                    c_bit = c_mask & (-c_mask)
                    c = c_bit.bit_length() - 1
                    c_mask ^= c_bit
                    
                    # d in N(c) excluding u, a, b
                    d_mask = adj_masks[c] & ~(1 << u) & ~(1 << a) & ~(1 << b)
                    if d_mask == 0:
                        continue
                    
                    # Check if any d in d_mask is connected to v
                    # v must be in N(d) and d != v
                    v_bit = 1 << v
                    # d must be in d_mask AND N(d) must contain v
                    # For each d in d_mask, check if v is in N(d)
                    d_check = d_mask
                    while d_check:
                        d_bit = d_check & (-d_check)
                        d = d_bit.bit_length() - 1
                        d_check ^= d_bit
                        if adj_masks[d] & v_bit:
                            return True
        
        return False

    # Local search: try to add edges by removing conflicting ones
    def local_search(adj_masks, edge_set, edge_list):
        """Try to improve by swapping edges."""
        improved = True
        while improved:
            improved = False
            # Try adding each non-edge
            for u, v in all_edges:
                if (u, v) in edge_set or (v, u) in edge_set:
                    continue
                if would_create_c6(u, v, adj_masks):
                    continue
                # Can add this edge
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                edge_set.add((u, v))
                edge_list.append((u, v))
                improved = True
        return edge_list

    # Greedy construction with multiple restarts
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        random.shuffle(all_edges)
        adj_masks = [0] * n
        edge_set = set()
        current_edges = []
        
        for u, v in all_edges:
            if would_create_c6(u, v, adj_masks):
                continue
            # Add edge
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            edge_set.add((u, v))
            current_edges.append((u, v))
        
        # Local search to try to add more edges
        current_edges = local_search(adj_masks, edge_set, current_edges)
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]
            
            # If we found a very good result, do more local search
            if best_count >= 95:
                # Try additional local search passes
                for _ in range(3):
                    if time.time() > deadline - 10:
                        break
                    # Try random edge swaps
                    improved = True
                    while improved and time.time() < deadline - 10:
                        improved = False
                        random.shuffle(current_edges)
                        for u, v in current_edges:
                            # Try removing this edge and adding a different one
                            # This is complex; skip for now
                            pass

    # Convert to list of lists [u, v]
    return [[u, v] for u, v in best_edges]
