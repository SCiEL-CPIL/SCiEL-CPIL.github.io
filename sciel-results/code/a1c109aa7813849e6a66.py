"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    n = 40
    
    # Precompute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def has_c6(adj_bits, u, v):
        """
        Check if adding edge (u,v) creates a C6.
        Path: u-a-b-c-d-v
        """
        # Quick checks
        if not adj_bits[u] or not adj_bits[v]:
            return False
            
        # Iterate a in N(u)
        a_candidates = adj_bits[u] & ~(1 << v)
        
        while a_candidates:
            a_bit = a_candidates & -a_candidates
            a = a_bit.bit_length() - 1
            a_candidates -= a_bit
            
            # Iterate b in N(a) \ {u, v}
            b_candidates = adj_bits[a] & ~(1 << u) & ~(1 << v)
            
            while b_candidates:
                b_bit = b_candidates & -b_candidates
                b = b_bit.bit_length() - 1
                b_candidates -= b_bit
                
                # Iterate c in N(b) \ {u, a, v}
                c_candidates = adj_bits[b] & ~(1 << u) & ~(1 << a) & ~(1 << v)
                
                while c_candidates:
                    c_bit = c_candidates & -c_candidates
                    c = c_bit.bit_length() - 1
                    c_candidates -= c_bit
                    
                    # Check if there exists d such that:
                    # d in N(c) AND d in N(v)
                    # d not in {u, a, b, c}
                    mask_c = adj_bits[c]
                    mask_v = adj_bits[v]
                    d_candidates = mask_c & mask_v
                    forbidden = (1 << u) | (1 << a) | (1 << b) | (1 << c)
                    
                    if d_candidates & ~forbidden:
                        return True
        return False

    def greedy_build(order, adj_bits, edge_set):
        """
        Greedily add edges from order to adj_bits if no C6 is formed.
        """
        added_edges = []
        for u, v in order:
            if (u, v) in edge_set or (v, u) in edge_set:
                continue
            if (adj_bits[u] >> v) & 1:
                continue
            
            if not has_c6(adj_bits, u, v):
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                edge_set.add((u, v))
                edge_set.add((v, u))
                added_edges.append([u, v])
        return added_edges

    random.seed(42)
    
    best_edges = []
    best_count = 0
    best_adj = [0] * n
    best_set = set()
    
    # Phase 1: Greedy Restarts
    # Try a few random orders to find a good starting point
    num_restarts = 30
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
            
        current_adj = [0] * n
        current_edge_set = set()
        current_order = all_pairs[:]
        random.shuffle(current_order)
        
        current_edges = greedy_build(current_order, current_adj, current_edge_set)
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]
            best_adj = current_adj[:]
            best_set = current_edge_set.copy()
            
        # If we hit a very high count (e.g., > 105), we might stop early
        # but let's use time to polish.
        
    # Phase 2: Local Search on the best graph found
    if best_count > 0 and time.time() < deadline:
        curr_adj = best_adj[:]
        curr_set = best_set.copy()
        curr_edges = best_edges[:]
        
        # Build list of non-edges
        curr_non_edges = [p for p in all_pairs if p not in curr_set]
        
        # Local search loop
        while time.time() < deadline:
            improved = False
            
            # Try to add any non-edge
            for i in range(len(curr_non_edges)):
                u, v = curr_non_edges[i]
                if not has_c6(curr_adj, u, v):
                    # Add edge
                    curr_adj[u] |= (1 << v)
                    curr_adj[v] |= (1 << u)
                    curr_set.add((u, v))
                    curr_set.add((v, u))
                    curr_edges.append([u, v])
                    curr_non_edges.pop(i)
                    improved = True
                    break
            
            if not improved:
                # Stuck: try a swap to unblock
                if not curr_edges or not curr_non_edges:
                    break
                
                # Pick a random edge to remove
                e1 = random.choice(curr_edges)
                u1, v1 = e1
                
                # Pick a random non-edge to add
                e2 = random.choice(curr_non_edges)
                u2, v2 = e2
                
                # Check if removing e1 and adding e2 is valid
                # Save state of e1 endpoints
                old_u1 = curr_adj[u1]
                old_v1 = curr_adj[v1]
                
                # Temporarily remove e1
                curr_adj[u1] &= ~(1 << v1)
                curr_adj[v1] &= ~(1 << u1)
                
                if not has_c6(curr_adj, u2, v2):
                    # Accept the swap
                    curr_adj[u2] |= (1 << v2)
                    curr_adj[v2] |= (1 << u2)
                    
                    curr_set.remove((u1, v1))
                    curr_set.remove((v1, u1))
                    curr_set.add((u2, v2))
                    curr_set.add((v2, u2))
                    
                    # Update lists
                    # Remove e1 from edges, add e2
                    curr_edges.remove(e1)
                    curr_edges.append([u2, v2])
                    
                    # Remove e2 from non_edges, add e1
                    # Find index of e2 in curr_non_edges
                    for j in range(len(curr_non_edges)):
                        if curr_non_edges[j] == e2:
                            curr_non_edges.pop(j)
                            break
                    curr_non_edges.append(e1)
                else:
                    # Revert removal of e1
                    curr_adj[u1] = old_u1
                    curr_adj[v1] = old_v1

    return best_edges
# EVOLVE-BLOCK-END
