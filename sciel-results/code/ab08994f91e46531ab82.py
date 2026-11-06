import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Canonical Q3 structure
    # Vertices 0-7
    # Edges:
    # 0-1, 0-2, 0-4
    # 1-3, 1-5
    # 2-3, 2-6
    # 3-7
    # 4-5, 4-6
    # 5-7
    # 6-7
    
    def find_q3_with_edge(adj, u, v):
        """
        Check if adding edge (u,v) to graph adj creates a Q3 containing (u,v).
        adj is a list of 16 integers representing adjacency bitmasks.
        Returns True if a Q3 is found, False otherwise.
        """
        n_u = adj[u]
        n_v = adj[v]
        
        # Get list of neighbors of u and v
        # Using list comprehension is fast enough for 16 elements
        neigh_u = [i for i in range(16) if (n_u >> i) & 1]
        neigh_v = [i for i in range(16) if (n_v >> i) & 1]
        
        used = [False] * 16
        used[u] = True
        used[v] = True
        
        # Assign 2 (neighbor of u)
        for v2 in neigh_u:
            if used[v2]: continue
            used[v2] = True
            
            # Assign 4 (neighbor of u, distinct from 2)
            for v4 in neigh_u:
                if used[v4]: continue
                used[v4] = True
                
                # Assign 3 (neighbor of v AND neighbor of v2)
                # Candidates: in neigh_v AND in adj[v2]
                # We can iterate neigh_v and check adj[v2]
                for v3 in neigh_v:
                    if used[v3]: continue
                    if not ((adj[v2] >> v3) & 1): continue
                    used[v3] = True
                    
                    # Assign 5 (neighbor of v AND neighbor of v4)
                    for v5 in neigh_v:
                        if used[v5]: continue
                        if not ((adj[v4] >> v5) & 1): continue
                        used[v5] = True
                        
                        # Assign 6 (neighbor of v2 AND neighbor of v4)
                        # Candidates: in adj[v2] AND in adj[v4]
                        # Let's compute common bits
                        common_24 = adj[v2] & adj[v4]
                        
                        # We need to pick v6 from common_24 that is not used
                        # Iterate through bits of common_24
                        temp_common = common_24
                        while temp_common:
                            # Get lowest set bit
                            lsb = temp_common & -temp_common
                            v6 = lsb.bit_length() - 1
                            temp_common -= lsb
                            
                            if used[v6]: continue
                            used[v6] = True
                            
                            # Assign 7 (neighbor of v3, v5, v6)
                            # Candidates: in adj[v3] AND adj[v5] AND adj[v6]
                            common_356 = adj[v3] & adj[v5] & adj[v6]
                            
                            # Check if any bit in common_356 is not used
                            # We can just check if (common_356 & ~used_mask) != 0
                            # But used is a list. Let's construct a mask.
                            # Or just iterate bits of common_356
                            temp_356 = common_356
                            found = False
                            while temp_356:
                                lsb = temp_356 & -temp_356
                                v7 = lsb.bit_length() - 1
                                temp_356 -= lsb
                                if not used[v7]:
                                    found = True
                                    break
                            
                            if found:
                                # Unwind
                                used[v6] = False
                                used[v5] = False
                                used[v3] = False
                                used[v4] = False
                                used[v2] = False
                                return True
                            used[v6] = False
                            
                        used[v5] = False
                    used[v3] = False
                used[v4] = False
            used[v2] = False
        return False

    best_edges = set()
    best_count = 0
    
    random.seed(42)
    
    # Strategy: Multiple restarts of Greedy Add + Simulated Annealing
    
    for attempt in range(50):
        if time.time() > deadline - 10:
            break
            
        adj = [0] * 16
        edges = set()
        
        # 1. Greedy Initialization
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        
        for u, v in shuffled_edges:
            if time.time() > deadline - 10:
                break
            if not find_q3_with_edge(adj, u, v):
                edges.add((u, v))
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
        current_count = len(edges)
        if current_count > best_count:
            best_count = current_count
            best_edges = set(edges)
            
        # 2. Simulated Annealing Refinement
        # We work on a copy of the state
        sa_adj = adj[:]
        sa_edges = set(edges)
        sa_count = current_count
        
        T = 5.0
        cooling = 0.995
        no_improve_count = 0
        max_no_improve = 2000
        
        # Precompute list of all edges for random selection
        # We need to pick from missing and existing
        
        while time.time() < deadline - 5 and no_improve_count < max_no_improve:
            
            # Generate Move
            move_type = random.random()
            
            if move_type < 0.5: # Add
                # Pick random missing edge
                missing = [e for e in all_edges if e not in sa_edges]
                if not missing: continue
                u, v = random.choice(missing)
                
                if not find_q3_with_edge(sa_adj, u, v):
                    # Valid add
                    sa_edges.add((u, v))
                    sa_adj[u] |= (1 << v)
                    sa_adj[v] |= (1 << u)
                    sa_count += 1
                    no_improve_count = 0
                    if sa_count > best_count:
                        best_count = sa_count
                        best_edges = set(sa_edges)
                else:
                    no_improve_count += 1
                    
            elif move_type < 0.7: # Remove
                # Pick random existing edge
                if not sa_edges: continue
                u, v = random.choice(list(sa_edges))
                
                # Delta E = -1. Accept with prob exp(1/T)
                if random.random() < (1.0 / T): # Approximation for exp(1/T) when T is small? 
                    # Actually exp(-delta/T) = exp(1/T)
                    # If T is small, exp(1/T) is large.
                    # Standard SA: accept if exp(-delta_E / T) > random.
                    # delta_E = -1. So exp(1/T).
                    # If T < 1, exp(1/T) > 1, so always accept.
                    # If T is large, exp(1/T) ~ 1.
                    # Let's just use: accept if random() < min(1.0, math.exp(1.0/T))
                    # To avoid import math, use:
                    # if T > 1: prob = 1.0
                    # else: prob = 1.0 (since exp(1/T) >= 1 for T>0)
                    # Wait. If we remove an edge, score decreases.
                    # We should accept with probability exp(-(-1)/T) = exp(1/T).
                    # Since T > 0, exp(1/T) >= 1.
                    # So we ALWAYS accept a removal? That would destroy the graph.
                    # 
                    # Correction:
                    # We want to maximize edges.
                    # E = -|E|.
                    # Move Remove: |E| -> |E|-1. E -> E+1. Delta E = +1.
                    # Accept if exp(-1/T) > random.
                    # This is correct.
                    
                    pass # Logic handled below
                    
                # Let's redo the acceptance logic clearly
                delta_e = 1.0 # Increase in energy (bad)
                # Accept if random < exp(-delta_e / T)
                # If T is large, exp(-1/T) ~ 1.
                # If T is small, exp(-1/T) ~ 0.
                if random.random() < (2.718281828 ** (-1.0 / T)): # e^(-1/T)
                    sa_edges.remove((u, v))
                    sa_adj[u] &= ~(1 << v)
                    sa_adj[v] &= ~(1 << u)
                    sa_count -= 1
                    no_improve_count += 1
            else: # Swap
                # Pick random existing and random missing
                if not sa_edges: continue
                missing = [e for e in all_edges if e not in sa_edges]
                if not missing: continue
                
                u1, v1 = random.choice(list(sa_edges))
                u2, v2 = random.choice(missing)
                
                # Remove u1-v1
                sa_edges.remove((u1, v1))
                sa_adj[u1] &= ~(1 << v1)
                sa_adj[v1] &= ~(1 << u1)
                
                # Try to add u2-v2
                if not find_q3_with_edge(sa_adj, u2, v2):
                    # Valid swap
                    sa_edges.add((u2, v2))
                    sa_adj[u2] |= (1 << v2)
                    sa_adj[v2] |= (1 << u2)
                    # Count unchanged
                    no_improve_count = 0 # Reset? Or not? 
                    # Swaps don't improve count, but change structure.
                    # Let's not reset no_improve for swaps, but cool down slower?
                    # Actually, if we do a swap, we might enable future adds.
                    # Let's treat it as a neutral step.
                else:
                    # Invalid swap, revert
                    sa_edges.add((u1, v1))
                    sa_adj[u1] |= (1 << v1)
                    sa_adj[v1] |= (1 << u1)
                    no_improve_count += 1
            
            T *= cooling
            if T < 0.1:
                # Restart temperature
                T = 5.0
                no_improve_count = 0
                
    result = [[u, v] for (u, v) in best_edges]
    return result
