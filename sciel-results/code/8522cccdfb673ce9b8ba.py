import random
import time
import math
import numpy as np

def construct():
    d = 7
    n_vertices = 1 << d
    n_edges_total = n_vertices * d // 2  # 448
    
    start_time = time.time()
    time_limit = 115.0
    
    # 1. Precompute edges and C4s
    # Edge representation: (u, i) where u < u^(1<<i)
    edges = []
    edge_map = {}
    idx = 0
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edges.append((u, i))
                edge_map[(u, i)] = idx
                idx += 1
    
    # Helper to get normalized edge index
    def get_edge_idx(u, i):
        v = u ^ (1 << i)
        if u < v:
            return edge_map[(u, i)]
        else:
            return edge_map[(v, i)]

    # Precompute C4s
    # A C4 is identified by (u, i, j) with i < j. 
    # The 4 edges are:
    # e1: (u, i)
    # e2: (u ^ (1<<i), j)
    # e3: (u, j)
    # e4: (u ^ (1<<j), i)
    
    c4_edges = [] # List of tuples of 4 edge indices
    edge_to_c4s = [[] for _ in range(n_edges_total)]
    
    for u in range(n_vertices):
        for i in range(d):
            for j in range(i + 1, d):
                # Only count each C4 once. 
                # A C4 is defined by the set of 4 vertices. 
                # The "minimal" corner is min(u, u^i, u^j, u^i^j) ? 
                # Actually, iterating all u, i, j generates each C4 4 times (once per corner).
                # We can just generate all and deduplicate, or use a set.
                # For efficiency, let's just generate all 4 corners and check if we've seen the C4.
                # Or simpler: iterate u from 0 to 127, i from 0 to 6, j from i+1 to 6.
                # This generates 128 * 21 = 2688 C4s.
                # Each unique C4 is generated 4 times.
                # Total unique C4s = 2688 / 4 = 672.
                
                e1_idx = get_edge_idx(u, i)
                e2_idx = get_edge_idx(u ^ (1 << i), j)
                e3_idx = get_edge_idx(u, j)
                e4_idx = get_edge_idx(u ^ (1 << j), i)
                
                # To avoid duplicates, we can sort the 4 indices and use a set, 
                # but that's slow. 
                # Alternative: Only generate if u is the "minimum" vertex of the C4?
                # The vertices are u, u^i, u^j, u^i^j.
                # We can require that u < u^i, u < u^j, u < u^i^j? No, u is just one corner.
                # Let's just use a set of frozensets of edge indices to deduplicate.
                c4_key = frozenset([e1_idx, e2_idx, e3_idx, e4_idx])
                
                # We need to map c4_key to an index in c4_edges
                # To avoid O(N) lookup, we can just append and later deduplicate?
                # Or use a dict.
                
                # Actually, let's just generate all and deduplicate at the end.
                c4_edges.append(tuple(sorted([e1_idx, e2_idx, e3_idx, e4_idx])))
                
    # Deduplicate C4s
    unique_c4s = list(set(c4_edges))
    c4_edges = unique_c4s
    n_c4s = len(c4_edges)
    
    # Rebuild edge_to_c4s
    edge_to_c4s = [[] for _ in range(n_edges_total)]
    for c4_idx, c4 in enumerate(c4_edges):
        for e_idx in c4:
            edge_to_c4s[e_idx].append(c4_idx)

    # 2. State Management
    # selected: bytearray of length n_edges_total
    # c4_counts: numpy array of length n_c4s, int
    
    def init_state():
        selected = bytearray(n_edges_total)
        c4_counts = np.zeros(n_c4s, dtype=np.int8)
        return selected, c4_counts

    def add_edge(selected, c4_counts, e_idx):
        # Check validity
        for c4_idx in edge_to_c4s[e_idx]:
            if c4_counts[c4_idx] >= 3:
                return False
        selected[e_idx] = 1
        for c4_idx in edge_to_c4s[e_idx]:
            c4_counts[c4_idx] += 1
        return True

    def remove_edge(selected, c4_counts, e_idx):
        selected[e_idx] = 0
        for c4_idx in edge_to_c4s[e_idx]:
            c4_counts[c4_idx] -= 1

    def get_count(selected):
        return sum(selected)

    # 3. Simulated Annealing
    def run_sa(initial_selected, initial_c4_counts, time_budget):
        selected = initial_selected[:]
        c4_counts = initial_c4_counts.copy()
        current_count = sum(selected)
        
        # Temperature schedule
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.999
        
        start = time.time()
        
        while time.time() - start < time_budget:
            if T < T_min:
                T = T_min
            
            # Choose move
            r = random.random()
            if r < 0.5:
                # Add
                e_idx = random.randint(0, n_edges_total - 1)
                if not selected[e_idx]:
                    if add_edge(selected, c4_counts, e_idx):
                        current_count += 1
                    else:
                        # Failed to add, try again? Or just skip.
                        pass
            elif r < 0.7:
                # Remove
                e_idx = random.randint(0, n_edges_total - 1)
                if selected[e_idx]:
                    remove_edge(selected, c4_counts, e_idx)
                    current_count -= 1
            else:
                # Swap (1-for-1)
                # Pick a selected edge to remove
                e_out = random.randint(0, n_edges_total - 1)
                if selected[e_out]:
                    # Pick an unselected edge to add
                    e_in = random.randint(0, n_edges_total - 1)
                    if not selected[e_in]:
                        # Check if e_in is valid AFTER removing e_out
                        # Temporarily remove e_out
                        remove_edge(selected, c4_counts, e_out)
                        
                        if add_edge(selected, c4_counts, e_in):
                            # Swap successful. Count unchanged.
                            # Accept with prob 1 (Delta E = 0)
                            pass
                        else:
                            # Swap failed. Revert.
                            # We need to re-add e_out. It should be valid because it was valid before.
                            add_edge(selected, c4_counts, e_out)
                        
            T *= cooling_rate
            
        return selected, c4_counts, current_count

    # 4. Main Loop: Multi-restart SA
    best_count = 0
    best_selected = None
    best_c4_counts = None
    
    # Initial Greedy to get a decent start
    def greedy_fill(selected, c4_counts):
        order = list(range(n_edges_total))
        random.shuffle(order)
        count = 0
        for e_idx in order:
            if add_edge(selected, c4_counts, e_idx):
                count += 1
        return count

    # Run multiple restarts
    while time.time() - start_time < time_limit:
        # Budget for one SA run
        # If we have 100s left, give each SA run ~5-10s?
        # Let's give each run 2 seconds, and restart.
        # Or adaptive: if we haven't improved in a while, increase time.
        
        time_left = time_limit - (time.time() - start_time)
        if time_left < 2.0:
            break
            
        # Greedy initialization
        selected, c4_counts = init_state()
        greedy_fill(selected, c4_counts)
        
        # SA
        selected, c4_counts, count = run_sa(selected, c4_counts, min(time_left, 5.0))
        
        if count > best_count:
            best_count = count
            best_selected = selected
            best_c4_counts = c4_counts
            # Reset cooling? No, just restart.
            
    # 5. Final Polish
    # Try to add any remaining edges
    if best_selected is not None:
        selected = best_selected[:]
        c4_counts = best_c4_counts.copy()
        count = sum(selected)
        
        # Try all edges
        for e_idx in range(n_edges_total):
            if not selected[e_idx]:
                if add_edge(selected, c4_counts, e_idx):
                    count += 1
                    best_selected = selected
                    best_c4_counts = c4_counts
                    best_count = count

    # 6. Output
    out = []
    for idx, (u, i) in enumerate(edges):
        if best_selected[idx]:
            v = u ^ (1 << i)
            out.append([u, v])
            
    return out
