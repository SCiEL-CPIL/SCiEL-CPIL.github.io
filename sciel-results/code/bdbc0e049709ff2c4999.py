import time
import random

def construct():
    d = 7
    n_vertices = 2 ** d
    n_edges = n_vertices * d // 2
    
    # Edge mapping
    # Edge (u, v) where u < v and u^v is power of 2
    edges = []
    edge_to_idx = {}
    idx_to_edge = []
    
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_to_idx[(u, v)] = idx
                idx_to_edge.append((u, v))
    
    # Precompute conflicts
    # For each edge, find the 6 squares it belongs to.
    # In each square, identify the other 3 edges.
    # If those 3 are present, adding this edge creates a C4.
    conflicts = [[] for _ in range(n_edges)]
    
    for idx, (u, v) in enumerate(idx_to_edge):
        diff = u ^ v
        dim_i = diff.bit_length() - 1
        
        for dim_j in range(d):
            if dim_j == dim_i:
                continue
            
            # Square in dimensions i and j
            # Vertices: u, v, u^2^j, v^2^j
            # Edges:
            # 1. (u, v) -> idx
            # 2. (u^2^j, v^2^j) -> opp
            # 3. (u, u^2^j) -> e1
            # 4. (v, v^2^j) -> e2
            
            u_j = u ^ (1 << dim_j)
            v_j = v ^ (1 << dim_j)
            
            # Get indices
            # Ensure u < v for lookup
            if u > u_j:
                e1 = (u_j, u)
            else:
                e1 = (u, u_j)
            
            if v > v_j:
                e2 = (v_j, v)
            else:
                e2 = (v, v_j)
            
            if u_j > v_j:
                e_opp = (v_j, u_j)
            else:
                e_opp = (u_j, v_j)
            
            idx_1 = edge_to_idx[e1]
            idx_2 = edge_to_idx[e2]
            idx_opp = edge_to_idx[e_opp]
            
            conflicts[idx].append((idx_opp, idx_1, idx_2))
            
    def can_add(idx, in_set):
        for opp, e1, e2 in conflicts[idx]:
            if in_set[opp] and in_set[e1] and in_set[e2]:
                return False
        return True

    def greedy_fill(in_set, order, start_time, deadline):
        added = 0
        for idx in order:
            if time.time() > deadline:
                break
            if not in_set[idx]:
                if can_add(idx, in_set):
                    in_set[idx] = True
                    added += 1
        return added

    best_count = 0
    best_set = [False] * n_edges
    
    deadline = time.time() + 110
    
    # Initial fixed greedy to get a baseline
    initial_order = list(range(n_edges))
    random.shuffle(initial_order)
    in_set = [False] * n_edges
    greedy_fill(in_set, initial_order, time.time(), deadline)
    best_count = sum(in_set)
    best_set = in_set[:]
    
    # Local Search
    while time.time() < deadline:
        # Start from best set
        in_set = best_set[:]
        current_count = best_count
        
        # Try to add any missing edge
        missing = [i for i in range(n_edges) if not in_set[i]]
        random.shuffle(missing)
        added_any = False
        for idx in missing:
            if time.time() > deadline:
                break
            if can_add(idx, in_set):
                in_set[idx] = True
                current_count += 1
                added_any = True
        if added_any:
            if current_count > best_count:
                best_count = current_count
                best_set = in_set[:]
            continue # Restart local search loop
        
        # If no edge can be added, try a kick
        # Pick a random edge to remove
        present = [i for i in range(n_edges) if in_set[i]]
        if not present:
            break
        remove_idx = random.choice(present)
        in_set[remove_idx] = False
        current_count -= 1
        
        # Try to add back as many as possible
        missing = [i for i in range(n_edges) if not in_set[i]]
        random.shuffle(missing)
        greedy_fill(in_set, missing, time.time(), deadline)
        current_count = sum(in_set)
        
        if current_count > best_count:
            best_count = current_count
            best_set = in_set[:]
            
    # Convert best_set to list of edges
    result = []
    for i in range(n_edges):
        if best_set[i]:
            result.append(list(idx_to_edge[i]))
            
    return result
