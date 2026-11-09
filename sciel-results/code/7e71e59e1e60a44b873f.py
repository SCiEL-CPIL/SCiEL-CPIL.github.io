import random
import itertools
import time
import math

# Precompute Q3 properties for exact check
q3_adj_matrix = [[False]*8 for _ in range(8)]
for i in range(8):
    for j in range(i+1, 8):
        if bin(i ^ j).count('1') == 1:
            q3_adj_matrix[i][j] = True
            q3_adj_matrix[j][i] = True

def contains_q3(subset, adj):
    """
    Checks if the subgraph induced by 'subset' (list of 8 vertex ids)
    contains a subgraph isomorphic to the 3-cube Q3.
    """
    n_sub = len(subset)
    if n_sub != 8:
        return False
        
    # Precompute degrees in induced subgraph
    degrees = [0]*8
    for i in range(8):
        for j in range(i+1, 8):
            if adj[subset[i]][subset[j]]:
                degrees[i] += 1
                degrees[j] += 1
                
    for d in degrees:
        if d < 3:
            return False
            
    edge_count = sum(degrees) // 2
    if edge_count < 12:
        return False
        
    # Precompute adjacency within subset for faster access
    adj_local = [[0]*8 for _ in range(8)]
    for i in range(8):
        for j in range(i+1, 8):
            if adj[subset[i]][subset[j]]:
                adj_local[i][j] = 1
                adj_local[j][i] = 1
                
    # Backtracking search for Q3 embedding
    mapping = [-1] * 8
    
    def backtrack(q3_v):
        if q3_v == 8:
            return True
        
        for s_idx in range(8):
            if s_idx in mapping[:q3_v]:
                continue
            
            valid = True
            for u in range(q3_v):
                if q3_adj_matrix[u][q3_v]:
                    if not adj_local[mapping[u]][s_idx]:
                        valid = False
                        break
            
            if valid:
                mapping[q3_v] = s_idx
                if backtrack(q3_v + 1):
                    return True
                mapping[q3_v] = -1
        return False
        
    for first_img in range(8):
        mapping[0] = first_img
        if backtrack(1):
            return True
        mapping[0] = -1
            
    return False

def construct():
    n = 16
    start_time = time.time()
    time_limit = 115
    
    # Precompute all 8-subsets
    all_subsets = list(itertools.combinations(range(n), 8))
    num_subsets = len(all_subsets)
    
    # Map subset to index
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # Precompute which subsets each edge belongs to
    # edge (u, v) -> list of subset indices
    edge_to_subsets = [[] for _ in range(n*n)] # Use u*n+v as key
    for u in range(n):
        for v in range(u+1, n):
            key = u*n + v
            # Find all subsets containing u and v
            # Subsets are combinations of 8 vertices.
            # We need 6 other vertices from the remaining 14.
            others = [x for x in range(n) if x != u and x != v]
            for comb in itertools.combinations(others, 6):
                subset = tuple(sorted(comb + (u, v)))
                idx = subset_to_idx[subset]
                edge_to_subsets[key].append(idx)
                
    # Initialize graph
    adj = [[0]*n for _ in range(n)]
    edge_count = 0
    
    # Start with a random graph, density ~0.5
    for u in range(n):
        for v in range(u+1, n):
            if random.random() < 0.5:
                adj[u][v] = 1
                adj[v][u] = 1
                edge_count += 1
                
    # Initialize subset degree sums
    subset_deg_sums = [0] * num_subsets
    for i, subset in enumerate(all_subsets):
        s = 0
        for a in range(8):
            for b in range(a+1, 8):
                if adj[subset[a]][subset[b]]:
                    s += 2
        subset_deg_sums[i] = s
        
    # Initial violation count (approximate: sum >= 24)
    viol_count = sum(1 for s in subset_deg_sums if s >= 24)
    
    # Simulated Annealing
    # Cost = viol_count * 1000 - edge_count
    # We want to minimize cost.
    T = 10.0
    T_min = 0.01
    cooling = 0.99
    
    current_cost = viol_count * 1000 - edge_count
    best_cost = current_cost
    best_adj = [row[:] for row in adj]
    best_edge_count = edge_count
    
    steps = 0
    max_steps = 50000
    
    while steps < max_steps and time.time() - start_time < time_limit - 10:
        steps += 1
        
        # Pick a random edge
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        key = u*n + v
        is_present = adj[u][v]
        
        # Calculate delta
        delta_edge_count = 1 if not is_present else -1
        
        # Calculate delta in viol_count
        delta_viol = 0
        for idx in edge_to_subsets[key]:
            old_sum = subset_deg_sums[idx]
            new_sum = old_sum + (2 if not is_present else -2)
            if old_sum < 24 and new_sum >= 24:
                delta_viol += 1
            elif old_sum >= 24 and new_sum < 24:
                delta_viol -= 1
        
        new_cost = (viol_count + delta_viol) * 1000 - (edge_count + delta_edge_count)
        delta_cost = new_cost - current_cost
        
        accept = False
        if delta_cost < 0:
            accept = True
        else:
            if T > T_min:
                prob = math.exp(-delta_cost / T)
                if random.random() < prob:
                    accept = True
                    
        if accept:
            # Apply move
            adj[u][v] = 0 if is_present else 1
            adj[v][u] = 0 if is_present else 1
            edge_count += delta_edge_count
            viol_count += delta_viol
            
            # Update subset_deg_sums
            for idx in edge_to_subsets[key]:
                subset_deg_sums[idx] += (2 if not is_present else -2)
                
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_adj = [row[:] for row in adj]
                best_edge_count = edge_count
                
        T *= cooling
        
    # Use the best graph found
    adj = best_adj
    edge_count = best_edge_count
    
    # Final Phase: Exact cleanup
    # Find all subsets with sum >= 24 and check for exact Q3
    # This list should be small
    candidate_subsets = []
    for i, s in enumerate(subset_deg_sums):
        if s >= 24:
            candidate_subsets.append(all_subsets[i])
            
    # Remove edges to destroy all Q3s
    # We iterate until no Q3 is found
    while time.time() - start_time < time_limit - 5:
        q3_found = False
        for subset in candidate_subsets:
            if contains_q3(subset, adj):
                q3_found = True
                # Find an edge in this subset to remove
                # Prefer edges that are in the Q3? 
                # We can just pick a random edge in the induced subgraph
                edges_in_subset = []
                for i in range(8):
                    for j in range(i+1, 8):
                        if adj[subset[i]][subset[j]]:
                            edges_in_subset.append((subset[i], subset[j]))
                if edges_in_subset:
                    u, v = random.choice(edges_in_subset)
                    adj[u][v] = 0
                    adj[v][u] = 0
                    edge_count -= 1
                    # Update subset_deg_sums for this edge
                    key = u*n + v
                    if u > v: u, v = v, u
                    key = u*n + v
                    for idx in edge_to_subsets[key]:
                        subset_deg_sums[idx] -= 2
                    # Update viol_count
                    # We don't strictly need viol_count anymore, but let's keep it consistent
                    # Actually, we can just break and restart candidate_subsets generation?
                    # Or just continue.
                    break
        if not q3_found:
            break
            
    # Regenerate candidate_subsets for the adding phase?
    # Or just try to add all missing edges and check.
    
    # Phase 2: Add edges
    missing_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if adj[u][v] == 0:
                missing_edges.append((u, v))
                
    random.shuffle(missing_edges)
    
    for u, v in missing_edges:
        if time.time() - start_time > time_limit - 2:
            break
            
        # Tentatively add
        adj[u][v] = 1
        adj[v][u] = 1
        
        # Check if it creates a Q3
        # Only check subsets containing u and v that have sum >= 24 (after adding)
        # We need to update the sums for subsets containing u,v
        key = u*n + v
        if u > v: u, v = v, u
        key = u*n + v
        
        creates_q3 = False
        # We need to check if ANY subset containing u,v now contains a Q3
        # But we only care about subsets that were "close" to being a Q3.
        # Let's just check all subsets containing u,v.
        # There are 3003 such subsets.
        # For each, we need to check if it contains a Q3.
        # This is expensive if done naively.
        # Optimization: Only check if the subset's degree sum (after adding) is >= 24.
        
        # Let's compute the sums for these subsets on the fly?
        # No, we have subset_deg_sums. But they are outdated for this edge.
        # Let's just check the candidates.
        
        # To speed this up, we can pre-filter:
        # A subset containing u,v can only form a Q3 if its degree sum is at least 24.
        # Before adding, sum was S. After adding, sum is S+2.
        # So we need S >= 22.
        
        for idx in edge_to_subsets[key]:
            if subset_deg_sums[idx] >= 22: # S+2 >= 24
                subset = all_subsets[idx]
                if contains_q3(subset, adj):
                    creates_q3 = True
                    break
                    
        if creates_q3:
            # Revert
            adj[u][v] = 0
            adj[v][u] = 0
        else:
            # Keep
            edge_count += 1
            # Update subset_deg_sums
            for idx in edge_to_subsets[key]:
                subset_deg_sums[idx] += 2

    # Construct list of edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append([i, j])
                
    return edges
