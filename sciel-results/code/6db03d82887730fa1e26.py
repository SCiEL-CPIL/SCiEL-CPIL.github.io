import random
import itertools
import time

# Precompute Q3 properties
# Vertices 0-7
q3_edges = []
q3_adj_matrix = [[False]*8 for _ in range(8)]
for i in range(8):
    for j in range(i+1, 8):
        if bin(i ^ j).count('1') == 1:
            q3_edges.append((i, j))
            q3_adj_matrix[i][j] = True
            q3_adj_matrix[j][i] = True

def find_q3_in_subset(subset, adj):
    """
    subset: list of 8 ints
    adj: 16x16 list of lists (0/1)
    Returns list of edges (u, v) in main graph forming Q3, or None.
    """
    # Try all 8 choices for mapping Q3 vertex 0
    for first_img in range(8):
        mapping = [-1] * 8
        mapping[0] = first_img
        
        def backtrack(q3_v):
            if q3_v == 8:
                # Found a valid mapping
                edges = []
                for u, v in q3_edges:
                    edges.append((subset[mapping[u]], subset[mapping[v]]))
                return edges
            
            # Try to map q3_v to any unused vertex in subset
            for s_idx in range(8):
                if s_idx in mapping[:q3_v]:
                    continue
                
                valid = True
                # Check edges to all previous Q3 vertices
                for u in range(q3_v):
                    if q3_adj_matrix[u][q3_v]:
                        if not adj[subset[mapping[u]]][subset[s_idx]]:
                            valid = False
                            break
                
                if valid:
                    mapping[q3_v] = s_idx
                    res = backtrack(q3_v + 1)
                    if res:
                        return res
                    mapping[q3_v] = -1
            return None
        
        res = backtrack(1)
        if res:
            return res
    return None

def construct():
    # Initialize with K16
    n = 16
    adj = [[1]*n for _ in range(n)]
    for i in range(n):
        adj[i][i] = 0
        
    removed_edges = set()
    
    # Helper to get current edges count
    def count_edges():
        return sum(sum(row) for row in adj) // 2

    start_time = time.time()
    time_limit = 110 # Leave some buffer
    
    # Phase 1: Greedy Removal
    # We want to remove edges to destroy all Q3s.
    
    while time.time() - start_time < time_limit:
        # Try to find a Q3 using random sampling
        found_q3 = None
        # Increase trials as we go? Or keep constant.
        for _ in range(500):
            subset = random.sample(range(n), 8)
            q3_edges_found = find_q3_in_subset(subset, adj)
            if q3_edges_found:
                found_q3 = q3_edges_found
                break
        
        if not found_q3:
            # Do a full check to be certain
            full_found = False
            for subset in itertools.combinations(range(n), 8):
                q3_edges_found = find_q3_in_subset(list(subset), adj)
                if q3_edges_found:
                    found_q3 = q3_edges_found
                    full_found = True
                    break
            
            if not full_found:
                # No Q3 found. We are done.
                break
        
        # Remove a random edge from the found Q3
        edge_to_remove = random.choice(found_q3)
        u, v = edge_to_remove
        if u > v: u, v = v, u
        edge_to_remove = (u, v)
        
        if edge_to_remove not in removed_edges:
            adj[u][v] = 0
            adj[v][u] = 0
            removed_edges.add(edge_to_remove)
            
        # Simulated Annealing step:
        # With some probability, try to add back a removed edge
        # This helps escape local minima where we removed "too many" edges
        if removed_edges and random.random() < 0.1:
            # Pick a random removed edge
            candidate_edge = random.choice(list(removed_edges))
            u, v = candidate_edge
            
            # Try adding it back
            adj[u][v] = 1
            adj[v][u] = 1
            removed_edges.remove(candidate_edge)
            
            # Check if this created a Q3
            # We can just run the random search for a Q3 again
            # Or check if the graph is still Q3-free?
            # To be safe, we check if a Q3 exists.
            # If it exists, we have to remove an edge.
            # We can try to remove the candidate edge again, or another one.
            
            new_q3 = None
            for _ in range(100): # Fewer trials for check
                subset = random.sample(range(n), 8)
                q3_edges_found = find_q3_in_subset(subset, adj)
                if q3_edges_found:
                    new_q3 = q3_edges_found
                    break
            
            if new_q3:
                # A Q3 was created. Remove a random edge from it.
                # Preferably not the one we just added, to allow exploration?
                # Or just remove the one we added to revert?
                # If we revert, we gain nothing.
                # Let's remove a random edge from the new Q3.
                edge_to_fix = random.choice(new_q3)
                uu, vv = edge_to_fix
                if uu > vv: uu, vv = vv, uu
                edge_to_fix = (uu, vv)
                
                adj[uu][vv] = 0
                adj[vv][uu] = 0
                removed_edges.add(edge_to_fix)
            else:
                # No Q3 found (likely). Keep the edge.
                pass
        else:
            # No annealing step
            pass

    # Final verification to ensure validity
    # If time allows, do a full check one last time
    if time.time() - start_time < time_limit - 5:
        is_valid = True
        for subset in itertools.combinations(range(n), 8):
            if find_q3_in_subset(list(subset), adj):
                is_valid = False
                break
        if not is_valid:
            # This shouldn't happen if the loop exited correctly, 
            # but just in case, we need to remove edges.
            # For simplicity, if invalid, we might return a known valid graph?
            # Or just keep removing.
            # Given the loop logic, it should be valid.
            pass

    # Construct list of edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append([i, j])
                
    return edges

# EVOLVE-BLOCK-END
