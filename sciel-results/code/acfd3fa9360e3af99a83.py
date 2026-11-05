import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n = 20
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_edges = len(all_edges)
    
    # Map edge index to (u, v) for fast lookup
    edge_map = {}
    for i, (u, v) in enumerate(all_edges):
        edge_map[(u, v)] = i
        edge_map[(v, u)] = i

    def is_valid(adj_mat):
        # C = adj @ adj.T gives counts of common neighbours for all pairs
        common_counts = adj_mat @ adj_mat.T
        np.fill_diagonal(common_counts, 0)
        return np.max(common_counts) <= 2

    def get_edge_count(adj_mat):
        return int(np.sum(adj_mat)) // 2

    best_adj = np.zeros((n, n), dtype=np.uint8)
    best_score = 0
    
    # Strategy: Joint Simulated Annealing on two subgraphs G1 and G2
    # G1 and G2 are edge-disjoint.
    # Constraint: G1 is C4-free, G2 is C4-free.
    # This implies G = G1 U G2 is K2,3-free.
    
    num_restarts = 50 # More restarts possible with faster SA steps?
    
    for restart_idx in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize G1 and G2 as empty
        G1 = np.zeros((n, n), dtype=np.uint8)
        G2 = np.zeros((n, n), dtype=np.uint8)
        
        # Initial random seed? Or empty?
        # Let's start empty and let SA add edges.
        # Or start with a greedy C4-free graph?
        # Greedy C4-free is easier to enforce.
        
        # Let's try a simple heuristic initialization:
        # Add edges to G1 until no more C4-free edges can be added.
        # Then add edges to G2 from the remaining edges, ensuring C4-free.
        
        # Helper to check if adding edge (u,v) to a C4-free graph keeps it C4-free
        # C4-free means no two vertices have > 1 common neighbor.
        def can_add_c4_free(adj, u, v):
            # Common neighbors of u and v in adj
            # If they already have 1 common neighbor, adding (u,v) doesn't change that count for the pair (u,v)?
            # Wait. C4-free means no cycle of length 4.
            # A C4 exists if there are u, v, w, x such that (u,v), (v,w), (w,x), (x,u) are edges.
            # Equivalent: no two vertices have 2 or more common neighbors.
            # If we add edge (u,v), we create new common neighbors for pairs involving u or v.
            # Specifically, if we add (u,v), then for any w, if (u,w) and (v,w) exist, w becomes a common neighbor of u and v.
            # But u and v were not connected.
            # The condition "no two vertices have > 1 common neighbor" must hold for ALL pairs.
            # Adding (u,v) increases the common neighbor count of (u,v) by 1? No.
            # Common neighbors of u and v are k such that (u,k) and (v,k) exist.
            # Adding (u,v) does not add new common neighbors for (u,v).
            # However, it adds u as a common neighbor of (v, k) for all k neighbor of u?
            # Let's check: Pair (v, k). Common neighbors are j such that (v,j) and (k,j) exist.
            # If we add (u,v), then u is a neighbor of v.
            # If u is also a neighbor of k, then u becomes a common neighbor of (v, k).
            # So, adding (u,v) is valid for C4-free if:
            # 1. (u,v) is not already an edge.
            # 2. For all k, if (u,k) exists, then (v,k) does NOT exist? No.
            #    If (u,k) and (v,k) exist, then k is a common neighbor of u and v.
            #    This doesn't violate C4-free for pair (u,v) unless they already had 1 common neighbor.
            #    Wait, C4-free constraint is on the pairs of vertices.
            #    Pair (u,v): If they have c common neighbors, adding (u,v) doesn't change c.
            #    Pair (v, k): u becomes a common neighbor if (u,k) exists.
            #    So if (v, k) already had 1 common neighbor, and (u,k) exists, then adding (u,v) makes (v,k) have 2 common neighbors. Violation.
            #    So condition: For all k, if (u,k) is an edge, then (v,k) must have 0 common neighbors?
            #    No, (v,k) common neighbors count increases by 1 if (u,k) is an edge.
            #    So we require: For all k, if (u,k) is an edge, then current_common_count(v, k) == 0.
            #    Similarly, for all k, if (v,k) is an edge, then current_common_count(u, k) == 0.
            
            # Let's compute common neighbor counts C = adj @ adj.T
            # We need C[v, k] == 0 for all k where adj[u, k] == 1 (k != v)
            # And C[u, k] == 0 for all k where adj[v, k] == 1 (k != u)
            
            # Optimization:
            # neighbors_u = np.where(adj[u] == 1)[0]
            # Check if any neighbor_u (excluding v) has C[v, k] > 0?
            # Actually, if C[v, k] >= 1, adding (u,v) makes it C[v, k] + 1.
            # If C[v, k] was 1, it becomes 2. Violation.
            # So we need C[v, k] == 0 for all k in neighbors_u.
            # And C[u, k] == 0 for all k in neighbors_v.
            
            if adj[u, v] == 1: return False
            
            C = adj @ adj.T
            np.fill_diagonal(C, 0)
            
            # Check neighbors of u
            # k such that adj[u, k] == 1 and k != v
            # We need C[v, k] == 0
            # Note: C[v, v] is 0.
            
            # Vectorized check
            # mask_u = (adj[u] == 1)
            # mask_u[v] = 0
            # if np.any(C[v, mask_u] > 0): return False
            
            # mask_v = (adj[v] == 1)
            # mask_v[u] = 0
            # if np.any(C[u, mask_v] > 0): return False
            
            # This is O(n^3) per check due to matmul.
            # We can maintain C incrementally.
            pass

        # Let's use a simpler initialization: Random sparse graphs?
        # Or just start empty and let SA do the work.
        
        curr_G1 = np.zeros((n, n), dtype=np.uint8)
        curr_G2 = np.zeros((n, n), dtype=np.uint8)
        
        # SA Parameters
        T = 10.0
        T_min = 0.1
        alpha = 0.999
        max_iter = 5000
        
        # Precompute edge indices for random selection
        edge_indices = list(range(num_edges))
        
        for i in range(max_iter):
            if time.time() > deadline:
                break
            
            # Pick a random edge
            e_idx = random.choice(edge_indices)
            u, v = all_edges[e_idx]
            
            # Determine which graph it belongs to, or if it's missing
            # State: 
            # 0: missing
            # 1: in G1
            # 2: in G2
            
            in_G1 = curr_G1[u, v]
            in_G2 = curr_G2[u, v]
            
            # Possible moves:
            # If missing: add to G1 or G2
            # If in G1: remove from G1, or move to G2
            # If in G2: remove from G2, or move to G1
            
            # Let's define a move as:
            # 1. Add edge to G1 (if missing)
            # 2. Add edge to G2 (if missing)
            # 3. Remove edge from G1 (if in G1)
            # 4. Remove edge from G2 (if in G2)
            # 5. Move edge from G1 to G2 (if in G1)
            # 6. Move edge from G2 to G1 (if in G2)
            
            # Randomly choose a valid move type
            moves = []
            if in_G1 == 0 and in_G2 == 0:
                moves = ['add_G1', 'add_G2']
            elif in_G1 == 1 and in_G2 == 0:
                moves = ['remove_G1', 'move_G1_to_G2']
            elif in_G1 == 0 and in_G2 == 1:
                moves = ['remove_G2', 'move_G2_to_G1']
            
            move = random.choice(moves)
            
            # Calculate delta in objective function
            # Objective: Maximize edges subject to C4-free constraints on G1 and G2.
            # We use a penalty method.
            # Score = (edges_G1 + edges_G2) - penalty * (violations_G1 + violations_G2)
            
            # To calculate penalty, we need to check C4-free constraints.
            # This is expensive if done from scratch.
            # Let's use a simpler objective:
            # Accept move if it maintains C4-free property.
            # This is a hard constraint SA.
            
            # Check if move is valid (maintains C4-free)
            valid = True
            
            if move == 'add_G1':
                # Check if adding (u,v) to G1 keeps it C4-free
                # We need C_G1[v, k] == 0 for all k in N_G1(u)
                # And C_G1[u, k] == 0 for all k in N_G1(v)
                C1 = curr_G1 @ curr_G1.T
                np.fill_diagonal(C1, 0)
                
                # Check neighbors of u in G1
                # k such that curr_G1[u, k] == 1 and k != v
                # We need C1[v, k] == 0
                # Note: If k == v, C1[v, v] is 0.
                
                # Vectorized
                mask_u = (curr_G1[u] == 1)
                mask_u[v] = 0
                if np.any(C1[v, mask_u] > 0):
                    valid = False
                
                if valid:
                    mask_v = (curr_G1[v] == 1)
                    mask_v[u] = 0
                    if np.any(C1[u, mask_v] > 0):
                        valid = False
            
            elif move == 'add_G2':
                C2 = curr_G2 @ curr_G2.T
                np.fill_diagonal(C2, 0)
                
                mask_u = (curr_G2[u] == 1)
                mask_u[v] = 0
                if np.any(C2[v, mask_u] > 0):
                    valid = False
                
                if valid:
                    mask_v = (curr_G2[v] == 1)
                    mask_v[u] = 0
                    if np.any(C2[u, mask_v] > 0):
                        valid = False
            
            elif move == 'remove_G1':
                valid = True # Always valid to remove
            elif move == 'remove_G2':
                valid = True
            elif move == 'move_G1_to_G2':
                # Remove from G1, add to G2
                # 1. Removing from G1 is always valid.
                # 2. Adding to G2 must be valid.
                # But wait, G1 and G2 are edge-disjoint.
                # If we move (u,v) from G1 to G2, we first remove it from G1.
                # Then we add it to G2.
                # The check for adding to G2 should be on the state AFTER removal from G1?
                # No, G2 is independent of G1.
                # So we just check if adding (u,v) to current G2 is valid.
                C2 = curr_G2 @ curr_G2.T
                np.fill_diagonal(C2, 0)
                
                mask_u = (curr_G2[u] == 1)
                mask_u[v] = 0
                if np.any(C2[v, mask_u] > 0):
                    valid = False
                
                if valid:
                    mask_v = (curr_G2[v] == 1)
                    mask_v[u] = 0
                    if np.any(C2[u, mask_v] > 0):
                        valid = False
            
            elif move == 'move_G2_to_G1':
                C1 = curr_G1 @ curr_G1.T
                np.fill_diagonal(C1, 0)
                
                mask_u = (curr_G1[u] == 1)
                mask_u[v] = 0
                if np.any(C1[v, mask_u] > 0):
                    valid = False
                
                if valid:
                    mask_v = (curr_G1[v] == 1)
                    mask_v[u] = 0
                    if np.any(C1[u, mask_v] > 0):
                        valid = False
            
            if not valid:
                # In hard constraint SA, we just reject.
                # Or we can use a penalty and accept with probability.
                # Let's stick to hard constraint for simplicity and correctness.
                # But this might get stuck.
                # Let's allow "illegal" moves with a penalty?
                # No, the final graph MUST be valid.
                # So we must ensure intermediate states are valid?
                # If we use hard constraints, we stay in the valid space.
                # If the valid space is disconnected, we might get stuck.
                # But C4-free graphs form a connected space under edge addition/removal?
                # Probably.
                
                # Let's try a soft constraint approach to allow exploration.
                # Score = edges - lambda * violations
                # This is complex to implement incrementally.
                
                # Let's just reject invalid moves.
                pass
            else:
                # Apply move
                if move == 'add_G1':
                    curr_G1[u, v] = 1
                    curr_G1[v, u] = 1
                elif move == 'add_G2':
                    curr_G2[u, v] = 1
                    curr_G2[v, u] = 1
                elif move == 'remove_G1':
                    curr_G1[u, v] = 0
                    curr_G1[v, u] = 0
                elif move == 'remove_G2':
                    curr_G2[u, v] = 0
                    curr_G2[v, u] = 0
                elif move == 'move_G1_to_G2':
                    curr_G1[u, v] = 0
                    curr_G1[v, u] = 0
                    curr_G2[u, v] = 1
                    curr_G2[v, u] = 1
                elif move == 'move_G2_to_G1':
                    curr_G2[u, v] = 0
                    curr_G2[v, u] = 0
                    curr_G1[u, v] = 1
                    curr_G1[v, u] = 1
                
                # Update best
                curr_adj = curr_G1 + curr_G2
                curr_score = get_edge_count(curr_adj)
                if curr_score > best_score:
                    best_score = curr_score
                    best_adj = curr_adj.copy()
            
            # Cool down
            T *= alpha
            if T < T_min:
                T = T_min

    # Final Greedy Fill on the best solution found
    if time.time() < deadline:
        final_adj = best_adj.copy()
        final_score = best_score
        improved = True
        while improved and time.time() < deadline:
            improved = False
            C = final_adj @ final_adj.T
            np.fill_diagonal(C, 0)
            
            shuffled_final = all_edges.copy()
            random.shuffle(shuffled_final)
            
            for u, v in shuffled_final:
                if final_adj[u, v] == 1:
                    continue
                    
                valid = True
                idx_v = np.where(C[:, v] == 2)[0]
                for i in idx_v:
                    if final_adj[i, u] == 1:
                        valid = False
                        break
                
                if valid:
                    idx_u = np.where(C[u, :] == 2)[0]
                    for j in idx_u:
                        if final_adj[v, j] == 1:
                            valid = False
                            break
                
                if valid:
                    final_adj[u, v] = 1
                    final_adj[v, u] = 1
                    final_score += 1
                    improved = True
                    break
        
        if final_score > best_score:
            best_score = final_score
            best_adj = final_adj

    # Convert back to list of edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                edges.append([u, v])
                
    return edges
# EVOLVE-BLOCK-END
