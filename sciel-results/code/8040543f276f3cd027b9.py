import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = 110  # Leave 10s margin
    
    n = 16
    # Structure:
    # 0 is hub
    # 1..9 are neighbors of 0 (N(u))
    # 10..15 are non-neighbors (W)
    # Edges:
    # (0, i) for i in 1..9
    # Edges in V' = 1..15, count 56
    # Constraint: No edges within 1..9
    
    N_u = list(range(1, 10))
    W = list(range(10, 16))
    
    # Possible edges in V'
    # All pairs (a,b) in V' such that not (a in N_u and b in N_u)
    possible_edges = []
    for i in range(1, 16):
        for j in range(i+1, 16):
            if i in N_u and j in N_u:
                continue
            possible_edges.append((i, j))
            
    # We need to select 56 edges from possible_edges
    # Initial random selection
    random.shuffle(possible_edges)
    current_edges_Vp = set(possible_edges[:56])
    
    # Fixed edges
    fixed_edges = [(0, i) for i in N_u]
    
    def get_full_edges(edges_Vp):
        return fixed_edges + list(edges_Vp)
        
    def calculate_score(edges_Vp):
        # Build adjacency list for V'
        adj = {i: set() for i in range(1, 16)}
        for u, v in edges_Vp:
            adj[u].add(v)
            adj[v].add(u)
            
        # Calculate degrees in G
        deg = [0] * 16
        deg[0] = 9
        for v in N_u:
            deg[v] = 1 + len(adj[v])
        for w in W:
            deg[w] = len(adj[w])
            
        max_sum = 0
        # Find triangles in V'
        # Iterate over edges in V'
        for u, v in edges_Vp:
            # Common neighbors
            common = adj[u] & adj[v]
            for w in common:
                # Triangle (u, v, w)
                s = deg[u] + deg[v] + deg[w]
                if s > max_sum:
                    max_sum = s
        return max_sum
        
    current_score = calculate_score(current_edges_Vp)
    best_edges = current_edges_Vp.copy()
    best_score = current_score
    
    # Local search
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Restart or perturb
            # Perturb: swap 2 edges
            if len(current_edges_Vp) > 0:
                edges_list = list(current_edges_Vp)
                random.shuffle(edges_list)
                # Remove 2
                e1, e2 = edges_list[:2]
                current_edges_Vp.remove(e1)
                current_edges_Vp.remove(e2)
                # Add 2 from non-edges
                non_edges = [e for e in possible_edges if e not in current_edges_Vp]
                random.shuffle(non_edges)
                if len(non_edges) >= 2:
                    current_edges_Vp.add(non_edges[0])
                    current_edges_Vp.add(non_edges[1])
                else:
                    # Revert if not enough non-edges (should not happen)
                    current_edges_Vp.add(e1)
                    current_edges_Vp.add(e2)
            no_improve_count = 0
            continue
            
        # Pick a random edge to remove
        edges_list = list(current_edges_Vp)
        if not edges_list:
            break
        e_remove = random.choice(edges_list)
        u_r, v_r = e_remove
        
        # Pick a random non-edge to add
        non_edges = [e for e in possible_edges if e not in current_edges_Vp]
        if not non_edges:
            break
        e_add = random.choice(non_edges)
        u_a, v_a = e_add
        
        # Apply
        new_edges_Vp = current_edges_Vp.copy()
        new_edges_Vp.remove(e_remove)
        new_edges_Vp.add(e_add)
        
        # Evaluate
        new_score = calculate_score(new_edges_Vp)
        
        if new_score < current_score:
            current_edges_Vp = new_edges_Vp
            current_score = new_score
            no_improve_count = 0
            if current_score < best_score:
                best_score = current_score
                best_edges = current_edges_Vp.copy()
        else:
            no_improve_count += 1
            
    # Construct final graph
    final_edges = fixed_edges + list(best_edges)
    return {'n': 16, 'edges': final_edges}
