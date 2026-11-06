import time
import random
from itertools import combinations

def check_non_2_colorable(n, edges):
    """
    Returns True if the hypergraph is NOT 2-colorable.
    Uses optimized backtracking with unit propagation.
    """
    if not edges:
        return False
    
    # Build incidence structure
    vertex_to_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_to_edges[v].append(i)
    
    m = len(edges)
    # edge_state: list of [count0, count1, unassigned]
    # We track for each edge how many vertices are color 0, color 1, and unassigned
    edge_state = [[0, 0, 5] for _ in range(m)]
    colors = [-1] * n
    
    def backtrack():
        # Unit propagation: check for forced moves and contradictions
        changed = True
        while changed:
            changed = False
            for i in range(m):
                c0, c1, c_un = edge_state[i]
                if c0 == 5 or c1 == 5:
                    return False  # Contradiction
                if c0 == 4 and c1 == 0:
                    # All unassigned vertices in this edge must be color 1
                    for v in edges[i]:
                        if colors[v] == -1:
                            colors[v] = 1
                            edge_state[i][1] += 1
                            edge_state[i][2] -= 1
                            for j in vertex_to_edges[v]:
                                if j != i:
                                    edge_state[j][1] += 1
                                    edge_state[j][2] -= 1
                            changed = True
                elif c1 == 4 and c0 == 0:
                    # All unassigned vertices in this edge must be color 0
                    for v in edges[i]:
                        if colors[v] == -1:
                            colors[v] = 0
                            edge_state[i][0] += 1
                            edge_state[i][2] -= 1
                            for j in vertex_to_edges[v]:
                                if j != i:
                                    edge_state[j][0] += 1
                                    edge_state[j][2] -= 1
                            changed = True
            
            # Check for contradictions again after propagation
            for i in range(m):
                c0, c1, c_un = edge_state[i]
                if c0 == 5 or c1 == 5:
                    return False
        
        # Find an unassigned vertex
        # Use a heuristic: pick the vertex involved in the most constrained edge
        # Or simply pick the first unassigned vertex for simplicity, but let's try a better heuristic
        
        # Heuristic: pick the unassigned vertex that appears in the edge with the smallest unassigned count
        min_unassigned = 6
        target_v = -1
        for i in range(m):
            c0, c1, c_un = edge_state[i]
            if c_un > 0 and c_un < min_unassigned:
                min_unassigned = c_un
                # Pick the first unassigned vertex in this edge
                for v in edges[i]:
                    if colors[v] == -1:
                        target_v = v
                        break
                if target_v != -1:
                    break
        
        if target_v == -1:
            # All vertices assigned, check if valid
            # If we get here without contradiction, it's a valid coloring
            return True
        
        # Try color 0
        saved_colors = []
        saved_edge_state = []
        for v in [target_v]:
            saved_colors.append(colors[v])
        for j in vertex_to_edges[target_v]:
            saved_edge_state.append(edge_state[j][:])
        
        colors[target_v] = 0
        for j in vertex_to_edges[target_v]:
            edge_state[j][0] += 1
            edge_state[j][2] -= 1
        
        if backtrack():
            return True
        
        # Restore
        colors[target_v] = -1
        for j, state in zip(vertex_to_edges[target_v], saved_edge_state):
            edge_state[j] = state
        
        # Try color 1
        saved_colors = []
        saved_edge_state = []
        for v in [target_v]:
            saved_colors.append(colors[v])
        for j in vertex_to_edges[target_v]:
            saved_edge_state.append(edge_state[j][:])
        
        colors[target_v] = 1
        for j in vertex_to_edges[target_v]:
            edge_state[j][1] += 1
            edge_state[j][2] -= 1
        
        if backtrack():
            return True
        
        # Restore
        colors[target_v] = -1
        for j, state in zip(vertex_to_edges[target_v], saved_edge_state):
            edge_state[j] = state
        
        return False
    
    return not backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    best_edges = None
    best_n = 0
    best_m = float('inf')
    
    # Start with a known non-2-colorable hypergraph
    # K_9^5 is definitely non-2-colorable
    n_init = 9
    init_edges = [list(e) for e in combinations(range(n_init), 5)]
    
    if check_non_2_colorable(n_init, init_edges):
        best_edges = init_edges
        best_n = n_init
        best_m = len(init_edges)
    else:
        # Fallback
        n_init = 10
        init_edges = [list(e) for e in combinations(range(n_init), 5)]
        if check_non_2_colorable(n_init, init_edges):
            best_edges = init_edges
            best_n = n_init
            best_m = len(init_edges)
        else:
            return {'n': 0, 'edges': []}
    
    current_n = best_n
    current_edges = best_edges[:]
    current_m = len(current_edges)
    
    # Simulated Annealing
    temp = 1.0
    cooling = 0.9995
    min_temp = 0.01
    
    all_possible_edges = set(tuple(e) for e in combinations(range(current_n), 5))
    
    while time.time() < deadline and temp > min_temp:
        # Generate a neighbor
        move_type = random.random()
        
        if move_type < 0.5 and current_m > 0:
            # Remove an edge
            idx = random.randint(0, current_m - 1)
            e_to_remove = current_edges[idx]
            new_edges = current_edges[:idx] + current_edges[idx+1:]
            
            if check_non_2_colorable(current_n, new_edges):
                current_edges = new_edges
                current_m -= 1
                if current_m < best_m:
                    best_m = current_m
                    best_edges = current_edges[:]
                    best_n = current_n
            else:
                # Rejected, but we might still try to swap
                pass
        
        elif move_type < 0.8:
            # Swap: remove one edge, add another
            if current_m > 0:
                idx = random.randint(0, current_m - 1)
                e_to_remove = current_edges[idx]
                current_set = set(tuple(e) for e in current_edges)
                available_adds = [e for e in all_possible_edges if e not in current_set]
                
                if available_adds:
                    e_to_add = random.choice(available_adds)
                    new_edges = [e for i, e in enumerate(current_edges) if i != idx]
                    new_edges.append(list(e_to_add))
                    
                    if check_non_2_colorable(current_n, new_edges):
                        current_edges = new_edges
                        # m unchanged
                    else:
                        pass
        
        else:
            # Add an edge (increase m) - only if we're stuck
            current_set = set(tuple(e) for e in current_edges)
            available_adds = [e for e in all_possible_edges if e not in current_set]
            
            if available_adds:
                e_to_add = random.choice(available_adds)
                new_edges = current_edges[:] + [list(e_to_add)]
                
                if check_non_2_colorable(current_n, new_edges):
                    # Accept with probability based on temperature
                    prob = min(1.0, temp)
                    if random.random() < prob:
                        current_edges = new_edges
                        current_m += 1
        
        temp *= cooling
    
    # Try other n values if time remains
    for n_try in [10, 11, 12, 13]:
        if time.time() > deadline - 10:
            break
        if n_try == current_n:
            continue
            
        init_edges_try = [list(e) for e in combinations(range(n_try), 5)]
        if not check_non_2_colorable(n_try, init_edges_try):
            continue
            
        cur_n = n_try
        cur_edges = init_edges_try[:]
        cur_m = len(cur_edges)
        
        t_local = 1.0
        cool_local = 0.9995
        min_t_local = 0.01
        
        all_possible_try = set(tuple(e) for e in combinations(range(n_try), 5))
        
        while time.time() < deadline - 5 and t_local > min_t_local:
            move_type = random.random()
            
            if move_type < 0.5 and cur_m > 0:
                idx = random.randint(0, cur_m - 1)
                e_to_remove = cur_edges[idx]
                new_edges = cur_edges[:idx] + cur_edges[idx+1:]
                
                if check_non_2_colorable(cur_n, new_edges):
                    cur_edges = new_edges
                    cur_m -= 1
                    if cur_m < best_m:
                        best_m = cur_m
                        best_edges = cur_edges[:]
                        best_n = cur_n
            
            elif move_type < 0.8:
                if cur_m > 0:
                    idx = random.randint(0, cur_m - 1)
                    e_to_remove = cur_edges[idx]
                    current_set = set(tuple(e) for e in cur_edges)
                    available_adds = [e for e in all_possible_try if e not in current_set]
                    
                    if available_adds:
                        e_to_add = random.choice(available_adds)
                        new_edges = [e for i, e in enumerate(cur_edges) if i != idx]
                        new_edges.append(list(e_to_add))
                        
                        if check_non_2_colorable(cur_n, new_edges):
                            cur_edges = new_edges
            
            else:
                current_set = set(tuple(e) for e in cur_edges)
                available_adds = [e for e in all_possible_try if e not in current_set]
                
                if available_adds:
                    e_to_add = random.choice(available_adds)
                    new_edges = cur_edges[:] + [list(e_to_add)]
                    
                    if check_non_2_colorable(cur_n, new_edges):
                        prob = min(1.0, t_local)
                        if random.random() < prob:
                            cur_edges = new_edges
                            cur_m += 1
            
            t_local *= cool_local
    
    # Final verification
    if best_edges is not None:
        if check_non_2_colorable(best_n, best_edges):
            return {'n': best_n, 'edges': best_edges}
    
    return {'n': best_n, 'edges': best_edges if best_edges else []}

if __name__ == "__main__":
    res = construct()
    print(f"Found hypergraph with n={res['n']}, m={len(res['edges'])}")
    assert check_non_2_colorable(res['n'], res['edges']), "Result is 2-colorable!"
    print("Valid.")
