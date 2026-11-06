import time
import random
from itertools import combinations

def is_2_colorable_fast(n, edges):
    """
    Optimized 2-colorability check using backtracking with unit propagation.
    """
    if not edges:
        return True
    
    num_edges = len(edges)
    
    # Convert edges to bitmasks
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
        
    # Precompute vertex -> list of edges
    vertex_edges = [[] for _ in range(n)]
    for i, em in enumerate(edge_masks):
        mm = em
        while mm:
            lsb = mm & -mm
            v = lsb.bit_length() - 1
            vertex_edges[v].append(i)
            mm ^= lsb
            
    # State
    c0 = [0] * num_edges
    c1 = [0] * num_edges
    color = [-1] * n
    
    def propagate():
        changed = True
        while changed:
            changed = False
            for i in range(num_edges):
                if c0[i] == 5 or c1[i] == 5:
                    return False
                
                if c0[i] == 4 and c1[i] == 0:
                    # Find unassigned vertex in edge i
                    mm = edge_masks[i]
                    unassigned_v = -1
                    while mm:
                        lsb = mm & -mm
                        v = lsb.bit_length() - 1
                        if color[v] == -1:
                            unassigned_v = v
                            break
                        mm ^= lsb
                    if unassigned_v != -1:
                        color[unassigned_v] = 1
                        c1[i] += 1
                        for other_e in vertex_edges[unassigned_v]:
                            c1[other_e] += 1
                        changed = True
                        if c1[i] == 5:
                            return False
                elif c1[i] == 4 and c0[i] == 0:
                    mm = edge_masks[i]
                    unassigned_v = -1
                    while mm:
                        lsb = mm & -mm
                        v = lsb.bit_length() - 1
                        if color[v] == -1:
                            unassigned_v = v
                            break
                        mm ^= lsb
                    if unassigned_v != -1:
                        color[unassigned_v] = 0
                        c0[i] += 1
                        for other_e in vertex_edges[unassigned_v]:
                            c0[other_e] += 1
                        changed = True
                        if c0[i] == 5:
                            return False
        return True

    def backtrack():
        # Find the most constrained edge with unassigned vertices
        best_edge = -1
        best_unassigned = 6
        best_max_color = 0
        
        for i in range(num_edges):
            if c0[i] == 5 or c1[i] == 5:
                return False
            unassigned = 5 - c0[i] - c1[i]
            if unassigned == 0:
                continue
            max_color = max(c0[i], c1[i])
            
            if unassigned < best_unassigned or (unassigned == best_unassigned and max_color > best_max_color):
                best_unassigned = unassigned
                best_max_color = max_color
                best_edge = i
                
        if best_edge == -1:
            # All edges fully assigned (or satisfied)
            return True
            
        # Get unassigned vertices in best_edge
        candidates = []
        mm = edge_masks[best_edge]
        while mm:
            lsb = mm & -mm
            v = lsb.bit_length() - 1
            if color[v] == -1:
                candidates.append(v)
            mm ^= lsb
            
        # Try assigning each candidate
        for v in candidates:
            # Try color 0
            saved_c0 = c0[:]
            saved_c1 = c1[:]
            
            color[v] = 0
            ok = True
            for e_idx in vertex_edges[v]:
                c0[e_idx] += 1
                if c0[e_idx] == 5:
                    ok = False
                    break
            
            if ok and propagate():
                if backtrack():
                    return True
            
            color[v] = -1
            c0[:] = saved_c0
            c1[:] = saved_c1
            
            # Try color 1
            saved_c0 = c0[:]
            saved_c1 = c1[:]
            
            color[v] = 1
            ok = True
            for e_idx in vertex_edges[v]:
                c1[e_idx] += 1
                if c1[e_idx] == 5:
                    ok = False
                    break
            
            if ok and propagate():
                if backtrack():
                    return True
            
            color[v] = -1
            c0[:] = saved_c0
            c1[:] = saved_c1
            
        return False

    return backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 140 # 10s buffer
    
    best_edges = None
    best_n = 0
    best_m = 10**9
    
    # Structural Change: Simulated Annealing
    # State: set of edges (represented as list of tuples for hashing)
    # Fitness: -len(edges) if non-2-colorable, else large penalty
    
    n = 20 # Start with a reasonable n
    all_possible = set(combinations(range(n), 5))
    
    # Initialize with a random non-2-colorable set if possible, or just random
    # To ensure we start valid, we can try to find a small non-2-colorable set first
    # Or just start with a complete graph and let SA remove edges?
    # SA with "Add/Remove/Swap" is better.
    
    # Let's try n=12, 13, 14 as well.
    for n_trial in [12, 13, 14, 15]:
        if time.time() > deadline: break
        
        all_possible = set(combinations(range(n_trial), 5))
        # Start with a random subset of size ~60
        current_edges = set(random.sample(list(all_possible), min(60, len(all_possible))))
        
        current_m = len(current_edges)
        # Check if it's non-2-colorable
        # If it is, we have a valid start. If not, we penalize.
        
        T = 1.0
        T_min = 0.01
        cooling = 0.99
        
        while T > T_min and time.time() < deadline:
            # Generate neighbor
            move_type = random.random()
            new_edges = set(current_edges)
            
            if move_type < 0.4:
                # Remove an edge
                if current_edges:
                    e_to_remove = random.choice(list(current_edges))
                    new_edges.remove(e_to_remove)
            elif move_type < 0.8:
                # Add an edge
                if len(all_possible) > len(current_edges):
                    # Pick a random edge not in current
                    # Efficient way: pick random from all, if in current, pick another
                    # Or maintain a list of missing edges?
                    # For simplicity, just pick random from all_possible
                    # If collision, try a few times
                    for _ in range(10):
                        e_to_add = random.choice(list(all_possible))
                        if e_to_add not in current_edges:
                            new_edges.add(e_to_add)
                            break
            else:
                # Swap
                if current_edges and len(all_possible) > len(current_edges):
                    e_to_remove = random.choice(list(current_edges))
                    for _ in range(10):
                        e_to_add = random.choice(list(all_possible))
                        if e_to_add not in current_edges:
                            new_edges.remove(e_to_remove)
                            new_edges.add(e_to_add)
                            break
            
            if new_edges == current_edges:
                continue
                
            # Evaluate
            is_colorable_new = is_2_colorable_fast(n_trial, [list(e) for e in new_edges])
            is_colorable_old = is_2_colorable_fast(n_trial, [list(e) for e in current_edges]) # Cache this?
            
            # We should cache the colorability of the current state to avoid re-checking
            # Let's assume we track `current_colorable`
            
            # Correction: We need to track the state of the current best and the current SA state.
            # Let's restructure the loop to keep `current_colorable` and `current_m`.
            
            # This part is getting complex. Let's simplify:
            # We will just do a local search with random restarts, but using SWAPS.
            # Greedy removal is bad. Swap allows us to trade one edge for another to "unlock" removals.
            
            # Actually, the simplest effective structural change is:
            # 1. Find a valid non-2-colorable hypergraph (e.g. via greedy removal from K_n).
            # 2. Use "Edge Swapping" local search: Replace edge E1 with E2.
            #    If the new set is non-2-colorable, accept.
            #    This keeps the number of edges constant but changes the structure.
            #    Then, try to remove edges again.
            #    Repeat.
            
            pass # Placeholder for the actual logic below
            
    # Re-implementing construct with the "Swap + Remove" strategy
    
    best_edges = None
    best_m = 10**9
    best_n = 0
    
    for n in range(10, 16):
        if time.time() > deadline: break
        
        all_possible = list(combinations(range(n), 5))
        
        # 1. Get an initial non-2-colorable set
        # Start with complete
        current_edges = set(all_possible)
        # Greedy remove
        order = list(range(len(current_edges)))
        random.shuffle(order)
        # This is slow. Let's just pick a random subset and check.
        # If we find a non-2-colorable set of size M, we start from there.
        
        # Better: Start with a known small non-2-colorable set if we had one.
        # Since we don't, we use the greedy removal to get a "seed" for SA.
        
        # Greedy removal to get a local minimum
        edges_list = [list(e) for e in all_possible]
        m = len(edges_list)
        # Fast greedy: try removing each edge once
        for i in range(m-1, -1, -1):
            if time.time() > deadline: break
            # Try remove i
            new_list = edges_list[:i] + edges_list[i+1:]
            if not is_2_colorable_fast(n, new_list):
                edges_list = new_list
                m -= 1
                
        if m < best_m:
            best_m = m
            best_edges = edges_list[:]
            best_n = n
            
        # 2. Local Search with Swaps
        # Current set: edges_list
        # Goal: Find a set of same size or smaller that is non-2-colorable,
        # but allows further removals.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try random swaps
            for _ in range(50): # 50 swaps per iteration
                if time.time() > deadline: break
                if not edges_list: break
                
                # Pick an edge to remove
                idx_remove = random.randint(0, len(edges_list)-1)
                e_remove = edges_list[idx_remove]
                
                # Pick an edge to add (from all_possible not in edges_list)
                # Efficient: pick random from all_possible
                # If it's in edges_list, skip
                e_add = None
                for _ in range(20):
                    candidate = random.choice(all_possible)
                    if candidate not in edges_list:
                        e_add = candidate
                        break
                        
                if e_add is None:
                    continue
                    
                # Perform swap
                new_edges_list = edges_list[:]
                new_edges_list[idx_remove] = list(e_add)
                
                # Check if non-2-colorable
                if not is_2_colorable_fast(n, new_edges_list):
                    # Accept swap
                    edges_list = new_edges_list
                    
                    # Now try to remove edges again (Greedy)
                    m_current = len(edges_list)
                    # Try to remove one edge
                    for i in range(m_current-1, -1, -1):
                        if time.time() > deadline: break
                        new_list = edges_list[:i] + edges_list[i+1:]
                        if not is_2_colorable_fast(n, new_list):
                            edges_list = new_list
                            m_current -= 1
                            improved = True
                            # If we improved, break inner removal loop to restart swap loop?
                            # Or continue removing?
                            # Let's continue removing until no more can be removed.
                            # But this is O(M^2).
                            # Let's just do one removal and then restart the swap loop.
                            break
                    
                    if m_current < best_m:
                        best_m = m_current
                        best_edges = edges_list[:]
                        best_n = n
                    break # Restart the "for _ in range(50)" loop?
                    # Actually, if we accepted a swap and removed an edge, we have improved.
                    # We should restart the "improved" loop to try more swaps on the new state.
            
    if best_edges is None:
        # Fallback
        n = 9
        best_edges = [list(e) for e in combinations(range(n), 5)]
        best_n = n
        
    return {'n': best_n, 'edges': best_edges}
