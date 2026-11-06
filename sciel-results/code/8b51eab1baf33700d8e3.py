import random
import time
import sys

# EVOLVE-BLOCK-START

def _build_graph(n, p, seed):
    """Build adjacency bitmasks from the same PRNG stream as the seed program."""
    rng = random.Random(seed)
    adj = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    return adj

def _popcount(x):
    """Count number of set bits in integer x."""
    return bin(x).count('1')

def _greedy_independent_set(adj, candidates, n):
    """
    Greedy heuristic to find an independent set within 'candidates'.
    Picks vertices with minimum degree in the induced subgraph.
    """
    is_set = 0
    rem = candidates
    while rem:
        # Find vertex with minimum degree in 'rem'
        min_deg = n
        v_idx = -1
        temp = rem
        while temp:
            v = (temp & -temp).bit_length() - 1
            temp &= temp - 1
            deg = _popcount(adj[v] & rem)
            if deg < min_deg:
                min_deg = deg
                v_idx = v
        if v_idx == -1:
            break
        is_set |= (1 << v_idx)
        # Remove v and its neighbors from remaining candidates
        rem &= ~(adj[v_idx] | (1 << v_idx))
    return is_set

def _local_search_is(adj, is_set, n, time_limit=5.0):
    """
    Local search to maximize the Independent Set.
    Tries to add vertices (if possible) and performs 1-1 swaps.
    """
    start_time = time.time()
    current_is = is_set
    current_size = _popcount(current_is)
    
    # Precompute neighbors for faster access
    # Note: adj is already bitmasks
    
    while True:
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        
        # 1. Try to add a vertex not in IS that has no neighbors in IS
        # This is a simple expansion step
        # Candidates: vertices not in current_is
        not_in_is = ((1 << n) - 1) & ~current_is
        # For a vertex v to be added, adj[v] & current_is must be 0
        
        # Heuristic: try vertices with fewest conflicts
        candidates_add = not_in_is
        best_add_v = -1
        min_conflicts = n + 1
        
        temp = candidates_add
        while temp:
            v = (temp & -temp).bit_length() - 1
            temp &= temp - 1
            conflicts = _popcount(adj[v] & current_is)
            if conflicts == 0:
                best_add_v = v
                min_conflicts = 0
                break
            elif conflicts < min_conflicts:
                min_conflicts = conflicts
                best_add_v = v
        
        if best_add_v != -1 and min_conflicts == 0:
            current_is |= (1 << best_add_v)
            current_size += 1
            improved = True
            # Continue loop to try adding more
            continue

        # 2. Try 1-1 Swaps
        # Remove v_in from IS, add v_out to IS if it results in a valid IS
        # We want to find a swap that increases size? No, 1-1 swap keeps size same.
        # But it might allow further additions later.
        # Actually, for maximizing size, we primarily rely on the "add" step.
        # If we can't add, we might be stuck in a local optimum.
        # A 1-1 swap doesn't increase size immediately.
        # A 2-1 swap (remove 1, add 2) increases size by 1.
        
        # Let's try 2-1 swaps: Remove v_in, add v_out1 and v_out2.
        # Condition: 
        # 1. v_in is in IS.
        # 2. v_out1, v_out2 are not in IS.
        # 3. v_out1 and v_out2 are not adjacent to each other.
        # 4. v_out1 and v_out2 are not adjacent to (IS \ {v_in}).
        
        is_mask = current_is
        temp_in = is_mask
        found_swap = False
        
        while temp_in and not found_swap:
            v_in = (temp_in & -temp_in).bit_length() - 1
            temp_in &= temp_in - 1
            
            is_without_in = is_mask & ~(1 << v_in)
            
            # Candidates for v_out: not in IS, not adjacent to v_in (since v_in is removed, 
            # actually v_out CAN be adjacent to v_in? No, v_in is removed from IS, 
            # so v_out only needs to be non-adjacent to the REMAINING IS).
            # Wait, if v_out is adjacent to v_in, it's fine because v_in is out.
            # So v_out just needs to be non-adjacent to (IS \ {v_in}).
            
            # Also v_out1 and v_out2 must not be adjacent to each other.
            
            # This search space is large O(n^3).
            # Let's restrict to a few attempts.
            
            # Find v_out1
            # v_out1 must not be in is_without_in
            # v_out1 must not be adjacent to is_without_in
            
            candidates_out = ((1 << n) - 1) & ~is_without_in
            # Filter: adj[v] & is_without_in == 0
            
            valid_outs = []
            temp_cand = candidates_out
            while temp_cand:
                v = (temp_cand & -temp_cand).bit_length() - 1
                temp_cand &= temp_cand - 1
                if _popcount(adj[v] & is_without_in) == 0:
                    valid_outs.append(v)
            
            if len(valid_outs) < 2:
                continue
                
            # Try pairs from valid_outs
            # To keep it fast, limit checks
            for i in range(len(valid_outs)):
                for j in range(i + 1, len(valid_outs)):
                    v1 = valid_outs[i]
                    v2 = valid_outs[j]
                    # Check if v1 and v2 are adjacent
                    if not (adj[v1] & (1 << v2)):
                        # Valid swap: remove v_in, add v1, v2
                        # New IS: is_without_in | (1<<v1) | (1<<v2)
                        # Size increases by 1
                        current_is = is_without_in | (1 << v1) | (1 << v2)
                        current_size += 1
                        improved = True
                        found_swap = True
                        break
                if found_swap:
                    break
            
        if not improved:
            break
            
    return current_is, current_size

def _max_independent_set_bnb(adj, n, time_limit=10.0, initial_set=0, initial_size=0):
    """
    Find the Maximum Independent Set (MIS) using Branch and Bound.
    """
    start_time = time.time()
    best_size = initial_size
    best_set = initial_set
    
    # Global deadline check
    def check_time():
        return time.time() - start_time > time_limit

    # Recursive B&B
    # State: current_set (bitmask), candidates (bitmask of vertices that can still be added)
    def bnb(current_set, candidates):
        nonlocal best_size, best_set
        
        if check_time():
            return
            
        if candidates == 0:
            if _popcount(current_set) > best_size:
                best_size = _popcount(current_set)
                best_set = current_set
            return
        
        # Bound: Max possible size is current_size + popcount(candidates)
        cur_size = _popcount(current_set)
        if cur_size + _popcount(candidates) <= best_size:
            return
            
        # Update incumbent with greedy solution on candidates
        # This is cheap and often improves the bound significantly
        g_set = _greedy_independent_set(adj, candidates, n)
        new_set = current_set | g_set
        new_size = _popcount(new_set)
        if new_size > best_size:
            best_size = new_size
            best_set = new_set
            
        # Re-check bound after incumbent update
        if cur_size + _popcount(candidates) <= best_size:
            return

        # Select a vertex to branch on
        # Heuristic: Pick vertex with highest degree in candidates? 
        # Or lowest? 
        # For MIS, picking a vertex with high degree might prune more?
        # Let's pick the vertex with the highest degree in the induced subgraph of candidates.
        v_idx = -1
        max_deg = -1
        temp = candidates
        while temp:
            v = (temp & -temp).bit_length() - 1
            temp &= temp - 1
            deg = _popcount(adj[v] & candidates)
            if deg > max_deg:
                max_deg = deg
                v_idx = v
        
        if v_idx == -1:
            return
            
        v_bit = 1 << v_idx
        
        # Branch 1: Include v_idx
        # New candidates: candidates minus v_idx and its neighbors
        new_candidates = candidates & ~(adj[v_idx] | v_bit)
        bnb(current_set | v_bit, new_candidates)
        
        if check_time():
            return

        # Branch 2: Exclude v_idx
        # New candidates: candidates minus v_idx
        new_candidates = candidates & ~v_bit
        bnb(current_set, new_candidates)

    # Initial candidates: all vertices
    all_vertices = (1 << n) - 1
    bnb(initial_set, all_vertices & ~initial_set)
    
    return best_size, best_set

def _min_vertex_cover(adj, n, time_budget=10.0):
    """
    Find Minimum Vertex Cover by finding Maximum Independent Set.
    MVC = V \ MIS.
    """
    # Step 1: Greedy IS
    all_vertices = (1 << n) - 1
    is_set, is_size = _greedy_independent_set(adj, all_vertices, n)
    
    # Step 2: Local Search to improve IS
    # Use a small time slice for local search
    ls_time = min(2.0, time_budget / 4.0)
    is_set, is_size = _local_search_is(adj, is_set, n, time_limit=ls_time)
    
    # Step 3: B&B to find optimal MIS
    bnb_time = time_budget - ls_time - 0.5
    if bnb_time < 0.5:
        bnb_time = 0.5
        
    best_size, best_set = _max_independent_set_bnb(
        adj, n, 
        time_limit=bnb_time, 
        initial_set=is_set, 
        initial_size=is_size
    )
    
    # The vertex cover is the complement of the independent set
    vc_set = all_vertices & ~best_set
    
    # Convert bitmask to list of vertex indices
    vc_list = []
    temp = vc_set
    while temp:
        v = (temp & -temp).bit_length() - 1
        vc_list.append(v)
        temp &= temp - 1
        
    return sorted(vc_list)

def _validate_solution(adj, selected, n):
    """Check if selected vertices cover all edges."""
    selected_set = set(selected)
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                if u not in selected_set and v not in selected_set:
                    return False
    return True

def _greedy_vc(adj, n):
    """Fast greedy fallback for Minimum Vertex Cover."""
    degs = [bin(adj[i]).count('1') for i in range(n)]
    selected = []
    remaining_edges = adj[:]
    
    for _ in range(n):
        if not any(remaining_edges):
            break
        max_deg = -1
        max_v = -1
        for i in range(n):
            if degs[i] > max_deg:
                max_deg = degs[i]
                max_v = i
        if max_v == -1:
            break
        selected.append(max_v)
        # Update degrees and remove edges
        for j in range(n):
            if remaining_edges[max_v] & (1 << j):
                degs[j] -= 1
                remaining_edges[j] &= ~(1 << max_v)
        remaining_edges[max_v] = 0
        degs[max_v] = 0
    return sorted(selected)

def construct():
    deadline = time.time() + 140  # 10s safety margin before 150s kill
    
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    results = []
    
    # Allocate time budget per instance
    # n=30 is fast, n=40 is slower. 
    # Total budget ~140s.
    # Give proportional time.
    time_per_instance = 45.0
    
    for n, p, seed in instances:
        if time.time() > deadline - 5:
            # Fallback: return greedy if out of time
            adj = _build_graph(n, p, seed)
            results.append(_greedy_vc(adj, n))
            continue
            
        adj = _build_graph(n, p, seed)
        
        # Adjust time budget based on n
        if n <= 30:
            t_budget = 15.0
        elif n <= 35:
            t_budget = 25.0
        else:
            t_budget = 35.0
            
        # Ensure we don't exceed global deadline
        remaining_global = deadline - time.time() - 5
        t_budget = min(t_budget, remaining_global)
        
        if t_budget < 1.0:
             results.append(_greedy_vc(adj, n))
             continue

        solution = _min_vertex_cover(adj, n, time_budget=t_budget)
        
        # Validate just in case
        if not _validate_solution(adj, solution, n):
             solution = _greedy_vc(adj, n)

        results.append(solution)
        
    return results

# EVOLVE-BLOCK-END
