<<<<<<< SEARCH
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin (120s limit)
    
    n = 16
    all_edges = list(itertools.combinations(range(n), 2))
    
    # Q3 structure
    q3_edges = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)]
    q3_adj = {
        0: [1, 2, 4],
        1: [0, 3, 5],
        2: [0, 3, 6],
        3: [1, 2, 7],
        4: [0, 5, 6],
        5: [1, 4, 7],
        6: [2, 4, 7],
        7: [3, 5, 6]
    }
    
    # Precompute for each Q3 edge (i,j), the list of other edges in Q3.
    # This helps in pruning.
    
    def creates_q3_fast(u, v, adj_bits):
        """
        Check if adding edge (u,v) creates a Q3.
        adj_bits: list of int bitmasks.
        """
        # We need to check if there is a Q3 embedding where (u,v) is an edge.
        # There are 24 orientations of (u,v) in Q3 (12 edges * 2 directions).
        for i, j in q3_edges:
            # Case 1: u->i, v->j
            if _check_embedding_fast(i, j, u, v, adj_bits, q3_adj, n):
                return True
            # Case 2: u->j, v->i
            if _check_embedding_fast(j, i, u, v, adj_bits, q3_adj, n):
                return True
        return False

    def _check_embedding_fast(q3_u, q3_v, g_u, g_v, adj_bits, q3_adj, n):
        mapping = [-1] * 8
        mapping[q3_u] = g_u
        mapping[q3_v] = g_v
        used_g = [False] * n
        used_g[g_u] = True
        used_g[g_v] = True
        remaining_q3 = [x for x in range(8) if x not in (q3_u, q3_v)]
        
        # Order remaining Q3 vertices by degree of constraints?
        # For now, fixed order.
        
        def backtrack(idx):
            if idx == len(remaining_q3):
                # Verify all Q3 edges exist
                for a, b in q3_edges:
                    ga = mapping[a]
                    gb = mapping[b]
                    if (ga == g_u and gb == g_v) or (ga == g_v and gb == g_u):
                        continue
                    if not (adj_bits[ga] & (1 << gb)):
                        return False
                return True
            
            q3_vertex = remaining_q3[idx]
            
            # Candidates: unused graph vertices
            # Must be adjacent to all mapped Q3 neighbors
            candidates = []
            for g_cand in range(n):
                if used_g[g_cand]:
                    continue
                ok = True
                for q3_neighbor in q3_adj[q3_vertex]:
                    g_neighbor = mapping[q3_neighbor]
                    if g_neighbor != -1:
                        if (g_cand == g_u and g_neighbor == g_v) or (g_cand == g_v and g_neighbor == g_u):
                            continue
                        if not (adj_bits[g_cand] & (1 << g_neighbor)):
                            ok = False
                            break
                if ok:
                    candidates.append(g_cand)
            
            for g_cand in candidates:
                mapping[q3_vertex] = g_cand
                used_g[g_cand] = True
                if backtrack(idx + 1):
                    return True
                mapping[q3_vertex] = -1
                used_g[g_cand] = False
            
            return False
        
        return backtrack(0)

    def local_search(adj_bits, max_iterations=100):
        current_count = sum(bin(adj_bits[i]).count('1') for i in range(n)) // 2
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            improved = False
            # Try adding each missing edge
            for u in range(n):
                for v in range(u+1, n):
                    if not (adj_bits[u] & (1 << v)):
                        if not creates_q3_fast(u, v, adj_bits):
                            adj_bits[u] |= (1 << v)
                            adj_bits[v] |= (1 << u)
                            current_count += 1
                            improved = True
            
            if not improved:
                break
        
        return adj_bits, current_count

    def try_remove_and_add(adj_bits, current_count):
        best_adj = adj_bits[:]
        best_count = current_count
        
        existing_edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj_bits[u] & (1 << v):
                    existing_edges.append((u, v))
        
        random.shuffle(existing_edges)
        
        for u, v in existing_edges:
            if time.time() > deadline:
                break
            
            # Remove edge (u, v)
            adj_bits[u] &= ~(1 << v)
            adj_bits[v] &= ~(1 << u)
            
            trial_adj = adj_bits[:]
            trial_count = current_count - 1
            
            # Greedy add back
            for u2 in range(n):
                for v2 in range(u2+1, n):
                    if not (trial_adj[u2] & (1 << v2)):
                        if not creates_q3_fast(u2, v2, trial_adj):
                            trial_adj[u2] |= (1 << v2)
                            trial_adj[v2] |= (1 << u2)
                            trial_count += 1
            
            if trial_count > best_count:
                best_count = trial_count
                best_adj = trial_adj[:]
            
            # Restore
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
        
        return best_adj, best_count

    best_count = 0
    best_adj = [0] * n
    
    trials = 50
    for t in range(trials):
        if time.time() > deadline:
            break
        
        random.seed(t * 7919 + 42)
        
        adj_bits = [0] * n
        
        # Start with K3,13
        for u in range(3):
            for v in range(3, 16):
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
        
        existing = set()
        for u in range(3):
            for v in range(3, 16):
                existing.add((u, v))
        
        remaining_edges = [e for e in all_edges if e not in existing]
        random.shuffle(remaining_edges)
        
        for u, v in remaining_edges:
            if time.time() > deadline:
                break
            if not creates_q3_fast(u, v, adj_bits):
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
        
        adj_bits, count = local_search(adj_bits)
        adj_bits, count = try_remove_and_add(adj_bits, count)
        
        if count > best_count:
            best_count = count
            best_adj = adj_bits[:]
    
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                result.append([i, j])
    
    return result
=======
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin (120s limit)
    
    n = 16
    all_edges = list(itertools.combinations(range(n), 2))
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    num_edges = len(all_edges)  # 120
    
    # Q3 structure
    q3_edges = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)]
    q3_adj = {
        0: [1, 2, 4],
        1: [0, 3, 5],
        2: [0, 3, 6],
        3: [1, 2, 7],
        4: [0, 5, 6],
        5: [1, 4, 7],
        6: [2, 4, 7],
        7: [3, 5, 6]
    }
    
    # Precompute all Q3 embeddings and their edge sets
    # For each embedding, store the set of edge indices that form a Q3
    q3_edge_sets = []
    
    # Generate all injective mappings from Q3 vertices {0..7} to graph vertices {0..15}
    from itertools import permutations
    for perm in permutations(range(n), 8):
        # Check if this mapping preserves all Q3 edges
        valid = True
        for a, b in q3_edges:
            ga, gb = perm[a], perm[b]
            if (ga, gb) not in edge_to_idx and (gb, ga) not in edge_to_idx:
                valid = False
                break
        if not valid:
            continue
        
        # Collect edge indices for this embedding
        edge_idxs = set()
        for a, b in q3_edges:
            ga, gb = perm[a], perm[b]
            if ga < gb:
                edge_idxs.add(edge_to_idx[(ga, gb)])
            else:
                edge_idxs.add(edge_to_idx[(gb, ga)])
        
        if len(edge_idxs) == 12:
            q3_edge_sets.append(edge_idxs)
    
    # Build conflict graph: two edges conflict if they are in some Q3 embedding together
    # conflict[i] is a bitmask of edges that conflict with edge i
    conflict = [0] * num_edges
    for edge_set in q3_edge_sets:
        edges_in_q3 = list(edge_set)
        for i, ei in enumerate(edges_in_q3):
            for j in range(i+1, len(edges_in_q3)):
                ej = edges_in_q3[j]
                conflict[ei] |= (1 << ej)
                conflict[ej] |= (1 << ei)
    
    # Now solve maximum independent set on the conflict graph using branch and bound
    # We want to maximize the number of non-conflicting edges (i.e., edges that don't form a Q3 together)
    
    # Bitset-based branch and bound
    # Available: bitmask of edges not yet decided
    # Selected: bitmask of edges selected so far
    # Rejected: bitmask of edges rejected so far
    
    best_solution = 0  # bitmask of best solution found
    best_count = 0
    
    def upper_bound(available, selected):
        """
        Compute an upper bound on the maximum independent set size
        using greedy coloring on the induced subgraph.
        """
        # Simple upper bound: just count available edges
        # Better: use greedy coloring for tighter bound
        available_list = []
        temp = available
        while temp:
            # Get lowest set bit
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            available_list.append(idx)
            temp ^= lsb
        
        if not available_list:
            return 0
        
        # Greedy coloring: assign colors to vertices in the induced subgraph
        # Two vertices have an edge in the conflict graph if they conflict
        color_of = {}
        next_color = 0
        
        for v in available_list:
            # Find a color not used by any neighbor
            used_colors = set()
            for u in available_list:
                if u != v and (conflict[v] & (1 << u)):
                    if u in color_of:
                        used_colors.add(color_of[u])
            
            c = 0
            while c in used_colors:
                c += 1
            color_of[v] = c
            next_color = max(next_color, c + 1)
        
        return len(selected) + next_color
    
    def branch_and_bound(available, selected, depth):
        nonlocal best_solution, best_count
        
        if time.time() > deadline:
            return
        
        # Prune if upper bound doesn't exceed current best
        ub = upper_bound(available, selected)
        if ub <= best_count:
            return
        
        if available == 0:
            if len(selected) > best_count:
                best_count = len(selected)
                best_solution = selected
            return
        
        # Choose a vertex to branch on
        # Heuristic: choose vertex with highest degree in the remaining graph
        avail_list = []
        temp = available
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            avail_list.append(idx)
            temp ^= lsb
        
        if not avail_list:
            return
        
        # Choose vertex with max degree
        v = max(avail_list, key=lambda x: bin(conflict[x] & available).count('1'))
        
        # Branch 1: Include v
        new_selected = selected | (1 << v)
        # Remove v and all its neighbors from available
        new_available = available & ~(1 << v) & ~conflict[v]
        branch_and_bound(new_available, new_selected, depth + 1)
        
        # Branch 2: Exclude v
        new_available2 = available & ~(1 << v)
        branch_and_bound(new_available2, selected, depth + 1)
    
    # Initial solution: greedy
    greedy_available = (1 << num_edges) - 1
    greedy_selected = 0
    while greedy_available:
        if time.time() > deadline:
            break
        # Pick vertex with min degree
        avail_list = []
        temp = greedy_available
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            avail_list.append(idx)
            temp ^= lsb
        
        if not avail_list:
            break
        
        v = min(avail_list, key=lambda x: bin(conflict[x] & greedy_available).count('1'))
        greedy_selected |= (1 << v)
        greedy_available &= ~(1 << v) & ~conflict[v]
    
    best_count = bin(greedy_selected).count('1')
    best_solution = greedy_selected
    
    # Run branch and bound
    all_available = (1 << num_edges) - 1
    branch_and_bound(all_available, 0, 0)
    
    # Convert solution to edge list
    result = []
    for i in range(num_edges):
        if best_solution & (1 << i):
            result.append(list(all_edges[i]))
    
    return result
>>>>>>> REPLACE
