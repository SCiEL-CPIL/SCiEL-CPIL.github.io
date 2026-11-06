import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    n = 18
    
    def edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def has_k4(adj):
        """Check if graph contains a K4 using bitmask operations."""
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b]
                for c in range(b + 1, n):
                    if not ((common >> c) & 1):
                        continue
                    d_candidates = common & adj[c]
                    d_candidates &= ~((1 << (c + 1)) - 1)
                    if d_candidates:
                        return True
        return False

    def max_triangle_free_size(adj, time_limit=5.0):
        """Find largest induced triangle-free subgraph using branch and bound."""
        best = [0]
        t0 = time.time()
        timeout = [False]
        
        # Order vertices by degree descending to prune faster?
        deg = [bin(adj[v]).count('1') for v in range(n)]
        order = sorted(range(n), key=lambda v: -deg[v])
        
        def backtrack(idx, current_mask, current_size):
            if timeout[0]:
                return
            if time.time() - t0 > time_limit:
                timeout[0] = True
                return
            if idx == n:
                if current_size > best[0]:
                    best[0] = current_size
                return
            if current_size + (n - idx) <= best[0]:
                return
            
            v = order[idx]
            
            # Option 1: Exclude v
            backtrack(idx + 1, current_mask, current_size)
            
            # Option 2: Include v
            if not timeout[0]:
                N_v_mask = adj[v]
                C = current_mask & N_v_mask
                
                # Check if subgraph induced by C has any edge
                tmp_c = C
                valid = True
                while tmp_c:
                    bit = tmp_c & (-tmp_c)
                    u = bit.bit_length() - 1
                    tmp_c ^= bit
                    if adj[u] & C:
                        valid = False
                        break
                
                if valid:
                    backtrack(idx + 1, current_mask | (1 << v), current_size + 1)
        
        backtrack(0, 0, 0)
        return best[0]

    def construct_paley_17():
        """Construct Paley(17) graph on vertices 0..16."""
        adj = [0] * 17
        residues = {1, 4, 9, 16, 8, 2, 15, 13}
        for i in range(17):
            for j in range(i + 1, 17):
                diff = (j - i) % 17
                if diff in residues:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    # Step 1: Get Paley(17)
    adj_17 = construct_paley_17()
    
    # Step 2: Find all max triangle-free sets in Paley(17)
    # We know T(Paley(17)) is likely 11.
    # Let's find one or all.
    # For efficiency, we just need to find a set of constraints.
    # Actually, we can just try to construct N(18) using a heuristic or search.
    
    # Heuristic: Try to find a triangle-free set N in Paley(17) that "hits" all max triangle-free sets.
    # Since finding ALL max triangle-free sets might be slow, we can use a randomized approach.
    # Or, we can just try a few candidates for N(18).
    
    # Candidate N(18) strategies:
    # 1. N(18) = Empty set (Isolated 18). T=12.
    # 2. N(18) = Single vertex.
    # 3. N(18) = Edge (2 vertices).
    # 4. N(18) = Larger triangle-free set.
    
    # We want to minimize T(G).
    # If we find N such that T(G)=11, we win.
    
    # Let's try to search for N.
    # Constraint: N must be triangle-free in Paley(17).
    # Objective: For all max triangle-free sets S in Paley(17), S ∩ N is not independent.
    
    # To check "S ∩ N is not independent", we can just check if there is an edge in N that is also in S?
    # No, we need an edge (u,v) in Paley(17) such that u,v in N AND u,v in S.
    # So N must contain an edge from S.
    
    # Let's find a few max triangle-free sets in Paley(17).
    # We can use the branch and bound to find one.
    # To find multiple, we can forbid the found set and repeat.
    
    max_ts = []
    adj_temp = adj_17[:]
    # Find first max TF set
    # We need a function to return the set, not just size.
    def find_max_tf_set(adj, n, time_limit=2.0):
        best_set = [0]
        t0 = time.time()
        timeout = [False]
        deg = [bin(adj[v]).count('1') for v in range(n)]
        order = sorted(range(n), key=lambda v: -deg[v])
        
        def backtrack(idx, current_mask, current_size):
            if timeout[0]:
                return
            if time.time() - t0 > time_limit:
                timeout[0] = True
                return
            if idx == n:
                if current_size > bin(best_set[0]).count('1'):
                    best_set[0] = current_mask
                return
            if current_size + (n - idx) <= bin(best_set[0]).count('1'):
                return
            v = order[idx]
            backtrack(idx + 1, current_mask, current_size)
            if not timeout[0]:
                N_v_mask = adj[v]
                C = current_mask & N_v_mask
                tmp_c = C
                valid = True
                while tmp_c:
                    bit = tmp_c & (-tmp_c)
                    u = bit.bit_length() - 1
                    tmp_c ^= bit
                    if adj[u] & C:
                        valid = False
                        break
                if valid:
                    backtrack(idx + 1, current_mask | (1 << v), current_size + 1)
        
        backtrack(0, 0, 0)
        return best_set[0]
    
    # Find a few max TF sets to use as constraints
    forbidden_masks = []
    for _ in range(5):
        if time.time() > deadline - 50:
            break
        mask = find_max_tf_set(adj_17, 17, 1.0)
        if mask not in forbidden_masks:
            forbidden_masks.append(mask)
    
    # Now search for N(18)
    # N is a subset of 0..16.
    # We want N to be triangle-free in Paley(17).
    # And for each S in forbidden_masks, S ∩ N contains an edge of Paley(17).
    
    best_N = 0
    best_score = 12
    
    # Try small N first? Or random N?
    # Let's try random triangle-free subsets of Paley(17).
    # We can generate random triangle-free sets by sampling.
    
    random.seed(42)
    for trial in range(200):
        if time.time() > deadline - 30:
            break
        
        # Generate a random triangle-free set in Paley(17)
        N_mask = 0
        # Greedy random
        vertices = list(range(17))
        random.shuffle(vertices)
        for v in vertices:
            # Check if v can be added to N_mask
            C = N_mask & adj_17[v]
            if not C:
                # No neighbors in N_mask, so no edge?
                # Wait, if v has no neighbors in N_mask, then adding v doesn't create a triangle?
                # Triangle in N requires 3 vertices.
                # If we just want N to be triangle-free, we need to ensure no triangle in N.
                # Condition: For any u, w in N_mask, if u~w, then v cannot be connected to both?
                # Actually, if v is added, a triangle is formed if there are u, w in N_mask such that u~v, w~v, u~w.
                # So we need: The set of neighbors of v in N_mask must be an independent set in Paley(17).
                neigh_in_N = N_mask & adj_17[v]
                is_indep = True
                tmp = neigh_in_N
                while tmp:
                    bit = tmp & (-tmp)
                    u = bit.bit_length() - 1
                    tmp ^= bit
                    if adj_17[u] & neigh_in_N:
                        is_indep = False
                        break
                if is_indep:
                    N_mask |= (1 << v)
        
        # Check if this N satisfies the constraints
        valid_N = True
        for S_mask in forbidden_masks:
            inter = S_mask & N_mask
            # Check if inter contains an edge in Paley(17)
            has_edge = False
            tmp = inter
            while tmp:
                bit = tmp & (-tmp)
                u = bit.bit_length() - 1
                tmp ^= bit
                if adj_17[u] & inter:
                    has_edge = True
                    break
            if not has_edge:
                valid_N = False
                break
        
        if valid_N:
            # Construct full graph
            adj_full = adj_17[:] + [0]
            for v in range(17):
                if (N_mask >> v) & 1:
                    adj_full[17] |= (1 << v)
                    adj_full[v] |= (1 << 17)
            
            # Check K4-free
            if not has_k4(adj_full):
                score = max_triangle_free_size(adj_full, 1.0)
                if score < best_score:
                    best_score = score
                    best_N = N_mask
    
    # If we found a better N, use it. Otherwise, fall back to isolated 18.
    if best_score < 12:
        adj_final = adj_17[:] + [0]
        for v in range(17):
            if (best_N >> v) & 1:
                adj_final[17] |= (1 << v)
                adj_final[v] |= (1 << 17)
    else:
        adj_final = adj_17[:] + [0]
    
    # Final check
    if has_k4(adj_final):
        # Fallback to safe K_{6,6,6}
        adj_final = [0] * 18
        for i in range(6):
            for j in range(6, 12):
                adj_final[i] |= (1 << j)
                adj_final[j] |= (1 << i)
            for j in range(12, 18):
                adj_final[i] |= (1 << j)
                adj_final[j] |= (1 << i)
        for i in range(6, 12):
            for j in range(12, 18):
                adj_final[i] |= (1 << j)
                adj_final[j] |= (1 << i)
    
    return edges_from_adj(adj_final)
