import numpy as np
import itertools
import time
import sys

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # 1. Generate candidates
    candidates = []
    candidate_types = [] # 1 for Type 1, 2 for Type 2
    
    # Type 1: One 2, rest 0
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            candidate_types.append(1)
            
    # Type 2: Four 1s, rest 0
    # Map support tuple to list of indices in candidates
    support_to_indices = {}
    base_idx = len(candidates)
    for idxs in itertools.combinations(range(D), 4):
        support_key = tuple(idxs)
        indices = []
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            candidate_types.append(2)
            indices.append(len(candidates) - 1)
        support_to_indices[support_key] = indices
        
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # 2. Build Conflict Graph using Bitsets
    # adj[i] is an integer bitmask of neighbors
    adj = [0] * N
    
    # Type 1 nodes (0 to 21) have no conflicts with anyone?
    # We established Type 1 dot Type 2 <= 2.
    # Type 1 dot Type 1:
    # (2,0..) . (-2,0..) = -4. OK.
    # (2,0..) . (2,0..) = 4. Conflict (Self? No, distinct nodes).
    # Wait, if I have (2,0..) and (-2,0..), they are distinct.
    # If I have two (2,0..)? No, I generate each once.
    # So Type 1 nodes:
    # i and j.
    # If i, j correspond to same axis and same sign: Impossible (unique generation).
    # If same axis, diff sign: Dot -4. OK.
    # If diff axis: Dot 0. OK.
    # So Type 1 nodes have NO conflicts.
    # We can include ALL of them.
    
    # Type 2 nodes:
    # Conflicts within same support: None (distinct sign patterns have dot <= 2).
    # Conflicts with other supports:
    # Iterate over supports
    supports = list(support_to_indices.keys())
    
    # Precompute intersection info for supports
    # For each support, find other supports intersecting in 3
    support_neighbors = {s: [] for s in supports}
    for i in range(len(supports)):
        s1 = supports[i]
        for j in range(i + 1, len(supports)):
            s2 = supports[j]
            inter = set(s1) & set(s2)
            if len(inter) == 3:
                support_neighbors[s1].append((s2, tuple(inter)))
                support_neighbors[s2].append((s1, tuple(inter)))
                
    # Build edges
    # For each support s1, and each sign vector v1 in s1
    # For each neighbor support s2, and each sign vector v2 in s2
    # Check if they conflict
    
    # To speed up, precompute sign patterns
    # sign_patterns: list of 8 tuples of 4 signs
    all_signs = list(itertools.product([1, -1], repeat=4))
    sign_to_idx = {sp: i for i, sp in enumerate(all_signs)}
    
    # For a support s, and a sign pattern sp, the candidate index is:
    # support_to_indices[s][sign_to_idx[sp]]
    
    # Conflict condition:
    # s1, s2 intersect in I (size 3).
    # v1 signs on I must equal v2 signs on I.
    
    for s1 in supports:
        idxs1 = support_to_indices[s1]
        # Map local sign index to global candidate index
        # idxs1[k] corresponds to all_signs[k]
        
        for s2, inter in support_neighbors[s1]:
            idxs2 = support_to_indices[s2]
            # Determine which positions in s1 and s2 correspond to inter
            # s1 is a tuple. inter is a tuple.
            pos_in_s1 = [s1.index(x) for x in inter]
            pos_in_s2 = [s2.index(x) for x in inter]
            
            # For each sign pattern k1 in s1
            for k1 in range(8):
                sp1 = all_signs[k1]
                # Extract signs on inter for sp1
                signs_on_inter_1 = [sp1[p] for p in pos_in_s1]
                
                # Find k2 in s2 such that signs on inter match
                # sp2[p] == signs_on_inter_1[j] for corresponding p
                # We can just check all 8, or precompute a lookup
                # Since 8 is small, check all
                for k2 in range(8):
                    sp2 = all_signs[k2]
                    signs_on_inter_2 = [sp2[p] for p in pos_in_s2]
                    if signs_on_inter_1 == signs_on_inter_2:
                        # Conflict
                        i1 = idxs1[k1]
                        i2 = idxs2[k2]
                        adj[i1] |= (1 << i2)
                        adj[i2] |= (1 << i1)
                        
    # 3. Solve MIS on Type 2 nodes
    # Type 2 nodes are from base_idx to N-1
    # We can solve on the subgraph of Type 2 nodes.
    # Remap indices to 0..N2-1 for bitset efficiency?
    # N2 = 5280. 1 << 5280 is a huge int, but Python handles it.
    # However, shifting 1 << 5280 is fast.
    
    N2 = N - base_idx
    # Create local adjacency for Type 2
    # local_adj[i] for i in 0..N2-1
    local_adj = [0] * N2
    
    # Map global index to local index
    # global i -> local i - base_idx
    for i in range(N2):
        gi = base_idx + i
        # Get neighbors of gi
        mask = adj[gi]
        # We only care about Type 2 neighbors
        # Type 1 neighbors are 0..21, which are < base_idx.
        # So mask already only has bits >= base_idx?
        # No, adj[gi] might have bits for Type 1?
        # We established no conflicts between T1 and T2.
        # So adj[gi] only has bits for T2.
        # Shift down by base_idx
        local_adj[i] = mask >> base_idx
        
    # Tomita's Algorithm
    best_set = []
    
    def color_sort(P):
        # P is a list of nodes (local indices)
        # Returns colored nodes (list of (node, color))
        # and color count
        if not P:
            return [], 0
        
        uncolored = P[:]
        colors = []
        color_count = 0
        
        while uncolored:
            color_count += 1
            current_color_nodes = []
            new_uncolored = []
            
            for v in uncolored:
                # Check if v conflicts with any node in current_color_nodes
                # If not, assign color
                can_color = True
                for u in current_color_nodes:
                    if (local_adj[v] >> u) & 1:
                        can_color = False
                        break
                if can_color:
                    current_color_nodes.append(v)
                else:
                    new_uncolored.append(v)
            
            colors.extend([(v, color_count) for v in current_color_nodes])
            uncolored = new_uncolored
            
        return colors, color_count

    def expand(P, current_size, current_set):
        nonlocal best_set
        
        if not P:
            if current_size > len(best_set):
                best_set = current_set[:]
            return
        
        # Bound: current_size + color_count(P) <= len(best_set)
        # We need color_count.
        # Tomita's algorithm typically sorts P by color.
        # Let's use a simplified bound: |P| is an upper bound.
        # Better: Use the coloring bound.
        
        # To be fast, we can just use |P| as a loose bound initially,
        # but coloring is much tighter.
        
        # Perform coloring
        colored_P, c_count = color_sort(P)
        
        if current_size + c_count <= len(best_set):
            return
            
        # Sort P by color (highest color first? Tomita uses specific order)
        # Standard Tomita:
        # 1. Color sort P.
        # 2. Expand in order of increasing color?
        # Let's follow standard implementation.
        
        # We will iterate through colored_P in reverse order (highest color first)
        # Actually, Tomita's algorithm:
        # Sort P such that colors are non-decreasing.
        # Then process nodes in that order.
        
        # Let's just pick the last node in the sorted list (highest color)
        # This is a heuristic.
        
        # For simplicity and speed, we pick a node with high degree or just the last one.
        # Let's pick the last node in colored_P (which has the highest color index in the list if we appended sequentially)
        # Wait, color_sort returns list.
        # Let's just use the last element of the sorted list.
        
        if not colored_P:
            return
            
        v = colored_P[-1][0]
        
        # Branch 1: Include v
        # New P = P \ {v} \ N(v)
        # We need to filter P.
        # P is the list of nodes.
        # We can create a new list.
        
        # Optimization: Use bitsets for P?
        # P is small (decreases). List is fine.
        
        # Remove v and its neighbors from P
        mask_v = local_adj[v]
        new_P = []
        for u in P:
            if u == v:
                continue
            if (mask_v >> u) & 1:
                continue
            new_P.append(u)
            
        expand(new_P, current_size + 1, current_set + [v])
        
        # Branch 2: Exclude v
        # Remove v from P
        P.remove(v)
        expand(P, current_size, current_set)
        
    # Initial P is all Type 2 nodes
    P0 = list(range(N2))
    # To improve performance, we can start with a greedy solution as a lower bound
    # Greedy: pick node with min degree, remove neighbors.
    greedy_set = []
    P_greedy = set(P0)
    deg = [local_adj[i].bit_count() for i in P0]
    
    while P_greedy:
        # Find min degree
        min_deg = min(deg[i] for i in P_greedy)
        candidates = [i for i in P_greedy if deg[i] == min_deg]
        v = candidates[0]
        greedy_set.append(v)
        
        # Remove v and neighbors
        to_remove = {v}
        mask = local_adj[v]
        for u in P_greedy:
            if (mask >> u) & 1:
                to_remove.add(u)
        
        for u in to_remove:
            P_greedy.remove(u)
            # Update degrees? Expensive.
            # Just restart degree calc? Or ignore and use 0?
            # For a lower bound, a simple greedy is enough.
            
    best_set = greedy_set[:]
    
    # Run B&B
    # Note: B&B might be slow if not optimized.
    # Given the time limit, we might need to stop early.
    
    # Let's run expand
    # To avoid recursion depth issues, we might need to increase limit
    sys.setrecursionlimit(10000)
    
    # We will run expand with a time check?
    # Python recursion is hard to interrupt cleanly.
    # We can add a global time check in expand.
    
    global_time_limit = deadline
    time_exceeded = False
    
    def expand_timed(P, current_size, current_set):
        nonlocal best_set, time_exceeded
        
        if time.time() > global_time_limit:
            time_exceeded = True
            return
            
        if not P:
            if current_size > len(best_set):
                best_set = current_set[:]
            return
            
        colored_P, c_count = color_sort(P)
        
        if current_size + c_count <= len(best_set):
            return
            
        if not colored_P:
            return
            
        v = colored_P[-1][0]
        
        mask_v = local_adj[v]
        new_P = []
        for u in P:
            if u == v:
                continue
            if (mask_v >> u) & 1:
                continue
            new_P.append(u)
            
        expand_timed(new_P, current_size + 1, current_set + [v])
        
        if time_exceeded:
            return
            
        P.remove(v)
        expand_timed(P, current_size, current_set)
        
    expand_timed(P0, 0, [])
    
    # Combine results
    # best_set contains local indices for Type 2
    final_indices = list(range(22)) # All Type 1
    final_indices.extend([i + base_idx for i in best_set])
    
    return candidates[final_indices].astype(np.int64)

