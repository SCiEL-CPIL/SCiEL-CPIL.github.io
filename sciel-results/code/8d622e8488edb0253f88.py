import time
import random
from itertools import combinations, permutations
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    n = 16
    num_edges_total = n * (n - 1) // 2  # 120

    # Map edge (u, v) to an index 0..119
    def edge_index(u, v):
        if u > v:
            u, v = v, u
        idx = 0
        for i in range(u):
            idx += (n - 1 - i)
        idx += (v - u - 1)
        return idx

    # Precompute edge indices for all pairs
    edge_idx_map = {}
    for u in range(n):
        for v in range(u + 1, n):
            edge_idx_map[(u, v)] = edge_index(u, v)

    # Inverse map: index -> (u, v)
    inv_edge_map = [None] * num_edges_total
    for (u, v), idx in edge_idx_map.items():
        inv_edge_map[idx] = (u, v)

    # Q3 abstract edges (standard labeling of 3-cube)
    # Vertices 0..7. Edges:
    # 0-1, 0-2, 0-4
    # 1-3, 1-5
    # 2-3, 2-6
    # 3-7
    # 4-5, 4-6
    # 5-7
    # 6-7
    q3_abstract_edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]

    # Local edge index mapping for K8 (28 edges)
    local_edge_to_idx = {}
    local_idx_to_edge = [None] * 28
    idx = 0
    for u in range(8):
        for v in range(u + 1, 8):
            local_edge_to_idx[(u, v)] = idx
            local_idx_to_edge[idx] = (u, v)
            idx += 1

    # Precompute local Q3 edge sets as bitmasks of 28 bits
    # We iterate over all permutations of 8 vertices to cover all isomorphisms
    # However, many permutations yield the same edge set. We use a set of frozensets.
    local_q3_edge_sets = set()
    for p in permutations(range(8)):
        edges_set = set()
        for u_abs, v_abs in q3_abstract_edges:
            u_val = p[u_abs]
            v_val = p[v_abs]
            le = local_edge_to_idx[(min(u_val, v_val), max(u_val, v_val))]
            edges_set.add(le)
        if len(edges_set) == 12:
            local_q3_edge_sets.add(frozenset(edges_set))
    
    # Convert to list of bitmasks for faster operations
    local_q3_bitmasks = []
    for mask_set in local_q3_edge_sets:
        mask_int = 0
        for e in mask_set:
            mask_int |= (1 << e)
        local_q3_bitmasks.append(mask_int)

    # Precompute global Q3 masks
    # For each subset of 8 vertices, map local edges to global edges
    global_q3_masks = []
    all_subsets = list(combinations(range(n), 8))
    
    if time.time() > deadline - 20:
        # If precomputation is too slow, fallback to a simple heuristic
        return [[i, i+1] for i in range(0, 16, 2)]

    # To speed up global mask generation, we can precompute for each local Q3 bitmask,
    # the list of local edge indices it contains.
    local_q3_edge_lists = []
    for mask in local_q3_bitmasks:
        edges = []
        m = mask
        while m:
            lsb = m & -m
            edges.append(lsb.bit_length() - 1)
            m ^= lsb
        local_q3_edge_lists.append(edges)

    for subset in all_subsets:
        # Map local edge idx to global edge idx
        l2g = [0] * 28
        for u_loc in range(8):
            for v_loc in range(u_loc + 1, 8):
                le = local_edge_to_idx[(u_loc, v_loc)]
                u_glob = subset[u_loc]
                v_glob = subset[v_loc]
                ge = edge_idx_map[(min(u_glob, v_glob), max(u_glob, v_glob))]
                l2g[le] = ge
                
        # For each local Q3, generate the global mask
        for edge_list in local_q3_edge_lists:
            global_mask = 0
            for le in edge_list:
                ge = l2g[le]
                global_mask |= (1 << ge)
            global_q3_masks.append(global_mask)
            
    if time.time() > deadline - 10:
        # Fallback if precomputation took too long
        return [[i, i+1] for i in range(0, 16, 2)]

    num_q3s = len(global_q3_masks)
    
    # Greedy Set Cover on the complement
    # We want to remove edges to hit all Q3s.
    # removed_edges[e] = True if edge e is removed (not in graph)
    removed_edges = [False] * num_edges_total
    
    # q3s_containing_edge[e] = list of indices of Q3s containing e
    q3s_containing_edge = [[] for _ in range(num_edges_total)]
    
    for i, mask in enumerate(global_q3_masks):
        m = mask
        while m:
            lsb = m & -m
            e_idx = lsb.bit_length() - 1
            q3s_containing_edge[e_idx].append(i)
            m ^= lsb
            
    # counts[e] = number of active Q3s containing e
    counts = [0] * num_edges_total
    for e in range(num_edges_total):
        counts[e] = len(q3s_containing_edge[e])
        
    # active_q3s: boolean list, True if Q3 is not yet hit
    active_q3s = [True] * num_q3s
    
    # Use a max-heap for faster greedy selection
    # Heap entries: (-count, edge_idx)
    heap = []
    for e in range(num_edges_total):
        if counts[e] > 0:
            heapq.heappush(heap, (-counts[e], e))
    
    # Greedy Loop with heap
    while heap:
        if time.time() > deadline - 20:
            break
            
        # Pop edges until we find one that's not removed and has correct count
        while heap:
            neg_count, e = heap[0]
            if removed_edges[e]:
                heapq.heappop(heap)
                continue
            if -neg_count != counts[e]:
                # Count changed, push updated entry
                heapq.heappop(heap)
                if counts[e] > 0:
                    heapq.heappush(heap, (-counts[e], e))
                continue
            break
            
        if not heap:
            break
            
        neg_count, best_edge = heapq.heappop(heap)
        max_count = -neg_count
        
        if max_count == 0:
            break
            
        # Remove best_edge
        removed_edges[best_edge] = True
        
        # Update active Q3s and counts
        for q3_idx in q3s_containing_edge[best_edge]:
            if active_q3s[q3_idx]:
                active_q3s[q3_idx] = False
                mask = global_q3_masks[q3_idx]
                m = mask
                while m:
                    lsb = m & -m
                    e_idx = lsb.bit_length() - 1
                    if e_idx != best_edge:
                        counts[e_idx] -= 1
                        if counts[e_idx] > 0:
                            heapq.heappush(heap, (-counts[e_idx], e_idx))
                    m ^= lsb

    # Local Search Polish: try to re-add removed edges
    # We iterate through removed edges. If adding edge e back does not create a Q3, we add it.
    # We maintain a count for each Q3 of how many of its edges are removed.
    # If count == 0, the Q3 is present in the graph.
    
    # Initialize removed_count_for_q3
    removed_count_for_q3 = [0] * num_q3s
    for q3_idx, mask in enumerate(global_q3_masks):
        m = mask
        cnt = 0
        while m:
            lsb = m & -m
            e_idx = lsb.bit_length() - 1
            if removed_edges[e_idx]:
                cnt += 1
            m ^= lsb
        removed_count_for_q3[q3_idx] = cnt
    
    # Also maintain for each edge, the list of Q3s it belongs to (already have q3s_containing_edge)
    
    # Try to add back removed edges
    # Order: random or by some heuristic? Let's try random first, then maybe deterministic.
    # We can do multiple passes.
    
    # For efficiency, we can just iterate through all removed edges in a random order and try to add them.
    # If adding e makes any Q3 have removed_count == 0, we reject.
    # When we add e (set removed_edges[e] = False), we decrement removed_count_for_q3 for all q3 in q3s_containing_edge[e].
    # We check if any becomes 0. If so, we revert (increment back) and skip.
    
    # To make it stronger, we can do multiple passes, shuffling the order of removed edges.
    # Or we can sort removed edges by some priority (e.g., degree in the removed graph, or just random).
    
    # Let's do a few passes with random shuffles.
    removed_list = [e for e in range(num_edges_total) if removed_edges[e]]
    
    for pass_num in range(15):
        if time.time() > deadline - 5:
            break
        random.seed(pass_num * 42 + 123)
        random.shuffle(removed_list)
        
        changed = False
        for e in removed_list:
            # Check if e is still removed
            if not removed_edges[e]:
                continue
            
            # Try to add e
            # Decrement counts for all Q3s containing e
            affected_q3s = q3s_containing_edge[e]
            made_zero = False
            for q3_idx in affected_q3s:
                removed_count_for_q3[q3_idx] -= 1
                if removed_count_for_q3[q3_idx] == 0:
                    made_zero = True
                    break
            
            if made_zero:
                # Revert
                for q3_idx in affected_q3s:
                    removed_count_for_q3[q3_idx] += 1
            else:
                # Accept
                removed_edges[e] = False
                changed = True
        
        if not changed:
            break
    
    # Phase 2: 2-swap local search
    # Try to swap: remove edge a (currently in graph) and add edge b (currently removed)
    # This can escape local optima where single additions are blocked.
    # We need: adding b doesn't create Q3, but after removing a, adding b is safe.
    # More precisely: we want to find a pair (a, b) where a is present, b is removed,
    # and the set of Q3s that become "complete" when adding b is a subset of Q3s that are "broken" by removing a.
    
    if time.time() < deadline - 10:
        present_list = [e for e in range(num_edges_total) if not removed_edges[e]]
        removed_list2 = [e for e in range(num_edges_total) if removed_edges[e]]
        
        # For each removed edge b, precompute the set of Q3s that would become complete
        # (i.e., have removed_count == 1 before adding b)
        blocked_q3s_for_edge = {}
        for b in removed_list2:
            blocked = set()
            for q3_idx in q3s_containing_edge[b]:
                if removed_count_for_q3[q3_idx] == 1:
                    blocked.add(q3_idx)
            blocked_q3s_for_edge[b] = blocked
        
        # For each present edge a, precompute the set of Q3s that are "critical"
        # (i.e., have removed_count == 1, so removing a would make them count 2)
        # Actually, we need: Q3s that contain a and have removed_count == 1
        # When we remove a, their count increases to 2, so they're no longer at risk.
        critical_q3s_for_edge = {}
        for a in present_list:
            critical = set()
            for q3_idx in q3s_containing_edge[a]:
                if removed_count_for_q3[q3_idx] == 1:
                    critical.add(q3_idx)
            critical_q3s_for_edge[a] = critical
        
        # Try swaps: for each removed b, find a present a such that blocked_q3s(b) ⊆ critical_q3s(a)
        # We do this in random order for diversity
        random.seed(999)
        random.shuffle(removed_list2)
        random.shuffle(present_list)
        
        swap_found = True
        while swap_found and time.time() < deadline - 5:
            swap_found = False
            for b in removed_list2:
                if not removed_edges[b]:
                    continue
                blocked_b = blocked_q3s_for_edge.get(b, set())
                if not blocked_b:
                    # Adding b is safe, should have been caught in phase 1
                    # But let's try anyway
                    for q3_idx in q3s_containing_edge[b]:
                        removed_count_for_q3[q3_idx] -= 1
                    removed_edges[b] = False
                    swap_found = True
                    continue
                
                for a in present_list:
                    if removed_edges[a]:
                        continue
                    critical_a = critical_q3s_for_edge.get(a, set())
                    # Check if blocked_b is subset of critical_a
                    if blocked_b.issubset(critical_a):
                        # Perform swap: remove a, add b
                        # Update counts for a's Q3s (remove a)
                        for q3_idx in q3s_containing_edge[a]:
                            removed_count_for_q3[q3_idx] += 1
                        removed_edges[a] = True
                        
                        # Update counts for b's Q3s (add b)
                        for q3_idx in q3s_containing_edge[b]:
                            removed_count_for_q3[q3_idx] -= 1
                        removed_edges[b] = False
                        
                        # Update blocked and critical sets
                        # Recalculate for affected edges
                        for edge in [a, b]:
                            if removed_edges[edge]:
                                new_blocked = set()
                                for q3_idx in q3s_containing_edge[edge]:
                                    if removed_count_for_q3[q3_idx] == 1:
                                        new_blocked.add(q3_idx)
                                blocked_q3s_for_edge[edge] = new_blocked
                            else:
                                new_critical = set()
                                for q3_idx in q3s_containing_edge[edge]:
                                    if removed_count_for_q3[q3_idx] == 1:
                                        new_critical.add(q3_idx)
                                critical_q3s_for_edge[edge] = new_critical
                        
                        swap_found = True
                        break
                if swap_found:
                    break
        
        # After swaps, do another round of single additions
        removed_list3 = [e for e in range(num_edges_total) if removed_edges[e]]
        for pass_num in range(5):
            if time.time() > deadline - 3:
                break
            random.seed(pass_num * 777 + 456)
            random.shuffle(removed_list3)
            
            changed = False
            for e in removed_list3:
                if not removed_edges[e]:
                    continue
                
                affected_q3s = q3s_containing_edge[e]
                made_zero = False
                for q3_idx in affected_q3s:
                    removed_count_for_q3[q3_idx] -= 1
                    if removed_count_for_q3[q3_idx] == 0:
                        made_zero = True
                        break
                
                if made_zero:
                    for q3_idx in affected_q3s:
                        removed_count_for_q3[q3_idx] += 1
                else:
                    removed_edges[e] = False
                    changed = True
            
            if not changed:
                break
    
    # Construct the final edge list
    final_edges = []
    for e in range(num_edges_total):
        if not removed_edges[e]:
            u, v = inv_edge_map[e]
            final_edges.append([u, v])
            
    return final_edges
