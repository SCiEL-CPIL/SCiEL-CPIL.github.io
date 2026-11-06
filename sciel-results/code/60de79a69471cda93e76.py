import time
import random
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 256
    
    # 1. Generate all 3-APs and 4-APs
    aps3 = []
    aps4 = []
    
    # Map element to list of 3-AP indices containing it
    elem_to_aps3 = [[] for _ in range(n+1)]
    # Map element to list of 4-AP indices containing it
    elem_to_aps4 = [[] for _ in range(n+1)]
    
    # Generate 3-APs
    ap_id = 0
    for d in range(1, n // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            if c <= n:
                aps3.append((a, b, c))
                elem_to_aps3[a].append(ap_id)
                elem_to_aps3[b].append(ap_id)
                elem_to_aps3[c].append(ap_id)
                ap_id += 1
                
    num_aps3 = len(aps3)
    
    # Generate 4-APs
    ap_id = 4
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= n:
                aps4.append((a, b, c, e))
                elem_to_aps4[a].append(ap_id)
                elem_to_aps4[b].append(ap_id)
                elem_to_aps4[c].append(ap_id)
                elem_to_aps4[e].append(ap_id)
                ap_id += 1

    # 2. Build Conflict Graph for 3-APs
    # Vertices: 3-APs
    # Edge between i and j if union of AP_i and AP_j contains a 4-AP
    
    # We need to efficiently determine conflicts.
    # A 4-AP (a, a+d, a+2d, a+3d) contains two 3-APs:
    # (a, a+d, a+2d) and (a+d, a+2d, a+3d)
    # If we select both of these 3-APs, we form a 4-AP. Thus they conflict.
    
    # Let's map each 3-AP to its ID
    ap3_id_map = {}
    for i, (a, b, c) in enumerate(aps3):
        ap3_id_map[(a, b, c)] = i
        
    # Adjacency list for conflict graph
    # Using sets for fast lookup, then converting to lists/bitmasks if needed
    # Given num_aps3 ~ 10k, a list of sets is okay, but bitmasks are faster for B&B
    # Let's use list of sets first for construction, then maybe optimize.
    
    conflicts = [set() for _ in range(num_aps3)]
    
    for a, b, c, d_val in aps4:
        # 3-AP 1: (a, b, c)
        # 3-AP 2: (b, c, d_val)
        id1 = ap3_id_map.get((a, b, c))
        id2 = ap3_id_map.get((b, c, d_val))
        
        if id1 is not None and id2 is not None:
            conflicts[id1].add(id2)
            conflicts[id2].add(id1)
            
    # Note: Are there other conflicts?
    # The problem states "no nonconstant four-term arithmetic progression".
    # A 4-AP is defined by 4 distinct terms in AP.
    # If we pick a set of 3-APs, does it guarantee no 4-AP?
    # Yes, if the only way to form a 4-AP is by having its two constituent 3-APs.
    # Is it possible to form a 4-AP using 3-APs that are not the consecutive triples?
    # A 4-AP is {x, x+d, x+2d, x+3d}.
    # The 3-APs within it are {x, x+d, x+2d} and {x+d, x+2d, x+3d}.
    # There are no other 3-APs formed by subsets of a 4-AP.
    # So, the conflict graph edges are exactly defined by the 4-APs.
    
    # 3. Branch and Bound for Maximum Independent Set
    
    # Convert conflicts to bitmasks for fast intersection?
    # num_aps3 is approx 10,000. Python ints are arbitrary precision, so bitmasks work.
    # However, 10k bits is ~1.25KB per int. 10k ints is ~12MB. This is fine.
    
    conflict_masks = [0] * num_aps3
    for i in range(num_aps3):
        mask = 0
        for j in conflicts[i]:
            mask |= (1 << j)
        conflict_masks[i] = mask
            
    # Greedy Upper Bound for B&B
    # We need a fast way to estimate the max independent set size in a subgraph.
    # A common heuristic is the size of a greedy maximal independent set.
    
    def greedy_mis_size(vertices_mask, current_size):
        # vertices_mask is a bitmask of available vertices
        # Returns an upper bound on the MIS size
        # Greedy: pick vertex with min degree, add to set, remove neighbors
        # This is O(V^2) in worst case if not careful, but V is small in subproblems?
        # Actually, we can use a simpler bound: |V| / (1 + avg_degree) or similar.
        # Or just use the count of vertices if we don't have a fast MIS solver.
        # Let's implement a simple greedy MIS for the bound.
        
        # To speed up, we can just return the number of set bits in vertices_mask?
        # No, that's too loose.
        # Let's use a heuristic: number of vertices / 2 is a weak bound.
        # Better: Greedy MIS.
        
        # Since we are in B&B, we want this to be fast.
        # Let's precompute degrees? No, degrees change as we prune.
        
        # Simple bound: popcount(vertices_mask)
        # This is valid but weak.
        
        # Let's try a greedy MIS with a time limit or simple heuristic.
        # For now, use popcount as a safe (though loose) upper bound.
        # We can improve this if the search is too slow.
        # Actually, let's use a slightly better bound:
        # Max independent set <= |V|
        # We can also use the fact that if we pick v, we remove N(v).
        
        # Let's stick to popcount for the bound to ensure speed, 
        # and rely on the branching order to prune effectively.
        return vertices_mask.bit_count()

    best_size = 0
    best_mask = 0  # Bitmask of selected 3-APs
    
    # Ordering of vertices for branching
    # Sort by degree (highest degree first) to prune faster?
    degrees = [len(conflicts[i]) for i in range(num_aps3)]
    order = sorted(range(num_aps3), key=lambda x: degrees[x], reverse=True)
    
    # Reverse order for indexing in recursion? 
    # Standard B&B: 
    # 1. Pick a vertex v.
    # 2. Branch 1: Include v. Remove v and N(v) from available.
    # 3. Branch 2: Exclude v. Remove v from available.
    
    # To make this efficient in Python, we represent the "available" set as a bitmask.
    # Initial available: all 1s.
    full_mask = (1 << num_aps3) - 1
    
    def solve(available_mask, current_size, current_selection_mask):
        nonlocal best_size, best_mask
        nonlocal deadline
        
        if time.time() > deadline:
            return
            
        if available_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_selection_mask
            return
            
        # Upper bound check
        # Simple bound: current_size + popcount(available_mask)
        if current_size + available_mask.bit_count() <= best_size:
            return
            
        # Pick a vertex to branch on
        # Heuristic: Pick vertex with highest degree in the induced subgraph?
        # Finding highest degree in induced subgraph is expensive.
        # Let's pick the first vertex in our pre-sorted order that is in available_mask.
        # Or just the lowest set bit?
        # Let's use the pre-sorted order.
        
        v = -1
        for u in order:
            if (available_mask >> u) & 1:
                v = u
                break
                
        if v == -1:
            return
            
        bit_v = 1 << v
        
        # Branch 1: Include v
        # New available: available_mask & ~bit_v & ~conflict_masks[v]
        new_avail_include = available_mask & ~bit_v & ~conflict_masks[v]
        solve(new_avail_include, current_size + 1, current_selection_mask | bit_v)
        
        # Branch 2: Exclude v
        new_avail_exclude = available_mask & ~bit_v
        solve(new_avail_exclude, current_size, current_selection_mask)
        
    # Initial call
    solve(full_mask, 0, 0)
    
    # 4. Reconstruct Set from Selected 3-APs
    # We have best_mask which is a bitmask of 3-AP indices.
    # We need to convert this back to a set of elements [1, n].
    
    A_set = set()
    for i in range(num_aps3):
        if (best_mask >> i) & 1:
            a, b, c = aps3[i]
            A_set.add(a)
            A_set.add(b)
            A_set.add(c)
            
    # 5. Validation and Polish
    # The B&B should guarantee no 4-APs.
    # But let's double check and potentially add isolated elements that don't form 4-APs
    # and don't conflict with existing 3-APs?
    # Wait, the MIS formulation maximizes the NUMBER of 3-APs.
    # It does NOT necessarily maximize the cardinality of the set A.
    # The task asks to maximize the number of distinct 3-APs.
    # So the MIS solution is directly optimal for the objective.
    
    # However, we might be able to add elements that don't participate in any 3-AP
    # or participate in 3-APs that are NOT in the selected set?
    # No, if we add an element, it might create NEW 3-APs.
    # If a new 3-AP is created, it wasn't in our MIS selection.
    # Does adding it violate the 4-AP constraint?
    # If the new 3-AP conflicts with any existing selected 3-AP, we have a 4-AP.
    # If it doesn't conflict, we can add it, increasing the count of 3-APs.
    # This means our MIS on the FULL graph of all possible 3-APs should have found it.
    # Because the MIS selects a subset of ALL possible 3-APs.
    # If a 3-AP was not selected, it was either:
    # a) Excluded because it conflicted with a selected one.
    # b) Excluded because it was not part of the optimal MIS (maybe it didn't add enough value? No, MIS is max cardinality).
    
    # Actually, the MIS problem is: Maximize |S| where S is a set of 3-APs such that no two in S conflict.
    # The objective is exactly the number of 3-APs.
    # So the MIS solution IS the optimal set of 3-APs.
    # The set of elements A is the union of these 3-APs.
    
    # One detail: Are there elements in [1,n] that are not in any selected 3-AP, 
    # but could be added to form NEW 3-APs that don't conflict?
    # If such a 3-AP exists, it is a vertex in our graph.
    # If it doesn't conflict with the current MIS, it should have been included in the MIS (since we maximize size).
    # So the MIS is maximal with respect to 3-AP inclusion.
    
    # However, we might be able to add elements that do NOT form any 3-AP at all.
    # Adding such elements doesn't increase the 3-AP count, but doesn't hurt.
    # The problem says "Maximize the number of distinct three-term arithmetic progressions".
    # It does NOT require maximizing cardinality.
    # So we don't need to add isolated elements.
    
    # Let's verify validity.
    A_mask = 0
    for x in A_set:
        A_mask |= (1 << (x-1))
        
    # Check for 4-APs
    has_4ap = False
    for a, b, c, d_val in aps4:
        mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1)) | (1 << (d_val-1))
        if (A_mask & mask) == mask:
            has_4ap = True
            break
            
    if has_4ap:
        # Fallback: This shouldn't happen if B&B is correct.
        # Use greedy fallback.
        A_set = set()
        candidates = list(range(1, n+1))
        random.shuffle(candidates)
        for x in candidates:
            # Check if adding x creates 4-AP
            valid = True
            for ap_idx in elem_to_aps4[x]:
                a, b, c, d_val = aps4[ap_idx]
                others = [a, b, c, d_val]
                others.remove(x)
                if all(o in A_set for o in others):
                    valid = False
                    break
            if valid:
                A_set.add(x)
                
    return sorted(A_set)

# EVOLVE-BLOCK-END
