import random
import time
import numpy as np

def construct(q, n, radius):
    total_words = q ** n
    
    # Precompute all words as tuples
    all_words = []
    for i in range(total_words):
        w = []
        temp = i
        for _ in range(n):
            w.append(temp % q)
            temp //= q
        all_words.append(tuple(reversed(w)))
        
    # Precompute powers of q for index calculation
    q_powers = [1] * n
    for i in range(n - 2, -1, -1):
        q_powers[i] = q_powers[i + 1] * q

    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % q
            idx //= q
        return tuple(t)

    # Precompute neighbors for distance <= 1 (for coverage check)
    # and distance <= 2 (for independent set graph)
    # For q=3, n=8, radius=1: 17 neighbors.
    # For dist <= 2: 1 + 16 + 112 = 129 neighbors.
    
    neighbors_r1 = [[] for _ in range(total_words)]
    neighbors_r2 = [[] for _ in range(total_words)]
    
    # Generate neighbors efficiently
    # For each word, generate all words at dist 1 and 2
    for i, w in enumerate(all_words):
        neighbors_r1[i].append(i)
        neighbors_r2[i].append(i)
        
        # Dist 1
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val == orig: continue
                diff = (new_val - orig) * q_powers[pos]
                j = i + diff
                neighbors_r1[i].append(j)
                neighbors_r2[i].append(j)
                
        # Dist 2
        for pos1 in range(n):
            for pos2 in range(pos1 + 1, n):
                v1 = w[pos1]
                v2 = w[pos2]
                # Change pos1 to v1', pos2 to v2'
                # v1' != v1, v2' != v2
                for nv1 in range(q):
                    if nv1 == v1: continue
                    for nv2 in range(q):
                        if nv2 == v2: continue
                        diff = (nv1 - v1) * q_powers[pos1] + (nv2 - v2) * q_powers[pos2]
                        j = i + diff
                        # Avoid duplicates if q=2? No q=3.
                        # Check if j already in list? 
                        # For q=3, dist 1 and dist 2 are disjoint sets of neighbors?
                        # Yes, a word cannot be at dist 1 and dist 2 simultaneously.
                        neighbors_r2[i].append(j)

    # Convert to sets for faster lookup if needed, but lists are fine for iteration
    # For "in" checks in local search, sets are better.
    # Given 6561 nodes, sets are manageable.
    neighbors_r1_set = [set(lst) for lst in neighbors_r1]
    neighbors_r2_set = [set(lst) for lst in neighbors_r2]

    # Step 1: Greedy Maximum Independent Set in G_2
    # G_2 has edges between nodes at dist <= 2.
    # Independent Set I: No two nodes in I are at dist <= 2.
    # This means dist(u, v) >= 3 for all u, v in I.
    
    active = [True] * total_words
    independent_set = set()
    
    # Heuristic: Pick node with min degree in active subgraph
    # To speed up, we can just iterate and pick first available or random.
    # Min-degree is better for MIS size.
    # Calculating degree dynamically is expensive.
    # Let's use a simple greedy: iterate through nodes, if not in I and not adjacent to any in I, add to I.
    # This is the "Greedy Independent Set" (not necessarily max, but good).
    # To improve, we can randomize the order.
    
    order = list(range(total_words))
    random.shuffle(order)
    
    for node in order:
        if not active[node]:
            continue
        # Check if node is adjacent to any node in independent_set
        # If yes, skip.
        # If no, add to I and deactivate its neighbors.
        
        # Optimization: Check neighbors in I
        conflict = False
        for nb in neighbors_r2_set[node]:
            if nb in independent_set:
                conflict = True
                break
        
        if not conflict:
            independent_set.add(node)
            # Deactivate node and its neighbors in active set
            # We don't strictly need to update 'active' if we just check 'independent_set' membership,
            # but 'active' helps if we were doing something else. 
            # Actually, the standard greedy IS algorithm is:
            # for v in V:
            #   if v not in I and no neighbor in I:
            #      I.add(v)
            # This is what I am doing above (implicitly, since we check conflict).
            # Wait, if I add v, I must ensure no FUTURE node picks a neighbor of v.
            # The check `if nb in independent_set` handles this.
            # So we don't need to modify 'active' or 'neighbors'.
            # We just build I.

    # Step 2: Initialize C = V \ I
    C = set()
    for i in range(total_words):
        if i not in independent_set:
            C.add(i)
            
    # Step 3: Verify Coverage and Repair
    # C must cover all words within radius 1.
    # covered = union of neighbors_r1[c] for c in C
    
    covered = [False] * total_words
    for c in C:
        for nb in neighbors_r1[c]:
            covered[nb] = True
            
    # Check for uncovered words
    uncovered = [i for i in range(total_words) if not covered[i]]
    
    # Repair: Add uncovered words to C (and remove from I)
    for u in uncovered:
        C.add(u)
        if u in independent_set:
            independent_set.remove(u)
        # Note: Adding u to C might cover others, but we just need to ensure u is covered.
        # Since u is in C, it covers itself.
        
    # After adding all uncovered, C is a valid covering code.
    # However, we might have added too many.
    # Also, the initial I might have been small, so C is large.
    
    # Step 4: Local Search to Minimize C
    # We maintain cover_count to check redundancy.
    cover_count = [0] * total_words
    for c in C:
        for nb in neighbors_r1[c]:
            cover_count[nb] += 1
            
    def is_redundant(c_idx):
        for nb in neighbors_r1[c_idx]:
            if cover_count[nb] == 1:
                return False
        return True
        
    def remove_center(c_idx):
        for nb in neighbors_r1[c_idx]:
            cover_count[nb] -= 1
        C.remove(c_idx)
        
    def add_center(c_idx):
        for nb in neighbors_r1[c_idx]:
            cover_count[nb] += 1
        C.add(c_idx)
        
    start_time = time.time()
    max_time = 115.0
    
    improved = True
    while improved and (time.time() - start_time) < max_time:
        improved = False
        
        # 1. Reduction Phase
        # Try to remove redundant centers
        # Iterate in random order
        centers_list = list(C)
        random.shuffle(centers_list)
        
        for c_idx in centers_list:
            if time.time() - start_time > max_time:
                break
            if is_redundant(c_idx):
                remove_center(c_idx)
                improved = True
                # Break to re-evaluate, as removal changes counts
                break
                
        if time.time() - start_time > max_time:
            break
            
        if improved:
            continue
            
        # 2. Swap Phase
        # Try to swap a center c with a non-center w
        # Condition: c is not redundant.
        # Critical neighbors of c: nb in N(c) with cover_count[nb] == 1.
        # w must cover all critical neighbors.
        
        # Pick a random center
        if not C:
            break
            
        c_idx = random.choice(list(C))
        critical_nb = []
        for nb in neighbors_r1[c_idx]:
            if cover_count[nb] == 1:
                critical_nb.append(nb)
                
        if not critical_nb:
            # Should have been removed
            remove_center(c_idx)
            improved = True
            continue
            
        # Find w such that critical_nb subset of N(w)
        # w should ideally be in V \ C to keep size same, or in C to reduce size (if w already there, it's just removal).
        # If w in C, it's a reduction. If w not in C, it's a swap.
        # We want to find ANY w that covers critical_nb.
        
        # Candidate w's must be in intersection of N(nb) for nb in critical_nb.
        # Let's pick a random critical nb0.
        # Candidates are in N(nb0).
        nb0 = critical_nb[0]
        candidates = list(neighbors_r1_set[nb0])
        
        # Filter candidates:
        # 1. Must cover all other critical_nbs.
        # 2. Prefer w not in C (to allow potential future reduction) or just any w.
        # Actually, if w is in C, we just remove c. This is a reduction.
        # If w is not in C, we swap c->w. Size stays same.
        # We want to enable reductions. Swapping to a "better" position might expose redundancies.
        
        valid_swap = False
        # Shuffle candidates for diversity
        random.shuffle(candidates)
        
        for w_idx in candidates:
            # Check if w covers all critical_nb
            ok = True
            for nb in critical_nb:
                if nb not in neighbors_r1_set[w_idx]:
                    ok = False
                    break
            if ok:
                # Perform swap
                # Remove c
                for nb in neighbors_r1[c_idx]:
                    cover_count[nb] -= 1
                C.remove(c_idx)
                
                # Add w
                if w_idx not in C:
                    for nb in neighbors_r1[w_idx]:
                        cover_count[nb] += 1
                    C.add(w_idx)
                else:
                    # w was already in C, so counts were already included?
                    # No, we just decremented c's counts.
                    # If w was in C, its counts were NOT decremented.
                    # So we don't need to increment.
                    pass
                    
                valid_swap = True
                improved = True
                break
                
        # If no valid swap found, try another random center
        # The loop continues, so we just let it pick another one next iteration?
        # The `improved` flag stays False, so we exit the while loop if no reduction and no swap.
        # We should try multiple swaps per "iteration" of the while loop to avoid getting stuck.
        
        # Let's add a loop for swaps inside the while loop
        swaps_attempted = 0
        while swaps_attempted < 20 and not improved and (time.time() - start_time) < max_time:
            swaps_attempted += 1
            if not C:
                break
            c_idx = random.choice(list(C))
            critical_nb = []
            for nb in neighbors_r1[c_idx]:
                if cover_count[nb] == 1:
                    critical_nb.append(nb)
                    
            if not critical_nb:
                remove_center(c_idx)
                improved = True
                break
                
            nb0 = critical_nb[0]
            candidates = list(neighbors_r1_set[nb0])
            random.shuffle(candidates)
            
            for w_idx in candidates:
                ok = True
                for nb in critical_nb:
                    if nb not in neighbors_r1_set[w_idx]:
                        ok = False
                        break
                if ok:
                    # Perform swap
                    for nb in neighbors_r1[c_idx]:
                        cover_count[nb] -= 1
                    C.remove(c_idx)
                    
                    if w_idx not in C:
                        for nb in neighbors_r1[w_idx]:
                            cover_count[nb] += 1
                        C.add(w_idx)
                    else:
                        # w in C, counts already there
                        pass
                        
                    improved = True
                    break
                    
            if improved:
                break

    # Convert C to list of lists
    result = [list(all_words[c]) for c in C]
    return result
