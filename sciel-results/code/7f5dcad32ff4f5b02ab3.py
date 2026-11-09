import numpy as np
import time

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    k = 2
    
    # 1. Compute Forbidden Differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, k, m))
    forbidden.discard(0)
    
    # 2. Precompute Conflict Masks
    # conflict_masks[i] is a set of nodes j that conflict with i
    # Using sets for easy intersection/union logic in the swap check
    conflict_masks = [set() for _ in range(m)]
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            d = (i - j) % m
            if d in forbidden:
                conflict_masks[i].add(j)
                # Symmetry: if i conflicts with j, j conflicts with i
                # conflict_masks[j].add(i) -- already handled in loop if we iterate all pairs, 
                # but here we only fill i's mask. Since (j-i)%m is also in forbidden if (i-j)%m is?
                # x^2 is symmetric. (i-j)^2 = (j-i)^2. So yes, symmetric.
    
    # Ensure symmetry is explicit for clarity/safety
    for i in range(m):
        for j in conflict_masks[i]:
            if i not in conflict_masks[j]:
                conflict_masks[j].add(i)

    def greedy_fill(order, initial_set=None):
        """
        Greedily fills a set based on the given order.
        Returns (set_of_nodes, conflict_counts_array)
        conflict_counts[i] is the number of nodes in the current set that conflict with i.
        """
        if initial_set is None:
            S = set()
            conf_counts = [0] * m
        else:
            S = set(initial_set)
            conf_counts = [0] * m
            for u in S:
                for v in conflict_masks[u]:
                    conf_counts[v] += 1
        
        for node in order:
            if node not in S:
                if conf_counts[node] == 0:
                    S.add(node)
                    # Update counts for neighbors of node
                    for neighbor in conflict_masks[node]:
                        conf_counts[neighbor] += 1
        return S, conf_counts

    def systematic_improve(S, conf_counts):
        """
        Attempts to improve S by swapping one node u in S with one node v not in S,
        and then adding a third node w if possible.
        Returns the best S found.
        """
        current_S = set(S)
        current_counts = list(conf_counts)
        
        improved = True
        while improved:
            improved = False
            # Iterate over all u in S
            S_list = list(current_S)
            for u in S_list:
                if improved:
                    break
                # Iterate over all v not in S
                for v in range(m):
                    if v in current_S:
                        continue
                    if improved:
                        break
                    
                    # Simulate swap u -> v
                    # 1. Remove u: decrement counts for neighbors of u
                    # 2. Add v: increment counts for neighbors of v
                    # We need to check if any w not in (S - {u} + {v}) has count 0.
                    
                    # To avoid modifying current_counts permanently until we decide,
                    # we can calculate the condition on the fly.
                    # new_count[w] = current_counts[w] - (1 if w in C(u) else 0) + (1 if w in C(v) else 0)
                    
                    # Candidates for w:
                    # w must not be in current_S - {u} + {v}
                    # So w != v, and w not in (current_S - {u})
                    
                    # Optimization:
                    # We can precompute C(u) and C(v) as sets (already done).
                    # We iterate w from 0 to m-1.
                    
                    found_w = -1
                    # We can break early if we find a w
                    for w in range(m):
                        # Check if w is in the new set S' = (S \ {u}) U {v}
                        if w == v:
                            continue
                        if w in current_S and w != u:
                            continue
                        
                        # Calculate new count for w
                        # delta = -1 if w in C(u) else 0
                        # delta += 1 if w in C(v) else 0
                        delta = 0
                        if w in conflict_masks[u]:
                            delta -= 1
                        if w in conflict_masks[v]:
                            delta += 1
                        
                        new_count = current_counts[w] + delta
                        
                        if new_count == 0:
                            found_w = w
                            break
                    
                    if found_w != -1:
                        # Accept the swap u -> v and add found_w
                        # 1. Remove u
                        current_S.remove(u)
                        for neighbor in conflict_masks[u]:
                            current_counts[neighbor] -= 1
                        
                        # 2. Add v
                        current_S.add(v)
                        for neighbor in conflict_masks[v]:
                            current_counts[neighbor] += 1
                        
                        # 3. Add found_w
                        current_S.add(found_w)
                        for neighbor in conflict_masks[found_w]:
                            current_counts[neighbor] += 1
                        
                        improved = True
                        # Break out of v loop and u loop
                        break
        
        return current_S, current_counts

    # Strategy: Try multiple deterministic starting orders
    orders = [
        list(range(m)),
        list(range(m-1, -1, -1)),
        list(range(0, m, 10)), # Step 10
        list(range(5, m, 10)), # Step 10, offset 5
        sorted(range(m), key=lambda x: x % 13), # Modulo 13
        sorted(range(m), key=lambda x: x % 17), # Modulo 17
    ]
    
    best_S = set()
    best_size = 0
    
    for order in orders:
        S, counts = greedy_fill(order)
        S, counts = systematic_improve(S, counts)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
    
    # Final check: maybe one more round of improvement on the best found?
    # The systematic_improve already loops until no improvement is found.
    
    result = np.array(list(best_S), dtype=np.int64)
    return result
