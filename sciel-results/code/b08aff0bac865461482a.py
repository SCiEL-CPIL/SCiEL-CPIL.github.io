"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11

    # Type A: Vectors with one non-zero entry +/- 2
    # Norm = 2.
    type_a_vectors = []
    for i in range(D):
        type_a_vectors.append(np.array([2 if j == i else 0 for j in range(D)], dtype=np.int64))
        type_a_vectors.append(np.array([-2 if j == i else 0 for j in range(D)], dtype=np.int64))
        
    # Type D: Vectors with four non-zero entries +/- 1.
    # Norm = 2.
    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports)
    
    support_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks.append(mask)
    
    # Precompute conflict lists: conflict if intersection >= 3
    # We treat this as a graph where nodes are supports, edges are conflicts.
    # We want Maximum Independent Set (MIS).
    conflict_lists = []
    for i in range(N_S):
        conf = []
        for j in range(N_S):
            if i == j:
                continue
            if (support_masks[i] & support_masks[j]).bit_count() >= 3:
                conf.append(j)
        conflict_lists.append(conf)
        
    # Precompute degrees for initial sorting
    degrees = [len(conf) for conf in conflict_lists]

    def min_degree_greedy(seed_order=None):
        """
        Greedy algorithm that always picks the available node with the minimum 
        current degree (number of available neighbors). 
        This is a standard heuristic for MIS.
        """
        x = np.zeros(N_S, dtype=bool)
        available = np.ones(N_S, dtype=bool)
        # current_deg[i] is the number of neighbors of i that are still available
        current_deg = np.array([len(conflict_lists[i]) for i in range(N_S)])
        
        if seed_order is not None:
            # This implementation doesn't strictly follow a fixed order if we pick min-degree dynamically.
            # However, we can break ties randomly.
            pass
            
        selected = 0
        while True:
            if time.time() > deadline - 5:
                break
            # Find available nodes
            avail_indices = np.where(available)[0]
            if len(avail_indices) == 0:
                break
            
            # Among available nodes, find the one with minimum current_deg
            # To handle ties randomly, we can add a small random noise or just pick randomly among min
            min_deg_val = np.min(current_deg[avail_indices])
            candidates = avail_indices[current_deg[avail_indices] == min_deg_val]
            
            if len(candidates) > 0:
                # Pick one randomly from candidates
                v = candidates[random.randint(0, len(candidates)-1)]
                
                x[v] = True
                available[v] = False
                selected += 1
                
                # Update degrees of neighbors
                for c in conflict_lists[v]:
                    if available[c]:
                        current_deg[c] -= 1
                        # If a neighbor's degree drops to 0, it becomes an isolated node in the remaining graph
                        # We don't explicitly remove it yet, but it will be picked if it's min degree.
                        
        return x, selected

    def local_search_refine(x, max_iters=200):
        """
        Simple local search: try to add a node, or swap.
        Since we want max cardinality, we mostly try to add.
        If we can't add, we try to remove one and add two? (2-opt)
        For speed, we stick to 1-opt (add) and random swaps.
        """
        current_x = x.copy()
        current_size = int(np.sum(current_x))
        
        # Compute blocked status
        blocked = np.zeros(N_S, dtype=bool)
        for i in range(N_S):
            if current_x[i]:
                for c in conflict_lists[i]:
                    blocked[c] = True
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Try to add a random unblocked, unselected node
            unselected = np.where(~current_x)[0]
            if len(unselected) == 0:
                break
            
            # Filter unselected to those not blocked
            addable = [u for u in unselected if not blocked[u]]
            if addable:
                v = random.choice(addable)
                current_x[v] = True
                current_size += 1
                for c in conflict_lists[v]:
                    blocked[c] = True
            else:
                # Try a swap: remove a random selected, add a random unselected
                selected = np.where(current_x)[0]
                if len(selected) > 0:
                    i = random.choice(selected)
                    j = random.choice(unselected)
                    
                    # Check if j is compatible with all selected except i
                    # This is O(N) per check, might be slow. 
                    # Optimization: check if j is blocked by any selected other than i.
                    # blocked[j] is True if ANY selected node conflicts with j.
                    # If blocked[j] is True, we need to know if i is the ONLY blocker.
                    # This is complex. Let's just try it and validate.
                    
                    new_x = current_x.copy()
                    new_x[i] = False
                    new_x[j] = True
                    
                    valid = True
                    # Check conflicts for j
                    for c in conflict_lists[j]:
                        if new_x[c]:
                            valid = False
                            break
                            
                    if valid:
                        current_x = new_x
                        # Size unchanged, but structure changed. Might allow future additions.
                        # Update blocked
                        blocked = np.zeros(N_S, dtype=bool)
                        for idx in range(N_S):
                            if current_x[idx]:
                                for c in conflict_lists[idx]:
                                    blocked[c] = True
                        continue # Loop continues to try more adds
                    
        return current_x, current_size

    best_x = None
    best_size = 0
    
    # Run multiple restarts
    # The key is to run min_degree_greedy many times with random tie-breaking
    n_restarts = 200
    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break
            
        x, size = min_degree_greedy()
        
        # Refine with local search
        x_ref, size_ref = local_search_refine(x, max_iters=100)
        
        if size_ref > best_size:
            best_size = size_ref
            best_x = x_ref.copy()
            
        # Early stopping if we hit a theoretical upper bound? 
        # Upper bound for 4-uniform packing in dim 11 with intersection <= 2:
        # Total pairs in support: C(11,2)=55. Each support has C(4,2)=6 pairs.
        # Max supports = floor(55/6) = 9? 
        # Wait, the "intersection <= 2" condition is stronger than "disjoint pairs".
        # Two supports intersect in 2 if they share 1 pair.
        # If they share 2 pairs, they share 4 elements? No.
        # If A={1,2,3,4} and B={1,2,5,6}, intersection is {1,2} (size 2).
        # Pairs in A: (1,2),(1,3)...
        # Pairs in B: (1,2),(1,5)...
        # They share pair (1,2).
        # So the "pair counting" bound is loose because pairs can be shared by multiple supports as long as the total intersection size is <= 2.
        # Actually, if A and B share 2 elements, they share exactly 1 pair.
        # If A and B share 1 element, they share 0 pairs.
        # If A and B share 0 elements, they share 0 pairs.
        # So each support consumes 6 "pair slots" but shares 1 slot with each neighbor.
        # This is a graph problem.
        
        # Let's just run until time runs out.

    if best_x is None:
        best_x, best_size = min_degree_greedy()

    # Construct the vectors
    type_d_vectors = []
    for i in range(N_S):
        if best_x[i]:
            support = supports[i]
            for signs in itertools.product([1, -1], repeat=4):
                v = np.zeros(D, dtype=np.int64)
                for idx, s in zip(support, signs):
                    v[idx] = s
                type_d_vectors.append(v)
                
    all_vectors = type_a_vectors + type_d_vectors
    result = np.array(all_vectors, dtype=np.int64)
    
    return result

# EVOLVE-BLOCK-END
