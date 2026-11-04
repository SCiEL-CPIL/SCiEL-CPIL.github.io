"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11

    # Type A: Vectors with one non-zero entry +/- 2
    # Norm = 2.
    # We will include all 22 of these.
    # They are mutually compatible (min dist 2 for distinct, 4 for opposite).
    # They are compatible with Type D (norm 2, weight 4) because min dist is 2.
    # Proof:
    # v_A = 2 e_i. v_D has weight 4.
    # If i not in supp(v_D), dist^2 = 4 + 4 = 8.
    # If i in supp(v_D), v_D[i] is +/-1.
    # dist^2 = (2 - v_D[i])^2 + sum_{j != i} v_D[j]^2
    # = (2 - 1)^2 + 3 = 4 or (2 - (-1))^2 + 3 = 12.
    # Min dist^2 = 4 => dist = 2. OK.
    
    type_a_vectors = []
    for i in range(D):
        type_a_vectors.append(np.array([2 if j == i else 0 for j in range(D)], dtype=np.int64))
        type_a_vectors.append(np.array([-2 if j == i else 0 for j in range(D)], dtype=np.int64))
        
    # Type D: Vectors with four non-zero entries +/- 1.
    # Norm = 2.
    # We need to select a set of supports (subsets of size 4) such that any two intersect in at most 2.
    # If |intersection| <= 2, then max dot product is 2.
    # dist^2 = 4 + 4 - 2 dot = 8 - 2 dot >= 8 - 4 = 4. Dist >= 2. OK.
    # For a fixed support, we can include all 16 sign vectors because any distinct pair has dot <= 2.
    # (dot = 4 - 2 * Hamming. Distinct => Hamming >= 1 => dot <= 2).
    
    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports)
    
    support_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks.append(mask)
    
    # Conflict matrix: conflict if intersection >= 3
    # We want to find a Maximum Independent Set in this graph.
    # Or equivalently, Maximum Clique in the compatibility graph (intersection <= 2).
    
    # Let's use a simple branch and bound or greedy with local search.
    # Given the small size (330 nodes), a decent heuristic should find a large set.
    # Theoretical max?
    # Turan bound?
    # Let's try a greedy approach with multiple restarts and local search (swap).
    
    def get_conflicts(mask):
        # Return list of indices that conflict with this mask
        # Intersection >= 3
        conf = []
        for j, m in enumerate(support_masks):
            if (mask & m).bit_count() >= 3:
                conf.append(j)
        return conf

    # Precompute conflict lists
    # This is O(N_S^2 * bit_count). 330^2 * 11 ~ 1.2M ops. Fast.
    conflict_lists = []
    for i in range(N_S):
        conflict_lists.append(get_conflicts(support_masks[i]))
        
    # Convert to sets for fast lookup? Or boolean arrays?
    # Boolean arrays are faster for vectorized operations if we do local search.
    # But for B&B, lists might be better.
    # Let's stick to a simple local search for now, it's robust.
    
    def greedy_init():
        x = np.zeros(N_S, dtype=bool)
        remaining = set(range(N_S))
        # Randomize order
        order = list(range(N_S))
        np.random.shuffle(order)
        for v in order:
            if v not in remaining: continue
            # Check if v conflicts with any selected
            # We can just check if x[v] is False and no conflict with selected
            # Actually, if we maintain 'blocked' set, it's faster.
            pass
            
        # Better greedy:
        x = np.zeros(N_S, dtype=bool)
        blocked = np.zeros(N_S, dtype=bool)
        order = list(range(N_S))
        np.random.shuffle(order)
        for v in order:
            if not x[v] and not blocked[v]:
                x[v] = True
                # Block all conflicts of v
                for c in conflict_lists[v]:
                    blocked[c] = True
        return x

    def local_search(x, max_iters=5000):
        # 1-swap: remove i, add j
        # Or 1-add: add j (if not blocked)
        # Or 1-remove: remove i (if it allows adding more later?)
        
        # Let's try 1-swap.
        # For each i in x, for each j not in x:
        # If we remove i, does j become unblocked?
        # j is blocked if it conflicts with any k in x.
        # If we remove i, j is unblocked if j did not conflict with any k in x \ {i}.
        # i.e., all conflicts of j in x must be exactly {i}.
        # And j must not conflict with itself (trivial).
        
        # This is expensive to check naively.
        # Let's try a simpler move: just try to add a node if possible.
        # Then try to remove a node that "frees up" the most nodes.
        
        current_x = x.copy()
        current_size = np.sum(current_x)
        
        # Compute blocked status
        blocked = np.zeros(N_S, dtype=bool)
        for i in range(N_S):
            if current_x[i]:
                for c in conflict_lists[i]:
                    blocked[c] = True
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            
            # Try to add a node
            # Find a node not in x and not blocked
            # To be smarter, pick one that blocks the fewest others?
            # Or just the first one.
            
            # Let's try swaps.
            # For each i in x:
            #   For each j not in x:
            #     If j is blocked only by i (among x), then swapping i->j is valid.
            #     Actually, we need j to not conflict with any k in x \ {i}.
            
            # Efficient check:
            # For each j not in x:
            #   Find conflicts of j in x.
            #   If len(conflicts) == 1, say {i}, then we can swap i->j.
            #   This maintains size.
            #   But we want to increase size.
            #   So first try to add.
            
            # Add:
            # Find j not in x, not blocked.
            # If exists, add it.
            
            # Since we recompute blocked after each move, this is slow.
            # Let's use a different local search:
            # Randomly remove a node, then greedily add as many as possible.
            
            if time.time() > deadline - 5:
                break
                
            # Pick a random node in x to remove
            selected_indices = np.where(current_x)[0]
            if len(selected_indices) == 0:
                break
            i_to_remove = selected_indices[np.random.randint(len(selected_indices))]
            current_x[i_to_remove] = False
            
            # Recompute blocked
            blocked = np.zeros(N_S, dtype=bool)
            for i in range(N_S):
                if current_x[i]:
                    for c in conflict_lists[i]:
                        blocked[c] = True
                        
            # Greedily add
            order = list(range(N_S))
            np.random.shuffle(order)
            for v in order:
                if not current_x[v] and not blocked[v]:
                    current_x[v] = True
                    for c in conflict_lists[v]:
                        blocked[c] = True
            # Note: we need to update blocked when we add v.
            # The loop above does that.
            
            new_size = np.sum(current_x)
            if new_size > current_size:
                current_size = new_size
                improved = True
            else:
                # Revert? Or keep?
                # If we didn't improve, maybe revert to previous state?
                # But we lost the previous state.
                # Let's just keep it, it might lead to a better state later.
                pass
                
        return current_x

    best_x = None
    best_size = 0
    
    # Run multiple restarts
    n_restarts = 50
    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break
            
        x = greedy_init()
        size = np.sum(x)
        
        # Local search
        x_ls = local_search(x, max_iters=1000)
        size_ls = np.sum(x_ls)
        
        if size_ls > best_size:
            best_size = size_ls
            best_x = x_ls.copy()
            
        if size > best_size:
            best_size = size
            best_x = x.copy()

    # If best_x is None, use a single greedy
    if best_x is None:
        best_x = greedy_init()
        best_size = np.sum(best_x)

    # Construct the vectors
    # For each selected support, add all 16 sign vectors.
    type_d_vectors = []
    for i in range(N_S):
        if best_x[i]:
            support = supports[i]
            # Generate all 16 sign vectors
            for signs in itertools.product([1, -1], repeat=4):
                v = np.zeros(D, dtype=np.int64)
                for idx, s in zip(support, signs):
                    v[idx] = s
                type_d_vectors.append(v)
                
    all_vectors = type_a_vectors + type_d_vectors
    result = np.array(all_vectors, dtype=np.int64)
    
    # Final validation check (optional, but good for safety)
    # Check min pairwise distance >= max norm
    # Max norm is 2.
    # We trust the construction logic.
    
    return result

# EVOLVE-BLOCK-END
