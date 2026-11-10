# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a maximum-size subset of Z/205Z with no two elements differing by a
    nonzero square modulo 205.

    Strategy:
      1. Build the forbidden-difference set F (nonzero quadratic residues mod 205).
      2. Use a randomized greedy construction with multiple random orderings to
         obtain diverse feasible subsets, accelerated by bitset conflict masks.
      3. Apply local search (additions and swaps) using vectorized NumPy operations
         to improve size efficiently.
      4. Track the best feasible subset found and return it.

    Returns:
        np.ndarray of distinct residues in 0..204, no pair has a forbidden difference.
    """
    m = 205
    deadline = time.time() + 580  # safety margin before 630s kill

    # Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)

    # Precompute compatibility matrix
    # compat[i, j] = 1 if i and j can coexist (difference not in F)
    # Note: The problem states "neither (a-b) mod 205 nor (b-a) mod 205 may be in F".
    # Since F contains squares, and (-x)^2 = x^2, if d is in F, then -d mod m is also in F?
    # Let's check: if d = x^2 mod m, is -d a square?
    # In Z/205Z, -1 is a square iff -1 is a square mod 5 and mod 41.
    # -1 mod 5 = 4 = 2^2. Yes.
    # -1 mod 41. 41 = 4*10 + 1. -1 is a square mod p if p % 4 == 1. 41 % 4 == 1. Yes.
    # So -1 is a square mod 205. Thus if d is a square, -d is a square.
    # So the condition "d not in F" is symmetric.
    
    compat = np.ones((m, m), dtype=bool)
    for i in range(m):
        for j in range(i + 1, m):
            d1 = (i - j) % m
            # Since -1 is a square, d1 in F implies (j-i)%m in F.
            if d1 in forbidden:
                compat[i, j] = False
                compat[j, i] = False

    # For bitset operations, we need to handle the "self" conflict.
    # We create a conflict mask where conflict[i, j] = True if i and j CANNOT coexist.
    # Note: A node cannot coexist with itself? No, distinct residues.
    # But for the logic "can I add x to S?", we check if x conflicts with ANY s in S.
    # Let's define `conflict_mask` such that `conflict_mask[x, s]` is True if x and s conflict.
    # This is just `~compat` but with diagonal True? 
    # Actually, if we maintain a bitset `active` of selected nodes, 
    # the set of nodes that conflict with `active` is `conflict_mask[:, active].any(axis=1)`.
    
    # Let's use a simpler representation for speed with m=205.
    # `conflict_mask` is a 205x205 boolean matrix.
    conflict_mask = ~compat
    # A node does not conflict with itself in the context of "distinct elements",
    # but we never add the same node twice.
    # However, if we check `conflict_mask[cand, selected]`, and `cand` is in `selected`,
    # it doesn't matter because we skip candidates in `selected`.
    
    def is_valid_subset(subset):
        """Check if subset is a valid square-difference-free set."""
        if not subset:
            return True
        arr = np.array(subset)
        # Check all pairs
        for i in range(len(arr)):
            for j in range(i+1, len(arr)):
                if conflict_mask[arr[i], arr[j]]:
                    return False
        return True

    def greedy_random_order(order):
        """
        Greedy construction using bitset masks for fast conflict checking.
        """
        selected = []
        selected_set = set()
        # active_mask is a boolean array of length m indicating which nodes are selected
        active_mask = np.zeros(m, dtype=bool)
        
        # candidates_mask indicates which nodes are still "available" (compatible with all selected)
        # Initially all are available.
        candidates_mask = np.ones(m, dtype=bool)
        
        for candidate in order:
            if not candidates_mask[candidate]:
                continue
            
            # If candidate is compatible with all selected, add it.
            # We maintain candidates_mask such that candidates_mask[i] is True iff i is compatible with ALL selected.
            # So we just need to check candidates_mask[candidate].
            
            selected.append(candidate)
            selected_set.add(candidate)
            active_mask[candidate] = True
            
            # Update candidates_mask:
            # Any node that conflicts with the new candidate must be removed from candidates.
            # conflict_mask[new_candidate, :] gives nodes that conflict with new_candidate.
            # We want to remove them from candidates_mask.
            # candidates_mask = candidates_mask & ~conflict_mask[new_candidate, :]
            
            # Optimization: Only update if there are many candidates left? 
            # m is small, vectorized op is fast.
            candidates_mask &= ~conflict_mask[candidate, :]
            
            # Also, the candidate itself is no longer a candidate (it's selected)
            # Though the loop iterates over `order` which is a permutation, so we won't see it again.
            # But for safety/cleanliness:
            candidates_mask[candidate] = False

        return selected

    def local_search_improve(subset, max_iter=500):
        """
        Improve subset by adding elements or swapping.
        Uses vectorized operations to find valid additions/swaps.
        """
        if not subset:
            return []
            
        selected = list(subset)
        selected_set = set(selected)
        active_mask = np.zeros(m, dtype=bool)
        for s in selected:
            active_mask[s] = True
            
        improved = True
        iters = 0
        
        while improved and iters < max_iter and time.time() < deadline:
            improved = False
            iters += 1
            
            # 1. Try adding any unselected element
            # Find all nodes not in selected_set that are compatible with ALL selected.
            # compatible_with_all = ~conflict_mask[:, active_mask].any(axis=1)
            # We need to exclude already selected nodes.
            
            # Compute conflict with active set
            # conflict_with_active is True for nodes that conflict with at least one active node
            conflict_with_active = np.any(conflict_mask[:, active_mask], axis=1)
            
            # Candidates are those NOT in conflict_with_active AND NOT in selected_set
            # We can use active_mask to filter out selected ones if we treat "not selected" as ~active_mask
            # But wait, active_mask is True for selected.
            # So valid_adds = ~conflict_with_active & ~active_mask
            
            valid_adds = ~conflict_with_active & ~active_mask
            add_indices = np.nonzero(valid_adds)[0]
            
            if len(add_indices) > 0:
                # Add the first one (or random one? First is deterministic and fast)
                # Adding any valid node increases size.
                cand = add_indices[0]
                selected.append(cand)
                selected_set.add(cand)
                active_mask[cand] = True
                improved = True
                continue
            
            # 2. Try swapping: remove one element, add another
            # For each element r in selected, compute what we can add if r is removed.
            # New active set A' = A \ {r}.
            # We need to find c not in A' such that c is compatible with all in A'.
            # This is equivalent to: c is compatible with all in A, EXCEPT possibly r.
            # Or: c is compatible with A \ {r}.
            
            # To speed this up, we can precompute for each r, the set of nodes that ONLY conflict with r (and no other active node)?
            # No, that's not quite right.
            # A node c is valid for swap with r if:
            # 1. c is not in A
            # 2. c is compatible with all s in A such that s != r.
            
            # Let's iterate through r in selected.
            # m is 205, selected size is ~20-30.
            # For each r, we can compute the conflict mask with A \ {r}.
            
            for r in selected:
                if time.time() >= deadline:
                    break
                
                # Create a mask for A \ {r}
                active_mask_minus_r = active_mask.copy()
                active_mask_minus_r[r] = False
                
                # Compute conflicts with A \ {r}
                # If A \ {r} is empty, all nodes are compatible.
                if not np.any(active_mask_minus_r):
                    # Can add any node not in A
                    valid_adds = ~active_mask
                    add_indices = np.nonzero(valid_adds)[0]
                else:
                    conflict_with_active_minus_r = np.any(conflict_mask[:, active_mask_minus_r], axis=1)
                    valid_adds = ~conflict_with_active_minus_r & ~active_mask
                    add_indices = np.nonzero(valid_adds)[0]
                
                if len(add_indices) > 0:
                    # We found a node c that can be added if we remove r.
                    # Does this increase size? No, size stays same.
                    # Does it help? Maybe it allows future additions.
                    # Heuristic: Swap if it leads to a "better" position?
                    # For now, just swap to escape local optima.
                    # Pick a random candidate from add_indices to diversify?
                    # Or pick the one that allows most future additions? (Expensive)
                    # Let's pick the first one for speed, or random.
                    # Random might be better for exploration.
                    idx = np.random.randint(len(add_indices))
                    cand = add_indices[idx]
                    
                    # Perform swap
                    selected.remove(r)
                    selected_set.remove(r)
                    active_mask[r] = False
                    
                    selected.append(cand)
                    selected_set.add(cand)
                    active_mask[cand] = True
                    
                    improved = True
                    break
            
        return selected

    # Multiple random restarts with greedy + local search
    best_subset = []
    best_size = 0

    # First, try a deterministic greedy in natural order
    init_subset = greedy_random_order(list(range(m)))
    init_subset = local_search_improve(init_subset)
    if len(init_subset) > best_size:
        best_size = len(init_subset)
        best_subset = list(init_subset)

    # Randomized greedy with local search
    num_restarts = 2000 # Increase restarts due to speedup
    for r in range(num_restarts):
        if time.time() >= deadline:
            break
        random.seed(r * 12345 + 67890)
        order = list(range(m))
        random.shuffle(order)
        subset = greedy_random_order(order)
        subset = local_search_improve(subset, max_iter=500)
        if len(subset) > best_size:
            best_size = len(subset)
            best_subset = list(subset)

    # Additional local search passes on the best subset found
    if time.time() < deadline - 30:
        improved = True
        while improved and time.time() < deadline - 30:
            improved = False
            best_subset = local_search_improve(best_subset, max_iter=1000)
            if len(best_subset) > best_size:
                best_size = len(best_subset)
                improved = True

    # Final validation
    if not is_valid_subset(best_subset):
        # Fallback to a known valid small set
        best_subset = [0]
        best_size = 1

    return np.array(best_subset, dtype=np.int64)


# EVOLVE-BLOCK-END
