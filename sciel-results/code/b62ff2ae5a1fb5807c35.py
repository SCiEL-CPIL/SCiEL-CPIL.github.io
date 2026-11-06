import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin

    TARGET = 200
    MAX_VAL = 100

    # Bitmask representation of reachable sums
    # Bit i is 1 if sum i is reachable
    # We care about sums 0..200.
    # A sum of 200 is invalid.

    def add_element(mask, x):
        """
        Returns new mask after adding element x, or None if 200 becomes reachable.
        """
        # If x > 200, it can't contribute to sum 200 (since all elements positive).
        # But x is in [1, 100].
        # new_mask = mask | (mask << x)
        # We need to check if bit 200 is set in the result.
        new_mask = mask | (mask << x)
        # Mask to keep only relevant bits (0..200) to prevent integer growth, though not strictly necessary for correctness if we check bit 200.
        # However, Python ints are arbitrary precision, so growth is fine but might slow down very large shifts.
        # Since x <= 100 and we only care up to 200, we can mask out bits > 200.
        # Let's use a mask for 0..200
        limit_mask = (1 << (TARGET + 1)) - 1
        new_mask &= limit_mask
        
        if (new_mask >> TARGET) & 1:
            return None
        return new_mask

    def greedy_build(order, mask_init=None):
        """Build a set using greedy addition in given order using bitmask DP."""
        if mask_init is None:
            mask = 1  # Only sum 0 is reachable
        else:
            mask = mask_init
        selected = []
        for x in order:
            new_mask = add_element(mask, x)
            if new_mask is not None:
                selected.append(x)
                mask = new_mask
        return selected, mask

    def local_search_improve(S, mask):
        """Improve set S by adding elements from complement."""
        S_set = set(S)
        all_elements = set(range(1, MAX_VAL + 1))
        complement = list(all_elements - S_set)
        improved = True
        current_S = list(S)
        current_mask = mask
        
        while improved:
            improved = False
            random.shuffle(complement)
            for x in complement:
                new_mask = add_element(current_mask, x)
                if new_mask is not None:
                    current_S.append(x)
                    S_set.add(x)
                    current_mask = new_mask
                    complement.remove(x)
                    improved = True
        return current_S, current_mask

    def local_search_with_swaps(S, mask, max_iterations=200):
        """
        Try to improve by swapping elements.
        Strategy: Remove one element, then greedily add from complement.
        If the resulting set is larger, accept.
        """
        S_set = set(S)
        all_elements = set(range(1, MAX_VAL + 1))
        current_S = list(S)
        current_mask = mask
        best_S = list(current_S)
        best_mask = current_mask
        best_size = len(current_S)
        
        for _ in range(max_iterations):
            if time.time() > deadline - 2:
                break
            if not current_S:
                break
            
            # Pick a random element to remove
            remove_idx = random.randint(0, len(current_S) - 1)
            elem_to_remove = current_S[remove_idx]
            
            # Temporarily remove
            current_S.pop(remove_idx)
            
            # Recalculate mask
            # This is O(N * 1) with bitmasks, so very fast.
            new_mask = 1
            for x in current_S:
                nm = add_element(new_mask, x)
                if nm is None:
                    # Should not happen if current_S was valid
                    break
                new_mask = nm
            
            # Now try to add elements from complement
            # Complement is all_elements - set(current_S)
            # Note: elem_to_remove is now in complement
            comp_set = all_elements - set(current_S)
            comp_list = list(comp_set)
            
            # Greedily add from comp_list in a randomized order?
            # Or sorted? Randomized is better for exploration.
            random.shuffle(comp_list)
            
            trial_S = list(current_S)
            trial_mask = new_mask
            
            for x in comp_list:
                nm = add_element(trial_mask, x)
                if nm is not None:
                    trial_S.append(x)
                    trial_mask = nm
            
            if len(trial_S) > best_size:
                best_size = len(trial_S)
                best_S = trial_S
                best_mask = trial_mask
                # Update current state to best to continue from there?
                # Or restart from original?
                # Restarting from best is a form of hill climbing.
                current_S = list(best_S)
                current_mask = best_mask
                
        return best_S, best_mask

    best_set = []
    best_size = -1
    best_mask = 1

    # Strategy 1: Standard Greedy (Descending)
    order = list(range(MAX_VAL, 0, -1))
    S, mask = greedy_build(order)
    S, mask = local_search_improve(S, mask)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S.copy()
        best_mask = mask

    if time.time() > deadline:
        return best_set if best_set else [MAX_VAL]

    # Strategy 2: Randomized Greedy + Local Search
    num_restarts = 300
    for _ in range(num_restarts):
        if time.time() > deadline - 10:
            break
        order = list(range(1, MAX_VAL + 1))
        random.shuffle(order)
        S, mask = greedy_build(order)
        S, mask = local_search_improve(S, mask)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            best_mask = mask

    if time.time() > deadline:
        return best_set if best_set else [MAX_VAL]

    # Strategy 3: Local Search with Swaps on Best Set
    if best_set:
        S, mask = local_search_with_swaps(best_set, best_mask)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            best_mask = mask

    # Strategy 4: More aggressive random restarts with swaps
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        order = list(range(1, MAX_VAL + 1))
        random.shuffle(order)
        S, mask = greedy_build(order)
        S, mask = local_search_improve(S, mask)
        S, mask = local_search_with_swaps(S, mask)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            best_mask = mask

    if not best_set:
        best_set = [MAX_VAL]
        
    return best_set
# EVOLVE-BLOCK-END
