import time
import random
import math

def construct():
    deadline = time.time() + 55  # Safety margin before 90s kill
    
    N = 256
    M = N + 1  # 1-indexed
    
    # Precompute 3-APs and 4-APs
    three_aps = []
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            three_aps.append((a, a + d, a + 2 * d))
    
    four_aps = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Data structures for fast lookup
    # For each element x, list of (ap_idx, position) for 3-APs
    elem_to_3aps = [[] for _ in range(M)]
    for i, (a, b, c) in enumerate(three_aps):
        elem_to_3aps[a].append((i, 0))
        elem_to_3aps[b].append((i, 1))
        elem_to_3aps[c].append((i, 2))
        
    # For each element x, list of (ap_idx, position) for 4-APs
    elem_to_4aps = [[] for _ in range(M)]
    for i, (a, b, c, d) in enumerate(four_aps):
        elem_to_4aps[a].append((i, 0))
        elem_to_4aps[b].append((i, 1))
        elem_to_4aps[c].append((i, 2))
        elem_to_4aps[d].append((i, 3))
        
    # State
    in_set = [False] * M
    current_score = 0
    
    # Helper to check validity of adding x
    # Returns True if valid, and the delta in score
    def try_add(x):
        if in_set[x]:
            return False, 0
        # Check 4-AP constraints
        for ap_idx, pos in elem_to_4aps[x]:
            a, b, c, d = four_aps[ap_idx]
            # Check if the other 3 are in set
            if pos == 0:
                if in_set[b] and in_set[c] and in_set[d]:
                    return False, 0
            elif pos == 1:
                if in_set[a] and in_set[c] and in_set[d]:
                    return False, 0
            elif pos == 2:
                if in_set[a] and in_set[b] and in_set[d]:
                    return False, 0
            else:
                if in_set[a] and in_set[b] and in_set[c]:
                    return False, 0
                    
        # Calculate score delta
        delta = 0
        for ap_idx, pos in elem_to_3aps[x]:
            a, b, c = three_aps[ap_idx]
            if pos == 0:
                if in_set[b] and in_set[c]:
                    delta += 1
            elif pos == 1:
                if in_set[a] and in_set[c]:
                    delta += 1
            else:
                if in_set[a] and in_set[b]:
                    delta += 1
        return True, delta

    def try_remove(x):
        if not in_set[x]:
            return False, 0
        # Calculate score loss
        delta = 0
        for ap_idx, pos in elem_to_3aps[x]:
            a, b, c = three_aps[ap_idx]
            if pos == 0:
                if in_set[b] and in_set[c]:
                    delta += 1
            elif pos == 1:
                if in_set[a] and in_set[c]:
                    delta += 1
            else:
                if in_set[a] and in_set[b]:
                    delta += 1
        return True, -delta

    def try_swap(x_out, x_in):
        if in_set[x_out] != True or in_set[x_in] != False:
            return False, 0
        # Temporarily remove x_out
        in_set[x_out] = False
        valid, delta_add = try_add(x_in)
        if valid:
            # Calculate loss from removing x_out (now that x_in is not in set, but x_out is)
            # Wait, try_add checks validity based on current state.
            # If valid, we need to calculate the net change.
            # Net change = (Gain from x_in) - (Loss from x_out)
            # But "Loss from x_out" depends on whether x_in was there? No, x_in is not there yet.
            # So Loss is calculated on state WITHOUT x_out.
            # We already removed x_out.
            # So we just need to calculate the loss that x_out *would* have caused in this new state.
            # Actually, it's easier to just calculate delta_add and delta_remove separately on the original state?
            # No, because they interact.
            
            # Let's do this:
            # 1. Remove x_out (done)
            # 2. Check if x_in can be added (done, valid=True)
            # 3. If valid, add x_in
            in_set[x_in] = True
            # 4. Calculate score change
            # Score change = (New 3-APs with x_in) - (Old 3-APs with x_out)
            # We already have delta_add (New 3-APs with x_in in state without x_out).
            # We need Old 3-APs with x_out in state with x_out (and without x_in).
            # Let's calculate loss_remove on the state BEFORE removal?
            # It's safer to just recalculate or store the loss before removal.
            
            # Let's refactor: 
            # 1. Calculate loss_remove on current state (with x_out, without x_in)
            # 2. Remove x_out
            # 3. Check validity of x_in
            # 4. If valid, calculate gain_add on new state (without x_out, without x_in)
            # 5. Add x_in
            # 6. Return valid, gain - loss
            
            pass # This logic is getting messy in the helper. Let's implement swap directly in the main loop for clarity or fix this helper.
            
            # For now, revert and return invalid to be safe, or implement correctly.
            # Let's implement correctly in the main loop to avoid complex state management in helpers.
            in_set[x_in] = False
            in_set[x_out] = True
            return False, 0
        else:
            in_set[x_out] = True
            return False, 0

    # Greedy Initialization
    # Sort elements by some heuristic? Or just random order?
    # Let's try a greedy approach: add elements that have high potential 3-APs but low 4-AP risks.
    # Simple greedy: iterate 1 to N, add if valid.
    for x in range(1, N + 1):
        valid, delta = try_add(x)
        if valid:
            in_set[x] = True
            current_score += delta

    best_score = current_score
    best_in_set = in_set[:]
    
    # Simulated Annealing
    random.seed(42)
    T = 10.0
    cooling = 0.9995
    min_T = 0.01
    all_elements = list(range(1, N + 1))
    
    # Precompute lists for faster access?
    # in_set is a list of bools.
    
    while time.time() < deadline:
        if T < min_T:
            T = min_T
            
        move_type = random.random()
        
        if move_type < 0.35:  # Add
            candidates = [x for x in all_elements if not in_set[x]]
            if candidates:
                x = random.choice(candidates)
                valid, delta = try_add(x)
                if valid:
                    if delta > 0 or (delta < 0 and random.random() < math.exp(delta / T)) or (delta == 0 and random.random() < 0.3):
                        in_set[x] = True
                        current_score += delta
                        if current_score > best_score:
                            best_score = current_score
                            best_in_set = in_set[:]
                            
        elif move_type < 0.55:  # Remove
            if any(in_set[1:]):
                x = random.choice([x for x in all_elements if in_set[x]])
                valid, delta = try_remove(x)
                if valid:
                    if random.random() < math.exp(delta / T):
                        in_set[x] = False
                        current_score += delta
                        
        else:  # Swap
            if any(in_set[1:]):
                x_out = random.choice([x for x in all_elements if in_set[x]])
                candidates = [x for x in all_elements if not in_set[x]]
                if candidates:
                    x_in = random.choice(candidates)
                    # Check validity of swap
                    # 1. Remove x_out
                    in_set[x_out] = False
                    # 2. Check if x_in is valid
                    valid_add, delta_add = try_add(x_in)
                    if valid_add:
                        # Calculate loss from x_out in the state BEFORE removal?
                        # We need to know how many 3-APs x_out was part of in the original state.
                        # Let's calculate it now? No, state is changed.
                        # Let's calculate loss on original state first.
                        # This requires saving state or recalculating.
                        # Let's just do:
                        # loss = count_3aps_involving(x_out) in original state
                        # gain = count_3aps_involving(x_in) in new state (without x_out)
                        
                        # To avoid complex state saving, let's just do a simple check:
                        # If valid_add is True, we tentatively accept?
                        # No, we need the delta.
                        
                        # Let's compute loss on the fly by re-adding x_out? No.
                        # Let's compute loss before removing x_out.
                        # Refactor Swap Logic:
                        # 1. Calculate loss_remove on current state (with x_out)
                        # 2. Remove x_out
                        # 3. Check validity of x_in
                        # 4. If valid, calculate gain_add on new state
                        # 5. Add x_in
                        # 6. Update score
                        # 7. If rejected, revert
                        
                        # I already removed x_out in the code above.
                        # Let's fix the swap logic properly.
                        
                        # Revert x_out removal for now to restart logic cleanly?
                        # No, let's just compute loss here? 
                        # We don't have the original state.
                        # Let's just use a simple heuristic: accept if delta_add > 0?
                        # Or just accept if valid?
                        # Let's try to calculate loss by iterating x_out's 3-APs again, but checking if the other elements are in the CURRENT state (which is without x_out).
                        # This is WRONG because the other elements might include x_in? No, x_in is not added yet.
                        # But the "other elements" for x_out's APs are the same set as before, just x_out is gone.
                        # So the loss is the number of 3-APs containing x_out where the other two are in the set.
                        # This is exactly what try_remove calculates, BUT try_remove assumes x_out is IN the set.
                        # So we can't use try_remove now.
                        
                        # Let's just calculate the loss manually:
                        loss = 0
                        for ap_idx, pos in elem_to_3aps[x_out]:
                            a, b, c = three_aps[ap_idx]
                            if pos == 0:
                                if in_set[b] and in_set[c]:
                                    loss += 1
                            elif pos == 1:
                                if in_set[a] and in_set[c]:
                                    loss += 1
                            else:
                                if in_set[a] and in_set[b]:
                                    loss += 1
                                    
                        net_delta = delta_add - loss
                        if net_delta > 0 or random.random() < math.exp(net_delta / T):
                            in_set[x_in] = True
                            current_score += net_delta
                            if current_score > best_score:
                                best_score = current_score
                                best_in_set = in_set[:]
                        else:
                            # Revert: x_out is already removed, x_in is not added.
                            # We need to put x_out back.
                            in_set[x_out] = True
                    else:
                        # Invalid, revert x_out removal
                        in_set[x_out] = True

        T *= cooling

    # Final Polish: Hill Climbing
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try additions
        for x in all_elements:
            if not in_set[x]:
                valid, delta = try_add(x)
                if valid and delta > 0:
                    in_set[x] = True
                    current_score += delta
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_in_set = in_set[:]
        
        # Try swaps
        in_set_list = [x for x in all_elements if in_set[x]]
        not_in_set_list = [x for x in all_elements if not in_set[x]]
        for x_out in in_set_list:
            for x_in in not_in_set_list:
                # Check validity
                in_set[x_out] = False
                valid_add, delta_add = try_add(x_in)
                if valid_add:
                    # Calculate loss
                    loss = 0
                    for ap_idx, pos in elem_to_3aps[x_out]:
                        a, b, c = three_aps[ap_idx]
                        if pos == 0:
                            if in_set[b] and in_set[c]:
                                loss += 1
                        elif pos == 1:
                            if in_set[a] and in_set[c]:
                                loss += 1
                        else:
                            if in_set[a] and in_set[b]:
                                loss += 1
                    net_delta = delta_add - loss
                    if net_delta > 0:
                        in_set[x_in] = True
                        current_score += net_delta
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_in_set = in_set[:]
                        break
                    else:
                        in_set[x_out] = True
                else:
                    in_set[x_out] = True
            if improved:
                break

    # Convert best_in_set to sorted list
    result = []
    for x in range(1, N + 1):
        if best_in_set[x]:
            result.append(x)
    return sorted(result)
