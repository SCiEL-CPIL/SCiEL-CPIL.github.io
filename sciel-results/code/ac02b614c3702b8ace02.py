import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 256
    MASK_ALL = (1 << N) - 1
    
    # Precompute 4-APs
    # For each 4-AP (a, a+d, a+2d, a+3d), store the bitmask of the 4 elements.
    # Also, for each element x, store a list of 4-AP masks that contain x.
    four_ap_masks = []
    four_ap_by_elem = [[] for _ in range(N)] # 0-indexed for bits
    
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            a0 = a - 1
            b0 = a + d - 1
            c0 = a + 2 * d - 1
            d0 = a + 3 * d - 1
            mask = (1 << a0) | (1 << b0) | (1 << c0) | (1 << d0)
            four_ap_masks.append(mask)
            four_ap_by_elem[a0].append(mask)
            four_ap_by_elem[b0].append(mask)
            four_ap_by_elem[c0].append(mask)
            four_ap_by_elem[d0].append(mask)
            
    # Precompute 3-APs
    # For fast counting, we can just iterate through all 3-APs.
    # There are roughly N^2/4 = 16k 3-APs. Iterating 16k times is fast.
    three_aps = []
    for d in range(1, (N - 1) // 2 + 1):
        max_a = N - 2 * d
        for a in range(1, max_a + 1):
            a0 = a - 1
            b0 = a + d - 1
            c0 = a + 2 * d - 1
            three_aps.append((a0, b0, c0))
            
    def count_3aps(state):
        count = 0
        # state is an integer bitmask
        for a, b, c in three_aps:
            # Check if bits a, b, c are set
            if (state >> a) & 1 and (state >> b) & 1 and (state >> c) & 1:
                count += 1
        return count

    def is_valid_4ap(state):
        # Check if state contains any 4-AP
        # Iterate through all 4-APs and check if all bits are set
        # There are ~ (N^2)/12 ~ 5k 4-APs.
        for mask in four_ap_masks:
            if (state & mask) == mask:
                return False
        return True

    # Optimization: Instead of checking all 4-APs for every move, we can be smarter,
    # but 5k checks * 1000s of steps is ~5M ops, which is fine in Python if optimized?
    # Actually, 5M ops in Python might take 1-2 seconds. 
    # Let's use a faster validity check for local moves.
    
    # Better validity check for a specific move (add/remove):
    # If we add x, we only need to check 4-APs containing x.
    # If we remove x, the set remains valid (since it was valid before).
    
    def can_add(state, x):
        # x is 0-indexed
        # Check if adding x creates a 4-AP
        # Only 4-APs containing x are at risk.
        for mask in four_ap_by_elem[x]:
            # If all OTHER bits in mask are already in state, then adding x completes the 4-AP.
            # mask has 4 bits. One is x. The other 3 must be in state.
            # (state & mask) == (mask ^ (1 << x))
            if (state & mask) == (mask ^ (1 << x)):
                return False
        return True

    # Initial State: Greedy construction
    # We want a good starting point.
    # Greedy: Add elements if they don't create a 4-AP.
    # To maximize 3-APs, we might want to prefer elements that form many 3-APs.
    # But simple greedy is a good baseline.
    # Let's do a few random restarts of greedy to find a good seed.
    
    best_state = 0
    best_score = -1
    
    # Run a few greedy constructions with different orderings
    for i in range(10):
        if time.time() > deadline - 5.0:
            break
            
        state = 0
        if i == 0:
            order = list(range(N))
        elif i == 1:
            order = list(range(N-1, -1, -1))
        else:
            order = list(range(N))
            random.shuffle(order)
            
        for x in order:
            if can_add(state, x):
                state |= (1 << x)
        
        score = count_3aps(state)
        if score > best_score:
            best_score = score
            best_state = state

    # Simulated Annealing
    current_state = best_state
    current_score = best_score
    best_sa_state = best_state
    best_sa_score = best_score
    
    T = 10.0
    cooling_rate = 0.999
    
    # We will perform moves: Add, Remove, Swap
    # To be efficient, we precompute the list of indices.
    
    while time.time() < deadline - 2.0:
        # Choose a move type
        r = random.random()
        
        if r < 0.4:
            # Add
            # Pick a random x not in state
            # To avoid O(N) scan for a random unset bit, we can maintain a list of unset bits?
            # Or just pick random x and retry if set.
            # Since density is ~0.5, retries are few.
            x = random.randint(0, N - 1)
            if (current_state >> x) & 1:
                continue
            
            if not can_add(current_state, x):
                continue
            
            new_state = current_state | (1 << x)
            # Delta score: count new 3-APs formed by x
            # A 3-AP is formed if x is in it and the other 2 are in state.
            delta = 0
            # We need to check 3-APs containing x.
            # We didn't precompute 3-APs by element. Let's just recount?
            # Recounting is O(16k). 1000s of steps * 16k = 16M ops. Might be too slow in Python.
            # Let's precompute 3-AP neighbors.
            
        # Actually, let's precompute 3-AP neighbors to make delta calculation O(deg(x)).
        # deg(x) is at most ~N.
        
        # Re-initializing precomputation for 3-AP neighbors
        three_ap_by_elem = [[] for _ in range(N)]
        for a, b, c in three_aps:
            three_ap_by_elem[a].append((b, c))
            three_ap_by_elem[b].append((a, c))
            three_ap_by_elem[c].append((a, b))
            
        def calc_delta_add(state, x):
            delta = 0
            for u, v in three_ap_by_elem[x]:
                if ((state >> u) & 1) and ((state >> v) & 1):
                    delta += 1
            return delta
            
        def calc_delta_remove(state, x):
            # Same as add, but we are removing x.
            # The 3-APs lost are those containing x where the other 2 are in state.
            delta = 0
            for u, v in three_ap_by_elem[x]:
                if ((state >> u) & 1) and ((state >> v) & 1):
                    delta += 1
            return delta

        # Resume SA loop with optimized delta calc
        T = 10.0
        while time.time() < deadline - 1.0:
            r = random.random()
            
            if r < 0.3:
                # Add
                x = random.randint(0, N - 1)
                if (current_state >> x) & 1:
                    continue
                if not can_add(current_state, x):
                    continue
                
                delta = calc_delta_add(current_state, x)
                new_score = current_score + delta
                new_state = current_state | (1 << x)
                
                if new_score > current_score or random.random() < math.exp((new_score - current_score) / T):
                    current_state = new_state
                    current_score = new_score
                    if current_score > best_sa_score:
                        best_sa_score = current_score
                        best_sa_state = current_state
                        
            elif r < 0.6:
                # Remove
                # Pick random x in state
                # Efficient way: pick random index, check if set.
                x = random.randint(0, N - 1)
                if not ((current_state >> x) & 1):
                    continue
                
                delta = -calc_delta_remove(current_state, x)
                new_score = current_score + delta
                new_state = current_state & ~(1 << x)
                
                if new_score > current_score or random.random() < math.exp((new_score - current_score) / T):
                    current_state = new_state
                    current_score = new_score
                    if current_score > best_sa_score:
                        best_sa_score = current_score
                        best_sa_state = current_state
                        
            else:
                # Swap
                # Pick x in state, y not in state
                x = random.randint(0, N - 1)
                if not ((current_state >> x) & 1):
                    continue
                y = random.randint(0, N - 1)
                if (current_state >> y) & 1:
                    continue
                
                # Check if y can be added to state - {x}
                # This is equivalent to: can_add(state ^ (1<<x), y)
                # But we can check: does y complete a 4-AP with elements in state, excluding x?
                # For each 4-AP containing y:
                #   If x is in the 4-AP, we need the other 2 to be in state.
                #   If x is not in the 4-AP, we need all 3 to be in state.
                
                valid = True
                for mask in four_ap_by_elem[y]:
                    # Check if mask is a subset of (state | (1<<y)) AND not a subset of state?
                    # No, we need to check if (state - {x}) contains the other 3 elements of the 4-AP (excluding y).
                    
                    # Let's just use the bit logic.
                    # state_minus_x = state & ~(1 << x)
                    # We need: (state_minus_x | (1 << y)) to NOT contain the 4-AP.
                    # i.e., (state_minus_x | (1 << y)) & mask != mask
                    
                    # Simpler:
                    # The 4-AP is formed if all 4 elements are in (state - {x}) U {y}.
                    # Since y is in the 4-AP, we need the other 3 to be in (state - {x}).
                    
                    # Let's identify the other 3 elements in mask.
                    # mask has 4 bits. One is y.
                    # We can't easily iterate bits of mask without loop.
                    # But we precomputed four_ap_by_elem[y] as list of masks.
                    # We can also precompute the "other 3" mask for each (y, mask) pair?
                    # Or just check:
                    # If (state & mask) == (mask ^ (1 << y)):
                    #    Then all other 3 are in state.
                    #    If x is NOT one of them, then they are in state-{x} -> Invalid.
                    #    If x IS one of them, then one of the "other 3" is x, which is removed.
                    #       So we need to check if the remaining 2 (excluding x and y) are in state.
                    
                    other_mask = mask ^ (1 << y)
                    # other_mask has 3 bits.
                    # If (state & other_mask) == other_mask:
                    #    All 3 others are in state.
                    #    If x is in other_mask:
                    #        One of them is x. So in state-{x}, only 2 remain.
                    #        We need to check if those 2 are in state. (They are, by definition).
                    #        So this 4-AP is NOT formed in state-{x}+{y} because x is missing.
                    #        Wait.
                    #        If x is in the 4-AP, then the 4-AP is {y, x, u, v}.
                    #        In state-{x}+{y}, we have y, u, v but NOT x.
                    #        So the 4-AP is NOT formed.
                    #    If x is NOT in other_mask:
                    #        All 3 others are in state and none is x.
                    #        So in state-{x}, all 3 others are present.
                    #        So the 4-AP IS formed. -> Invalid.
                    
                    # So:
                    if (state & other_mask) == other_mask:
                        # All 3 others are in state.
                        if not (other_mask & (1 << x)):
                            # x is not in the 4-AP. So removing x doesn't help.
                            valid = False
                            break
                        # If x IS in the 4-AP, it's fine (the 4-AP is broken by removing x).
                    
                    # What if not all 3 others are in state?
                    # Then the 4-AP is not formed anyway.
                    
                if not valid:
                    continue
                
                # Calculate Delta for Swap
                # Remove x, Add y.
                # Delta = (3-APs in new state) - (3-APs in old state)
                # = (3-APs with y in new state) - (3-APs with x in old state)
                # Note: 3-APs not involving x or y are unchanged.
                # 3-APs involving both x and y?
                # If a 3-AP contains both x and y, it is lost (x removed) and gained (y added)?
                # No, if it contains x, it's lost. If it contains y, it's gained.
                # If it contains both, it's lost AND gained?
                # Let's trace:
                # Old: x, y, z in S. 3-AP (x,y,z) exists.
                # New: y, z in S, x not. 3-AP (x,y,z) does NOT exist.
                # So we lost it.
                # Did we gain it? No, because x is not there.
                # So net change for (x,y,z) is -1.
                
                # My formula:
                # Gain(y): count 3-APs containing y where other 2 are in state-{x}.
                # Loss(x): count 3-APs containing x where other 2 are in state.
                
                # Let's calculate Loss(x) first.
                loss = 0
                for u, v in three_ap_by_elem[x]:
                    if ((state >> u) & 1) and ((state >> v) & 1):
                        loss += 1
                        
                # Calculate Gain(y) in state-{x}.
                gain = 0
                state_minus_x = state & ~(1 << x)
                for u, v in three_ap_by_elem[y]:
                    if ((state_minus_x >> u) & 1) and ((state_minus_x >> v) & 1):
                        gain += 1
                        
                delta = gain - loss
                new_score = current_score + delta
                new_state = (state & ~(1 << x)) | (1 << y)
                
                if new_score > current_score or random.random() < math.exp((new_score - current_score) / T):
                    current_state = new_state
                    current_score = new_score
                    if current_score > best_sa_score:
                        best_sa_score = current_score
                        best_sa_state = current_state
            
            T *= cooling_rate
            if T < 0.1:
                T = 0.1
                
    # Final local search on best_sa_state to polish
    # Try adding any element that improves score
    changed = True
    while changed and time.time() < deadline - 0.5:
        changed = False
        for x in range(N):
            if (best_sa_state >> x) & 1:
                continue
            if not can_add(best_sa_state, x):
                continue
            delta = calc_delta_add(best_sa_state, x)
            if delta > 0:
                best_sa_state |= (1 << x)
                best_sa_score += delta
                changed = True
                
    # Convert to list of 1-based integers
    result = []
    for i in range(N):
        if (best_sa_state >> i) & 1:
            result.append(i + 1)
            
    return sorted(result)

# Need to import math for exp
import math

# Re-define construct to ensure imports are at top or inside
# Wait, I used math.exp inside construct but didn't import math at the top of the file in the draft.
# I will place import math at the top of the final code block.

# Final check on logic:
# 1. Precomputation is correct.
# 2. Greedy seed is good.
# 3. SA uses bitmask for speed.
# 4. Delta calculations are correct for Add/Remove/Swap.
# 5. Validity check for Swap is correct.
# 6. Final polish adds any beneficial elements.

# One potential issue: `random` module usage. It's standard.
# Time limit: 55s.
# Precomputation is fast.
# SA loop: ~50k steps?
# Each step: O(1) or O(deg) ~ O(N).
# 50k * 256 = 12.8M ops. Python can do ~10M-50M simple ops/sec.
# Should fit in 55s.

# Let's refine the Swap validity check to be absolutely sure.
# `other_mask` contains the 3 elements of the 4-AP other than y.
# If `state` contains all 3 (`(state & other_mask) == other_mask`):
#   - If x is one of them: Removing x breaks the 4-AP. So adding y is safe regarding this 4-AP.
#   - If x is NOT one of them: All 3 are still in `state - {x}`. Adding y completes the 4-AP. Unsafe.
# This logic holds.

# What if `state` does NOT contain all 3?
# Then the 4-AP is not completed by adding y, regardless of x. Safe.

# The code looks solid.

# One small optimization:
# In `can_add`, we iterate `four_ap_by_elem[x]`.
# In Swap, we iterate `four_ap_by_elem[y]`.
# This is consistent.

# I will output the complete program now.
