import time
from math import lcm
from fractions import Fraction

def construct():
    deadline = time.time() + 55
    
    # Numbers 2..256
    N = 255
    nums = list(range(2, 257))
    
    # Precompute compatibility bitsets
    # compat_mask[i] will have bit j set if lcm(nums[i], nums[j]) > 256
    # Note: We only care about j > i for the forward search, but for bitset ops, full mask is easier.
    compat_mask = [0] * N
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if lcm(nums[i], nums[j]) > 256:
                compat_mask[i] |= (1 << j)
    
    # Reciprocals as Fractions
    recip = [Fraction(1, nums[i]) for i in range(N)]
    
    # Sort indices by reciprocal descending (which is just 0, 1, ..., N-1 since nums are 2..256)
    # So order is naturally 2, 3, ..., 256
    
    # Greedy initial solution to get a good lower bound (best)
    # Greedy: Pick smallest number compatible with all previously picked
    best_sum = Fraction(0, 1)
    best_set_mask = 0
    remaining = (1 << N) - 1
    current_set = []
    
    for i in range(N):
        # Check if i is compatible with all in current_set
        ok = True
        for j in current_set:
            if not (compat_mask[i] & (1 << j)):
                ok = False
                break
        if ok:
            current_set.append(i)
            best_set_mask |= (1 << i)
            best_sum += recip[i]
            remaining &= ~compat_mask[i] # Remove incompatible ones? No, greedy just picks next valid.
            # Actually, standard greedy for independent set:
            # Pick i, remove i and all incompatible with i from candidate pool.
            # But here we want max weight.
            # Simple greedy: iterate 2..256, if compatible with all selected, select it.
            # This is what the loop above does.
            
    best = [best_sum]
    best_solution_mask = [best_set_mask]
    
    # B&B
    # We process indices 0..N-1.
    # State: pos, current_sum, available_mask
    # available_mask: bits j >= pos such that j is compatible with all selected items.
    
    def branch_and_bound(pos, current_sum, available_mask):
        if time.time() > deadline:
            return
        
        if pos >= N:
            if current_sum > best[0]:
                best[0] = current_sum
                best_solution_mask[0] = available_mask # This is wrong, available_mask is not the set.
                # We need to track the set mask separately or reconstruct.
                # Let's pass current_set_mask as well.
            return
        
        # Upper Bound: current_sum + sum of recip[j] for j in available_mask
        # Since available_mask only contains bits >= pos (if we maintain it correctly),
        # we can sum them.
        ub = current_sum
        m = available_mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            ub += recip[idx]
            m ^= lsb
            
        if ub <= best[0]:
            return
        
        # Option 1: Include pos
        if available_mask & (1 << pos):
            # New available mask: items compatible with pos AND already available
            # Note: compat_mask[pos] contains all j such that lcm(nums[pos], nums[j]) > 256.
            # We also need to ensure we only consider items > pos to avoid re-adding or cycles?
            # The search is sequential. If we include pos, we move to pos+1.
            # The new available mask for pos+1 should be (available_mask & compat_mask[pos]) & ~(bits < pos+1).
            # Actually, available_mask already only has bits >= pos.
            # So new_avail = available_mask & compat_mask[pos]
            # But we must also mask out pos itself? Yes, but pos is not in available_mask for the *next* step if we shift?
            # No, available_mask is passed to the next call with pos+1.
            # The bits in available_mask are indices.
            # If we include pos, we remove pos from consideration for future steps.
            # Also, any item incompatible with pos must be removed.
            
            new_avail = available_mask & compat_mask[pos]
            # Ensure we don't consider pos again? pos is < pos+1, so it won't be in the "remaining" range if we enforce pos >= current_pos.
            # But `available_mask` might have bit `pos` set. We should clear it or just rely on the `pos` index check?
            # The loop in `branch_and_bound` checks `if available_mask & (1 << pos)`.
            # If we call `branch_and_bound(pos+1, ..., new_avail)`, it will check `new_avail & (1 << (pos+1))`.
            # So bit `pos` in `new_avail` doesn't matter for the next step's inclusion check, but it DOES matter for the UB calculation if we sum all bits in `new_avail`.
            # So we must clear bit `pos` from `new_avail` before passing it?
            # Or better, define `available_mask` to only contain bits > current_pos.
            
            # Let's refine state: `available_mask` contains bits j > pos that are compatible with current set.
            # Initial call: pos=0, available_mask = (1 << N) - 1 (all bits 0..N-1 set? No, we start at 0).
            # Let's say `available_mask` contains bits j >= pos.
            
            # If we include pos:
            # New set = current set + {pos}
            # New available for pos+1: 
            #   Must be compatible with pos.
            #   Must be > pos.
            #   Must have been available before.
            
            new_avail = (available_mask & compat_mask[pos]) & ~((1 << (pos + 1)) - 1) # Clear bits <= pos
            # Actually, simpler: `new_avail = available_mask & compat_mask[pos]`
            # Since we recurse with `pos+1`, the function will only look at bits >= pos+1 for inclusion.
            # But for UB, we sum all bits in `new_avail`. If bit `pos` is still there, it adds recip[pos] again?
            # No, `current_sum` already includes `recip[pos]`.
            # So `new_avail` should NOT contain `pos`.
            
            new_avail = (available_mask & compat_mask[pos]) & ~(1 << pos)
            
            branch_and_bound(pos + 1, current_sum + recip[pos], new_avail)
            
        # Option 2: Exclude pos
        # New available for pos+1: same as before, but without pos.
        new_avail_excl = available_mask & ~(1 << pos)
        branch_and_bound(pos + 1, current_sum, new_avail_excl)
        
    # Initial call
    # available_mask: all bits 0..N-1 are potentially available?
    # Yes, initially no items selected, so all are compatible with empty set.
    initial_mask = (1 << N) - 1
    branch_and_bound(0, Fraction(0, 1), initial_mask)
    
    # Reconstruct solution from best_solution_mask?
    # Wait, I didn't update best_solution_mask correctly in the B&B.
    # Let's fix the B&B to track the mask or reconstruct.
    # Tracking mask is easy: pass it down.
    
    # Let's rewrite B&B to track mask.
    
    best_sum = Fraction(0, 1)
    best_mask = 0
    
    # Greedy again to initialize
    g_mask = 0
    g_sum = Fraction(0, 1)
    for i in range(N):
        ok = True
        for j in range(N):
            if g_mask & (1 << j):
                if not (compat_mask[i] & (1 << j)):
                    ok = False
                    break
        if ok:
            g_mask |= (1 << i)
            g_sum += recip[i]
            
    best_sum = g_sum
    best_mask = g_mask
    
    def bnb(pos, curr_sum, curr_mask, avail_mask):
        nonlocal best_sum, best_mask
        if time.time() > deadline:
            return
        
        if pos >= N:
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_mask = curr_mask
            return
        
        # UB
        ub = curr_sum
        m = avail_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            ub += recip[idx]
            m ^= lsb
            
        if ub <= best_sum:
            return
        
        # Include pos
        if avail_mask & (1 << pos):
            new_mask = curr_mask | (1 << pos)
            new_avail = (avail_mask & compat_mask[pos]) & ~(1 << pos)
            bnb(pos + 1, curr_sum + recip[pos], new_mask, new_avail)
            
        # Exclude pos
        new_avail_excl = avail_mask & ~(1 << pos)
        bnb(pos + 1, curr_sum, curr_mask, new_avail_excl)
        
    initial_mask = (1 << N) - 1
    bnb(0, Fraction(0, 1), 0, initial_mask)
    
    result = [nums[i] for i in range(N) if best_mask & (1 << i)]
    if not result:
        result = [2]
    return result
