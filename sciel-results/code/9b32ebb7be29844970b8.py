import time
import random

def construct():
    n = 12
    S = [1 << i for i in range(n)]
    current_max = max(S)
    
    def is_valid(set_list):
        sums = set()
        for mask in range(1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += set_list[i]
            if s in sums:
                return False
            sums.add(s)
        return True

    start_time = time.time()
    deadline = start_time + 85.0
    
    used = set(S)
    
    while time.time() < deadline:
        max_idx = S.index(current_max)
        # Generate candidates smaller than current_max
        candidates = [x for x in range(1, current_max) if x not in used]
        random.shuffle(candidates)
        
        improved = False
        for cand in candidates:
            old_val = S[max_idx]
            S[max_idx] = cand
            used.discard(old_val)
            used.add(cand)
            
            if is_valid(S):
                current_max = cand
                improved = True
                break
            else:
                S[max_idx] = old_val
                used.discard(cand)
                used.add(old_val)
                
        if not improved:
            # Try to reduce a non-max element
            idx = random.randint(0, n - 1)
            if idx == max_idx:
                continue
            old_val = S[idx]
            candidates = [x for x in range(1, old_val) if x not in used]
            random.shuffle(candidates)
            
            for cand in candidates:
                S[idx] = cand
                used.discard(old_val)
                used.add(cand)
                
                if is_valid(S):
                    # Accept
                    if old_val == current_max:
                        # Max might change, but we are reducing a non-max element
                        # So current_max stays same unless old_val was max (handled above)
                        pass
                    improved = True
                    break
                else:
                    S[idx] = old_val
                    used.discard(cand)
                    used.add(old_val)
                    
            if not improved:
                # No single reduction worked.
                # Try a random swap to escape local minimum?
                # Or just continue, random seed might help next iteration?
                # Actually, if no reduction works, we are stuck.
                # Let's try a restart with a different random seed or perturbation.
                # For now, just break or continue?
                # If we continue, we will keep trying the same thing.
                # Let's add a small perturbation: swap two random elements.
                i, j = random.sample(range(n), 2)
                S[i], S[j] = S[j], S[i]
                # Note: set of values is same, so 'used' is same.
                # This changes the structure, might allow reductions.
                # But it doesn't change current_max.
                # We should check if this swap was valid? 
                # Swapping elements in a set doesn't change the set of subset sums.
                # So validity is preserved.
                # So this is a safe move to change the "order" or indices, 
                # but since we only care about the set of values, swapping indices 
                # does NOT change the set of values.
                # So this does nothing for the search space of sets.
                # The search space is the set of values.
                # If no value can be reduced, we are stuck.
                
                # To escape, we might need to INCREASE one element to allow another to decrease?
                # This is complex.
                # Given the large gap, simple descent likely works.
                # If stuck, we might just return the current best.
                pass

    return S
