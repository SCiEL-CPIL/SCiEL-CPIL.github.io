import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    TARGET_MASK = (1 << 30) - 1  # bits 0..29 for diffs 1..30
    
    def is_valid_sidon_covering(A):
        if 0 not in A:
            return False
        A = sorted(A)
        diffs = set()
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                d = A[j] - A[i]
                if d in diffs:
                    return False
                diffs.add(d)
        for d in range(1, 31):
            if d not in diffs:
                return False
        return True

    def search_for_M(M, deadline):
        if M < 30:
            return None
        
        # Bitmask for used differences.
        # We need to track diffs up to M.
        # Python ints are arbitrary precision, so we can use bits 0..M.
        
        init_set = frozenset([0, M])
        # used_diffs bitmask
        # diff d corresponds to bit d
        init_used = 0
        init_used |= (1 << M)
        
        init_mask = 0
        if M <= 30:
            init_mask |= (1 << (M - 1))
            
        best_result = [None]
        
        def backtrack(current_set, used_diffs_mask, covered_mask, depth):
            if time.time() > deadline:
                return None
            
            if covered_mask == TARGET_MASK:
                return sorted(current_set)
            
            if len(current_set) > 32:
                return None
            
            candidates = []
            
            # Generate candidates
            # To speed up, we can pre-check against used_diffs_mask
            # d is invalid if (used_diffs_mask >> d) & 1
            
            for x in range(1, M):
                if x in current_set:
                    continue
                
                valid = True
                new_diffs_list = []
                
                for a in current_set:
                    d = abs(x - a)
                    # Check if d is already used
                    if (used_diffs_mask >> d) & 1:
                        valid = False
                        break
                    # Check if d is already in new_diffs (collision within new set)
                    # Since we check against current_set one by one, 
                    # and current_set elements are distinct, 
                    # collisions between new_diffs arise if |x-a1| == |x-a2| for a1 != a2 in current_set.
                    # This implies a1 + a2 = 2x.
                    # We can just add to a set for this x.
                    if d in new_diffs_list: # O(k) check, k is small
                        valid = False
                        break
                    new_diffs_list.append(d)
                
                if not valid:
                    continue
                
                new_mask = covered_mask
                for d in new_diffs_list:
                    if 1 <= d <= 30:
                        new_mask |= (1 << (d - 1))
                
                gain = bin(new_mask ^ covered_mask).count('1')
                
                # Heuristic: prioritize higher gain
                # Also, maybe prioritize x that are "central" or "spread out"?
                # For now, just gain.
                candidates.append((gain, x, new_diffs_list, new_mask))
            
            if not candidates:
                return None
            
            # Sort by gain desc, then randomize ties to allow exploration
            candidates.sort(key=lambda c: (-c[0], random.random()))
            
            # Limit branching factor
            max_branch = 10
            for gain, x, new_diffs, new_mask in candidates[:max_branch]:
                if gain == 0:
                    # If best candidate has 0 gain, we are stuck?
                    # Not necessarily, but unlikely to be useful.
                    # However, if ALL have 0 gain, we break.
                    if candidates[0][0] == 0:
                        break
                    # If this specific one has 0 gain, skip? 
                    # The loop iterates sorted, so if top has 0, all are 0.
                    # So break is correct if top is 0.
                    # If top > 0, we might encounter 0 gain later. We should skip those.
                    continue
                
                new_set = current_set | {x}
                new_used = used_diffs_mask
                for d in new_diffs:
                    new_used |= (1 << d)
                
                result = backtrack(new_set, new_used, new_mask, depth + 1)
                if result is not None:
                    return result
            
            return None
        
        # Run multiple restarts with different random seeds/orders if possible?
        # The current sort uses random.random(), so each call to search_for_M 
        # (if we called it again) would be different. 
        # But we call it once per M.
        
        return backtrack(init_set, init_used, init_mask, 0)

    # Search from lower bound upward
    for M in range(30, 150):
        if time.time() > deadline:
            break
        
        result = search_for_M(M, deadline)
        if result is not None:
            if is_valid_sidon_covering(result):
                return result
    
    # Fallback
    return [0, 1, 4, 10, 12, 17, 21, 25, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120]
