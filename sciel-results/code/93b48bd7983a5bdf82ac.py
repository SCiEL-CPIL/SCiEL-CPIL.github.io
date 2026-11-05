import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    best_score = 0
    best_A = [1]
    best_B = [2]
    
    def rebuild_diffs_sums(S):
        diffs = set()
        sums = set()
        for i, a in enumerate(S):
            sums.add(2 * a)
            for j in range(i + 1, len(S)):
                b = S[j]
                d = abs(a - b)
                diffs.add(d)
                sums.add(a + b)
        return diffs, sums

    def can_add(x, S, diffs_S, sums_S, diffs_other):
        for a in S:
            d = abs(x - a)
            if d in diffs_S:
                return False
            if d in diffs_other:
                return False
        if 2 * x in sums_S:
            return False
        return True

    def add_to_set(x, S, diffs_S, sums_S):
        S.append(x)
        sums_S.add(2 * x)
        for a in S:
            if a != x:
                d = abs(x - a)
                diffs_S.add(d)
                sums_S.add(x + a)

    def try_local_search(A, B, diffs_A, sums_A, diffs_B, sums_B):
        """Try to improve by removing one element and adding two, or swapping."""
        improved = True
        while improved:
            improved = False
            # Try: remove one element from A or B, then try to add two new elements
            for set_idx in [0, 1]:
                if set_idx == 0:
                    S, other_S = A, B
                    diffs_S, sums_S = diffs_A, sums_A
                    diffs_other = diffs_B
                else:
                    S, other_S = B, A
                    diffs_S, sums_S = diffs_B, sums_B
                    diffs_other = diffs_A
                
                if len(S) < 2:
                    continue
                
                for i in range(len(S)):
                    removed = S[i]
                    # Temporarily remove
                    S.pop(i)
                    # Rebuild diffs and sums
                    new_diffs_S, new_sums_S = rebuild_diffs_sums(S)
                    
                    # Try to add two new elements
                    added_count = 0
                    added_elements = []
                    for x in range(1, 257):
                        if x in S or x in other_S:
                            continue
                        if can_add(x, S, new_diffs_S, new_sums_S, diffs_other):
                            add_to_set(x, S, new_diffs_S, new_sums_S)
                            added_elements.append(x)
                            added_count += 1
                            if added_count >= 2:
                                break
                    
                    if added_count >= 2:
                        # Net gain of 1
                        S.pop(i)  # This is wrong, we already popped. Let me fix.
                        # Actually we need to restore properly
                        pass
                    # Restore: put removed back
                    S.insert(i, removed)
                    
                    # Rebuild to original state
                    new_diffs_S, new_sums_S = rebuild_diffs_sums(S)
                    # Update the original sets
                    if set_idx == 0:
                        diffs_A, sums_A = new_diffs_S, new_sums_S
                    else:
                        diffs_B, sums_B = new_diffs_S, new_sums_S
                    break  # This logic is getting complex, let me simplify
                
                break  # Only try one set per iteration for simplicity
        
        return diffs_A, sums_A, diffs_B, sums_B

    def do_local_search(A, B):
        """Perform local search: try swaps to improve total size."""
        diffs_A, sums_A = rebuild_diffs_sums(A)
        diffs_B, sums_B = rebuild_diffs_sums(B)
        
        # Try multiple rounds of local search
        for _round in range(20):
            if time.time() > deadline:
                break
            
            made_change = False
            
            # Try: for each element in A, see if removing it allows adding 2+ elements
            for i in range(len(A)):
                removed = A[i]
                A_temp = A[:i] + A[i+1:]
                diffs_A_temp, sums_A_temp = rebuild_diffs_sums(A_temp)
                
                # Try to add as many new elements as possible
                added = []
                for x in range(1, 257):
                    if x in A_temp or x in B:
                        continue
                    if can_add(x, A_temp, diffs_A_temp, sums_A_temp, diffs_B):
                        add_to_set(x, A_temp, diffs_A_temp, sums_A_temp)
                        added.append(x)
                
                if len(added) >= 2:
                    # Accept this change
                    A = A_temp
                    diffs_A, sums_A = diffs_A_temp, sums_A_temp
                    made_change = True
                    break
            
            if made_change:
                continue
            
            # Try: for each element in B, see if removing it allows adding 2+ elements
            for i in range(len(B)):
                removed = B[i]
                B_temp = B[:i] + B[i+1:]
                diffs_B_temp, sums_B_temp = rebuild_diffs_sums(B_temp)
                
                added = []
                for x in range(1, 257):
                    if x in A or x in B_temp:
                        continue
                    if can_add(x, B_temp, diffs_B_temp, sums_B_temp, diffs_A):
                        add_to_set(x, B_temp, diffs_B_temp, sums_B_temp)
                        added.append(x)
                
                if len(added) >= 2:
                    B = B_temp
                    diffs_B, sums_B = diffs_B_temp, sums_B_temp
                    made_change = True
                    break
            
            if made_change:
                continue
            
            # Try simple swaps: remove one from A, add one to A that was previously blocked
            # Or: try to add any remaining element to either set
            any_added = False
            for x in range(1, 257):
                if x in A or x in B:
                    continue
                if can_add(x, A, diffs_A, sums_A, diffs_B):
                    add_to_set(x, A, diffs_A, sums_A)
                    any_added = True
                    break
                if can_add(x, B, diffs_B, sums_B, diffs_A):
                    add_to_set(x, B, diffs_B, sums_B)
                    any_added = True
                    break
            
            if not any_added:
                break
        
        return A, B

    attempts = 0
    while time.time() < deadline:
        attempts += 1
        random.seed(attempts * 12345 + 67890)
        
        A = []
        B = []
        diffs_A = set()
        diffs_B = set()
        sums_A = set()
        sums_B = set()
        
        # Candidate list
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        # Greedy Phase
        for x in candidates:
            valid_A = can_add(x, A, diffs_A, sums_A, diffs_B)
            valid_B = can_add(x, B, diffs_B, sums_B, diffs_A)
            
            if valid_A and valid_B:
                if len(A) < len(B):
                    add_to_set(x, A, diffs_A, sums_A)
                elif len(B) < len(A):
                    add_to_set(x, B, diffs_B, sums_B)
                else:
                    if random.random() < 0.5:
                        add_to_set(x, A, diffs_A, sums_A)
                    else:
                        add_to_set(x, B, diffs_B, sums_B)
            elif valid_A:
                add_to_set(x, A, diffs_A, sums_A)
            elif valid_B:
                add_to_set(x, B, diffs_B, sums_B)
        
        # Second greedy pass with different order
        remaining = [x for x in range(1, 257) if x not in A and x not in B]
        random.shuffle(remaining)
        
        for x in remaining:
            valid_A = can_add(x, A, diffs_A, sums_A, diffs_B)
            valid_B = can_add(x, B, diffs_B, sums_B, diffs_A)
            
            if valid_A and valid_B:
                if len(A) < len(B):
                    add_to_set(x, A, diffs_A, sums_A)
                elif len(B) < len(A):
                    add_to_set(x, B, diffs_B, sums_B)
                else:
                    if random.random() < 0.5:
                        add_to_set(x, A, diffs_A, sums_A)
                    else:
                        add_to_set(x, B, diffs_B, sums_B)
            elif valid_A:
                add_to_set(x, A, diffs_A, sums_A)
            elif valid_B:
                add_to_set(x, B, diffs_B, sums_B)
        
        # Local search phase
        A, B = do_local_search(A, B)
        
        score = len(A) + len(B)
        if score > best_score:
            best_score = score
            best_A = A.copy()
            best_B = B.copy()

    return [best_A, best_B]
