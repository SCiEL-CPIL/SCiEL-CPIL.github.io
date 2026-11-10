import time
import random

def _check_sums_valid(x, S, sums):
    if x + x in sums:
        return False
    for a in S:
        if x + a in sums:
            return False
    return True

def _check_diffs_valid(x, S, other_diffs):
    for a in S:
        d1 = x - a
        d2 = a - x
        if d1 in other_diffs or d2 in other_diffs:
            return False
    return True

def _add_to_set(x, S, sums, diffs):
    S.append(x)
    sums.add(x + x)
    for a in S[:-1]:
        sums.add(x + a)
        diffs.add(x - a)
        diffs.add(a - x)

def _local_improve(A, B, sums_A, sums_B, diffs_A, diffs_B, start_time, timeout):
    """Try to add remaining candidates that were skipped."""
    all_used = set(A) | set(B)
    candidates = [x for x in range(1, 257) if x not in all_used]
    
    for x in candidates:
        if time.time() - start_time > timeout:
            break
        
        valid_A = _check_sums_valid(x, A, sums_A) and _check_diffs_valid(x, A, diffs_B)
        valid_B = _check_sums_valid(x, B, sums_B) and _check_diffs_valid(x, B, diffs_A)
        
        if valid_A and not valid_B:
            _add_to_set(x, A, sums_A, diffs_A)
        elif valid_B and not valid_A:
            _add_to_set(x, B, sums_B, diffs_B)
        elif valid_A and valid_B:
            if len(A) <= len(B):
                _add_to_set(x, A, sums_A, diffs_A)
            else:
                _add_to_set(x, B, sums_B, diffs_B)
    
    return len(A) + len(B)

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin
    best_size = 0
    best_A = []
    best_B = []
    
    # We will try multiple random restarts
    max_restarts = 5000 # Adjust based on speed
    
    for _ in range(max_restarts):
        if time.time() - start_time > timeout:
            break
            
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        # Candidates are 1..256, ordered with smaller-first bias for better packing
        candidates = list(range(1, 257))
        # Shuffle but preserve some order bias: interleave random with sorted
        random.shuffle(candidates)
        # Create a mixed ordering: some fraction sorted, some random
        n_sorted = random.randint(128, 256)
        sorted_part = sorted(candidates[:n_sorted])
        random_part = candidates[n_sorted:]
        candidates = sorted_part + random_part
        random.shuffle(random_part)
        candidates = sorted_part + random_part
        
        for x in candidates:
            if time.time() - start_time > timeout:
                break
            
            # Try adding to A
            valid_A = _check_sums_valid(x, A, sums_A) and _check_diffs_valid(x, A, diffs_B)
            
            # Try adding to B
            valid_B = _check_sums_valid(x, B, sums_B) and _check_diffs_valid(x, B, diffs_A)
            
            # Decision logic with difference-count heuristic
            if valid_A and valid_B:
                # Choose side that results in fewer total differences (less blocking)
                new_diffs_A = sum(1 for a in A if (x - a) not in diffs_B and (a - x) not in diffs_B)
                new_diffs_B = sum(1 for b in B if (x - b) not in diffs_A and (b - x) not in diffs_A)
                
                # Prefer the side with fewer new conflicting potential differences
                # and balance sizes
                score_A = len(A) + 0.1 * new_diffs_A
                score_B = len(B) + 0.1 * new_diffs_B
                
                if score_A < score_B:
                    side = 'A'
                elif score_B < score_A:
                    side = 'B'
                else:
                    side = random.choice(['A', 'B'])
            elif valid_A:
                side = 'A'
            elif valid_B:
                side = 'B'
            else:
                continue # Cannot add x to either
            
            if side == 'A':
                _add_to_set(x, A, sums_A, diffs_A)
            else:
                _add_to_set(x, B, sums_B, diffs_B)
        
        # Local improvement pass
        current_size = _local_improve(A, B, sums_A, sums_B, diffs_A, diffs_B, start_time, timeout)
        
        if current_size > best_size:
            best_size = current_size
            best_A = A[:]
            best_B = B[:]

    return [best_A, best_B]
