import random
import time

def _is_sidon(subset):
    sums = set()
    for i in range(len(subset)):
        for j in range(i, len(subset)):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size(A):
    n = len(A)
    best = 0
    # Iterate masks in increasing order of popcount? No, just iterate all.
    # Optimization: if we find a Sidon set of size k, we can stop checking smaller ones?
    # No, we want the MAX. So we should check larger ones first?
    # Or just check all. 2^12 is small.
    # To speed up, we can iterate masks by descending popcount and break early if found.
    # But generating masks by popcount is complex.
    # Let's just iterate 1..2^n and track max.
    # Or, iterate from (1<<n)-1 down to 1? No, that doesn't guarantee popcount order.
    
    # A better approach for finding MAX size:
    # Check if there is a Sidon subset of size n. If yes, return n.
    # Check if there is a Sidon subset of size n-1. If yes, return n-1.
    # ...
    # This requires generating combinations.
    
    # Given n=12, 2^12 = 4096. Checking all is fast enough.
    # But we can prune: if len(sub) <= best, skip.
    
    for mask in range(1, 1 << n):
        # Quick popcount check?
        # If popcount <= best, skip.
        # bin(mask).count('1') is fast enough for n=12.
        if bin(mask).count('1') <= best:
            continue
        sub = [A[i] for i in range(n) if (mask >> i) & 1]
        if _is_sidon(sub):
            best = len(sub)
            if best == n:
                break
    return best

def _find_sidon_subset(A, target_size):
    n = len(A)
    # Find any Sidon subset of size target_size
    # Iterate masks with popcount == target_size
    for mask in range(1, 1 << n):
        if bin(mask).count('1') != target_size:
            continue
        sub = [A[i] for i in range(n) if (mask >> i) & 1]
        if _is_sidon(sub):
            return sub
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # 5s safety margin for 90s limit
    
    best_set = list(range(12))
    best_val = _max_sidon_size(best_set)
    
    random.seed(42)
    
    # Try a few random initializations
    for _ in range(50):
        if time.time() > deadline:
            break
        current_set = sorted(random.sample(range(129), 12))
        current_val = _max_sidon_size(current_set)
        if current_val < best_val:
            best_val = current_val
            best_set = current_set
            if best_val == 3:
                break
    
    # Local search
    if best_val > 3:
        current_set = best_set[:]
        current_val = best_val
        no_improve_count = 0
        
        while time.time() < deadline:
            if current_val <= 3:
                break
            
            # Find a Sidon subset of size current_val
            found_sub = _find_sidon_subset(current_set, current_val)
            if not found_sub:
                # Should not happen if current_val is correct, but just in case
                current_val = _max_sidon_size(current_set)
                continue
            
            # Strategy 1: Replace an element in found_sub with a random new element
            # Strategy 2: Swap an element in found_sub with an element outside
            # Strategy 3: Replace an element outside with a random new element (to change the landscape)
            
            # Let's try Strategy 1 first
            idx_to_replace = random.choice([current_set.index(x) for x in found_sub])
            old_elem = current_set[idx_to_replace]
            
            # Pick a new element
            # Bias towards elements that might break sums? Hard to know.
            # Just random.
            new_elem = random.choice(range(129))
            
            if new_elem in current_set:
                # If collision, try a swap instead
                # Find an element outside found_sub to swap with
                outside_indices = [i for i in range(12) if current_set[i] not in found_sub]
                if outside_indices:
                    idx_out = random.choice(outside_indices)
                    # Swap current_set[idx_to_replace] and current_set[idx_out]
                    current_set[idx_to_replace], current_set[idx_out] = current_set[idx_out], current_set[idx_to_replace]
                    current_set.sort()
                    new_val = _max_sidon_size(current_set)
                    
                    if new_val < current_val:
                        current_val = new_val
                        no_improve_count = 0
                        if current_val < best_val:
                            best_val = current_val
                            best_set = current_set[:]
                            if best_val == 3:
                                break
                    elif new_val == current_val:
                        if random.random() < 0.5:
                            no_improve_count += 1
                        else:
                            # Revert
                            current_set[idx_to_replace], current_set[idx_out] = current_set[idx_out], current_set[idx_to_replace]
                            current_set.sort()
                            no_improve_count += 1
                    else:
                        # Revert
                        current_set[idx_to_replace], current_set[idx_out] = current_set[idx_out], current_set[idx_to_replace]
                        current_set.sort()
                        no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                # Replace
                current_set[idx_to_replace] = new_elem
                current_set.sort()
                new_val = _max_sidon_size(current_set)
                
                if new_val < current_val:
                    current_val = new_val
                    no_improve_count = 0
                    if current_val < best_val:
                        best_val = current_val
                        best_set = current_set[:]
                        if best_val == 3:
                            break
                elif new_val == current_val:
                    if random.random() < 0.5:
                        no_improve_count += 1
                    else:
                        # Revert
                        current_set[idx_to_replace] = old_elem
                        current_set.sort()
                        no_improve_count += 1
                else:
                    # Revert
                    current_set[idx_to_replace] = old_elem
                    current_set.sort()
                    no_improve_count += 1
            
            # Occasionally restart if stuck
            if no_improve_count > 50:
                current_set = sorted(random.sample(range(129), 12))
                current_val = _max_sidon_size(current_set)
                no_improve_count = 0
                if current_val < best_val:
                    best_val = current_val
                    best_set = current_set[:]
                    if best_val == 3:
                        break

    return best_set
