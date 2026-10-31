import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
_START = time.time()
_DEADLINE = _START + 115.0  # Safety margin vs 150 s kill

def _score_from_set(u):
    """
    Compute the score for a set of integers u.
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    Uses NumPy for vectorized computation of sum and difference sets.
    """
    if not u or len(u) < 2:
        return 0.0
    m = max(u)
    if m < 1:
        return 0.0
    
    u_arr = np.array(u, dtype=np.int64)
    n = len(u_arr)
    
    # Compute all pairwise sums and differences
    # u_arr[:, None] + u_arr[None, :] creates an (n, n) matrix
    sums = u_arr[:, None] + u_arr[None, :]
    diffs = u_arr[:, None] - u_arr[None, :]
    
    # Count unique elements
    # np.unique is efficient for moderate sizes. 
    # For very large n, this might be slow, but we cap n in the search.
    unique_sums = np.unique(sums)
    unique_diffs = np.unique(diffs)
    
    if len(unique_sums) == 0:
        return 0.0
        
    ratio = len(unique_diffs) / float(len(unique_sums))
    if ratio <= 0:
        return 0.0
        
    return 1.0 + math.log(ratio) / math.log(2 * m + 1)

def _generate_random_set(n, max_val, base=2.0):
    """
    Generate a random set of size n with values up to max_val.
    Uses a mix of geometric and random steps to create structure.
    """
    if n < 2:
        return [0, 1]
    
    # Start with 0
    vals = [0]
    current = 0
    
    # Choose a random growth factor for the "backbone"
    # This creates a spread-out set which tends to have good ratios
    growth = random.uniform(1.5, 3.0)
    
    for i in range(1, n):
        # Next value is current * growth * random_noise
        noise = random.uniform(0.8, 1.5)
        next_val = int(current * growth * noise)
        
        # Ensure it's strictly increasing and within bounds
        if next_val <= current:
            next_val = current + 1
        if next_val > max_val:
            next_val = current + 1
            if next_val > max_val:
                break
                
        vals.append(next_val)
        current = next_val
        
    # If we didn't reach n elements, fill with random small increments
    while len(vals) < n:
        next_val = vals[-1] + random.randint(1, max(1, max_val // n))
        if next_val > max_val:
            next_val = vals[-1] + 1
        if next_val in vals:
            continue
        vals.append(next_val)
        
    # Ensure 0 is present (it is, by construction)
    # Return sorted list
    return sorted(set(vals))

def _local_search(u, max_steps=5000, timeout=10.0):
    """
    Perform local search on the set u.
    Moves:
    1. Replace an element with a nearby value.
    2. Add a new element.
    3. Remove an element.
    """
    u = sorted(set(u))
    if len(u) < 2:
        u = [0, 1]
        
    current_score = _score_from_set(u)
    best_u = u[:]
    best_score = current_score
    
    start_time = time.time()
    
    for step in range(max_steps):
        if time.time() > _DEADLINE - 5.0 or time.time() > start_time + timeout:
            break
            
        move_type = random.choice(['replace', 'replace', 'add', 'remove'])
        new_u = u[:]
        
        if move_type == 'replace':
            if len(u) < 2:
                continue
            idx = random.randint(1, len(u) - 1)  # Don't replace 0
            old_val = u[idx]
            # Try a few nearby values
            for _ in range(10):
                delta = random.randint(-max(1, old_val//10), max(1, old_val//10))
                new_val = old_val + delta
                if new_val < 0:
                    new_val = 0
                if new_val in u:
                    continue
                test_u = u[:]
                test_u[idx] = new_val
                test_u = sorted(set(test_u))
                if len(test_u) != len(u):
                    continue
                s = _score_from_set(test_u)
                if s > current_score:
                    u = test_u
                    current_score = s
                    if current_score > best_score:
                        best_score = current_score
                        best_u = u[:]
                    break
        elif move_type == 'add':
            if len(u) > 60:  # Cap size for performance
                continue
            max_val = max(u)
            # Try adding a value in a random gap or extending the range
            # Pick a random position to insert
            if random.random() < 0.5:
                # Extend range
                new_val = max_val + random.randint(1, max(1, max_val//10))
            else:
                # Fill a gap
                if len(u) < 2:
                    continue
                idx = random.randint(0, len(u)-2)
                low = u[idx]
                high = u[idx+1]
                if high - low > 1:
                    new_val = random.randint(low + 1, high - 1)
                else:
                    continue
                    
            if new_val in u or new_val < 0:
                continue
            test_u = sorted(u + [new_val])
            s = _score_from_set(test_u)
            if s > current_score:
                u = test_u
                current_score = s
                if current_score > best_score:
                    best_score = current_score
                    best_u = u[:]
        elif move_type == 'remove':
            if len(u) < 3:
                continue
            idx = random.randint(1, len(u) - 1)  # Don't remove 0
            test_u = u[:]
            test_u.pop(idx)
            s = _score_from_set(test_u)
            if s > current_score:
                u = test_u
                current_score = s
                if current_score > best_score:
                    best_score = current_score
                    best_u = u[:]
                    
    return best_u, best_score

def construct():
    global _DEADLINE
    
    random.seed(42)
    np.random.seed(42)
    
    best_u = [0, 1]
    best_s = 0.0
    
    # Phase 1: Generate diverse initial candidates
    # Try different sizes and growth factors
    sizes = [5, 10, 15, 20, 30, 40, 50]
    max_vals = [100, 500, 1000, 5000, 10000]
    
    candidates = []
    
    # Geometric progressions
    for base in [1.5, 2.0, 2.5, 3.0]:
        for n in [10, 20, 30]:
            u = [0]
            curr = 1
            for i in range(1, n):
                curr = int(curr * base)
                u.append(curr)
            candidates.append(u)
            
    # Random structured sets
    for n in sizes:
        for mv in max_vals:
            if time.time() > _DEADLINE - 20.0:
                break
            u = _generate_random_set(n, mv)
            candidates.append(u)
            
    # Evaluate candidates
    for u in candidates:
        if time.time() > _DEADLINE - 10.0:
            break
        s = _score_from_set(u)
        if s > best_s:
            best_s = s
            best_u = u[:]
            
    # Phase 2: Local search from the best candidate and a few others
    # Sort candidates by score to prioritize
    scored_candidates = []
    for u in candidates:
        s = _score_from_set(u)
        scored_candidates.append((s, u))
    scored_candidates.sort(reverse=True)
    
    # Run local search on the top 3 candidates
    for i, (init_s, init_u) in enumerate(scored_candidates[:3]):
        if time.time() > _DEADLINE - 5.0:
            break
        if init_s < best_s * 0.9:  # Skip if too far behind
            continue
            
        u_new, s_new = _local_search(init_u, max_steps=2000, timeout=15.0)
        if s_new > best_s:
            best_s = s_new
            best_u = u_new[:]
            
    # Phase 3: Final polish on the best result
    if time.time() < _DEADLINE - 2.0:
        u_new, s_new = _local_search(best_u, max_steps=1000, timeout=5.0)
        if s_new > best_s:
            best_s = s_new
            best_u = u_new[:]
            
    # Final sanity check
    best_u = sorted(set(int(x) for x in best_u))
    if 0 not in best_u:
        best_u.append(0)
    if len(best_u) < 2:
        best_u = [0, 1]
        
    return best_u
# EVOLVE-BLOCK-END
