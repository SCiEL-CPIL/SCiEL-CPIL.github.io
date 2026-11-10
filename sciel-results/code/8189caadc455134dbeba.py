import time
import random
import sys

_START_TIME = time.time()
_DEADLINE = 85.0  # Safety margin before 90s kill

def _exact_max_sidon_subset_size(arr, time_limit=0.05):
    """
    Exact B&B solver to find the maximum Sidon subset size.
    Optimized for N=12. Returns the size.
    Uses a bitset for sums to speed up membership tests.
    """
    n = len(arr)
    if n <= 1:
        return n
    
    arr_sorted = sorted(arr)
    best = 0
    start_time = time.time()
    
    # Precompute all pair sums (including diagonals) for each element
    # For element i, the sums it creates with elements j (j <= i) are:
    #   arr[i] + arr[j] for all j in current_elements (j < i in sorted order)
    #   2*arr[i] (diagonal)
    
    # Use a list of integers as bitsets: sums range 0..256, so we need 257 bits
    # Python ints can handle this natively
    
    def dfs(idx, current_size, used_sums_bits, current_elements):
        nonlocal best, start_time
        
        # Time check every few levels
        if (idx % 3 == 0) and (time.time() - start_time > time_limit):
            return False
            
        # Pruning: if we can't beat best
        if current_size + (n - idx) <= best:
            return False
            
        # Update best
        if current_size > best:
            best = current_size
            if best == n:
                return True  # Can't do better than n
            
        if idx == n:
            return False
        
        # Stronger Pruning: Sum Count Bound
        # Count set bits in used_sums_bits
        used_count = bin(used_sums_bits).count('1')
        available_sums = 257 - used_count
        # Max additional elements 'k' that can be added such that k(k+1)/2 <= available_sums
        max_addable = int((-1 + (1 + 8 * available_sums) ** 0.5) / 2)
        
        if current_size + max_addable <= best:
            return False
            
        x = arr_sorted[idx]
        
        # Option 1: Include x
        # Check if x can be added to current_elements
        can_add = True
        new_sums = used_sums_bits
        
        # Check collisions with existing elements
        for y in current_elements:
            s = x + y
            if (new_sums >> s) & 1:
                can_add = False
                break
            new_sums |= (1 << s)
        
        if can_add:
            s_diag = 2 * x
            if (new_sums >> s_diag) & 1:
                can_add = False
            else:
                new_sums |= (1 << s_diag)
        
        if can_add:
            # Try including x first (often finds larger solutions faster, improving pruning)
            if dfs(idx + 1, current_size + 1, new_sums, current_elements + [x]):
                return True
        
        # Option 2: Exclude x
        if dfs(idx + 1, current_size, used_sums_bits, current_elements):
            return True
            
        return False

    # Initial sums: all False (0)
    dfs(0, 0, 0, [])
    return best

def construct():
    global _START_TIME, _DEADLINE
    
    # Ensure valid output: 12 distinct integers in [0, 128]
    
    candidates = []
    
    # 1. Arithmetic Progressions (known to have low Sidon numbers)
    for step in range(1, 12):
        end = 11 * step
        if end <= 128:
            candidates.append(list(range(0, end + 1, step)))
    
    # 2. Dense Clusters - two clusters
    for gap in [50, 55, 60, 64, 65, 66, 70]:
        for cluster_size in [6]:
            c1 = list(range(0, cluster_size))
            c2 = list(range(gap, gap + cluster_size))
            candidates.append(c1 + c2)
    
    # 3. Dense Clusters - three clusters
    for gap1 in [35, 40, 42, 45]:
        for gap2 in [75, 80, 84, 85, 90]:
            c1 = list(range(0, 4))
            c2 = list(range(gap1, gap1 + 4))
            c3 = list(range(gap2, gap2 + 4))
            if gap2 + 4 <= 129:
                candidates.append(c1 + c2 + c3)
    
    # 4. Very dense: 0-11 (already covered by AP step 1)
    
    # 5. Perturbed APs
    base_ap = list(range(12))
    rng = random.Random(42)
    for _ in range(100):
        c = base_ap[:]
        for _ in range(rng.randint(1, 3)):
            idx = rng.randint(0, 11)
            shift = rng.choice([-3, -2, -1, 1, 2, 3])
            new_val = c[idx] + shift
            if 0 <= new_val <= 128 and new_val not in c:
                c[idx] = new_val
        if len(set(c)) == 12:
            candidates.append(sorted(c))
            
    # 6. Random dense sets in small range
    for _ in range(200):
        c = sorted(rng.sample(range(0, 18), 12))
        candidates.append(c)
        
    # 7. Random dense sets in medium range  
    for _ in range(100):
        c = sorted(rng.sample(range(0, 35), 12))
        candidates.append(c)
    
    # 8. Structured: pairs that create sum collisions
    # e.g., {0, a, b, a+b, ...} creates collisions
    for a in range(1, 20):
        for b in range(a+1, 30):
            if a + b <= 128:
                c = [0, a, b, a+b]
                # Add more elements that collide
                for d in range(1, 10):
                    val = a + d
                    if val <= 128 and val not in c:
                        c.append(val)
                    if len(c) >= 12:
                        break
                if len(c) < 12:
                    # Fill with more collision-prone values
                    for d in range(1, 50):
                        val = b + d
                        if val <= 128 and val not in c:
                            c.append(val)
                        if len(c) >= 12:
                            break
                if len(c) == 12 and len(set(c)) == 12:
                    candidates.append(sorted(c))
    
    best_arr = None
    best_score = 12  # Max possible
    
    # Evaluate candidates with tight time limit
    for c in candidates:
        if len(set(c)) != 12:
            continue
        if any(x < 0 or x > 128 for x in c):
            continue
        score = _exact_max_sidon_subset_size(c, time_limit=0.015)
        if score < best_score:
            best_score = score
            best_arr = c[:]
            if best_score <= 4:
                break  # Very good, stop searching
                
    # If no candidate worked (shouldn't happen), fallback
    if best_arr is None:
        best_arr = list(range(12))
        best_score = _exact_max_sidon_subset_size(best_arr, time_limit=0.05)
        
    # Local Search / Simulated Annealing
    # Objective: Minimize best_score (max Sidon subset size)
    
    current_arr = best_arr[:]
    current_score = best_score
    best_arr_found = best_arr[:]
    best_score_found = best_score
    
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.997
    
    rng = random.Random(123)
    
    # Helper to check if a value is valid for the array
    def is_valid(val, arr, idx):
        if val < 0 or val > 128:
            return False
        if val == arr[idx]:
            return False
        for j, v in enumerate(arr):
            if j != idx and v == val:
                return False
        return True

    # Precompute for faster collision checking
    def count_sum_collisions(arr):
        """Count number of duplicate pair sums (including diagonals)"""
        sums = set()
        collisions = 0
        for i in range(len(arr)):
            for j in range(i, len(arr)):
                s = arr[i] + arr[j]
                if s in sums:
                    collisions += 1
                else:
                    sums.add(s)
        return collisions

    while time.time() < _DEADLINE:
        # Generate neighbor
        new_arr = current_arr[:]
        idx = rng.randint(0, 11)
        old_val = new_arr[idx]
        
        # Strategy: Try to move the element to a value that causes a sum collision
        # with existing pairs.
        if rng.random() < 0.6:
            # Small perturbation
            delta = rng.randint(-4, 4)
            new_val = old_val + delta
        elif rng.random() < 0.3:
            # Try to create collision: pick two other elements, set new_val so that
            # new_val + a = b + c for some existing b, c
            other_indices = [i for i in range(12) if i != idx]
            if len(other_indices) >= 2:
                i1 = rng.choice(other_indices)
                i2 = rng.choice([i for i in other_indices if i != i1])
                # Want new_val + arr[i1] = arr[i2] + arr[j] for some j
                # new_val = arr[i2] + arr[j] - arr[i1]
                i3 = rng.choice([i for i in other_indices if i not in [i1, i2]])
                new_val = arr[i2] + arr[i3] - arr[i1]
                if not (0 <= new_val <= 128):
                    new_val = old_val + rng.randint(-3, 3)
            else:
                new_val = old_val + rng.randint(-3, 3)
        else:
            # Random jump
            new_val = rng.randint(0, 128)
            
        # Ensure distinctness and range
        if not is_valid(new_val, new_arr, idx):
            continue
            
        new_arr[idx] = new_val
        new_arr.sort()  # Keep sorted for solver efficiency
        
        # Evaluate
        new_score = _exact_max_sidon_subset_size(new_arr, time_limit=0.015)
        
        delta_score = new_score - current_score
        
        if delta_score < 0:
            # Improvement
            current_arr = new_arr
            current_score = new_score
            if current_score < best_score_found:
                best_score_found = current_score
                best_arr_found = new_arr[:]
        else:
            # Accept with probability
            if delta_score == 0:
                if rng.random() < 0.3:
                    current_arr = new_arr
            else:
                # Since score is integer, delta_score >= 1
                prob = 2.718281828 ** (-delta_score / T)
                if rng.random() < prob:
                    current_arr = new_arr
                    current_score = new_score
                    
        T *= cooling_rate
        if T < T_min:
            T = T_min

    # Final validation
    assert len(best_arr_found) == 12
    assert len(set(best_arr_found)) == 12
    assert all(0 <= x <= 128 for x in best_arr_found)
    
    return best_arr_found
