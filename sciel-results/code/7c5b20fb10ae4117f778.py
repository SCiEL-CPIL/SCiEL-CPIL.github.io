"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def _is_sidon_fast(subset):
    """Fast check if a subset has all unique pair sums (including diagonals)."""
    sums = set()
    n = len(subset)
    for i in range(n):
        si = subset[i]
        for j in range(i, n):
            s = si + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _count_duplicate_sums(arr):
    """Count the number of duplicate pair sums in arr (including diagonals).
    Higher means more collisions, which correlates with smaller max Sidon subset."""
    sums = {}
    n = len(arr)
    for i in range(n):
        ai = arr[i]
        for j in range(i, n):
            s = ai + arr[j]
            sums[s] = sums.get(s, 0) + 1
    duplicates = 0
    for count in sums.values():
        if count > 1:
            duplicates += count - 1
    return duplicates

def _max_sidon_subset_size(arr):
    """Compute the largest subset of arr with all unordered pair sums unique (including diagonals)."""
    n = len(arr)
    # Check k from n down to 1.
    for k in range(n, 0, -1):
        for subset in combinations(arr, k):
            if _is_sidon_fast(subset):
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    # Initial best: try some structured sets first
    candidates = [
        list(range(12)),
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 3, 7, 15],
        [0, 1, 2, 3, 6, 12, 24, 48, 96, 5, 11, 23],
        [0, 1, 2, 4, 8, 16, 32, 64, 127, 126, 125, 124],
        [0, 1, 2, 4, 8, 16, 32, 64, 100, 110, 120, 128],
    ]
    
    best_arr = None
    best_size = 12
    
    for cand in candidates:
        if len(cand) == 12 and len(set(cand)) == 12 and all(0 <= x <= 128 for x in cand):
            sz = _max_sidon_subset_size(cand)
            if sz < best_size:
                best_size = sz
                best_arr = cand
    
    if best_arr is None:
        best_arr = list(range(12))
        best_size = _max_sidon_subset_size(best_arr)
    
    random.seed(42)
    
    # We will perform multiple restarts of a local search.
    # Local search minimizes _max_sidon_subset_size.
    # We use a surrogate function _count_duplicate_sums for fast evaluation.
    
    # Track the best surrogate score found
    best_surrogate = _count_duplicate_sums(best_arr)
    
    for restart in range(10000):
        if time.time() > deadline:
            break
            
        # Multiple initialization strategies
        strategy = restart % 8
        if strategy == 0:
            # Random set
            arr = sorted(random.sample(range(129), 12))
        elif strategy == 1:
            # Arithmetic progression with random start and step
            step = random.randint(1, 12)
            base = random.randint(0, max(0, 128 - 11 * step))
            arr = sorted([base + i * step for i in range(12)])
            if max(arr) > 128:
                continue
        elif strategy == 2:
            # Clustered: groups of 2-4 close elements to create sum collisions
            arr = []
            num_clusters = random.randint(4, 6)
            for _ in range(num_clusters):
                center = random.randint(0, 120)
                size = random.randint(2, 4)
                for d in range(size):
                    val = center + d
                    if 0 <= val <= 128 and val not in arr:
                        arr.append(val)
            while len(arr) < 12:
                val = random.randint(0, 128)
                if val not in arr:
                    arr.append(val)
            arr = sorted(arr[:12])
        elif strategy == 3:
            # Geometric-like: powers or near-powers
            arr = [0]
            current = 1
            while len(arr) < 12 and current <= 128:
                arr.append(current)
                current = current * random.randint(2, 3) + random.randint(0, 5)
            while len(arr) < 12:
                val = random.randint(0, 128)
                if val not in arr:
                    arr.append(val)
            arr = sorted(arr[:12])
        elif strategy == 4:
            # Hybrid: some clustered, some spread
            arr = []
            # 3-4 clusters of 3
            for _ in range(random.randint(3, 4)):
                center = random.randint(0, 100)
                for d in range(3):
                    val = center + d
                    if 0 <= val <= 128 and val not in arr:
                        arr.append(val)
            while len(arr) < 12:
                val = random.randint(0, 128)
                if val not in arr:
                    arr.append(val)
            arr = sorted(arr[:12])
        elif strategy == 5:
            # Pairs that sum to same value: e.g., (a, b), (c, d) where a+b=c+d
            arr = []
            target_sums = random.sample(range(20, 120), 4)
            for ts in target_sums:
                a = random.randint(max(0, ts - 128), min(128, ts))
                b = ts - a
                if 0 <= a <= 128 and 0 <= b <= 128 and a != b and a not in arr and b not in arr:
                    arr.append(a)
                    arr.append(b)
            while len(arr) < 12:
                val = random.randint(0, 128)
                if val not in arr:
                    arr.append(val)
            arr = sorted(arr[:12])
        elif strategy == 6:
            # Symmetric around center: x and 128-x pairs
            arr = []
            used = set()
            for _ in range(6):
                x = random.randint(1, 63)
                if x not in used and (128 - x) not in used:
                    arr.append(x)
                    arr.append(128 - x)
                    used.add(x)
                    used.add(128 - x)
            while len(arr) < 12:
                val = random.randint(0, 128)
                if val not in arr:
                    arr.append(val)
            arr = sorted(arr[:12])
        else:
            # Quadratic sequence: i^2 mod something
            base = random.randint(0, 20)
            scale = random.randint(1, 5)
            arr = sorted(set([(base + i * i * scale) % 129 for i in range(12)]))
            while len(arr) < 12:
                val = random.randint(0, 128)
                if val not in arr:
                    arr.append(val)
            arr = sorted(arr[:12])
        
        if len(arr) != 12 or len(set(arr)) != 12:
            continue
            
        current_surrogate = _count_duplicate_sums(arr)
        
        # Only check exact size if surrogate is promising
        if current_surrogate > best_surrogate or restart % 10 == 0:
            current_size = _max_sidon_subset_size(arr)
            if current_size < best_size:
                best_arr = arr
                best_size = current_size
                best_surrogate = current_surrogate
                if best_size <= 4:
                    return best_arr
        
        # Local Search with multiple move types, using surrogate for speed
        no_improve_count = 0
        for step in range(500):
            if time.time() > deadline:
                break
                
            move_type = random.randint(0, 5)
            
            if move_type == 0:
                # Replace one element
                i = random.randint(0, 11)
                new_val = random.randint(0, 128)
                if new_val in arr:
                    continue
                new_arr = arr.copy()
                new_arr[i] = new_val
            elif move_type == 1:
                # Swap two elements
                i, j = random.sample(range(12), 2)
                new_arr = arr.copy()
                new_arr[i], new_arr[j] = new_arr[j], new_arr[i]
            elif move_type == 2:
                # Replace two elements
                i, j = random.sample(range(12), 2)
                new_val1 = random.randint(0, 128)
                new_val2 = random.randint(0, 128)
                if new_val1 in arr or new_val2 in arr or new_val1 == new_val2:
                    continue
                new_arr = arr.copy()
                new_arr[i] = new_val1
                new_arr[j] = new_val2
            elif move_type == 3:
                # Shift: add random delta to one element
                i = random.randint(0, 11)
                delta = random.randint(-15, 15)
                new_val = arr[i] + delta
                if new_val < 0 or new_val > 128 or new_val in arr:
                    continue
                new_arr = arr.copy()
                new_arr[i] = new_val
            elif move_type == 4:
                # Create sum collision: pick two pairs that should have same sum
                i, j = random.sample(range(12), 2)
                k, l = random.sample([x for x in range(12) if x not in (i, j)], 2)
                target_sum = arr[i] + arr[j]
                # Try to make arr[k] + arr[l] = target_sum by adjusting one
                delta_needed = target_sum - (arr[k] + arr[l])
                new_val = arr[l] + delta_needed
                if 0 <= new_val <= 128 and new_val not in arr:
                    new_arr = arr.copy()
                    new_arr[l] = new_val
                else:
                    continue
            else:
                # Replace three elements
                indices = random.sample(range(12), 3)
                new_vals = []
                valid = True
                for _ in range(3):
                    nv = random.randint(0, 128)
                    if nv in arr or nv in new_vals:
                        valid = False
                        break
                    new_vals.append(nv)
                if not valid:
                    continue
                new_arr = arr.copy()
                for idx, nv in zip(indices, new_vals):
                    new_arr[idx] = nv
            
            new_arr = sorted(new_arr)
            if len(set(new_arr)) != 12:
                continue
            
            new_surrogate = _count_duplicate_sums(new_arr)
            
            if new_surrogate > current_surrogate:
                arr = new_arr
                current_surrogate = new_surrogate
                no_improve_count = 0
                
                # Check exact size periodically or if surrogate improves significantly
                if step % 10 == 0 or new_surrogate > best_surrogate:
                    new_size = _max_sidon_subset_size(arr)
                    if new_size < best_size:
                        best_arr = arr
                        best_size = new_size
                        best_surrogate = new_surrogate
                        if best_size <= 4:
                            return best_arr
            else:
                no_improve_count += 1
                if no_improve_count > 50:
                    break
        
    return best_arr
# EVOLVE-BLOCK-END
