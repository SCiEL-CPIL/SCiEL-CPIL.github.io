import time
import random

# EVOLVE-BLOCK-START
def _is_valid_bitset(arr):
    """Check if subset sums of different cardinalities are disjoint using bitsets."""
    n = len(arr)
    if n == 0:
        return True
    sums_by_card = [0] * (n + 1)
    sums_by_card[0] = 1
    
    for x in arr:
        for j in range(n - 1, -1, -1):
            if sums_by_card[j]:
                sums_by_card[j + 1] |= (sums_by_card[j] << x)
    
    for j in range(n + 1):
        for k in range(j + 1, n + 1):
            if sums_by_card[j] & sums_by_card[k]:
                return False
    return True

def _validate_full(arr):
    if len(arr) != len(set(arr)):
        return False
    if any(v < 1 or v > 256 for v in arr):
        return False
    return _is_valid_bitset(arr)

def construct():
    max_val = 256
    deadline = time.time() + 75
    
    # Step 1: Best AP
    best_set = None
    best_k = 0
    for k in range(60, 0, -1):
        max_j_val = max(j * (k - 1 - j) for j in range(1, k))
        required_x_1 = max_j_val + 1
        if required_x_1 <= 256 - k + 1:
            x_1 = required_x_1
            set_vals = list(range(x_1, x_1 + k))
            if len(set(set_vals)) == k and all(1 <= v <= max_val for v in set_vals):
                best_set = set_vals
                best_k = k
                break
    if best_set is None:
        return [1]
    
    # Step 2: Greedy add
    current = list(best_set)
    current_set = set(current)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for v in range(1, 257):
            if v in current_set:
                continue
            if time.time() >= deadline:
                break
            new_set = current + [v]
            if _is_valid_bitset(new_set):
                current = new_set
                current_set = current_set | {v}
                improved = True
                if len(current) > best_k:
                    best_set = list(current)
                    best_k = len(current)
    
    # Step 3: Random restarts with different AP bases
    if time.time() < deadline:
        for _ in range(50):
            if time.time() >= deadline:
                break
            # Try random initial sets
            k_try = random.randint(20, 40)
            if k_try <= best_k:
                continue
            cand = random.sample(range(1, 257), k_try)
            if _is_valid_bitset(cand):
                current = cand
                current_set = set(cand)
                if len(current) > best_k:
                    best_set = list(current)
                    best_k = len(current)
                # Try to grow
                improved = True
                while improved and time.time() < deadline:
                    improved = False
                    for v in range(1, 257):
                        if v in current_set:
                            continue
                        if time.time() >= deadline:
                            break
                        new_set = current + [v]
                        if _is_valid_bitset(new_set):
                            current = new_set
                            current_set = current_set | {v}
                            improved = True
                            if len(current) > best_k:
                                best_set = list(current)
                                best_k = len(current)
    
    if not _validate_full(best_set):
        for k in range(60, 0, -1):
            max_j_val = max(j * (k - 1 - j) for j in range(1, k))
            required_x_1 = max_j_val + 1
            if required_x_1 <= 256 - k + 1:
                x_1 = required_x_1
                set_vals = list(range(x_1, x_1 + k))
                if len(set(set_vals)) == k and all(1 <= v <= max_val for v in set_vals):
                    best_set = set_vals
                    break
        else:
            best_set = [1]
    
    return list(best_set)
# EVOLVE-BLOCK-END
