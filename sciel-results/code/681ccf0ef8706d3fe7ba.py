import time
import random

N = 256
DEADLINE = 85.0  # Safety margin before 90s limit


def _precompute_4ap_masks():
    """Precompute for each element x, the bitmask of elements that would form a 4-AP with x."""
    # For each position x (0-indexed), find all 4-APs that include x
    # Then for each such 4-AP, the other 3 elements form a "conflict" set
    # If all 3 are present, adding x creates a 4-AP
    # We store for each x a list of frozensets of 3 elements (the other members)
    # But for bitmask speed, we store the bitmask of the other 3 elements
    masks = [0] * N  # bitmask of elements that, if all present along with x, form a 4-AP
    # Actually, we need: for element x, the set of 3-element subsets S such that S ∪ {x} is a 4-AP
    # We'll store a list of 3-element masks for each x
    conflict_masks = [[] for _ in range(N)]  # list of 3-bit masks per element
    
    for a in range(N):
        for d in range(1, (N - 1 - a) // 3 + 1):
            b = a + d
            c = b + d
            e = c + d
            if e >= N:
                break
            # 4-AP: a, b, c, e (0-indexed)
            # For each element in this 4-AP, the other 3 form a conflict
            quad = [a, b, c, e]
            for i, x in enumerate(quad):
                others = [quad[j] for j in range(4) if j != i]
                mask = (1 << others[0]) | (1 << others[1]) | (1 << others[2])
                conflict_masks[x].append(mask)
    
    return conflict_masks


def _precompute_3ap_pairs():
    """Precompute all 3-APs as pairs (a, c) where b = (a+c)//2."""
    # For counting: a 3-AP (a, b, c) with a < b < c, b-a = c-b
    # We'll precompute for each middle element b, the list of (a, c) pairs
    pairs_by_mid = [[] for _ in range(N)]
    for b in range(N):
        for d in range(1, min(b, N - 1 - b) + 1):
            a = b - d
            c = b + d
            pairs_by_mid[b].append((a, c))
    return pairs_by_mid


def _count_3ap_bitmask(bits):
    """Count 3-APs using precomputed pairs."""
    count = 0
    for b in range(N):
        if not (bits >> b) & 1:
            continue
        for (a, c) in _pairs_by_mid[b]:
            if (bits >> a) & 1 and (bits >> c) & 1:
                count += 1
    return count


def _is_valid_4ap_free(bits, x, conflict_masks):
    """Check if adding element x to bits keeps it 4-AP free."""
    for mask in conflict_masks[x]:
        if (bits & mask) == mask:
            return False
    return True


def _remove_is_valid(bits, x, conflict_masks):
    """Removing x is always valid (fewer elements can't create new 4-APs)."""
    return True


def construct():
    start_time = time.time()
    
    # Precompute
    conflict_masks = _precompute_4ap_masks()
    _pairs_by_mid_local = _pairs_by_mid
    
    # Build initial set using greedy
    bits = 0
    for x in range(N):
        if time.time() - start_time > DEADLINE - 5:
            break
        if _is_valid_4ap_free(bits, x, conflict_masks):
            bits |= (1 << x)
    
    current_score = _count_3ap_bitmask(bits)
    best_bits = bits
    best_score = current_score
    
    # Simulated annealing
    if time.time() - start_time < DEADLINE - 3:
        T = 10.0
        T_min = 0.01
        cooling = 0.995
        rng = random.Random(12345)
        
        while time.time() - start_time < DEADLINE - 3:
            T *= cooling
            if T < T_min:
                T = T_min
            
            # Pick a random element to flip
            x = rng.randint(0, N - 1)
            is_present = (bits >> x) & 1
            
            if is_present:
                # Remove x
                new_bits = bits & ~(1 << x)
                new_score = _count_3ap_bitmask(new_bits)
                delta = new_score - current_score
                if delta > 0 or rng.random() < min(1.0, 2.0 * delta / max(T, 0.01)):
                    bits = new_bits
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_bits = bits
            else:
                # Add x
                if not _is_valid_4ap_free(bits, x, conflict_masks):
                    continue
                new_bits = bits | (1 << x)
                new_score = _count_3ap_bitmask(new_bits)
                delta = new_score - current_score
                if delta > 0 or rng.random() < min(1.0, 2.0 * delta / max(T, 0.01)):
                    bits = new_bits
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_bits = bits
    
    # Convert best_bits to sorted list (1-indexed)
    result = [i + 1 for i in range(N) if (best_bits >> i) & 1]
    return result


# Precompute at module level for speed
_conflict_masks_global = _precompute_4ap_masks()
_pairs_by_mid = _precompute_3ap_pairs()


def _count_3ap_bitmask(bits):
    """Count 3-APs using precomputed pairs (module-level)."""
    count = 0
    for b in range(N):
        if not (bits >> b) & 1:
            continue
        for (a, c) in _pairs_by_mid[b]:
            if (bits >> a) & 1 and (bits >> c) & 1:
                count += 1
    return count


def _is_valid_4ap_free(bits, x, conflict_masks):
    """Check if adding element x to bits keeps it 4-AP free."""
    for mask in conflict_masks[x]:
        if (bits & mask) == mask:
            return False
    return True


def construct():
    start_time = time.time()
    conflict_masks = _conflict_masks_global
    
    # Build initial set using greedy
    bits = 0
    for x in range(N):
        if time.time() - start_time > DEADLINE - 5:
            break
        if _is_valid_4ap_free(bits, x, conflict_masks):
            bits |= (1 << x)
    
    current_score = _count_3ap_bitmask(bits)
    best_bits = bits
    best_score = current_score
    
    # Simulated annealing with multiple restarts
    rng = random.Random(12345)
    
    while time.time() - start_time < DEADLINE - 3:
        T = 8.0
        T_min = 0.005
        cooling = 0.997
        
        while time.time() - start_time < DEADLINE - 3 and T > T_min:
            T *= cooling
            
            # Pick a random element to flip
            x = rng.randint(0, N - 1)
            is_present = (bits >> x) & 1
            
            if is_present:
                # Remove x - always valid
                new_bits = bits & ~(1 << x)
                new_score = _count_3ap_bitmask(new_bits)
                delta = new_score - current_score
                if delta >= 0:
                    bits = new_bits
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_bits = bits
                else:
                    # Accept with probability based on temperature
                    prob = min(1.0, -2.0 * delta / max(T, 0.01))
                    if rng.random() < prob:
                        bits = new_bits
                        current_score = new_score
            else:
                # Add x - check validity
                if not _is_valid_4ap_free(bits, x, conflict_masks):
                    continue
                new_bits = bits | (1 << x)
                new_score = _count_3ap_bitmask(new_bits)
                delta = new_score - current_score
                if delta >= 0:
                    bits = new_bits
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_bits = bits
                else:
                    prob = min(1.0, -2.0 * delta / max(T, 0.01))
                    if rng.random() < prob:
                        bits = new_bits
                        current_score = new_score
        
        # After each cooling schedule, try a few random restarts from best
        if time.time() - start_time > DEADLINE - 10:
            break
        # Occasionally restart from best
        if rng.random() < 0.1:
            bits = best_bits
            current_score = best_score
    
    # Convert best_bits to sorted list (1-indexed)
    result = [i + 1 for i in range(N) if (best_bits >> i) & 1]
    return result
