import time
from math import log, floor
import random
import numpy as np

def score_set(U):
    """Compute the score for a set U of nonnegative integers containing 0."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1

    arr = np.array(U, dtype=np.int64)
    # All pairwise sums (i <= j) and differences
    sums = np.add.outer(arr, arr)
    sum_count = len(np.unique(sums))

    # All pairwise differences (including 0 and negatives)
    diffs = np.subtract.outer(arr, arr)
    diff_count = len(np.unique(diffs))

    if sum_count == 0:
        return -1

    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1

    max_u = U[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1

    return 1 + log(ratio) / log(D)

def geometric_set(r, k):
    """
    Generate a geometric-like set: {0, 1, r, r^2, ..., r^(k-1)} plus some structured additions.
    Geometric progressions have good difference/sum ratio properties.
    """
    U = {0, 1}
    val = 1
    for i in range(k):
        val *= r
        U.add(val)
    # Add some intermediate points to improve the ratio
    for i in range(1, k):
        base = r ** i
        U.add(base + 1)
        if i + 1 < k:
            U.add(base + r ** (i + 1) // 2 if r ** (i + 1) % 2 == 0 else base + r ** (i + 1) // 2 + 1)
    return sorted(U)

def base_expansion_set(B, k, max_elements=None):
    """
    Generate a set using base-B expansion hierarchy.
    Elements are u = sum_{j=0}^{k-1} b_j * B^j where b_j in {0, 1}.
    This structurally limits sumset collisions to local digit-wise interactions.
    """
    if max_elements is None:
        max_elements = 2 ** k
    
    U = set()
    U.add(0)
    
    for mask in range(1, min(2 ** k, max_elements)):
        val = 0
        for j in range(k):
            if mask & (1 << j):
                val += B ** j
        U.add(val)
    
    return sorted(U)

def base_expansion_mixed(B, k, sparsity=0.5, max_elements=None):
    """
    Generate a set using base-B expansion with sparsity control.
    Only include elements with a certain fraction of active digits.
    """
    if max_elements is None:
        max_elements = 2 ** k
    
    U = set()
    U.add(0)
    
    target_count = int(sparsity * (2 ** k))
    count = 0
    
    for mask in range(1, min(2 ** k, max_elements)):
        # Count active digits
        active = bin(mask).count('1')
        if active <= k * sparsity + 0.5:  # Include if density is below sparsity threshold
            val = 0
            for j in range(k):
                if mask & (1 << j):
                    val += B ** j
            if val not in U:
                U.add(val)
                count += 1
                if count >= target_count:
                    break
    
    return sorted(U)

def generate_candidates(U, limit_val):
    """
    Generate candidate elements for addition based on sums and differences of existing elements.
    """
    U_set = set(U)
    candidates = set()
    
    for a in U:
        for b in U:
            s = a + b
            d = abs(a - b)
            if 0 < s <= limit_val and s not in U_set:
                candidates.add(s)
            if 0 < d <= limit_val and d not in U_set:
                candidates.add(d)
            
    return list(candidates)

def local_search_dynamic(U, deadline, max_val_limit=1000, max_n_limit=40):
    """
    Local search with dynamic candidate generation.
    Uses numpy-accelerated scoring for speed.
    """
    U = sorted(set(U))
    current_score = score_set(U)
    best_local_score = current_score
    best_local_set = list(U)
    
    # Initial candidate pool
    candidates = generate_candidates(U, max_val_limit)
    
    no_improvement_count = 0
    max_no_improvement = 20  # Reduced since scoring is faster
    
    while time.time() < deadline - 2:
        if no_improvement_count > max_no_improvement:
            break
            
        improved = False
        move_type = random.random()
        
        # Update candidate pool periodically or if empty
        if not candidates or random.random() < 0.15:
            # Expand limit if we are growing
            current_max = max(U)
            new_limit = max(max_val_limit, current_max * 2)
            candidates = generate_candidates(U, new_limit)
        
        if move_type < 0.5:
            # Try to ADD an element
            if len(U) < max_n_limit and candidates:
                # Try more random candidates since scoring is fast
                random.shuffle(candidates)
                for c in candidates[:20]:
                    if c in U:
                        continue
                    new_U = U + [c]
                    new_U.sort()
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        # Regenerate candidates immediately as structure changed significantly
                        candidates = generate_candidates(U, max(max_val_limit, max(U)*2))
                        break
        elif move_type < 0.7:
            # Try to REMOVE an element
            if len(U) > 2:
                removable = [x for x in U if x != 0]
                if removable:
                    c = random.choice(removable)
                    new_U = [x for x in U if x != c]
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_candidates(U, max_val_limit)
        else:
            # Try to REPLACE an element
            if len(U) > 1 and candidates:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                # Pick a random candidate - try more since scoring is fast
                random.shuffle(candidates)
                for new_val in candidates[:20]:
                    if new_val == old_val or new_val in U:
                        continue
                    new_U = U.copy()
                    new_U[idx] = new_val
                    new_U.sort()
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_candidates(U, max(max_val_limit, max(U)*2))
                        break
            
        if improved:
            no_improvement_count = 0
            if current_score > best_local_score:
                best_local_score = current_score
                best_local_set = list(U)
        else:
            no_improvement_count += 1
                
    return best_local_score, best_local_set

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    best_set = [0, 1, 4, 9, 11]
    best_score = score_set(best_set)

    # 1. Structured Candidates
    structured_candidates = [
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81], # Squares
        [0, 1, 2, 4, 8, 16, 32, 64, 128], # Powers of 2
        [0, 1, 3, 7, 15, 31, 63, 127], # Mersenne-like
        [0, 1, 4, 10, 20, 35, 56, 84], # Triangular
        [0, 1, 3, 4, 9, 10, 12, 13], # Small Sidon-like
        [0, 1, 4, 10, 18, 28, 40, 54], # Quadratic-ish
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256], # Larger powers of 2
        [0, 1, 3, 9, 27, 81, 243], # Powers of 3
    ]
    
    # Add geometric set candidates
    for r in [2, 3, 4, 5]:
        for k in range(3, 10):
            if time.time() > deadline - 80:
                break
            geo = geometric_set(r, k)
            if len(geo) >= 2:
                structured_candidates.append(geo)
    
    for cand in structured_candidates:
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # 2. Base Expansion Search
    # Search over different bases and depths for base expansion sets
    for B in [2, 3, 4, 5, 6, 7, 8, 10]:
        if time.time() > deadline - 50:
            break
        for k in range(2, 12):
            if time.time() > deadline - 50:
                break
            U = base_expansion_set(B, k)
            if len(U) < 2:
                continue
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = list(U)
            
            # Also try mixed sparsity
            for sparsity in [0.3, 0.5, 0.7]:
                if time.time() > deadline - 50:
                    break
                U_mixed = base_expansion_mixed(B, k, sparsity)
                if len(U_mixed) < 2:
                    continue
                s_mixed = score_set(U_mixed)
                if s_mixed > best_score:
                    best_score = s_mixed
                    best_set = list(U_mixed)

    # 3. Dynamic Local Search
    # Use the best found so far as the primary seed, and geometric sets as secondary seeds
    seeds = [best_set]
    
    # Prioritize geometric sets as seeds
    for cand in structured_candidates:
        if cand != best_set and len(cand) >= 3:
            # Check if it's a geometric-like set (contains 0, 1, and powers)
            is_geometric = False
            for r in [2, 3, 4, 5]:
                expected = {0, 1}
                val = 1
                for i in range(8):
                    val *= r
                    expected.add(val)
                if set(cand) & expected:
                    is_geometric = True
                    break
            if is_geometric:
                seeds.append(cand)
        if len(seeds) >= 6:
            break
    
    # Add a few more structured candidates if we don't have enough seeds
    for cand in structured_candidates:
        if cand not in seeds and len(cand) >= 3:
            seeds.append(cand)
        if len(seeds) >= 8:
            break
    
    max_val_limit = 500
    max_n_limit = 30
    
    restarts = 0
    while time.time() < deadline - 10:
        restarts += 1
        # Cycle through seeds
        base = seeds[restarts % len(seeds)]
        U = list(base)
        
        # Run local search
        local_score, local_set = local_search_dynamic(U, deadline, max_val_limit, max_n_limit)
        
        if local_score > best_score:
            best_score = local_score
            best_set = local_set
            # Add the new best to seeds for future restarts
            if len(seeds) < 5:
                seeds.append(local_set)
            else:
                seeds[-1] = local_set

    # 4. Final Polish: Fine-grained local search on the best set
    if time.time() < deadline - 2:
        current = list(best_set)
        current_score = best_score
        improved = True
        iteration = 0
        while improved and time.time() < deadline - 1 and iteration < 500:
            improved = False
            iteration += 1
            n = len(current)
            # Try replacing each element with nearby values (small delta)
            for i in range(1, n):
                if time.time() > deadline - 1:
                    break
                old_val = current[i]
                # Try deltas in a wider range but smaller steps
                for delta in range(-10, 11):
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val <= 0:
                        continue
                    new_set = [current[j] if j != i else new_val for j in range(n)]
                    new_set = sorted(set(new_set))
                    if len(new_set) != n:
                        continue
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            
            # Try adding an element near the max
            if not improved and n < 25 and time.time() < deadline - 1:
                max_val = max(current)
                for new_val in range(max_val - 10, max_val + 11):
                    if new_val <= 0:
                        continue
                    if new_val in current:
                        continue
                    new_set = sorted(current + [new_val])
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break

            # Try removing an element
            if not improved and n > 3 and time.time() < deadline - 1:
                for i in range(1, n):
                    new_set = [current[j] for j in range(n) if j != i]
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break

        if current_score > best_score:
            best_score = current_score
            best_set = current

    return best_set

if __name__ == "__main__":
    # Quick test
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
