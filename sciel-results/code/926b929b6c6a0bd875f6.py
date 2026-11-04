import time
from math import log, floor, sqrt
import random
import numpy as np

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    sums = set()
    diffs = set()
    
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def fast_score_set(U):
    """Vectorized scoring using numpy for speed."""
    U = np.array(sorted(set(U)))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    # Compute all pairwise sums (i <= j)
    i_idx, j_idx = np.triu_indices(n)
    sums = U[i_idx] + U[j_idx]
    sum_count = len(np.unique(sums))
    
    # Compute all pairwise differences (i < j)
    i_idx2, j_idx2 = np.triu_indices(n, k=1)
    diffs_pos = U[j_idx2] - U[i_idx2]
    diffs = np.concatenate([diffs_pos, -diffs_pos, [0]])
    diff_count = len(np.unique(diffs))
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = int(U.max())
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def greedy_binary_construction(max_bits=12, max_elements=30, deadline=None):
    """
    Greedily construct U by selecting elements that maximize |U-U|/|U+U| ratio increase.
    Uses binary representation to prefer elements with sparse bit patterns (reduces carry collisions).
    """
    U = [0]
    U_set = {0}
    
    # Generate candidate elements with sparse binary representations
    candidates = []
    for bits in range(1, max_bits + 1):
        # Generate all numbers with exactly 'bits' set bits in the lower max_bits positions
        from itertools import combinations
        for combo in combinations(range(max_bits), bits):
            val = 0
            for b in combo:
                val |= (1 << b)
            if val > 0 and val not in U_set:
                candidates.append(val)
    
    # Also add some structured candidates
    for k in range(1, max_bits + 1):
        val = (1 << k) - 1  # Mersenne numbers
        if val > 0 and val not in U_set:
            candidates.append(val)
        val2 = 1 << k  # Powers of 2
        if val2 > 0 and val2 not in U_set:
            candidates.append(val2)
    
    candidates = sorted(set(candidates))
    
    current_score = fast_score_set(U)
    
    for _ in range(max_elements - 1):
        if deadline is not None and time.time() > deadline:
            break
        
        best_candidate = None
        best_new_score = current_score
        best_ratio_gain = 0
        
        # Sample candidates for efficiency
        if len(candidates) > 200:
            sample_size = min(200, len(candidates))
            sampled = random.sample(candidates, sample_size)
        else:
            sampled = candidates
        
        for c in sampled:
            if c in U_set:
                continue
            
            new_U = U + [c]
            new_score = fast_score_set(new_U)
            
            if new_score > best_new_score:
                best_new_score = new_score
                best_candidate = c
                best_ratio_gain = new_score - current_score
        
        if best_candidate is None:
            break
        
        U.append(best_candidate)
        U_set.add(best_candidate)
        current_score = best_new_score
        
        # Remove used candidate
        candidates = [c for c in candidates if c != best_candidate]
        
        if len(candidates) < 50:
            # Generate more candidates
            for bits in range(1, max_bits + 1):
                from itertools import combinations
                for combo in combinations(range(max_bits + 1), bits):
                    val = 0
                    for b in combo:
                        val |= (1 << b)
                    if val > 0 and val not in U_set and val not in candidates:
                        candidates.append(val)
            candidates = sorted(set(candidates))
    
    return sorted(U)

def geometric_set(b, m):
    """Generate U = {floor(b^k) for k=0..m}, ensuring 0 is included and all are nonnegative distinct integers."""
    U = [0]
    seen = {0}
    for k in range(1, m + 1):
        val = floor(b ** k)
        if val <= 0:
            val = 1
        if val not in seen:
            U.append(val)
            seen.add(val)
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
    
    # Precompute powers
    powers = [B ** j for j in range(k)]
    
    for mask in range(1, min(2 ** k, max_elements)):
        # Count bits
        bits = bin(mask).count('1')
        # Simple sparsity check: roughly between sparsity*0.5 and sparsity*1.5
        if 0.3 * k <= bits <= 0.8 * k:
            val = 0
            for j in range(k):
                if mask & (1 << j):
                    val += powers[j]
            U.add(val)
    
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
    """
    U = sorted(set(U))
    current_score = fast_score_set(U)
    best_local_score = current_score
    best_local_set = list(U)
    
    # Initial candidate pool
    candidates = generate_candidates(U, max_val_limit)
    
    no_improvement_count = 0
    max_no_improvement = 30
    
    while time.time() < deadline - 2:
        if no_improvement_count > max_no_improvement:
            break
            
        improved = False
        move_type = random.random()
        
        # Update candidate pool periodically or if empty
        if not candidates or random.random() < 0.1:
            # Expand limit if we are growing
            current_max = max(U)
            new_limit = max(max_val_limit, current_max * 2)
            candidates = generate_candidates(U, new_limit)
        
        if move_type < 0.5:
            # Try to ADD an element
            if len(U) < max_n_limit and candidates:
                # Try a few random candidates
                random.shuffle(candidates)
                for c in candidates[:10]:
                    if c in U:
                        continue
                    new_U = U + [c]
                    new_U.sort()
                    new_score = fast_score_set(new_U)
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
                    new_score = fast_score_set(new_U)
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
                # Pick a random candidate
                random.shuffle(candidates)
                for new_val in candidates[:10]:
                    if new_val == old_val or new_val in U:
                        continue
                    new_U = U.copy()
                    new_U[idx] = new_val
                    new_U.sort()
                    new_score = fast_score_set(new_U)
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

def high_precision_polish(U, deadline):
    """
    Final high-precision polish pass with tighter tolerance and more iterations.
    Uses exact scoring and explores a wider neighborhood.
    """
    U = sorted(set(U))
    current_score = score_set(U)  # Use exact scoring for final polish
    
    improved = True
    iteration = 0
    max_iterations = 200
    
    while improved and time.time() < deadline - 1 and iteration < max_iterations:
        improved = False
        iteration += 1
        n = len(U)
        
        # Try replacing each element with nearby values
        for i in range(1, n):
            if time.time() > deadline - 1:
                break
            old_val = U[i]
            # Wider range for high precision
            for delta in range(-20, 21):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val <= 0:
                    continue
                new_set = [U[j] if j != i else new_val for j in range(n)]
                new_set = sorted(set(new_set))
                if len(new_set) != n:
                    continue
                new_score = score_set(new_set)
                if new_score > current_score:
                    U = new_set
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break
        
        # Try adding an element
        if not improved and n < 30 and time.time() < deadline - 1:
            max_val = max(U)
            for new_val in range(max(1, max_val - 20), max_val + 21):
                if new_val in U:
                    continue
                new_set = sorted(U + [new_val])
                new_score = score_set(new_set)
                if new_score > current_score:
                    U = new_set
                    current_score = new_score
                    improved = True
                    break
        
        # Try removing an element
        if not improved and n > 3 and time.time() < deadline - 1:
            for i in range(1, n):
                new_set = [U[j] for j in range(n) if j != i]
                new_score = score_set(new_set)
                if new_score > current_score:
                    U = new_set
                    current_score = new_score
                    improved = True
                    break
    
    return U, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

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
    
    for cand in structured_candidates:
        s = fast_score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # 2. Greedy Binary Construction (New Strategy)
    # Uses binary representation to prevent carries in addition
    greedy_deadline = deadline - 40
    greedy_set = greedy_binary_construction(max_bits=12, max_elements=25, deadline=greedy_deadline)
    if len(greedy_set) >= 2:
        s = fast_score_set(greedy_set)
        if s > best_score:
            best_score = s
            best_set = list(greedy_set)

    # 3. Base Expansion Candidates
    for B in [2, 3, 4, 5, 6]:
        for k in range(3, 12):
            if time.time() > deadline - 30:
                break
            # Full expansion
            U = base_expansion_set(B, k, max_elements=50)
            if len(U) >= 2:
                s = fast_score_set(U)
                if s > best_score:
                    best_score = s
                    best_set = U
            
            # Mixed expansion (sparsity)
            U = base_expansion_mixed(B, k, sparsity=0.5, max_elements=50)
            if len(U) >= 2:
                s = fast_score_set(U)
                if s > best_score:
                    best_score = s
                    best_set = U

    # 4. Geometric Search
    b_candidates = [b_int / 10.0 for b_int in range(12, 101)]
    best_b = None
    best_m = None
    best_grid_score = -1

    for b in b_candidates:
        if time.time() > deadline - 25:
            break
        for m in range(2, 21):
            U = geometric_set(b, m)
            if len(U) < 2:
                continue
            s = fast_score_set(U)
            if s > best_grid_score:
                best_grid_score = s
                best_b = b
                best_m = m
                if s > best_score:
                    best_score = s
                    best_set = U

    # 5. Dynamic Local Search
    seeds = [best_set]
    for cand in structured_candidates[:3]:
        if cand != best_set:
            seeds.append(cand)
    
    if greedy_set and greedy_set != best_set:
        seeds.append(greedy_set)
        
    max_val_limit = 500
    max_n_limit = 30
    
    restarts = 0
    while time.time() < deadline - 15:
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

    # 6. High-Precision Polish (Final Pass)
    if time.time() < deadline - 2:
        polished_set, polished_score = high_precision_polish(best_set, deadline)
        if polished_score > best_score:
            best_score = polished_score
            best_set = polished_set

    # Final verification with exact scoring
    final_score = score_set(best_set)
    if final_score > best_score:
        best_score = final_score

    return best_set

if __name__ == "__main__":
    # Quick test
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
