"""Improved program: cardinality-indexed subset-sum DP to maximize set size."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

DEADLINE_SECONDS = 55.0  # safety margin before 60s kill

# Maximum possible sum: 60 * 256 = 15360
# We'll use bitsets with enough bits to cover all possible sums
MAX_SUM = 60 * 256  # 15360


def _bitset_add_to_existing(existing_bits, candidate, n_elements):
    """
    Check if adding 'candidate' to a set with existing bitsets creates collisions.
    existing_bits[k] = integer where bit i is set if sum i is achievable with k elements.
    Returns (True, new_bits) if valid, (False, None) otherwise.
    """
    n = n_elements  # current number of elements
    new_bits = [0] * (n + 2)
    new_bits[0] = 1  # sum 0 achievable with 0 elements
    
    # For each cardinality k, new sums = existing sums at k-1 shifted by candidate
    # Check for collisions: new sum at cardinality k must not exist at any other cardinality
    for k in range(1, n + 2):
        if k - 1 < len(existing_bits):
            shifted = existing_bits[k - 1] << candidate
            new_bits[k] = existing_bits[k] | shifted
    
    # Check for collisions: no sum should be achievable at two different cardinalities
    # Build a "sum used at any cardinality" mask and check overlaps
    for k in range(1, n + 2):
        if new_bits[k] == 0:
            continue
        # Check if any bit in new_bits[k] is also set in new_bits[j] for j != k
        # More efficient: build cumulative mask
        mask = 0
        for j in range(n + 2):
            if j != k:
                mask |= new_bits[j]
        if new_bits[k] & mask:
            return False, None
    
    return True, new_bits


def _check_add_bitset(bits, candidate, n_elements):
    """
    Fast collision check using bitsets.
    bits[k] = integer where bit i is set if sum i is achievable with k elements.
    Returns (True, new_bits) if valid, (False, None) otherwise.
    """
    n = n_elements
    new_bits = [0] * (n + 2)
    new_bits[0] = 1
    
    for k in range(1, n + 2):
        if k - 1 < len(bits):
            shifted = bits[k - 1] << candidate
            new_bits[k] = bits[k] | shifted
    
    # Check for collisions between different cardinalities
    for k in range(1, n + 2):
        if new_bits[k] == 0:
            continue
        for j in range(k + 1, n + 2):
            if new_bits[j] and (new_bits[k] & new_bits[j]):
                return False, None
    
    return True, new_bits


def _build_bitset(S):
    """Build bitset representation for set S."""
    n = len(S)
    bits = [0] * (n + 1)
    bits[0] = 1  # sum 0 with 0 elements
    
    for x in S:
        for k in range(n, 0, -1):
            if bits[k - 1]:
                bits[k] |= bits[k - 1] << x
    
    return bits


def _greedy_build_bitset(candidates, deadline):
    """Greedily build a valid set using bitset-based collision checks."""
    current = []
    n = 0
    bits = [1]  # bits[0] = 1 (sum 0 achievable with 0 elements)
    
    for x in candidates:
        if time.time() > deadline:
            break
        if len(current) >= 60:
            break
        
        valid, new_bits = _check_add_bitset(bits, x, n)
        if valid:
            current.append(x)
            n += 1
            bits = new_bits
    
    return current


def _score_candidate_bitset(bits, candidate, n_elements):
    """
    Score candidate using bitsets: count how many new sums would collide.
    Lower is better.
    """
    n = n_elements
    collision_count = 0
    
    for k in range(1, n + 2):
        if k - 1 < len(bits) and bits[k - 1]:
            shifted = bits[k - 1] << candidate
            # Check overlap with existing sums at cardinality k (other than the shifted part)
            # Actually, we need to check if any bit in shifted is already set in bits[j] for j != k
            # But that's expensive. Instead, just check if shifted has any bits that would collide.
            # For now, return 0 if no immediate collision, else count
            for j in range(n + 2):
                if j != k and bits[j]:
                    if shifted & bits[j]:
                        collision_count += 1
                        break
    
    return collision_count


def _greedy_build_scored_bitset(candidates, deadline):
    """Greedy build with scoring using bitsets."""
    current = []
    n = 0
    bits = [1]
    remaining = list(candidates)
    
    while remaining and time.time() < deadline and len(current) < 60:
        best_x = None
        best_score = float('inf')
        
        for x in remaining:
            sc = _score_candidate_bitset(bits, x, n)
            if sc < best_score:
                best_score = sc
                best_x = x
            if sc == 0:
                break
        
        if best_x is None:
            break
        
        valid, new_bits = _check_add_bitset(bits, best_x, n)
        if valid:
            current.append(best_x)
            n += 1
            bits = new_bits
            remaining.remove(best_x)
        else:
            remaining.remove(best_x)
    
    return current


def _local_search_bitset(best_set, deadline, max_iter=50):
    """Local search improvement using bitset checks."""
    current = best_set[:]
    if not current:
        return current
    
    n = len(current)
    bits = _build_bitset(current)
    in_set = set(current)
    
    # Try to add elements
    remaining = [x for x in range(1, 257) if x not in in_set]
    random.shuffle(remaining)
    
    for x in remaining:
        if time.time() > deadline:
            break
        if len(current) >= 60:
            break
        valid, new_bits = _check_add_bitset(bits, x, n)
        if valid:
            current.append(x)
            n += 1
            bits = new_bits
            in_set.add(x)
    
    # Try swaps
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        if len(current) >= 60:
            break
        
        found_swap = False
        current_list = current[:]
        random.shuffle(current_list)
        
        for elem in current_list:
            if time.time() > deadline:
                break
            
            without = [x for x in current if x != elem]
            if not without:
                continue
            
            n_wo = len(without)
            bits_wo = _build_bitset(without)
            in_wo = set(without)
            
            candidates_to_try = [x for x in range(1, 257) if x not in in_wo]
            random.shuffle(candidates_to_try)
            
            best_addition = None
            best_further = 0
            
            for x in candidates_to_try[:30]:  # Limit candidates per swap attempt
                if time.time() > deadline - 5:
                    break
                
                valid, new_bits = _check_add_bitset(bits_wo, x, n_wo)
                if valid:
                    # Try to add more elements after this swap
                    further = 0
                    test_bits = new_bits
                    test_n = n_wo + 1
                    test_set = without + [x]
                    test_in = set(test_set)
                    
                    remaining2 = [y for y in range(1, 257) if y not in test_in]
                    random.shuffle(remaining2)
                    
                    for y in remaining2[:15]:
                        if time.time() > deadline - 3:
                            break
                        valid2, new_bits2 = _check_add_bitset(test_bits, y, test_n)
                        if valid2:
                            further += 1
                    
                    if further > best_further:
                        best_further = further
                        best_addition = x
                
                if time.time() > deadline - 5:
                    break
            
            if best_addition is not None and best_further > 0:
                current = without + [best_addition]
                n = len(current)
                bits = _build_bitset(current)
                in_set = set(current)
                found_swap = True
                break
        
        if not found_swap:
            break
    
    return current


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE_SECONDS
    
    best_set = []
    best_size = 0
    
    # Strategy 1: Deterministic ascending
    candidates = list(range(1, 257))
    current = _greedy_build_bitset(candidates, deadline)
    if len(current) > best_size:
        best_size = len(current)
        best_set = current[:]
    
    # Strategy 2: Deterministic descending
    candidates = list(range(256, 0, -1))
    current = _greedy_build_bitset(candidates, deadline)
    if len(current) > best_size:
        best_size = len(current)
        best_set = current[:]
    
    # Strategy 3: Scored greedy
    candidates = list(range(1, 257))
    current = _greedy_build_scored_bitset(candidates, deadline)
    if len(current) > best_size:
        best_size = len(current)
        best_set = current[:]
    
    # Strategy 4: Random restarts
    random.seed(42)
    for attempt in range(2000):
        if time.time() > deadline:
            break
        
        if attempt % 5 == 0:
            candidates = list(range(1, 257))
            random.shuffle(candidates)
            current = _greedy_build_bitset(candidates, deadline)
        elif attempt % 5 == 1:
            # Bias toward smaller values
            candidates = list(range(1, 129))
            random.shuffle(candidates)
            candidates += list(range(129, 257))
            random.shuffle(candidates)
            current = _greedy_build_bitset(candidates, deadline)
        elif attempt % 5 == 2:
            # Mix of small and large
            small = list(range(1, 65))
            large = list(range(192, 257))
            random.shuffle(small)
            random.shuffle(large)
            candidates = small + large
            random.shuffle(candidates)
            current = _greedy_build_bitset(candidates, deadline)
        elif attempt % 5 == 3:
            # Scored greedy on random order
            candidates = list(range(1, 257))
            random.shuffle(candidates)
            current = _greedy_build_scored_bitset(candidates, deadline)
        else:
            # Powers of 2 based lattice
            base_candidates = [1, 2, 4, 8, 16, 32, 64, 128]
            extras = [x for x in range(1, 257) if x not in set(base_candidates)]
            random.shuffle(extras)
            candidates = base_candidates + extras[:100]
            random.shuffle(candidates)
            current = _greedy_build_bitset(candidates, deadline)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
        
        # Try to extend best_set with remaining candidates
        if best_size >= 15 and time.time() < deadline - 5:
            in_set = set(best_set)
            remaining = [x for x in range(1, 257) if x not in in_set]
            random.shuffle(remaining)
            
            n = len(best_set)
            bits = _build_bitset(best_set)
            
            improved = best_set[:]
            for x in remaining:
                if time.time() > deadline:
                    break
                if len(improved) >= 60:
                    break
                valid, new_bits = _check_add_bitset(bits, x, n)
                if valid:
                    improved.append(x)
                    n += 1
                    bits = new_bits
            
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved[:]
    
    # Final local search improvement
    if best_set and time.time() < deadline - 3:
        improved = _local_search_bitset(best_set, deadline)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved[:]
    
    # Ensure all elements are Python int
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
