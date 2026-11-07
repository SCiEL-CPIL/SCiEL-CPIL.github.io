import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _is_valid_set(S):
    """
    Check if set S (list/tuple of ints) satisfies the constraint:
    No three distinct a, b, c in S such that a+b, a+c, b+c are all in S.
    """
    S_set = set(S)
    n = len(S)
    for i in range(n):
        a = S[i]
        for j in range(i + 1, n):
            b = S[j]
            if a + b in S_set:
                for k in range(j + 1, n):
                    c = S[k]
                    if a + c in S_set and b + c in S_set:
                        return False
    return True

def _is_valid_mask(mask):
    """
    Fast check using bitmasks. Bit i represents integer i+1 (1..128).
    Condition: No three distinct members a, b, c in S such that a+b, a+c, b+c are also in S.
    """
    # For each pair (a, b) with a < b where a+b is in S,
    # check if there exists c > b such that a+c and b+c are in S
    for i in range(128):
        if not (mask >> i) & 1:
            continue
        for j in range(i + 1, 128):
            if not (mask >> j) & 1:
                continue
            
            # Check if a+b is in S
            # a = i+1, b = j+1 => a+b = i+j+2 => bit index i+j+1
            if i + j + 1 >= 128:
                continue
            if not (mask >> (i + j + 1)) & 1:
                continue
            
            # If a+b is in S, check if there exists c > b such that a+c in S and b+c in S
            # c is value k+1, where k > j
            # a+c = i+1 + k+1 = i+k+2 => Bit index i+k+1
            # b+c = j+1 + k+1 = j+k+2 => Bit index j+k+1
            # Need: bit k set, bit (i+k+1) set, bit (j+k+1) set, for some k > j
            
            # candidates = mask & (mask >> (i+1)) & (mask >> (j+1))
            # But we only care about k > j
            # For k > j: (mask >> (i+1)) bit k corresponds to original bit k+i+1
            # (mask >> (j+1)) bit k corresponds to original bit k+j+1
            
            candidates = mask & (mask >> (i + 1)) & (mask >> (j + 1))
            
            # We want bits k where k > j, i.e., k >= j+1
            # Mask out bits 0..j
            if candidates & ~((1 << (j + 1)) - 1):
                return False
    return True

def _try_add(mask, val):
    """Try to add value val to mask. Return new mask if valid, else None."""
    bit = 1 << (val - 1)
    if mask & bit:
        return mask
    trial = mask | bit
    if _is_valid_mask(trial):
        return trial
    return None

def _greedy_construct(deadline):
    """
    Greedy construction: start from empty, add elements in order of increasing 'difficulty'
    (elements that conflict with fewer existing elements are easier to add).
    """
    # Start with high values first (they're easier to add since their sums go out of range)
    mask = 0
    
    # First, add all values 65..128 (they can't form valid triples with sums in range)
    for v in range(65, 129):
        bit = 1 << (v - 1)
        mask |= bit
    
    # Now try to add values 1..64 in order
    for v in range(1, 65):
        if time.time() > deadline:
            break
        new_mask = _try_add(mask, v)
        if new_mask is not None:
            mask = new_mask
    
    return mask

def _local_search_add(mask, deadline):
    """
    Local search: try to add each missing element one at a time.
    Return improved mask.
    """
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for v in range(1, 129):
            if time.time() > deadline:
                break
            if mask & (1 << (v - 1)):
                continue
            new_mask = _try_add(mask, v)
            if new_mask is not None:
                mask = new_mask
                improved = True
    return mask

def _local_search_swap(mask, deadline):
    """
    Local search: try removing one element and adding two others.
    Return improved mask.
    """
    improved = True
    while improved and time.time() < deadline:
        improved = False
        set_bits = [v for v in range(1, 129) if mask & (1 << (v - 1))]
        for rem in set_bits:
            if time.time() > deadline:
                break
            base = mask & ~(1 << (rem - 1))
            # Try adding elements
            for add1 in range(1, 129):
                if time.time() > deadline:
                    break
                if add1 == rem or (base & (1 << (add1 - 1))):
                    continue
                trial1 = base | (1 << (add1 - 1))
                if not _is_valid_mask(trial1):
                    continue
                # Try adding a second element
                for add2 in range(add1 + 1, 129):
                    if time.time() > deadline:
                        break
                    if add2 == rem or (trial1 & (1 << (add2 - 1))):
                        continue
                    trial2 = trial1 | (1 << (add2 - 1))
                    if _is_valid_mask(trial2):
                        if trial2.bit_count() > mask.bit_count():
                            mask = trial2
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    return mask

def _randomized_search(mask, deadline, seed=42):
    """
    Randomized local search with restarts.
    """
    random.seed(seed)
    best_mask = mask
    best_size = mask.bit_count()
    
    for restart in range(10):
        if time.time() > deadline:
            break
        
        # Perturb current best
        current = best_mask
        # Remove a few random elements
        for _ in range(random.randint(1, 3)):
            set_vals = [v for v in range(1, 129) if current & (1 << (v - 1))]
            if not set_vals:
                break
            rem = random.choice(set_vals)
            current &= ~(1 << (rem - 1))
        
        # Add back greedily
        current = _local_search_add(current, deadline)
        
        if current.bit_count() > best_size:
            best_mask = current
            best_size = current.bit_count()
        
        # Try swaps
        if time.time() < deadline - 5:
            current2 = _local_search_swap(current, deadline)
            if current2.bit_count() > best_size:
                best_mask = current2
                best_size = current2.bit_count()
    
    return best_mask

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    # Start with seed: 65..128
    best_mask = 0
    for v in range(65, 129):
        best_mask |= (1 << (v - 1))
    best_size = best_mask.bit_count()
    
    # Step 1: Greedy add
    if time.time() < deadline - 40:
        greedy_mask = _greedy_construct(deadline - 40)
        if greedy_mask.bit_count() > best_size:
            best_mask = greedy_mask
            best_size = best_mask.bit_count()
    
    # Step 2: Local search add
    if time.time() < deadline - 30:
        ls_mask = _local_search_add(best_mask, deadline - 30)
        if ls_mask.bit_count() > best_size:
            best_mask = ls_mask
            best_size = best_mask.bit_count()
    
    # Step 3: Randomized search
    if time.time() < deadline - 10:
        rs_mask = _randomized_search(best_mask, deadline - 10)
        if rs_mask.bit_count() > best_size:
            best_mask = rs_mask
            best_size = rs_mask.bit_count()
    
    # Final validity check
    if not _is_valid_mask(best_mask):
        # Fallback to seed
        best_mask = 0
        for v in range(65, 129):
            best_mask |= (1 << (v - 1))
    
    # Convert to list
    result = [i + 1 for i in range(128) if (best_mask >> i) & 1]
    return result

# EVOLVE-BLOCK-END
