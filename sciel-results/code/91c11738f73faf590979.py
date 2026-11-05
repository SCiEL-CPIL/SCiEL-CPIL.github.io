# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def is_valid_fast(s):
    """
    Fast validity check using residue-based approach.
    For each element a in s, check if any 3 distinct OTHERS sum to a multiple of a.
    Key optimization: for large a (> 3*max_possible_sum/2), only a itself can be hit.
    """
    lst = sorted(s)
    n = len(lst)
    if n < 4:
        return True
    
    # Precompute all 3-sums' residues for each potential divisor
    # Strategy: for each a, we need to check if any triple of others sums to 0 mod a
    # For large a (a > max_sum3), only sum == a is possible (since all positive)
    # max_sum3 <= 192+191+190 = 573
    
    for idx_a, a in enumerate(lst):
        others = lst[:idx_a] + lst[idx_a+1:]
        m = len(others)
        if m < 3:
            continue
        
        # If a > max possible 3-sum, then no 3-sum can equal a (or any multiple)
        max_others = others[-1] + others[-2] + others[-3]
        if a > max_others:
            continue
        
        # Check if any triple of others sums to a multiple of a
        # Multiples to check: a, 2a, 3a, ... up to max_others
        # For each multiple M, check if any triple sums to M
        # Use a set-based approach: for each pair, check if M - pair_sum is in others
        
        # Build a set of others for O(1) lookup, but we need to ensure distinctness
        others_set = set(others)
        
        # Only check multiples that are in achievable range
        min_others = others[0] + others[1] + others[2]
        if min_others > max_others:
            continue
        
        for M in range(a, max_others + 1, a):
            if M < min_others:
                continue
            # Check if any triple of distinct others sums to M
            # For each i, j with i < j, check if M - others[i] - others[j] is in others and > others[j]
            found = False
            for i in range(m):
                for j in range(i + 1, m):
                    s2 = others[i] + others[j]
                    rem = M - s2
                    # rem must be in others, distinct from others[i] and others[j]
                    if rem in others_set and rem > others[j]:
                        # Ensure rem is not equal to others[i] or others[j]
                        if rem != others[i] and rem != others[j]:
                            found = True
                            break
                    elif rem in others_set and rem > others[j]:
                        found = True
                        break
                if found:
                    break
            if found:
                return False
    
    return True

def is_valid_pairwise_check(s):
    """
    Even faster check: only need to verify small elements as potential divisors.
    Large elements (> 573) can never divide a 3-sum since max 3-sum is 573.
    Actually max 3-sum of numbers <= 192 is at most 192+191+190=573.
    So any element > 573 is automatically safe as a divisor.
    But since all elements <= 192, all could potentially divide a 3-sum.
    However, for element a, the 3-sum must be >= 3*1=3 and <= 573.
    So a must be <= 573 to be a problem, which is always true here.
    But more importantly: if a > 3*max_other, then a can't divide any 3-sum 
    unless the 3-sum is 0 (impossible) or >= a.
    Actually the key insight: for a given a, the 3-sum S must satisfy S >= a 
    (since S is a positive multiple of a). And S <= 3*max(others).
    So if a > 3*max(others), a is safe.
    """
    lst = sorted(s)
    n = len(lst)
    if n < 4:
        return True
    
    for idx_a, a in enumerate(lst):
        others = lst[:idx_a] + lst[idx_a+1:]
        m = len(others)
        if m < 3:
            continue
        
        max_others = others[-1]
        # If a > 3 * max_others, no 3-sum can reach a
        if a > 3 * max_others:
            continue
        
        # Check if any triple of others sums to a multiple of a
        min_others = others[0] + others[1] + others[2]
        max_sum3 = others[-1] + others[-2] + others[-3]
        
        if min_others > max_sum3:
            continue
        
        # Check multiples of a in [min_others, max_sum3]
        for M in range(a, max_sum3 + 1, a):
            if M < min_others:
                continue
            # Check if any triple sums to M
            found = False
            for i in range(m):
                for j in range(i + 1, m):
                    s2 = others[i] + others[j]
                    rem = M - s2
                    if rem <= others[j]:
                        continue
                    if rem in set(others):
                        if rem != others[i] and rem != others[j]:
                            found = True
                            break
                if found:
                    break
            if found:
                return False
    
    return True

def greedy_construct(seed_rng=None):
    """Greedy from largest to smallest."""
    selected = []
    for x in range(192, 0, -1):
        candidate = selected + [x]
        if is_valid_fast(candidate):
            selected.append(x)
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s deadline to be safe
    rng = random.Random(42)
    
    # Start with greedy
    best = greedy_construct()
    best_set = frozenset(best)
    best_size = len(best)
    
    all_numbers = set(range(1, 193))
    
    def local_search(s, rng, max_iters=200):
        """Hill-climb: try adding elements, removing and replacing."""
        s = set(s)
        cur_size = len(s)
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            # Try adding a random element not in s
            candidates = list(all_numbers - s)
            rng.shuffle(candidates)
            for x in candidates[:12]:  # try up to 12 random additions
                new_s = s | {x}
                if is_valid_fast(new_s):
                    s = new_s
                    cur_size += 1
                    improved = True
                    break
            if improved:
                continue
            # Try swap: remove one, add another
            elems = list(s)
            rng.shuffle(elems)
            for rem in elems[:6]:
                candidates = list(all_numbers - s)
                rng.shuffle(candidates)
                for add in candidates[:12]:
                    new_s = (s - {rem}) | {add}
                    if is_valid_fast(new_s):
                        s = new_s
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return s
    
    # Run multiple local search rounds with diverse starting points
    for round_num in range(500):
        if time.time() > deadline:
            break
        
        # Generate diverse starting points
        if round_num == 0:
            start_s = set(best)
        elif round_num == 1:
            start_s = set(greedy_construct())
        elif round_num % 5 == 2:
            # Greedy but skip some small numbers to allow more large ones
            selected = []
            for x in range(192, 0, -1):
                if x < 20 and rng.random() < 0.5:
                    continue  # sometimes skip small numbers
                candidate = selected + [x]
                if is_valid_fast(candidate):
                    selected.append(x)
            start_s = set(selected)
        else:
            # Random start: take a random subset and filter to valid
            k = rng.randint(35, 48)
            start_s = set(rng.sample(range(1, 193), k))
            valid_s = set()
            for x in sorted(start_s, reverse=True):
                new_s = valid_s | {x}
                if is_valid_fast(new_s):
                    valid_s = new_s
            start_s = valid_s
        
        # Local search
        result = local_search(start_s, rng, max_iters=150)
        
        if len(result) > best_size:
            best_size = len(result)
            best_set = frozenset(result)
            best = list(result)
    
    return list(best_set)
# EVOLVE-BLOCK-END
