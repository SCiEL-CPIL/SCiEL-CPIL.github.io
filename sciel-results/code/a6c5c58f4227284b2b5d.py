import random
import time
from itertools import combinations

def is_valid_fast(s):
    """
    Fast validity check: for each element a, check if any 3 distinct others sum to multiple of a.
    Uses pair iteration and set lookup for the third element.
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
        
        # Max sum of 3 distinct others
        max_sum3 = others[-1] + others[-2] + others[-3]
        # Min sum of 3 distinct others
        min_sum3 = others[0] + others[1] + others[2]
        
        # If a is larger than the max possible sum, no multiple can be formed
        if a > max_sum3:
            continue
            
        others_set = set(others)
        
        # Check all multiples of a in the achievable range [min_sum3, max_sum3]
        # The smallest multiple is a (since sums are positive and >= 3, but a could be small)
        # Actually, we just iterate k such that k*a is in [min_sum3, max_sum3]
        
        # Optimize: if a is large, there are very few multiples.
        # If a is small, there are more, but the pair check is fast.
        
        start_mult = (min_sum3 + a - 1) // a
        end_mult = max_sum3 // a
        
        if start_mult > end_mult:
            continue
            
        for k in range(start_mult, end_mult + 1):
            target = k * a
            # Check if any triple sums to target
            found = False
            # Iterate pairs (i, j) with i < j
            # We can prune if others[i] + others[j] + others[j+1] > target (assuming sorted)
            # But simple iteration is fine for m~48
            for i in range(m - 2):
                xi = others[i]
                # If xi + others[i+1] + others[i+2] > target, break
                if xi + others[i+1] + others[i+2] > target:
                    break
                for j in range(i + 1, m - 1):
                    xj = others[j]
                    # If xi + xj + others[j+1] > target, break
                    if xi + xj + others[j+1] > target:
                        break
                    rem = target - xi - xj
                    # rem must be > xj to be distinct and not yet seen
                    if rem > xj and rem in others_set:
                        found = True
                        break
                if found:
                    break
            if found:
                return False
    return True

def greedy_construct(seed_rng=None):
    """Greedy from largest to smallest."""
    rng = seed_rng or random
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
            for x in candidates[:10]:  # try up to 10 random additions
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
            for rem in elems[:5]:
                candidates = list(all_numbers - s)
                rng.shuffle(candidates)
                for add in candidates[:10]:
                    new_s = (s - {rem}) | {add}
                    if is_valid_fast(new_s):
                        s = new_s
                        improved = True
                        break
                if improved:
                    break
            
            # If no simple improvement, try a 2-swap (remove 2, add 2) occasionally
            if not improved and rng.random() < 0.2:
                elems = list(s)
                rng.shuffle(elems)
                rem1, rem2 = elems[0], elems[1]
                candidates = list(all_numbers - s)
                rng.shuffle(candidates)
                for add1 in candidates[:5]:
                    for add2 in candidates[:5]:
                        if add1 == add2: continue
                        new_s = (s - {rem1, rem2}) | {add1, add2}
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
    for round_num in range(300):
        if time.time() > deadline:
            break
        
        # Generate diverse starting points
        if round_num == 0:
            start_s = set(best)
        elif round_num == 1:
            start_s = set(greedy_construct())
        else:
            # Random start: take a random subset and filter to valid
            k = rng.randint(35, 45)
            start_s = set(rng.sample(range(1, 193), k))
            valid_s = set()
            # Filter to valid set by greedily adding
            for x in sorted(start_s, reverse=True):
                new_s = valid_s | {x}
                if is_valid_fast(new_s):
                    valid_s = new_s
            start_s = valid_s
        
        # Local search
        result = local_search(start_s, rng, max_iters=100)
        
        if len(result) > best_size:
            best_size = len(result)
            best_set = frozenset(result)
            best = list(result)
            
    return list(best_set)
