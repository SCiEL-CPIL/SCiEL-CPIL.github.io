import time
import random

def _creates_4ap(x, arr_set):
    """Fast check if adding x creates a 4-AP with elements in arr_set."""
    # x as 1st: need x+d, x+2d, x+3d
    d = 1
    while x + 3*d <= 512:
        if (x+d) in arr_set and (x+2*d) in arr_set and (x+3*d) in arr_set:
            return True
        d += 1
    # x as 2nd: need x-d, x+d, x+2d
    d = 1
    while x - d >= 1 and x + 2*d <= 512:
        if (x-d) in arr_set and (x+d) in arr_set and (x+2*d) in arr_set:
            return True
        d += 1
    # x as 3rd: need x-2d, x-d, x+d
    d = 1
    while x - 2*d >= 1 and x + d <= 512:
        if (x-2*d) in arr_set and (x-d) in arr_set and (x+d) in arr_set:
            return True
        d += 1
    # x as 4th: need x-3d, x-2d, x-d
    d = 1
    while x - 3*d >= 1:
        if (x-3*d) in arr_set and (x-2*d) in arr_set and (x-d) in arr_set:
            return True
        d += 1
    return False

def _verify_no_4ap(arr_set):
    """Verify no 4-AP exists."""
    for a in arr_set:
        d = 1
        while a + 3*d <= 512:
            if (a+d) in arr_set and (a+2*d) in arr_set and (a+3*d) in arr_set:
                return False
            d += 1
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    random.seed(12345)
    
    # Greedy initial construction
    best_set = set()
    for x in range(1, 513):
        if not _creates_4ap(x, best_set):
            best_set.add(x)
    best_size = len(best_set)
    
    # Iterated local search
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        # Perturb
        perturbed = set(best_set)
        if perturbed:
            num_remove = max(2, len(perturbed) // 8)
            to_remove = random.sample(list(perturbed), min(num_remove, len(perturbed)))
            for x in to_remove:
                perturbed.discard(x)
        
        # Local search
        cur_set = set(perturbed)
        cur_size = len(cur_set)
        stalled = 0
        
        while time.time() < deadline - 5 and stalled < 1000:
            improved = False
            
            # Try adds
            candidates = [x for x in range(1, 513) if x not in cur_set]
            if candidates:
                random.shuffle(candidates)
                for x in candidates[:15]:
                    if not _creates_4ap(x, cur_set):
                        cur_set.add(x)
                        improved = True
                        break
            
            # Try swaps
            if not improved and cur_set:
                rem = random.choice(list(cur_set))
                cur_set.discard(rem)
                candidates = [x for x in range(1, 513) if x not in cur_set]
                random.shuffle(candidates)
                found = False
                for x in candidates[:20]:
                    if not _creates_4ap(x, cur_set):
                        cur_set.add(x)
                        found = True
                        break
                if not found:
                    cur_set.add(rem)
            
            if improved:
                stalled = 0
                if len(cur_set) > best_size:
                    best_set = set(cur_set)
                    best_size = len(cur_set)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                stalled += 1
        
        no_improve_count += 1
    
    # Verify
    if not _verify_no_4ap(best_set):
        # Fallback
        best_set = set()
        for x in range(1, 513):
            if not _creates_4ap(x, best_set):
                best_set.add(x)
    
    return sorted(best_set)
