import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Use more of the 150s budget
    
    prescribed = [1, 2, 5, 11]
    prescribed_set = frozenset(prescribed)
    
    def is_valid_sidon(s):
        sums = set()
        sl = sorted(s)
        for i in range(len(sl)):
            for j in range(i, len(sl)):
                sm = sl[i] + sl[j]
                if sm in sums:
                    return False
                sums.add(sm)
        return True
    
    def get_sums(s):
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                sums.add(s[i] + s[j])
        return sums
    
    def can_add(x, s, sums):
        for y in s:
            if x + y in sums:
                return False
        if 2 * x in sums:
            return False
        return True
    
    def get_new_sums(x, s):
        return {x + y for y in s} | {2 * x}
    
    best_set = list(prescribed)
    best_size = len(best_set)
    
    def try_greedy(order, time_limit=2.0):
        nonlocal best_set, best_size
        if time.time() > deadline - 5:
            return
        s = list(prescribed)
        sums = get_sums(s)
        start = time.time()
        
        for x in order:
            if time.time() - start > time_limit:
                break
            if x in s:
                continue
            if can_add(x, s, sums):
                s.append(x)
                sums = sums | get_new_sums(x, s)
        
        if len(s) > best_size:
            best_set = list(s)
            best_size = len(s)
    
    remaining = list(range(12, 285))
    rng = random.Random(12345)
    
    # Ascending greedy
    try_greedy(remaining, time_limit=1.0)
    # Descending greedy  
    try_greedy(list(reversed(remaining)), time_limit=1.0)
    
    # Several random orderings
    for i in range(30):
        if time.time() > deadline - 10:
            break
        order = remaining[:]
        rng.shuffle(order)
        try_greedy(order, time_limit=0.2)
    
    # Strategy: Systematic local search
    def local_search_systematic(start_s, time_limit=60.0):
        nonlocal best_set, best_size
        if time.time() > deadline - 5:
            return
        s = list(start_s)
        s_set = set(s)
        sums = get_sums(s)
        start = time.time()
        
        all_vals = list(range(1, 285))
        no_improve_count = 0
        
        while time.time() - start < time_limit and time.time() < deadline - 5:
            improved = False
            
            # Phase 1: Try to add any element (systematic, in order)
            for x in all_vals:
                if time.time() > deadline - 5:
                    return
                if x in s_set:
                    continue
                if can_add(x, s, sums):
                    s.append(x)
                    s_set.add(x)
                    sums = sums | get_new_sums(x, s)
                    if len(s) > best_size:
                        best_set = list(s)
                        best_size = len(s)
                    improved = True
                    no_improve_count = 0
                    break
            
            if improved:
                continue
            
            # Phase 2: Try 1-swap moves (remove one non-prescribed, add one new)
            removable = [x for x in s if x not in prescribed_set]
            for x_remove in removable:
                if time.time() > deadline - 5:
                    return
                # Remove x_remove
                s.remove(x_remove)
                s_set.discard(x_remove)
                sums = get_sums(s)
                
                found = False
                for x_add in all_vals:
                    if time.time() > deadline - 5:
                        return
                    if x_add in s_set:
                        continue
                    if can_add(x_add, s, sums):
                        s.append(x_add)
                        s_set.add(x_add)
                        sums = sums | get_new_sums(x_add, s)
                        if len(s) > best_size:
                            best_set = list(s)
                            best_size = len(s)
                        found = True
                        break
                
                if found:
                    improved = True
                    no_improve_count = 0
                    break
                else:
                    # Restore
                    s.append(x_remove)
                    s_set.add(x_remove)
                    sums = get_sums(s)
            
            if improved:
                continue
            
            # Phase 3: Randomized escape - remove a random non-prescribed, try random additions
            no_improve_count += 1
            removable = [x for x in s if x not in prescribed_set]
            if removable:
                x_remove = rng.choice(removable)
                s.remove(x_remove)
                s_set.discard(x_remove)
                sums = get_sums(s)
                
                candidates = [x for x in all_vals if x not in s_set]
                rng.shuffle(candidates)
                for x_add in candidates[:100]:
                    if time.time() > deadline - 5:
                        return
                    if can_add(x_add, s, sums):
                        s.append(x_add)
                        s_set.add(x_add)
                        sums = sums | get_new_sums(x_add, s)
                        if len(s) > best_size:
                            best_set = list(s)
                            best_size = len(s)
                        break
    
    # Run systematic local search from best solution
    if time.time() < deadline - 30:
        local_search_systematic(best_set, time_limit=60.0)
    
    # Additional random restarts with local search
    for trial in range(50):
        if time.time() > deadline - 20:
            break
        s = list(prescribed)
        sums = get_sums(s)
        candidates = remaining[:]
        rng.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline - 20:
                break
            if can_add(x, s, sums):
                s.append(x)
                sums = sums | get_new_sums(x, s)
        if len(s) > best_size:
            best_set = list(s)
            best_size = len(s)
        
        # Quick local search on this solution
        if time.time() < deadline - 10 and len(s) >= best_size - 1:
            local_search_systematic(s, time_limit=5.0)
    
    # Final polish
    if time.time() < deadline - 5 and len(best_set) > 4:
        local_search_systematic(best_set, time_limit=10.0)
    
    # Verify and return
    result = sorted(best_set)
    if not is_valid_sidon(result):
        return list(prescribed)
    
    return result

if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    print(f"Result: {result}")
    sums = set()
    valid = True
    for i, x in enumerate(result):
        for j in range(i, len(result)):
            s = x + result[j]
            if s in sums:
                valid = False
                break
            sums.add(s)
        if not valid:
            break
    print(f"Valid: {valid}")
