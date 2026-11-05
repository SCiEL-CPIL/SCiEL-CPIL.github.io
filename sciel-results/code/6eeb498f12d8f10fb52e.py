import time
import random
import math

N = 5000
START_TIME = time.time()
TIME_LIMIT = 300.0  # 300s deadline, keep margin

def _primes(limit):
    sieve = [True] * (limit + 1)
    if limit >= 0:
        sieve[0] = False
    if limit >= 1:
        sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def _is_admissible(points, primes_list):
    """
    Check if the set of points is admissible.
    Admissible: for every prime p <= len(points), some residue class mod p is missed.
    """
    k = len(points)
    if k == 0:
        return True
    
    for p in primes_list:
        if p > k:
            break
        
        seen = [False] * p
        count = 0
        for v in points:
            r = v % p
            if not seen[r]:
                seen[r] = True
                count += 1
                if count == p:
                    return False
    return True

def construct():
    global START_TIME
    start_time = START_TIME
    if start_time == 0:
        start_time = time.time()
        START_TIME = start_time
    
    primes = _primes(N)
    
    best_set = set()
    best_size = 0
    
    num_restarts = 300
    
    for restart in range(num_restarts):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
            
        current_set = set(range(N))
        removed_elements = []
        
        while True:
            if time.time() - start_time > TIME_LIMIT - 2:
                break
                
            k = len(current_set)
            if k == 0:
                break
            
            violating_p = None
            for p in primes:
                if p > k:
                    break
                seen = [False] * p
                count = 0
                for v in current_set:
                    r = v % p
                    if not seen[r]:
                        seen[r] = True
                        count += 1
                        if count == p:
                            violating_p = p
                            break
                if violating_p:
                    break
            
            if violating_p is None:
                break
            
            classes = [[] for _ in range(violating_p)]
            for v in current_set:
                classes[v % violating_p].append(v)
            
            class_sizes = [(len(c), i) for i, c in enumerate(classes)]
            class_sizes.sort()
            
            if random.random() < 0.7:
                min_idx = class_sizes[0][1]
            else:
                top_k = min(3, len(class_sizes))
                min_idx = class_sizes[random.randint(0, top_k - 1)][1]
            
            to_remove = set(classes[min_idx])
            current_set -= to_remove
            removed_elements.extend(classes[min_idx])
        
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
        
        max_add_passes = 10
        for _ in range(max_add_passes):
            if time.time() - start_time > TIME_LIMIT - 2:
                break
            
            candidates = removed_elements[:]
            random.shuffle(candidates)
            
            added_this_pass = 0
            check_limit = 800
            for x in candidates[:check_limit]:
                if time.time() - start_time > TIME_LIMIT - 1:
                    break
                if x in current_set:
                    continue
                
                new_size = len(current_set) + 1
                is_adm = True
                for p in primes:
                    if p > new_size:
                        break
                    seen = [False] * p
                    count = 0
                    for v in current_set:
                        r = v % p
                        if not seen[r]:
                            seen[r] = True
                            count += 1
                            if count == p:
                                is_adm = False
                                break
                    if not is_adm:
                        break
                    rx = x % p
                    if not seen[rx]:
                        count += 1
                        if count == p:
                            is_adm = False
                            break
                if is_adm:
                    current_set.add(x)
                    added_this_pass += 1
            
            if added_this_pass == 0:
                break
        
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
            
    return list(best_set)

construct()
