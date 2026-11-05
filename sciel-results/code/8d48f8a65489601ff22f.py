import time
import random
import math
import numpy as np

N = 5000
START_TIME = time.time()
TIME_LIMIT = 320.0  # 320s wall-clock deadline, keep margin for safety

def _primes(limit):
    """Sieve of Eratosthenes to find primes up to limit."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = 0
    sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = 0
    return [p for p in range(2, limit + 1) if sieve[p]]

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
        
        seen = bytearray(p)
        count = 0
        for v in points:
            r = v % p
            if not seen[r]:
                seen[r] = 1
                count += 1
                if count == p:
                    return False
    return True

def _find_violating_prime(points, primes_list):
    """Find a prime p <= len(points) where all residues are covered, or None if admissible."""
    k = len(points)
    if k == 0:
        return None
    
    for p in primes_list:
        if p > k:
            break
        
        seen = bytearray(p)
        count = 0
        for v in points:
            r = v % p
            if not seen[r]:
                seen[r] = 1
                count += 1
                if count == p:
                    return p
    return None

def construct():
    global START_TIME
    start_time = START_TIME
    
    # Precompute primes up to N (5000)
    primes = _primes(N)
    prime_set = set(primes)
    
    best_set = set()
    best_size = 0
    
    num_restarts = 200
    
    for restart in range(num_restarts):
        if time.time() - start_time > TIME_LIMIT - 15:
            break
        
        # Start with all numbers
        current_set_set = set(range(N))
        
        # Greedy Repair Loop with efficient residue tracking
        max_iterations = 5000
        for _ in range(max_iterations):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
                
            k = len(current_set_set)
            if k == 0:
                break
            
            # Find a violating prime using efficient checking
            violating_p = None
            for p in primes:
                if p > k:
                    break
                seen = bytearray(p)
                count = 0
                for v in current_set_set:
                    r = v % p
                    if not seen[r]:
                        seen[r] = 1
                        count += 1
                        if count == p:
                            violating_p = p
                            break
                if violating_p:
                    break
            
            if violating_p is None:
                break
            
            # Group elements by residue class mod violating_p
            classes = [[] for _ in range(violating_p)]
            for v in current_set_set:
                classes[v % violating_p].append(v)
            
            # Choose a class to remove. Try different strategies:
            # 1. Smallest class (greedy)
            # 2. Random small class
            # 3. Class that helps most other constraints (heuristic)
            
            class_sizes = [(len(c), i) for i, c in enumerate(classes)]
            class_sizes.sort()
            
            strategy = random.random()
            if strategy < 0.4:
                min_idx = class_sizes[0][1]
            elif strategy < 0.7:
                top_k = min(5, len(class_sizes))
                min_idx = class_sizes[random.randint(0, top_k - 1)][1]
            elif strategy < 0.9:
                # Pick from bottom 10%
                bottom_frac = max(1, len(class_sizes) // 10)
                min_idx = class_sizes[random.randint(0, bottom_frac - 1)][1]
            else:
                # Random choice
                min_idx = random.randint(0, len(class_sizes) - 1)
            
            to_remove = set(classes[min_idx])
            current_set_set -= to_remove
        
        # Now current_set is admissible (or empty)
        if len(current_set_set) > best_size:
            best_set = set(current_set_set)
            best_size = len(current_set_set)
        
        # Add-back phase with efficient checking
        max_add_passes = 15
        for pass_num in range(max_add_passes):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
                
            k = len(current_set_set)
            if k == 0:
                break
            
            # Precompute residue counts for current set
            residue_counts = {}
            for p in primes:
                if p > k + 1:  # We'll check up to new_size = k+1
                    break
                cnt = [0] * p
                for v in current_set_set:
                    cnt[v % p] += 1
                residue_counts[p] = cnt
            
            # Precompute zero counts for each prime
            zero_counts = {}
            for p, cnt in residue_counts.items():
                zero_counts[p] = sum(1 for c in cnt if c == 0)
            
            candidates = [x for x in range(N) if x not in current_set_set]
            random.shuffle(candidates)
            
            added_any = False
            check_limit = 3000
            for x in candidates[:check_limit]:
                if time.time() - start_time > TIME_LIMIT - 3:
                    break
                
                new_size = k + 1
                can_add = True
                
                for p in primes:
                    if p > new_size:
                        break
                    if p not in residue_counts:
                        continue
                    
                    cnt = residue_counts[p]
                    zeros = zero_counts[p]
                    r = x % p
                    
                    if zeros == 0:
                        # Shouldn't happen if current_set is admissible
                        can_add = False
                        break
                    elif zeros == 1:
                        # Find the missing residue
                        for i in range(p):
                            if cnt[i] == 0:
                                if r == i:
                                    can_add = False
                                break
                        if not can_add:
                            break
                    # else: zeros >= 2, any x is fine for this prime
                
                if can_add:
                    current_set_set.add(x)
                    k += 1
                    added_any = True
                    # Update residue counts and zero counts
                    for p in primes:
                        if p > k:
                            break
                        if p in residue_counts:
                            old_cnt = residue_counts[p][x % p]
                            residue_counts[p][x % p] += 1
                            if old_cnt == 0:
                                zero_counts[p] -= 1
                        else:
                            # Need to add this prime
                            cnt = [0] * p
                            for v in current_set_set:
                                cnt[v % p] += 1
                            residue_counts[p] = cnt
                            zero_counts[p] = sum(1 for c in cnt if c == 0)
            
            if not added_any:
                break
        
        if len(current_set_set) > best_size:
            best_set = set(current_set_set)
            best_size = len(current_set_set)
        
        # Local search: try to swap elements to improve
        if time.time() - start_time < TIME_LIMIT - 10:
            for _ in range(8):
                if time.time() - start_time > TIME_LIMIT - 5:
                    break
                
                k = len(current_set_set)
                if k == 0:
                    break
                
                # Try removing one element and adding two others
                removed = None
                added_count = 0
                
                # Find a good element to remove
                candidates_remove = list(current_set_set)
                random.shuffle(candidates_remove)
                
                for rem in candidates_remove[:20]:
                    if time.time() - start_time > TIME_LIMIT - 3:
                        break
                    
                    test_set = set(current_set_set)
                    test_set.remove(rem)
                    
                    if not _is_admissible(test_set, primes):
                        continue
                    
                    # Try to add back
                    residue_counts = {}
                    for p in primes:
                        if p > len(test_set) + 2:
                            break
                        cnt = [0] * p
                        for v in test_set:
                            cnt[v % p] += 1
                        residue_counts[p] = cnt
                    
                    zero_counts = {}
                    for p, cnt in residue_counts.items():
                        zero_counts[p] = sum(1 for c in cnt if c == 0)
                    
                    candidates_add = [x for x in range(N) if x not in test_set and x != rem]
                    random.shuffle(candidates_add)
                    
                    added = []
                    for x in candidates_add[:200]:
                        if len(added) >= 2:
                            break
                        
                        new_size = len(test_set) + len(added) + 1
                        can_add = True
                        
                        for p in primes:
                            if p > new_size:
                                break
                            if p not in residue_counts:
                                continue
                            
                            cnt = residue_counts[p]
                            zeros = zero_counts[p]
                            r = x % p
                            
                            if zeros == 0:
                                can_add = False
                                break
                            elif zeros == 1:
                                for i in range(p):
                                    if cnt[i] == 0:
                                        if r == i:
                                            can_add = False
                                        break
                                if not can_add:
                                    break
                            # else: zeros >= 2, any x is fine for this prime
                        
                        if can_add:
                            test_set.add(x)
                            added.append(x)
                            # Update counts
                            for p in primes:
                                if p > len(test_set):
                                    break
                                if p in residue_counts:
                                    old_cnt = residue_counts[p][x % p]
                                    residue_counts[p][x % p] += 1
                                    if old_cnt == 0:
                                        zero_counts[p] -= 1
                                else:
                                    cnt = [0] * p
                                    for v in test_set:
                                        cnt[v % p] += 1
                                    residue_counts[p] = cnt
                                    zero_counts[p] = sum(1 for c in cnt if c == 0)
                    
                    if len(added) >= 2:
                        current_set_set = test_set
                        removed = rem
                        added_count = len(added)
                        break
                
                if removed is None:
                    break
        
        if len(current_set_set) > best_size:
            best_set = set(current_set_set)
            best_size = len(current_set_set)
    
    # Final polish pass: try to add back elements in a smarter order
    if best_set and time.time() - start_time < TIME_LIMIT - 2:
        k = len(best_set)
        residue_counts = {}
        for p in primes:
            if p > k + 1:
                break
            cnt = [0] * p
            for v in best_set:
                cnt[v % p] += 1
            residue_counts[p] = cnt
        
        zero_counts = {}
        for p, cnt in residue_counts.items():
            zero_counts[p] = sum(1 for c in cnt if c == 0)
        
        candidates = [x for x in range(N) if x not in best_set]
        
        # Sort candidates by how "safe" they are (fewer constraints they might violate)
        def safety_score(x):
            score = 0
            for p in primes:
                if p > k + 1:
                    break
                if p not in residue_counts:
                    continue
                cnt = residue_counts[p]
                zeros = zero_counts[p]
                r = x % p
                if zeros == 1 and cnt[r] == 0:
                    score += 1  # This would violate
            return score
        
        candidates.sort(key=safety_score)
        
        for x in candidates[:500]:
            if time.time() - start_time > TIME_LIMIT - 1:
                break
            
            new_size = k + 1
            can_add = True
            
            for p in primes:
                if p > new_size:
                    break
                if p not in residue_counts:
                    continue
                
                cnt = residue_counts[p]
                zeros = zero_counts[p]
                r = x % p
                
                if zeros == 0:
                    can_add = False
                    break
                elif zeros == 1:
                    for i in range(p):
                        if cnt[i] == 0:
                            if r == i:
                                can_add = False
                            break
                    if not can_add:
                        break
            
            if can_add:
                best_set.add(x)
                k += 1
                for p in primes:
                    if p > k:
                        break
                    if p in residue_counts:
                        old_cnt = residue_counts[p][x % p]
                        residue_counts[p][x % p] += 1
                        if old_cnt == 0:
                            zero_counts[p] -= 1
                    else:
                        cnt = [0] * p
                        for v in best_set:
                            cnt[v % p] += 1
                        residue_counts[p] = cnt
                        zero_counts[p] = sum(1 for c in cnt if c == 0)
    
    return list(best_set)

# Initial call to set start time and warm up
construct()
