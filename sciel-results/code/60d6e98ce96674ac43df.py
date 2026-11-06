"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Powers of 4 form a valid B_3[1] base set
    # In base 4, sum of 3 elements each with digit 0-3 gives max digit 9 = 21_4
    # so no carry propagates to next position
    base = [4**i for i in range(10) if 4**i <= 4096]
    # 1, 4, 16, 64, 256, 1024, 4096 -> 7 elements
    
    def compute_triple_sums(lst):
        """Compute all unordered triple sums (with repetition) for lst."""
        n = len(lst)
        sums = set()
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    sums.add(lst[i] + lst[j] + lst[k])
        return sums
    
    def can_add_with_sums(lst, candidate, existing_sums):
        """Check if adding candidate to lst keeps all triple sums unique.
        Returns (bool, new_sums_set) if valid."""
        n = len(lst)
        new_sums = set(existing_sums)
        
        # Triples with one candidate: candidate + lst[i] + lst[j] for i <= j
        for i in range(n):
            for j in range(i, n):
                s = candidate + lst[i] + lst[j]
                if s in new_sums:
                    return False, None
                new_sums.add(s)
        
        # Triples with two candidates: 2*candidate + lst[i]
        for i in range(n):
            s = 2 * candidate + lst[i]
            if s in new_sums:
                return False, None
            new_sums.add(s)
        
        # Triple with three candidates: 3*candidate
        s = 3 * candidate
        if s in new_sums:
            return False, None
        
        return True, new_sums
    
    def greedy_insert(start_set, candidates, deadline, max_size=80):
        """Greedy insertion of candidates into start_set."""
        current = list(start_set)
        existing_sums = compute_triple_sums(current)
        
        for cand in candidates:
            if time.time() > deadline:
                break
            if len(current) >= max_size:
                break
            ok, new_sums = can_add_with_sums(current, cand, existing_sums)
            if ok:
                current.append(cand)
                existing_sums = new_sums
        
        return current
    
    def local_search(current, deadline, max_size=80, max_swaps=500):
        """Try to improve by swapping elements out and inserting new ones."""
        current = list(current)
        existing_sums = compute_triple_sums(current)
        in_set = set(current)
        
        swaps_done = 0
        improved = True
        while improved and time.time() < deadline and swaps_done < max_swaps:
            improved = False
            swaps_done += 1
            
            # Try removing each element and see if we can add 2 new ones
            for idx in range(len(current)):
                if time.time() > deadline:
                    break
                if len(current) >= max_size:
                    break
                
                removed = current[idx]
                temp = current[:idx] + current[idx+1:]
                temp_sums = compute_triple_sums(temp)
                
                # Try to add candidates not in set
                added_count = 0
                new_elements = []
                for cand in range(1, 4097):
                    if cand in in_set or cand == removed:
                        continue
                    ok, ns = can_add_with_sums(temp, cand, temp_sums)
                    if ok:
                        new_elements.append(cand)
                        temp_sums = ns
                        added_count += 1
                        if added_count >= 2:
                            break
                
                if added_count > 0:
                    # Accept the swap: remove one, add the new ones
                    current = temp + new_elements
                    in_set = set(current)
                    existing_sums = compute_triple_sums(current)
                    improved = True
                    break
        
        return current
    
    # Strategy 1: Greedy insertion in sorted order
    best_set = list(base)
    
    candidates_sorted = [x for x in range(1, 4097) if x not in set(base)]
    result1 = greedy_insert(base, candidates_sorted, deadline)
    if len(result1) > len(best_set):
        best_set = result1
    
    # Strategy 2: Greedy insertion in random order (multiple attempts)
    random.seed(42)
    for attempt in range(30):
        if time.time() > deadline:
            break
        
        candidates_random = [x for x in range(1, 4097) if x not in set(base)]
        random.shuffle(candidates_random)
        result = greedy_insert(base, candidates_random, deadline)
        if len(result) > len(best_set):
            best_set = result
    
    # Strategy 3: Local search on best found
    if time.time() < deadline - 10:
        improved = local_search(best_set, deadline)
        if len(improved) > len(best_set):
            best_set = improved
    
    # Strategy 4: Try different base sets (shifted powers of 4)
    if time.time() < deadline - 15:
        for shift in range(1, 4):
            if time.time() > deadline:
                break
            alt_base = [4**i + shift for i in range(10) if 4**i + shift <= 4096]
            if len(alt_base) >= 5:
                candidates_alt = [x for x in range(1, 4097) if x not in set(alt_base)]
                result_alt = greedy_insert(alt_base, candidates_alt, deadline)
                if len(result_alt) > len(best_set):
                    best_set = result_alt
    
    # Final: try one more round of local search
    if time.time() < deadline - 5 and len(best_set) < 80:
        improved = local_search(best_set, deadline)
        if len(improved) > len(best_set):
            best_set = improved
    
    return best_set[:80]
# EVOLVE-BLOCK-END
