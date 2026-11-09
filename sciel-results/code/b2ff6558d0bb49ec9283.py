import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    best_set = [31, 32, 34]
    best_size = 3
    
    def get_subset_sums(S):
        sums = set()
        n = len(S)
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += S[i]
            sums.add(s)
        return sums

    def is_valid(S):
        if not S:
            return True
        sums = get_subset_sums(S)
        if len(sums) != (1 << len(S)) - 1:
            return False
        sum_list = sorted(sums)
        max_sum = sum_list[-1]
        for i in range(len(sum_list)):
            s_i = sum_list[i]
            if s_i * 2 > max_sum:
                break
            mult = 2 * s_i
            while mult <= max_sum:
                if mult in sums:
                    return False
                mult += s_i
        return True

    # Focus on larger numbers to minimize divisibility risks
    # and keep sums distinct.
    # Range 100-256 is a good sweet spot.
    high_candidates = list(range(100, 257))
    all_candidates = list(range(1, 257))
    
    # 1. Try to build sets from high_candidates
    for _ in range(50000):
        if time.time() > deadline:
            break
        
        # Randomly pick a starting size between 3 and 6
        start_size = random.randint(3, 6)
        if start_size > len(high_candidates):
            start_size = len(high_candidates)
            
        # Pick random distinct numbers
        current_set = random.sample(high_candidates, start_size)
        
        if not is_valid(current_set):
            continue
            
        # Try to expand
        # Sort remaining candidates randomly
        remaining = [x for x in high_candidates if x not in current_set]
        random.shuffle(remaining)
        
        for c in remaining:
            new_set = current_set + [c]
            if is_valid(new_set):
                current_set = new_set
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set.copy()
                    
    # 2. If we found a good set, try to optimize it further with swaps?
    # Or just run a few more general random searches on all_candidates
    # but biased towards the end of the list.
    
    for _ in range(20000):
        if time.time() > deadline:
            break
        
        # Biased random selection
        # Pick numbers weighted towards 256
        current_set = []
        # Try to add 4-8 numbers
        target_size = random.randint(4, 8)
        
        # Greedy add from a shuffled, biased list
        biased_list = all_candidates.copy()
        random.shuffle(biased_list)
        # Move larger numbers to front?
        # Just shuffle is fine, but we can filter out small numbers if we want
        # Let's try to avoid 1..50 as they cause many divisibility issues
        filtered = [x for x in biased_list if x > 50]
        
        for c in filtered:
            if len(current_set) >= target_size:
                break
            new_set = current_set + [c]
            if is_valid(new_set):
                current_set = new_set
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()

    return best_set
