import random
import itertools
import time

def construct():
    start_time = time.time()
    TIME_LIMIT = 55.0

    def is_valid(S):
        n = len(S)
        sums = set()
        for r in range(1, n + 1):
            for combo in itertools.combinations(S, r):
                s = sum(combo)
                if s in sums:
                    return False
                sums.add(s)
        
        sum_list = sorted(sums)
        max_sum = sum_list[-1]
        sum_set = sums
        
        for a in sum_list:
            multiple = 2 * a
            while multiple <= max_sum:
                if multiple in sum_set:
                    return False
                multiple += a
        return True

    best_set = None
    best_size = 0

    for k in range(3, 8):
        low = 2 ** k
        high = 2 ** k + 2 ** (k-1)
        high = min(high, 257)
        if low >= 257:
            continue
            
        candidates = list(range(low, high))
        
        # Greedy construction
        # Try different orderings? Random shuffle + greedy.
        for _ in range(20): # 20 random orderings
            if time.time() - start_time > TIME_LIMIT:
                break
            shuffled = candidates.copy()
            random.shuffle(shuffled)
            current_set = []
            for c in shuffled:
                new_set = current_set + [c]
                if is_valid(new_set):
                    current_set = new_set
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set

    if best_set:
        return list(best_set)
    
    return [31, 32, 34]
