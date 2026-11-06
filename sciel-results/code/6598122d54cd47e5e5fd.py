import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    base = [1, 2, 5, 11]
    base_set = set(base)
    
    # Pre-compute initial sums
    initial_sums = set()
    for i in range(len(base)):
        for j in range(i, len(base)):
            initial_sums.add(base[i] + base[j])

    all_numbers = list(range(1, 285))
    candidates_pool = [x for x in all_numbers if x not in base_set]

    def get_sums_and_energy(S):
        """Returns (sums_set, max_sum) for set S."""
        sums = set()
        max_s = 0
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i, len(S_list)):
                s = S_list[i] + S_list[j]
                sums.add(s)
                if s > max_s: max_s = s
        return sums, max_s

    def is_valid_add(x, S, sums):
        """Check if x can be added to S with existing sums."""
        # 2x
        if 2*x in sums: return False
        # x + y
        for y in S:
            if x + y in sums:
                return False
        return True

    def greedy_fill(S, sums, pool, deadline):
        """
        Greedily add elements from pool to S.
        Returns (new_S, new_sums).
        """
        S = set(S)
        sums = set(sums)
        # Filter pool to those currently valid
        valid_pool = [x for x in pool if is_valid_add(x, S, sums)]
        
        # Sort by value? Or random?
        # Sorting by value (ascending) is a good heuristic for Sidon sets (packing tight).
        # Or we can try a few strategies. Let's stick to ascending for determinism/speed in local search.
        valid_pool.sort()
        
        added = 0
        for x in valid_pool:
            if time.time() > deadline: break
            if is_valid_add(x, S, sums):
                S.add(x)
                sums.add(2*x)
                for y in S:
                    if y != x:
                        sums.add(x + y)
                added += 1
        
        return S, sums

    # 1. Initial Greedy Construction
    # Try a few different random orders to get a good starting point
    best_S = set(base)
    best_sums = initial_sums
    
    for restart in range(50):
        if time.time() > deadline: break
        random.shuffle(candidates_pool)
        S, sums = greedy_fill(base_set, initial_sums, candidates_pool, deadline)
        if len(S) > len(best_S):
            best_S = S
            best_sums = sums
            if len(best_S) >= 20:
                return sorted(best_S)

    # 2. Local Search: Remove and Refill
    # If we haven't reached 20, try to improve.
    if len(best_S) < 20:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing each non-base element
            for x in list(best_S):
                if x in base_set: continue
                
                S_temp = set(best_S)
                S_temp.remove(x)
                sums_temp = set(best_sums)
                
                # Recalculate sums for S_temp is expensive if we just remove x.
                # Better to just use greedy_fill which checks validity.
                # But greedy_fill needs initial sums.
                # Let's just rebuild sums for S_temp? No, S_temp is small.
                # Actually, greedy_fill takes sums. We need sums of S_temp.
                # Let's just call a helper to get sums of S_temp.
                sums_temp, _ = get_sums_and_energy(S_temp)
                
                S_new, sums_new = greedy_fill(S_temp, sums_temp, candidates_pool, deadline)
                
                if len(S_new) > len(best_S):
                    best_S = S_new
                    best_sums = sums_new
                    improved = True
                    if len(best_S) >= 20:
                        return sorted(best_S)
                    break # Restart the removal loop with new best

    return sorted(best_S)

if __name__ == "__main__":
    res = construct()
    print(res)
    print(len(res))
