import time
import random
import sys
import numpy as np

def construct():
    N = 1000
    max_sum = 2 * N
    best_set = []
    best_len = 0
    
    def check_and_add(chosen, counts, x):
        """Check if x can be added. Return True if added."""
        if len(chosen) > 0:
            sums = x + np.array(chosen)
            if np.max(counts[sums]) + 1 > 2:
                return False
        if counts[2 * x] + 1 > 2:
            return False
        
        if len(chosen) > 0:
            counts[x + np.array(chosen)] += 1
        counts[2 * x] += 1
        chosen.append(x)
        return True

    def remove_from_set(chosen, counts, x):
        """Remove x from chosen set."""
        idx = chosen.index(x)
        others = [y for y in chosen if y != x]
        if others:
            others_arr = np.array(others)
            counts[x + others_arr] -= 1
        counts[2 * x] -= 1
        chosen.pop(idx)

    def can_add_fast(chosen, counts, x, chosen_set):
        """Fast check if x can be added."""
        if x in chosen_set:
            return False
        if counts[2 * x] + 1 > 2:
            return False
        if len(chosen) > 0:
            sums = x + np.array(chosen)
            if np.max(counts[sums]) + 1 > 2:
                return False
        return True

    def run_greedy(order='random', seed=None):
        if seed is not None:
            random.seed(seed)
        if order == 'random':
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
        elif order == 'ascending':
            candidates = list(range(1, N + 1))
        elif order == 'descending':
            candidates = list(range(N, 0, -1))
        elif order == 'sparse':
            candidates = list(range(1, N + 1, 2))
            random.shuffle(candidates)
            candidates += list(range(2, N + 1, 2))
            random.shuffle(candidates)
        else:
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
        
        chosen = []
        chosen_set = set()
        counts = np.zeros(max_sum + 1, dtype=np.int32)
        
        for x in candidates:
            if can_add_fast(chosen, counts, x, chosen_set):
                check_and_add(chosen, counts, x)
                chosen_set.add(x)
                
        return chosen, counts, chosen_set
    
    def local_search(chosen, counts, chosen_set, time_limit=10):
        """Try to improve by inserting or swapping elements."""
        start_time = time.time()
        improved = True
        max_iters = 500
        iters = 0
        all_candidates = list(range(1, N + 1))
        
        while improved and iters < max_iters and time.time() - start_time < time_limit:
            improved = False
            iters += 1
            
            # 1. Try pure insertions
            random.shuffle(all_candidates)
            for x in all_candidates[:50]:
                if can_add_fast(chosen, counts, x, chosen_set):
                    check_and_add(chosen, counts, x)
                    chosen_set.add(x)
                    improved = True
                    break
            
            if improved:
                continue
            
            # 2. Try swaps
            if not chosen:
                continue
                
            x_to_remove = random.choice(chosen)
            random.shuffle(all_candidates)
            found_swap = False
            
            for y in all_candidates:
                if y in chosen_set:
                    continue
                
                remove_from_set(chosen, counts, x_to_remove)
                chosen_set.discard(x_to_remove)
                
                if can_add_fast(chosen, counts, y, chosen_set):
                    check_and_add(chosen, counts, y)
                    chosen_set.add(y)
                    improved = True
                    found_swap = True
                    break
                else:
                    check_and_add(chosen, counts, x_to_remove)
                    chosen_set.add(x_to_remove)
                    
                if time.time() - start_time > time_limit:
                    break
            
            if not found_swap and not improved:
                if time.time() - start_time > time_limit:
                    break
                 
        return chosen, counts, chosen_set
    
    def structured_build():
        """Build using a structured approach."""
        chosen = []
        chosen_set = set()
        counts = np.zeros(max_sum + 1, dtype=np.int32)
        
        initial = random.sample(range(1, N + 1), 20)
        for x in initial:
            if can_add_fast(chosen, counts, x, chosen_set):
                check_and_add(chosen, counts, x)
                chosen_set.add(x)
        
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        for x in candidates:
            if can_add_fast(chosen, counts, x, chosen_set):
                check_and_add(chosen, counts, x)
                chosen_set.add(x)
        
        return chosen, counts, chosen_set
    
    start_time = time.time()
    timeout = 115
    
    current_set, current_counts, current_set_set = structured_build()
    current_set, current_counts, current_set_set = local_search(current_set, current_counts, current_set_set, time_limit=5)
    if len(current_set) > best_len:
        best_len = len(current_set)
        best_set = current_set[:]
    
    orders = ['random', 'sparse', 'ascending', 'descending']
    attempt = 0
    
    while time.time() - start_time < timeout:
        order = random.choice(orders)
        seed = attempt * 7919 + 12345
        current_set, current_counts, current_set_set = run_greedy(order, seed=seed)
        
        time_left = timeout - (time.time() - start_time)
        if time_left > 10:
            ls_time = min(5, time_left - 2)
        else:
            ls_time = max(1, time_left - 1)
        
        current_set, current_counts, current_set_set = local_search(current_set, current_counts, current_set_set, time_limit=ls_time)
        
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = current_set[:]
        
        attempt += 1
        
    return best_set
