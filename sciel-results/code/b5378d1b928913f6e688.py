import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 112.0  # Safety margin for 120s limit
    
    total = q ** n
    ball_size = 1 + n * (q - 1)
    
    # Precompute place values for index conversion
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q
        
    # Precompute all words as indices
    words = np.zeros((total, n), dtype=np.int32)
    for i in range(total):
        temp = i
        for j in range(n - 1, -1, -1):
            words[i, j] = temp % q
            temp //= q
            
    # Precompute ball_indices: for each word, list of indices within Hamming distance 1
    ball_indices = [[] for _ in range(total)]
    for i in range(total):
        w = words[i]
        ball_indices[i].append(i)
        for coord in range(n):
            orig = w[coord]
            for v in range(q):
                if v == orig:
                    continue
                new_idx = i + (v - orig) * base[coord]
                ball_indices[i].append(new_idx)
                
    # Convert to numpy arrays for faster vectorized operations
    ball_indices_np = [np.array(b, dtype=np.int32) for b in ball_indices]
    
    # Precompute ball membership matrix: ball_mem[i, j] = True if j is in ball(i)
    ball_mem = np.zeros((total, total), dtype=bool)
    for i in range(total):
        ball_mem[i, ball_indices_np[i]] = True

    # Greedy construction with randomization
    def greedy_build(seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
            
        covered = np.zeros(total, dtype=bool)
        code = []
        
        while not np.all(covered):
            best_c = -1
            max_new = -1
            candidates = list(range(total))
            rng.shuffle(candidates)
            
            for c in candidates:
                ball = ball_indices_np[c]
                new_cov = np.sum(~covered[ball])
                if new_cov > max_new:
                    max_new = new_cov
                    best_c = c
                    if max_new == ball_size:
                        break
            code.append(best_c)
            covered[ball_indices_np[best_c]] = True
        return code

    def run_delete_phase(code_set, cnt_arr):
        """Removes redundant codewords. Returns number removed."""
        removed = 0
        changed = True
        while changed:
            changed = False
            for c in list(code_set):
                ball = ball_indices_np[c]
                if np.min(cnt_arr[ball]) >= 2:
                    code_set.remove(c)
                    cnt_arr[ball] -= 1
                    removed += 1
                    changed = True
        return removed

    def get_valid_swaps(c, code_set, cnt_arr):
        """Return list of valid c_add for removing c that preserve coverage."""
        ball = ball_indices_np[c]
        # Find words in ball(c) with count == 1 (unique to c)
        unique_mask = cnt_arr[ball] == 1
        unique_words = ball[unique_mask]
        
        if len(unique_words) == 0:
            # No unique words: any word not in code_set works
            return None  # Signal: any non-code word works
        
        # For each unique word u, we need c_add such that u is in ball(c_add)
        # i.e., ball_mem[c_add, u] == True
        # Find all c_add that cover ALL unique words
        # Start with all candidates, intersect constraints
        valid_mask = np.ones(total, dtype=bool)
        for u in unique_words:
            # c_add must be such that ball_mem[c_add, u] is True
            # ball_mem[:, u] gives all c that cover u
            valid_mask &= ball_mem[:, u]
        
        # Exclude words already in code_set
        for c_code in code_set:
            valid_mask[c_code] = False
        
        valid_list = np.where(valid_mask)[0]
        if len(valid_list) == 0:
            return []
        return valid_list

    def local_search(code_set, cnt_arr, time_limit):
        """
        Perform local search (delete + swap) starting from code_set.
        Returns the best code list found.
        """
        best_code = list(code_set)
        best_size = len(code_set)
        no_improve_counter = 0
        max_no_improve = 20

        while time.time() < time_limit:
            # 1. Delete Phase
            removed = run_delete_phase(code_set, cnt_arr)
            if removed > 0:
                if len(code_set) < best_size:
                    best_size = len(code_set)
                    best_code = list(code_set)
                    no_improve_counter = 0
                else:
                    no_improve_counter += 1
            else:
                no_improve_counter += 1
            
            # 2. Swap Phase with smarter candidate selection
            swaps_attempted = 0
            max_swaps = 300
            while swaps_attempted < max_swaps and time.time() < time_limit:
                swaps_attempted += 1
                if len(code_set) == 0:
                    break
                    
                # Pick random codeword to remove
                c = random.choice(list(code_set))
                
                # Get valid swaps
                valid_swaps = get_valid_swaps(c, code_set, cnt_arr)
                
                if valid_swaps is None:
                    # Any non-code word works
                    c_prime = random.randint(0, total - 1)
                    while c_prime in code_set:
                        c_prime = random.randint(0, total - 1)
                elif len(valid_swaps) == 0:
                    no_improve_counter += 1
                    continue
                else:
                    # Pick from valid swaps
                    c_prime = int(random.choice(valid_swaps))
                
                # Perform swap
                code_set.remove(c)
                code_set.add(c_prime)
                cnt_arr[ball_indices_np[c]] -= 1
                cnt_arr[ball_indices_np[c_prime]] += 1
                
                # After swap, run delete phase
                removed = run_delete_phase(code_set, cnt_arr)
                if removed > 0:
                    if len(code_set) < best_size:
                        best_size = len(code_set)
                        best_code = list(code_set)
                        no_improve_counter = 0
                    else:
                        no_improve_counter += 1
                else:
                    no_improve_counter += 1
                    
            if no_improve_counter > max_no_improve:
                break
                
        return best_code, best_size

    # --- Main search: multiple greedy starts + local search ---
    
    # Try several greedy initializations
    best_initial = None
    best_initial_size = float('inf')
    
    for seed in range(8):
        if time.time() > deadline - 30.0:
            break
        init_code = greedy_build(seed=seed)
        if len(init_code) < best_initial_size:
            best_initial_size = len(init_code)
            best_initial = set(init_code)
    
    current_code = set(best_initial)
    best_code = list(current_code)
    best_size = len(current_code)
    
    # Initialize counts
    cnt = np.zeros(total, dtype=np.int32)
    for c in current_code:
        cnt[ball_indices_np[c]] += 1
        
    # Local search on the best initial
    if time.time() < deadline - 5.0:
        best_code, best_size = local_search(current_code, cnt, deadline - 5.0)
    
    # Try a few more restarts with local search if time permits
    for restart in range(5):
        if time.time() > deadline - 10.0:
            break
        seed = restart + 100
        init_code = greedy_build(seed=seed)
        if len(init_code) > best_size + 5:
            continue  # Skip if too far from best
        
        current_code = set(init_code)
        cnt = np.zeros(total, dtype=np.int32)
        for c in current_code:
            cnt[ball_indices_np[c]] += 1
        
        # Quick local search
        time_left = deadline - 10.0 - time.time()
        if time_left < 2.0:
            break
        search_time = min(15.0, time_left * 0.6)
        lc_code, lc_size = local_search(current_code, cnt, time.time() + search_time)
        
        if lc_size < best_size:
            best_size = lc_size
            best_code = lc_code
            # Rebuild cnt for continued search
            current_code = set(lc_code)
            cnt = np.zeros(total, dtype=np.int32)
            for c in current_code:
                cnt[ball_indices_np[c]] += 1
    
    # Final result conversion
    result = []
    for idx in best_code:
        w = []
        temp = idx
        for i in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result.append(w)
        
    return result
