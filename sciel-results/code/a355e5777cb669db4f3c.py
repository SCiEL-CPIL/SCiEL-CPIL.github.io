import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 113.0  # Safety margin for 120s limit
    
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
    
    # Precompute sets for fast membership testing
    ball_sets = [set(b) for b in ball_indices]
    
    # Precompute ball membership matrix for vectorized operations
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
    
    # Multiple greedy restarts to find good starting points
    best_initial = None
    best_initial_size = float('inf')
    
    init_code = greedy_build(seed=0)
    best_initial = set(init_code)
    best_initial_size = len(init_code)
    
    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        if time.time() > deadline - 45.0:
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
    
    def try_swap(c_remove, c_add, code_set, cnt_arr):
        """Try a swap. Returns True if successful and swap was performed."""
        if c_add in code_set:
            return False
        if c_remove == c_add:
            return False
            
        ball_rem = ball_indices_np[c_remove]
        ball_add = ball_indices_np[c_add]
        
        # Check validity: for all x in ball_rem \ ball_add, cnt[x] >= 2
        set_add = ball_sets[c_add]
        valid = True
        for x in ball_rem:
            if x not in set_add:
                if cnt_arr[x] < 2:
                    valid = False
                    break
                    
        if not valid:
            return False
            
        # Perform swap
        code_set.remove(c_remove)
        code_set.add(c_add)
        cnt_arr[ball_rem] -= 1
        cnt_arr[ball_add] += 1
        return True
    
    def get_swap_candidates(c_remove, code_set, cnt_arr):
        """Get list of valid c_add candidates for removing c_remove."""
        ball_rem = ball_indices_np[c_remove]
        
        # Find words in ball_rem that are uniquely covered (cnt == 1)
        unique_words = [x for x in ball_rem if cnt_arr[x] == 1]
        
        if len(unique_words) == 0:
            # No unique words; any non-code word works
            return None
        
        # Find intersection of balls of all unique words
        # These are the only valid c_add positions
        valid_set = set(ball_indices_np[unique_words[0]].tolist())
        for u in unique_words[1:]:
            valid_set &= set(ball_indices_np[u].tolist())
            if len(valid_set) == 0:
                return []
        
        # Filter out words already in code_set
        valid_list = [v for v in valid_set if v not in code_set]
        return valid_list
    
    # Local Search Loop
    no_improve_counter = 0
    max_no_improve = 12
    rng = random.Random(42)
    
    while time.time() < deadline - 3.0:
        # 1. Delete Phase
        removed = run_delete_phase(current_code, cnt)
        if removed > 0:
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = list(current_code)
                no_improve_counter = 0
            else:
                no_improve_counter += 1
        else:
            no_improve_counter += 1
        
        # 2. Swap Phase with smarter candidate selection
        swaps_attempted = 0
        max_swaps = 300
        while swaps_attempted < max_swaps and time.time() < deadline - 3.0:
            swaps_attempted += 1
            if len(current_code) == 0:
                break
                
            # Pick random codeword to remove
            c = rng.choice(list(current_code))
            
            # Get valid swaps
            valid_swaps = get_swap_candidates(c, current_code, cnt)
            
            if valid_swaps is None:
                # Any non-code word works; pick one that maximizes new coverage
                # of words with cnt==1 in ball(c)
                # Sample a few candidates and pick the best
                c_prime = None
                best_new_cov = -1
                for _ in range(10):
                    cand = rng.randint(0, total - 1)
                    if cand in current_code:
                        continue
                    # Count how many unique words in ball(c) are covered by ball(cand)
                    ball_c = ball_indices_np[c]
                    unique_in_c = ball_c[cnt[ball_c] == 1]
                    if len(unique_in_c) == 0:
                        c_prime = cand
                        break
                    # Check how many unique words are in ball(cand)
                    set_cand = ball_sets[cand]
                    new_cov = sum(1 for u in unique_in_c if u in set_cand)
                    if new_cov > best_new_cov:
                        best_new_cov = new_cov
                        c_prime = cand
                    if new_cov == len(unique_in_c):
                        break
                if c_prime is None:
                    c_prime = rng.randint(0, total - 1)
                    while c_prime in current_code:
                        c_prime = rng.randint(0, total - 1)
            elif len(valid_swaps) == 0:
                no_improve_counter += 1
                continue
            else:
                # Pick from valid swaps, preferring those that cover many unique words
                c_prime = rng.choice(valid_swaps)
            
            # Perform swap
            if try_swap(c, c_prime, current_code, cnt):
                # After swap, run delete phase
                removed = run_delete_phase(current_code, cnt)
                if removed > 0:
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = list(current_code)
                        no_improve_counter = 0
                    else:
                        no_improve_counter += 1
                else:
                    no_improve_counter += 1
            else:
                no_improve_counter += 1
        
        # 3. 2-for-1 move: try removing 2 codewords and adding 1
        if time.time() < deadline - 5.0 and len(current_code) >= 2:
            # Sample a few pairs to try
            for _ in range(20):
                if time.time() > deadline - 5.0:
                    break
                pair = rng.sample(list(current_code), 2)
                c1, c2 = pair
                
                # Check if removing both is valid
                ball1 = ball_indices_np[c1]
                ball2 = ball_indices_np[c2]
                
                # Words uniquely covered by c1 or c2 (after removing both)
                # A word x is uniquely covered if cnt[x] == 1 and x is in ball1 or ball2
                # After removing both, cnt'[x] = cnt[x] - (1 if x in ball1) - (1 if x in ball2)
                # We need cnt'[x] >= 0 for all x, and cnt'[x] >= 1 for all x not in ball1∪ball2
                
                # Find words that would become uncovered
                would_uncover = []
                for x in ball1:
                    if cnt[x] == 1 and x not in ball_sets[c2]:
                        would_uncover.append(x)
                for x in ball2:
                    if cnt[x] == 1 and x not in ball_sets[c1]:
                        would_uncover.append(x)
                
                if len(would_uncover) == 0:
                    # Valid to remove both
                    current_code.remove(c1)
                    current_code.remove(c2)
                    cnt[ball1] -= 1
                    cnt[ball2] -= 1
                    removed = run_delete_phase(current_code, cnt)
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = list(current_code)
                        no_improve_counter = 0
                    else:
                        no_improve_counter += 1
                elif len(would_uncover) <= 3:
                    # Try to find a single codeword that covers all would_uncover words
                    valid_set = set(range(total))
                    for u in would_uncover:
                        valid_set &= ball_sets[u]
                        if len(valid_set) == 0:
                            break
                    valid_list = [v for v in valid_set if v not in current_code]
                    if valid_list:
                        c_new = rng.choice(valid_list)
                        current_code.remove(c1)
                        current_code.remove(c2)
                        current_code.add(c_new)
                        cnt[ball1] -= 1
                        cnt[ball2] -= 1
                        cnt[ball_indices_np[c_new]] += 1
                        removed = run_delete_phase(current_code, cnt)
                        if len(current_code) < best_size:
                            best_size = len(current_code)
                            best_code = list(current_code)
                            no_improve_counter = 0
                        else:
                            no_improve_counter += 1
        
        # If no improvement for many iterations, try a restart
        if no_improve_counter > max_no_improve and time.time() < deadline - 15.0:
            # Try a new greedy initialization
            seed = int(time.time() * 1000) % 10000
            init_code = greedy_build(seed=seed)
            if len(init_code) < len(current_code) + 5:
                current_code = set(init_code)
                cnt = np.zeros(total, dtype=np.int32)
                for c in current_code:
                    cnt[ball_indices_np[c]] += 1
                no_improve_counter = 0
                
        if best_size <= 160:
            break

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
