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
    
    # Precompute sets for faster membership testing
    ball_sets = [set(b) for b in ball_indices]
    
    # Coset structure: partition by sum(coords) mod 3
    # Each coset has exactly 3 words; any codeword covers all 3 in its coset
    coset_of = np.array([i % 3 for i in range(total)], dtype=np.int32)
    coset_members = [[] for _ in range(total // 3)]
    for i in range(total):
        coset_members[i // 3].append(i)
    coset_members_np = [np.array(m, dtype=np.int32) for m in coset_members]
    
    # For each coset, precompute which codeword positions cover it
    # (a codeword c covers coset k iff c is in coset k, since ball(c) includes all 3 words of coset(c))
    
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
    
    for seed in [1, 2, 3, 4, 5]:
        if time.time() > deadline - 40.0:
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
        """Try a swap. Returns True if successful."""
        if c_add in code_set:
            return False
        if c_remove == c_add:
            return False
        # Coset constraint: c_add must be in same coset as c_remove
        if coset_of[c_remove] != coset_of[c_add]:
            return False
            
        ball_rem = ball_indices_np[c_remove]
        ball_add = ball_indices_np[c_add]
        
        # Vectorized validity check: for all x in ball_rem \ ball_add, cnt[x] >= 2
        set_add = ball_sets[c_add]
        mask = np.array([x not in set_add for x in ball_rem], dtype=bool)
        if np.any(cnt_arr[ball_rem[mask]] < 2):
            return False
            
        # Perform swap
        code_set.remove(c_remove)
        code_set.add(c_add)
        cnt_arr[ball_rem] -= 1
        cnt_arr[ball_add] += 1
        return True
    
    def get_valid_swaps(c, code_set, cnt_arr):
        """Return list of valid c_add for removing c, using coset constraint."""
        # Coset constraint: c_add must be in same coset as c
        coset_id = c // 3
        coset_words = coset_members_np[coset_id]
        
        # Filter to words not in code_set
        candidates = [v for v in coset_words if v not in code_set]
        if len(candidates) == 0:
            return []
        
        # Vectorized validity check for all candidates
        ball_rem = ball_indices_np[c]
        # For each candidate, check: for all x in ball_rem \ ball(cand), cnt[x] >= 2
        valid_list = []
        for v in candidates:
            set_v = ball_sets[v]
            mask = np.array([x not in set_v for x in ball_rem], dtype=bool)
            if not np.any(cnt_arr[ball_rem[mask]] < 2):
                valid_list.append(v)
        return valid_list
    
    # Local Search Loop
    no_improve_counter = 0
    max_no_improve = 15
    
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
        max_swaps = 500
        while swaps_attempted < max_swaps and time.time() < deadline - 3.0:
            swaps_attempted += 1
            if len(current_code) == 0:
                break
                
            # Pick random codeword to remove
            c = random.choice(list(current_code))
            
            # Get valid swaps
            valid_swaps = get_valid_swaps(c, current_code, cnt)
            
            if len(valid_swaps) == 0:
                no_improve_counter += 1
                continue
            else:
                # Pick from valid swaps (already coset-constrained)
                c_prime = random.choice(valid_swaps)
            
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
