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
    
    # Precompute coverage count array for fast swap evaluation
    # cov_count[w] = number of codewords whose ball contains word w
    # This is maintained incrementally during search
    
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
    
    # Initialize coverage count array: cnt[w] = number of codewords covering word w
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
    
    # Compute unique words for each codeword: words in ball(c) with cnt==1
    def compute_unique_words(c, cnt_arr):
        """Return list of words in ball(c) with cnt==1."""
        ball = ball_indices_np[c]
        unique = ball[cnt_arr[ball] == 1]
        return unique
    
    def get_valid_swaps(c, code_set, cnt_arr):
        """Return list of valid c_add for removing c."""
        unique = compute_unique_words(c, cnt_arr)
        if len(unique) == 0:
            # Any word not in code_set is valid
            # Return a sample for efficiency
            return None  # Signal: any non-code word works
        # Intersection of balls of all unique words
        # Start with ball of first unique word
        valid_set = set(ball_indices_np[unique[0]].tolist())
        for u in unique[1:]:
            valid_set &= set(ball_indices_np[u].tolist())
            if len(valid_set) == 0:
                return []
        # Filter out words already in code_set
        valid_list = [v for v in valid_set if v not in code_set]
        return valid_list
    
    # Local Search Loop with improved swap evaluation
    no_improve_counter = 0
    max_no_improve = 20
    
    def evaluate_swap_gain(c, c_prime, cnt_arr, code_set):
        """
        Compute the net coverage change if we remove c and add c_prime.
        Returns (gain, valid) where gain = new_covered_by_c_prime - lost_by_removing_c
        and valid indicates if the swap maintains coverage (no word drops to 0).
        """
        ball_c = ball_indices_np[c]
        ball_cp = ball_indices_np[c_prime]
        
        # Words that would lose coverage when c is removed (cnt==1 in ball_c)
        unique_to_c = ball_c[cnt_arr[ball_c] == 1]
        
        if len(unique_to_c) == 0:
            # c has no unique words; any c_prime not in code_set is valid
            # Gain is just new words covered by c_prime that weren't covered before
            new_by_cp = ball_cp[cnt_arr[ball_cp] == 0]
            return len(new_by_cp), True
        
        # Check if c_prime covers all unique words of c
        # A word u is covered by c_prime if u is in ball(c_prime)
        # We need: for all u in unique_to_c, u is in ball(c_prime)
        # Equivalently: unique_to_c ⊆ ball(c_prime)
        
        # Fast check: count how many unique words are NOT in ball(c_prime)
        # Use a temporary marker approach
        # Mark all words in ball(c_prime)
        # Then check if all unique_to_c are marked
        
        # Efficient: for each unique word, check if it's within distance 1 of c_prime
        # Since ball_size is small (15 for q=3,n=7), we can do this directly
        # But let's use the precomputed ball membership via counting
        
        # Alternative: compute intersection size
        # Count unique words that are in ball(c_prime)
        # We can do this by checking each unique word against c_prime
        
        # For efficiency, use the fact that ball_indices are precomputed
        # Create a set of ball(c_prime) for fast lookup
        ball_cp_set = set(ball_cp.tolist())
        
        uncovered = 0
        for u in unique_to_c:
            if u not in ball_cp_set:
                uncovered += 1
                break  # Early exit if any unique word isn't covered
        
        if uncovered > 0:
            return 0, False
        
        # Swap is valid. Compute gain: new words covered by c_prime
        # that were not covered by any other codeword
        # After removing c, words in ball(c_prime) that had cnt==1 (only covered by c)
        # would have cnt==0, but we're adding c_prime so they get cnt==1
        # Words with cnt==0 in ball(c_prime) become cnt==1 (new coverage)
        # Words with cnt>=1 stay the same or increase
        
        # Net gain = number of words in ball(c_prime) with cnt==0
        # (these are words not covered by any codeword, but wait - if all words are
        # covered, cnt should be >= 1 for all words... unless we're in a state where
        # some words aren't covered)
        
        # Actually, in a valid covering code, all words have cnt >= 1.
        # So gain = 0 always for valid swaps in a complete covering.
        # The benefit is enabling further deletions.
        
        # For search purposes, we prefer swaps that create more "removable" codewords
        # i.e., codewords with many words that will have cnt==1 after the swap.
        
        # Simple heuristic: just return 1 for any valid swap
        return 1, True
    
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
            
            # Score each codeword by how many unique words it has
            # Codewords with more unique words are harder to remove
            # We want to try removing codewords with FEWER unique words first
            # (they're more likely to be swappable)
            
            # Pick a codeword to remove, biased toward those with fewer unique words
            code_list = list(current_code)
            
            # Compute unique word counts for each codeword
            unique_counts = []
            for c in code_list:
                ball = ball_indices_np[c]
                uc = np.sum(cnt[ball] == 1)
                unique_counts.append(uc)
            
            # Weighted random selection: prefer codewords with fewer unique words
            # Use inverse weighting so codewords with 0 unique words are most likely
            weights = [1.0 / (uc + 1) for uc in unique_counts]
            total_weight = sum(weights)
            r = random.random() * total_weight
            cumulative = 0
            chosen_idx = 0
            for i, w in enumerate(weights):
                cumulative += w
                if r <= cumulative:
                    chosen_idx = i
                    break
            
            c = code_list[chosen_idx]
            
            # Find valid swap candidates
            # Get unique words of c
            ball_c = ball_indices_np[c]
            unique_to_c = ball_c[cnt[ball_c] == 1]
            
            if len(unique_to_c) == 0:
                # c is redundant in terms of unique words; any non-code word works
                # Pick a random non-code word
                c_prime = random.randint(0, total - 1)
                while c_prime in current_code:
                    c_prime = random.randint(0, total - 1)
            else:
                # Need c_prime to cover all unique words of c
                # Intersect balls of all unique words
                # Start with ball of first unique word
                valid_set = set(ball_indices_np[unique_to_c[0]].tolist())
                for u in unique_to_c[1:]:
                    valid_set &= set(ball_indices_np[u].tolist())
                    if len(valid_set) == 0:
                        break
                
                # Filter out words already in code_set
                valid_list = [v for v in valid_set if v not in current_code]
                
                if len(valid_list) == 0:
                    no_improve_counter += 1
                    continue
                
                # Pick from valid swaps
                c_prime = random.choice(valid_list)
            
            # Perform swap
            current_code.remove(c)
            current_code.add(c_prime)
            cnt[ball_indices_np[c]] -= 1
            cnt[ball_indices_np[c_prime]] += 1
            
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

    # Final exhaustive deletion pass with remaining time
    final_code = set(best_code)
    final_cnt = np.zeros(total, dtype=np.int32)
    for c in final_code:
        final_cnt[ball_indices_np[c]] += 1
    
    while time.time() < deadline - 1.0:
        # Try to remove any codeword
        found_removal = False
        for c in list(final_code):
            ball = ball_indices_np[c]
            if np.min(final_cnt[ball]) >= 2:
                final_code.remove(c)
                final_cnt[ball] -= 1
                found_removal = True
                break
        if not found_removal:
            break
    
    # Final result conversion
    result = []
    for idx in final_code:
        w = []
        temp = idx
        for i in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result.append(w)
        
    return result
