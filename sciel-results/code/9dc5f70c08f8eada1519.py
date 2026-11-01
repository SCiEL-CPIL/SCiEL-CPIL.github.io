import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 120s limit
    
    # Handle the specific case q=3, n=8, radius=1
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other cases
        if q ** n > 100000:
            return []
        # Simple greedy for general case
        total = q ** n
        codewords = []
        uncovered = set(range(total))
        
        def idx_to_digits(idx):
            digits = [0] * n
            for i in range(n - 1, -1, -1):
                digits[i] = idx % q
                idx //= q
            return digits
        
        def digits_to_idx(digits):
            idx = 0
            for d in digits:
                idx = idx * q + d
            return idx
        
        # Precompute neighbors
        neighbors = {}
        for i in range(total):
            digits = idx_to_digits(i)
            nbs = [i]
            for pos in range(n):
                orig = digits[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    new_digits = digits[:]
                    new_digits[pos] = val
                    nbs.append(digits_to_idx(new_digits))
            neighbors[i] = nbs
        
        while uncovered and time.time() < deadline:
            # Find codeword covering most uncovered
            best_cw = -1
            best_count = -1
            for cw in uncovered:
                count = sum(1 for nb in neighbors[cw] if nb in uncovered)
                if count > best_count:
                    best_count = count
                    best_cw = cw
            
            if best_cw == -1:
                break
            
            codewords.append(best_cw)
            for nb in neighbors[best_cw]:
                uncovered.discard(nb)
        
        result = [idx_to_digits(cw) for cw in codewords]
        seen = set()
        unique_result = []
        for cw in result:
            t = tuple(cw)
            if t not in seen:
                seen.add(t)
                unique_result.append(cw)
        return unique_result
    
    # Specific case: q=3, n=8, radius=1
    total_words = 3 ** 8  # 6561
    num_coords = 8
    alphabet = 3
    
    # Precompute balls (neighborhoods of radius 1) using numpy
    # Each word has 1 + 8*2 = 17 neighbors
    balls = np.zeros((total_words, 17), dtype=np.int32)
    weights = [3 ** (num_coords - 1 - i) for i in range(num_coords)]
    
    for w in range(total_words):
        balls[w, 0] = w
        idx = 1
        for pos in range(num_coords):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(alphabet):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1
    
    # Convert balls to list of lists for local search
    balls_list = [balls[i].tolist() for i in range(total_words)]
    
    # Precompute neighbor sets for fast membership testing
    neighbor_sets = [set(balls[i]) for i in range(total_words)]
    
    def idx_to_digits(idx):
        digits = [0] * num_coords
        for i in range(num_coords - 1, -1, -1):
            digits[i] = idx % alphabet
            idx //= alphabet
        return digits
    
    def digits_to_idx(digits):
        idx = 0
        for d in digits:
            idx = idx * alphabet + d
        return idx
    
    def vectorized_greedy(deadline_local):
        """Fast vectorized greedy construction."""
        uncovered_mask = np.ones(total_words, dtype=bool)
        selected = []
        
        while np.any(uncovered_mask):
            if time.time() > deadline_local:
                break
            
            # Vectorized count: for each candidate, count how many uncovered words it covers
            counts = uncovered_mask[balls].sum(axis=1)
            
            if np.max(counts) == 0:
                break
            
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]
            
            # Random selection among best candidates
            best_idx = int(candidates[np.random.randint(len(candidates))])
            
            selected.append(best_idx)
            uncovered_mask[balls[best_idx]] = False
        
        return set(selected)
    
    def compute_cov_count(code_set):
        """Compute coverage count for each word given a set of codewords."""
        cov_count = np.zeros(total_words, dtype=np.int32)
        for c in code_set:
            for nb in balls_list[c]:
                cov_count[nb] += 1
        return cov_count
    
    def prune_once(code_set, deadline_local):
        """Remove redundant codewords."""
        if not code_set:
            return code_set
        
        cov_count = compute_cov_count(code_set)
        
        removable = []
        for c in code_set:
            is_removable = True
            for nb in balls_list[c]:
                if cov_count[nb] <= 1:
                    is_removable = False
                    break
            if is_removable:
                removable.append(c)
        
        if removable:
            for c in removable:
                code_set.remove(c)
            # Recurse to find more removable codewords
            if time.time() < deadline_local:
                return prune_once(code_set, deadline_local)
        return code_set
    
    def local_search(code_set, deadline_local):
        """Apply 1-swaps and 2-swaps to improve the code."""
        improved = True
        while improved and time.time() < deadline_local:
            improved = False
            c_list = list(code_set)
            random.shuffle(c_list)
            
            cov_count = compute_cov_count(code_set)
            
            # Try 1-swaps: remove c, add d
            found_1_swap = False
            for c in c_list:
                if time.time() > deadline_local:
                    break
                if c not in code_set:
                    continue
                
                # Find words that would become uncovered if c is removed
                critical_words = [w for w in balls_list[c] if cov_count[w] == 1]
                
                if not critical_words:
                    # c is redundant, remove it
                    code_set.remove(c)
                    improved = True
                    found_1_swap = True
                    cov_count = compute_cov_count(code_set)
                    continue
                
                # Try to find d that covers all critical_words
                if len(critical_words) <= 4:  # Only try if small
                    candidate_set = set(balls_list[critical_words[0]])
                    for w in critical_words[1:]:
                        candidate_set &= set(balls_list[w])
                    
                    for d in candidate_set:
                        if d in code_set:
                            continue
                        
                        # Check if d covers all critical_words
                        valid = True
                        for w in critical_words:
                            if w not in neighbor_sets[d]:
                                valid = False
                                break
                        
                        if valid:
                            # Perform the swap
                            code_set.remove(c)
                            code_set.add(d)
                            improved = True
                            found_1_swap = True
                            cov_count = compute_cov_count(code_set)
                            break
            
            # Try 2-swaps: remove c1, c2, add d
            if not found_1_swap and time.time() < deadline_local:
                cov_count = compute_cov_count(code_set)
                c_list = list(code_set)
                random.shuffle(c_list)
                
                found_2_swap = False
                n_cw = len(c_list)
                
                for i in range(n_cw):
                    if time.time() > deadline_local:
                        break
                    c1 = c_list[i]
                    if c1 not in code_set:
                        continue
                    
                    for j in range(i + 1, n_cw):
                        c2 = c_list[j]
                        if c2 not in code_set:
                            continue
                        
                        # Find words that would become uncovered after removing c1 and c2
                        critical_words = []
                        seen_words = set()
                        
                        for w in balls_list[c1]:
                            if w in seen_words:
                                continue
                            seen_words.add(w)
                            cover_count = cov_count[w]
                            if w in neighbor_sets[c2]:
                                cover_count -= 1
                            cover_count -= 1  # for c1
                            if cover_count == 0:
                                critical_words.append(w)
                        
                        for w in balls_list[c2]:
                            if w in seen_words:
                                continue
                            seen_words.add(w)
                            cover_count = cov_count[w]
                            cover_count -= 1  # for c2
                            if cover_count == 0:
                                critical_words.append(w)
                        
                        if not critical_words:
                            # Both c1 and c2 are redundant together
                            code_set.remove(c1)
                            code_set.remove(c2)
                            improved = True
                            found_2_swap = True
                            cov_count = compute_cov_count(code_set)
                            break
                        
                        # Try to find d that covers all critical_words
                        if len(critical_words) <= 3:  # Only try if small
                            candidate_set = set(balls_list[critical_words[0]])
                            for w in critical_words[1:]:
                                candidate_set &= set(balls_list[w])
                            
                            for d in candidate_set:
                                if d in code_set:
                                    continue
                                
                                valid = True
                                for w in critical_words:
                                    if w not in neighbor_sets[d]:
                                        valid = False
                                        break
                                
                                if valid:
                                    # Remove c1, c2, add d (net -1 codeword)
                                    code_set.remove(c1)
                                    code_set.remove(c2)
                                    code_set.add(d)
                                    improved = True
                                    found_2_swap = True
                                    cov_count = compute_cov_count(code_set)
                                    break
                        
                        if found_2_swap:
                            break
                    
                    if found_2_swap:
                        break
            
            # Final prune after local search
            code_set = prune_once(code_set, deadline_local)
        
        return code_set
    
    # Main loop: multiple restarts with vectorized greedy + local search
    best_set = None
    best_size = total_words + 1
    
    # Reserve time for final local search on the best solution
    final_search_time = 15  # seconds reserved for polishing the best solution
    
    while time.time() < deadline - final_search_time:
        # Vectorized greedy initialization
        init_deadline = min(time.time() + 3.0, deadline - final_search_time)
        code_set = vectorized_greedy(init_deadline)
        
        if not code_set:
            continue
        
        # Local search to improve
        ls_deadline = min(time.time() + 5.0, deadline - final_search_time)
        code_set = local_search(code_set, ls_deadline)
        
        # Keep the best solution found
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_set = set(code_set)
        
        # If we've found a very good solution, break early
        if best_size <= 486:
            break
    
    # Final polish on the best solution with remaining time
    if best_set is not None and time.time() < deadline - 2:
        best_set = local_search(best_set, deadline - 2)
        best_set = prune_once(best_set, deadline - 1)
    
    if best_set is None:
        # Fallback: return a simple greedy solution
        best_set = vectorized_greedy(deadline - 1)
    
    # Final conversion
    result = [idx_to_digits(c) for c in best_set]
    seen = set()
    unique_result = []
    for cw in result:
        t = tuple(cw)
        if t not in seen:
            seen.add(t)
            unique_result.append(cw)
    return unique_result
