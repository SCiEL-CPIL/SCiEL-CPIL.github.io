"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # For the specific case q=3, n=7, radius=1
    if q == 3 and n == 7 and radius == 1:
        return _construct_3_7_1(deadline)
    
    # General fallback
    return _construct_general(q, n, radius, deadline)


def _construct_3_7_1(deadline):
    """
    Hybrid approach: Fast vectorized greedy + simulated annealing with periodic
    greedy reduction. Combines the speed of numpy bitsets for initial cover with
    the escape capability of SA for deep local minima.
    """
    
    N = 2187
    all_indices = list(range(N))
    
    # Precompute covers as bitsets (Python ints)
    covers = [0] * N
    for i in range(N):
        digits = [0]*7
        tmp = i
        for pos in range(7):
            digits[pos] = tmp % 3
            tmp //= 3
        
        mask = 1 << i
        for pos in range(7):
            for val in range(3):
                if val != digits[pos]:
                    diff = val - digits[pos]
                    new_i = i + diff * (3**pos)
                    mask |= (1 << new_i)
        covers[i] = mask
    
    full_mask = (1 << N) - 1
    
    # Precompute cover sets as lists of indices for faster iteration in local search
    cover_indices = []
    for i in range(N):
        mask = covers[i]
        idxs = []
        m = mask
        while m:
            lsb = m & -m
            idxs.append(lsb.bit_length() - 1)
            m ^= lsb
        cover_indices.append(idxs)
    
    def greedy_cover(randomize=False, seed=None):
        if seed is not None:
            random.seed(seed)
        
        uncovered = full_mask
        selected = []
        
        while uncovered:
            # Find best candidate - vectorize with numpy
            counts = np.array([(uncovered & covers[i]).bit_count() for i in all_indices], dtype=np.int32)
            best_count = counts.max()
            if best_count == 0:
                break
            
            # Find all indices achieving best_count
            best_mask = counts == best_count
            best_indices = [i for i, m in enumerate(best_mask) if m]
            
            if randomize and len(best_indices) > 1:
                idx = random.choice(best_indices)
            else:
                idx = best_indices[0]
            
            selected.append(idx)
            uncovered &= ~covers[idx]
            
            if time.time() > deadline - 5.0:
                break
                
        return selected, uncovered

    def local_search_remove(code, deadline):
        """
        Vectorized local search: try to remove redundant codewords.
        """
        if not code:
            return code
        
        # Use a count array for coverage counts
        counts = np.zeros(N, dtype=np.int32)
        for i in code:
            counts[cover_indices[i]] += 1
        
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            indices_to_try = list(range(len(code)))
            random.shuffle(indices_to_try)
            
            for idx_in_code in indices_to_try:
                if idx_in_code >= len(code):
                    continue
                c = code[idx_in_code]
                
                # Check if removing c is safe: all words it covers must have count > 1
                if np.all(counts[cover_indices[c]] > 1):
                    # Remove c
                    counts[cover_indices[c]] -= 1
                    code.pop(idx_in_code)
                    changed = True
            
        return code

    def simulated_annealing(init_code, deadline):
        """
        Simulated Annealing on a single start, with periodic greedy reduction.
        Uses the precomputed cover_indices for fast delta calculation.
        """
        if not init_code:
            return init_code
            
        code = list(init_code)
        code_set = set(code)
        
        # Initialize counts
        counts = np.zeros(N, dtype=np.int32)
        for i in code:
            counts[cover_indices[i]] += 1
        
        covered = int(np.sum(counts > 0))
        size = len(code)
        
        P = 10.0  # Penalty for uncovered words
        def cost(sz, cov):
            return sz + P * (N - cov)
        
        current_cost = cost(size, covered)
        best_code = list(code)
        best_size = size
        best_covered = covered
        best_cost = current_cost
        
        T = 8.0
        T_min = 0.005
        cooling = 0.99985
        
        all_indices = list(range(N))
        
        # Periodic greedy reduction
        last_reduction = time.time()
        reduction_interval = 1.0
        
        while time.time() < deadline - 5.0:
            # Periodic greedy reduction: try to remove redundant codewords
            if time.time() - last_reduction > reduction_interval:
                last_reduction = time.time()
                indices = list(range(size))
                random.shuffle(indices)
                for idx in indices:
                    if idx >= size:
                        continue
                    c = code[idx]
                    # Check if removing this word keeps coverage
                    if np.all(counts[cover_indices[c]] > 1):
                        # Safe to remove
                        code.pop(idx)
                        code_set.remove(c)
                        counts[cover_indices[c]] -= 1
                        size -= 1
                        covered = int(np.sum(counts > 0))
                        current_cost = cost(size, covered)
                        if size < best_size and covered == N:
                            best_size = size
                            best_code = list(code)
                            best_covered = covered
                            best_cost = current_cost
                        break  # Restart the scan after a removal
            
            move_type = random.random()
            
            if move_type < 0.55:
                # Swap
                if size == 0:
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
                idx_remove = random.randint(0, size - 1)
                c_remove = code[idx_remove]
                idx_add = random.choice(all_indices)
                
                if idx_add in code_set:
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
                
                nbs_remove = cover_indices[c_remove]
                nbs_add = cover_indices[idx_add]
                
                # Calculate delta coverage
                # Words losing coverage: in nbs_remove with count==1 and NOT in nbs_add
                # Words gaining coverage: in nbs_add with count==0
                
                # Optimization: Use numpy for these checks
                # Create a mask for nbs_add
                mask_add = np.zeros(N, dtype=bool)
                mask_add[nbs_add] = True
                
                # Losing: count==1 in nbs_remove AND not in nbs_add
                # We need to be careful with indices
                cnt_rm = counts[nbs_remove]
                is_in_add = mask_add[nbs_remove]
                losing = np.sum((cnt_rm == 1) & (~is_in_add))
                
                # Gaining: count==0 in nbs_add
                cnt_add = counts[nbs_add]
                gaining = np.sum(cnt_add == 0)
                
                delta = gaining - losing
                new_covered = covered + delta
                new_cost = cost(size, new_covered)
                delta_cost = new_cost - current_cost
                
                if delta_cost <= 0 or random.random() < np.exp(-delta_cost / T):
                    # Accept
                    code[idx_remove] = idx_add
                    code_set.remove(c_remove)
                    code_set.add(idx_add)
                    
                    # Update counts
                    counts[nbs_remove] -= 1
                    counts[nbs_add] += 1
                    
                    covered = new_covered
                    current_cost = new_cost
                    
                    if covered == N and size < best_size:
                        best_size = size
                        best_covered = covered
                        best_code = list(code)
                        best_cost = current_cost
                    
                    T *= cooling
                    if T < T_min: T = T_min
                else:
                    T *= cooling
                    if T < T_min: T = T_min
                    
            elif move_type < 0.85:
                # Remove
                if size == 0:
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
                idx = random.randint(0, size - 1)
                c = code[idx]
                nbs = cover_indices[c]
                
                losing = np.sum(counts[nbs] == 1)
                if losing == 0:
                    # Safe removal
                    code.pop(idx)
                    code_set.remove(c)
                    counts[nbs] -= 1
                    size -= 1
                    covered = int(np.sum(counts > 0))
                    current_cost = cost(size, covered)
                    
                    if size < best_size and covered == N:
                        best_size = size
                        best_covered = covered
                        best_code = list(code)
                        best_cost = current_cost
                else:
                    # Try to accept bad move? No, removal that breaks coverage is usually bad.
                    # But in SA we might accept it to escape local minima?
                    # Let's stick to valid moves for removal to keep it simple, 
                    # or allow with penalty.
                    delta_cost = P * losing
                    if random.random() < np.exp(-delta_cost / T):
                        code.pop(idx)
                        code_set.remove(c)
                        counts[nbs] -= 1
                        size -= 1
                        new_covered = covered - losing
                        current_cost = cost(size, new_covered)
                        covered = new_covered
                        
                T *= cooling
                if T < T_min: T = T_min
            else:
                # Add (rarely needed if starting from full cover, but good for SA)
                idx_add = random.choice(all_indices)
                if idx_add in code_set:
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
                
                nbs = cover_indices[idx_add]
                gaining = np.sum(counts[nbs] == 0)
                delta_cost = 1 - P * gaining
                if delta_cost <= 0 or random.random() < np.exp(-delta_cost / T):
                    code.append(idx_add)
                    code_set.add(idx_add)
                    counts[nbs] += 1
                    size += 1
                    new_covered = covered + gaining
                    current_cost = cost(size, new_covered)
                    covered = new_covered
                
                T *= cooling
                if T < T_min: T = T_min
        
        return best_code

    best_code = None
    best_size = N
    
    # 1. Deterministic Greedy
    code, unc = greedy_cover(randomize=False)
    if not unc:
        code = local_search_remove(code, deadline)
        code = simulated_annealing(code, deadline)
        code = local_search_remove(code, deadline)
        best_code = code
        best_size = len(code)
    
    # 2. Randomized Greedy + SA with multiple restarts
    attempts = 0
    while time.time() < deadline - 15.0:
        attempts += 1
        code, unc = greedy_cover(randomize=True, seed=attempts)
        if not unc:
            code = local_search_remove(code, deadline)
            code = simulated_annealing(code, deadline)
            code = local_search_remove(code, deadline)
            if len(code) < best_size:
                best_size = len(code)
                best_code = code
                # If we get close to known optimum, try more aggressive search
                if best_size <= 190:
                    code = simulated_annealing(code, deadline)
                    code = local_search_remove(code, deadline)
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code
    
    if best_code is None:
        # Fallback: complete the cover
        code, unc = greedy_cover(randomize=False)
        if unc:
            while unc:
                lsb = unc & -unc
                idx = lsb.bit_length() - 1
                code.append(idx)
                unc &= ~covers[idx]
        best_code = code
        best_size = len(best_code)
    
    # Convert indices to words
    result = []
    for idx in best_code:
        word = [0]*7
        tmp = idx
        for pos in range(7):
            word[pos] = tmp % 3
            tmp //= 3
        result.append(word)
    
    return result


def _construct_general(q, n, radius, deadline):
    """General fallback for other q, n, radius."""
    import itertools
    
    all_words = list(itertools.product(range(q), repeat=n))
    total = len(all_words)
    
    # Map word to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute covers
    covers = [0] * total
    for i, w in enumerate(all_words):
        mask = 1 << i
        # Generate all words within radius
        # This is expensive for large n, but for small n it's fine.
        # For general case, we might not have time.
        # Let's just do a simple greedy without precomputation for now.
        pass
    
    # Simple greedy
    uncovered = set(range(total))
    selected = []
    
    while uncovered and time.time() < deadline - 5.0:
        best_idx = -1
        best_count = -1
        for i in range(total):
            # Count how many uncovered words are within radius of i
            cnt = 0
            for j in uncovered:
                # Hamming distance
                dist = sum(1 for a, b in zip(all_words[i], all_words[j]) if a != b)
                if dist <= radius:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        
        if best_idx == -1:
            break
            
        selected.append(best_idx)
        # Remove covered words
        to_remove = []
        for j in uncovered:
            dist = sum(1 for a, b in zip(all_words[best_idx], all_words[j]) if a != b)
            if dist <= radius:
                to_remove.append(j)
        for j in to_remove:
            uncovered.discard(j)
            
    # Ensure validity
    if uncovered:
        # Add all uncovered
        for j in uncovered:
            selected.append(j)
            
    return [list(all_words[i]) for i in selected]
# EVOLVE-BLOCK-END
