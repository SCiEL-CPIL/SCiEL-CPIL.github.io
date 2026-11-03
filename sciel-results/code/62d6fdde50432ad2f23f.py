import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    """
    Construct a ternary covering code K_3(6,1).
    Returns a list of codewords (lists of integers).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Fallback for non-standard parameters
    if q != 3 or n != 6 or radius != 1:
        if n <= 4:
            return [list(t) for t in itertools.product(range(q), repeat=n)]
        else:
            return [list(t) for t in itertools.product(range(q), repeat=n)]
    
    total_words = 3 ** n  # 729
    
    # --- 1. Precompute Data Structures ---
    
    # Helper: Convert index to word tuple
    def idx_to_word(idx):
        w = [0] * n
        temp = idx
        for i in range(n):
            w[n - 1 - i] = temp % 3
            temp //= 3
        return tuple(w)
    
    # Helper: Convert word tuple to index
    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * 3 + val
        return idx
    
    # Precompute spheres (Hamming balls of radius 1) as sets for fast intersection
    spheres = [None] * total_words
    sphere_lists = [None] * total_words  # Also keep list form
    for idx in range(total_words):
        w = list(idx_to_word(idx))
        sphere_set = {idx}
        sphere_list = [idx]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(3):
                if new_val == orig_val:
                    continue
                w[pos] = new_val
                new_idx = word_to_idx(w)
                if new_idx not in sphere_set:
                    sphere_set.add(new_idx)
                    sphere_list.append(new_idx)
                w[pos] = orig_val
        spheres[idx] = sphere_set
        sphere_lists[idx] = sphere_list
    
    # Precompute for each word w, which codewords cover it (static)
    coverers = [[] for _ in range(total_words)]
    for c in range(total_words):
        for w in spheres[c]:
            coverers[w].append(c)
    
    # --- 2. Initial Code Construction ---
    # Tetracode seed
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            initial_codewords.append(h + list(tail))
    
    initial_indices = [word_to_idx(cw) for cw in initial_codewords]
    
    # --- 3. Simulated Annealing ---
    
    def compute_counts(code_set):
        counts = np.zeros(total_words, dtype=np.int32)
        for idx in code_set:
            for w in sphere_lists[idx]:
                counts[w] += 1
        return counts
    
    def is_valid(code_set, counts):
        return np.all(counts >= 1)
    
    def get_uncovered(code_set, counts):
        return np.where(counts == 0)[0].tolist()
    
    def can_remove(c, code_set, counts):
        """Check if removing c keeps code valid."""
        for w in sphere_lists[c]:
            if counts[w] == 1:
                return False
        return True
    
    def find_replacements(c_remove, code_set, counts):
        """Find all c_add such that swapping c_remove with c_add keeps validity."""
        # Words that would become uncovered
        at_risk = [w for w in sphere_lists[c_remove] if counts[w] == 1]
        if not at_risk:
            # c_remove is redundant, any c_add works (but we don't want to add)
            return []
        
        # c_add must cover all at_risk words
        candidates = []
        # Start with coverers of first at_risk word
        first_risk = at_risk[0]
        for c_add in coverers[first_risk]:
            if c_add in code_set:
                continue
            # Check if c_add covers all at_risk
            valid = True
            for w in at_risk:
                if w not in spheres[c_add]:
                    valid = False
                    break
            if valid:
                candidates.append(c_add)
        return candidates
    
    # Run simulated annealing
    rng = random.Random(12345)
    
    best_code = None
    best_counts = None
    best_len = float('inf')
    
    # Multiple restarts
    max_restarts = 8
    time_per_restart = (deadline - start_time) / max_restarts
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Initialize
        if restart == 0:
            current_code = set(initial_indices)
        else:
            # Perturb best code
            if best_code is None:
                current_code = set(initial_indices)
            else:
                current_code = set(best_code)
                # Randomly swap some codewords
                swaps = rng.randint(3, 8)
                for _ in range(swaps):
                    if len(current_code) < 2:
                        break
                    c_remove = rng.choice(list(current_code))
                    replacements = find_replacements(c_remove, current_code, compute_counts(current_code))
                    if replacements:
                        c_add = rng.choice(replacements)
                        current_code.remove(c_remove)
                        current_code.add(c_add)
        
        current_counts = compute_counts(current_code)
        
        if not is_valid(current_code, current_counts):
            # Repair: add random codewords until valid
            while not is_valid(current_code, current_counts):
                uncovered = get_uncovered(current_code, current_counts)
                if not uncovered:
                    break
                w = rng.choice(uncovered)
                # Pick a random codeword covering w
                candidates = [c for c in coverers[w] if c not in current_code]
                if not candidates:
                    # All codewords covering w are already in code; pick any new codeword
                    all_candidates = [c for c in range(total_words) if c not in current_code]
                    if not all_candidates:
                        break
                    c_add = rng.choice(all_candidates)
                else:
                    c_add = rng.choice(candidates)
                current_code.add(c_add)
                for w2 in sphere_lists[c_add]:
                    current_counts[w2] += 1
        
        # Simulated annealing
        T = 10.0
        T_min = 0.01
        cooling = 0.995
        max_iter = 50000
        iter_count = 0
        
        cur_len = len(current_code)
        
        while time.time() < deadline - 3.0 and iter_count < max_iter:
            iter_count += 1
            
            # Try random move
            move_type = rng.random()
            
            if move_type < 0.6:
                # Try to remove a codeword
                c_remove = rng.choice(list(current_code))
                if can_remove(c_remove, current_code, current_counts):
                    # Valid removal
                    current_code.remove(c_remove)
                    for w in sphere_lists[c_remove]:
                        current_counts[w] -= 1
                    cur_len -= 1
                    if cur_len < best_len:
                        best_len = cur_len
                        best_code = current_code.copy()
                        best_counts = current_counts.copy()
                else:
                    # Try to swap
                    replacements = find_replacements(c_remove, current_code, current_counts)
                    if replacements:
                        c_add = rng.choice(replacements)
                        current_code.remove(c_remove)
                        current_code.add(c_add)
                        for w in sphere_lists[c_remove]:
                            current_counts[w] -= 1
                        for w in sphere_lists[c_add]:
                            current_counts[w] += 1
                        # cur_len unchanged
                        if cur_len < best_len:
                            best_len = cur_len
                            best_code = current_code.copy()
                            best_counts = current_counts.copy()
                    else:
                        # No valid swap; try adding a random codeword (worsening)
                        if len(current_code) < 100:
                            c_add = rng.randint(0, total_words - 1)
                            if c_add not in current_code:
                                current_code.add(c_add)
                                for w in sphere_lists[c_add]:
                                    current_counts[w] += 1
                                cur_len += 1
            elif move_type < 0.8:
                # Try to add a codeword (worsening move, helps escape local optima)
                if len(current_code) < 100:
                    c_add = rng.randint(0, total_words - 1)
                    if c_add not in current_code:
                        current_code.add(c_add)
                        for w in sphere_lists[c_add]:
                            current_counts[w] += 1
                        cur_len += 1
            else:
                # Try to remove a random codeword even if invalid (will be repaired)
                c_remove = rng.choice(list(current_code))
                current_code.remove(c_remove)
                for w in sphere_lists[c_remove]:
                    current_counts[w] -= 1
                cur_len -= 1
                # Repair
                while not is_valid(current_code, current_counts):
                    uncovered = get_uncovered(current_code, current_counts)
                    if not uncovered:
                        break
                    w = rng.choice(uncovered)
                    candidates = [c for c in coverers[w] if c not in current_code]
                    if not candidates:
                        all_candidates = [c for c in range(total_words) if c not in current_code]
                        if not all_candidates:
                            break
                        c_add = rng.choice(all_candidates)
                    else:
                        c_add = rng.choice(candidates)
                    current_code.add(c_add)
                    for w2 in sphere_lists[c_add]:
                        current_counts[w2] += 1
                    cur_len += 1
                    if cur_len < best_len:
                        best_len = cur_len
                        best_code = current_code.copy()
                        best_counts = current_counts.copy()
            
            # Cool down
            T *= cooling
            if T < T_min:
                T = T_min
        
        # After SA, try greedy reduction
        changed = True
        while changed:
            changed = False
            for c in list(current_code):
                if can_remove(c, current_code, current_counts):
                    current_code.remove(c)
                    for w in sphere_lists[c]:
                        current_counts[w] -= 1
                    cur_len -= 1
                    changed = True
            if cur_len < best_len:
                best_len = cur_len
                best_code = current_code.copy()
                best_counts = current_counts.copy()
    
    # Final validation
    if best_code is None:
        best_code = set(initial_indices)
        best_counts = compute_counts(best_code)
    
    # Ensure validity
    if not is_valid(best_code, best_counts):
        # Fallback
        fallback = set(initial_indices)
        fb_counts = compute_counts(fallback)
        while not is_valid(fallback, fb_counts):
            uncovered = get_uncovered(fallback, fb_counts)
            if not uncovered:
                break
            w = rng.choice(uncovered)
            candidates = [c for c in coverers[w] if c not in fallback]
            if not candidates:
                all_candidates = [c for c in range(total_words) if c not in fallback]
                if not all_candidates:
                    break
                c_add = rng.choice(all_candidates)
            else:
                c_add = rng.choice(candidates)
            fallback.add(c_add)
            for w2 in sphere_lists[c_add]:
                fb_counts[w2] += 1
        best_code = fallback
        best_counts = fb_counts
    
    return [list(idx_to_word(idx)) for idx in best_code]
