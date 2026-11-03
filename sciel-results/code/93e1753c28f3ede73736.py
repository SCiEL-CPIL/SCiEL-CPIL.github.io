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
    deadline = start_time + 118.0  # Safety margin for 120s limit

    # Fallback for non-standard parameters
    if q != 3 or n != 6 or radius != 1:
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

    # Precompute spheres (Hamming balls of radius 1)
    spheres = [None] * total_words
    sphere_lists = [None] * total_words
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

    # Precompute for each word w, which codewords cover it
    coverers = [[] for _ in range(total_words)]
    for c in range(total_words):
        for w in spheres[c]:
            coverers[w].append(c)

    # Precompute distance-2 graph adjacency for greedy independent set
    # Two codewords conflict if their spheres overlap (distance <= 2)
    dist2_adj = [set() for _ in range(total_words)]
    for c in range(total_words):
        for w in spheres[c]:
            for c2 in coverers[w]:
                if c2 != c:
                    dist2_adj[c].add(c2)

    # --- 2. Initial Code Construction via Greedy Independent Set ---
    def greedy_independent_set(order):
        """Select maximum independent set on distance-2 graph using given vertex order."""
        selected = set()
        blocked = set()
        for c in order:
            if c in blocked:
                continue
            selected.add(c)
            for nb in dist2_adj[c]:
                if nb not in selected:
                    blocked.add(nb)
        return selected

    # Tetracode seed for diversity
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    tetracode_indices = set()
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            tetracode_indices.add(word_to_idx(h + list(tail)))

    # Greedy IS with random ordering
    rng_init = random.Random(42)
    order = list(range(total_words))
    rng_init.shuffle(order)
    initial_indices = greedy_independent_set(order)

    # Ensure it's a valid cover (should be, but verify and repair if needed)
    init_counts = np.zeros(total_words, dtype=np.int32)
    for idx in initial_indices:
        for w in sphere_lists[idx]:
            init_counts[w] += 1
    if np.any(init_counts == 0):
        # Repair: add codewords to cover uncovered words
        while np.any(init_counts == 0):
            uncovered = np.where(init_counts == 0)[0]
            w = uncovered[0]
            best_c = None
            best_gain = -1
            for c in coverers[w]:
                if c in initial_indices:
                    continue
                gain = 0
                for w2 in sphere_lists[c]:
                    if init_counts[w2] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            if best_c is None:
                break
            initial_indices.add(best_c)
            for w2 in sphere_lists[best_c]:
                init_counts[w2] += 1

    # --- 3. Helper Functions ---

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
        at_risk = [w for w in sphere_lists[c_remove] if counts[w] == 1]
        if not at_risk:
            return []
        
        first_risk = at_risk[0]
        candidates = []
        for c_add in coverers[first_risk]:
            if c_add in code_set:
                continue
            valid = True
            for w in at_risk:
                if w not in spheres[c_add]:
                    valid = False
                    break
            if valid:
                candidates.append(c_add)
        return candidates

    # --- 4. Search Algorithm ---
    rng = random.Random(12345)

    best_code = None
    best_counts = None
    best_len = float('inf')

    # Multiple restarts with different random seeds
    max_restarts = 15

    for restart in range(max_restarts):
        if time.time() > deadline - 8.0:
            break
        
        # Initialize with different strategies per restart
        if restart == 0:
            current_code = set(initial_indices)
        elif restart == 1:
            # Greedy IS with different ordering
            order2 = list(range(total_words))
            rng.shuffle(order2)
            current_code = greedy_independent_set(order2)
            # Repair if needed
            temp_counts = compute_counts(current_code)
            while not is_valid(current_code, temp_counts) and time.time() < deadline - 1.0:
                uncovered = get_uncovered(current_code, temp_counts)
                if not uncovered:
                    break
                w = uncovered[0]
                best_c = None
                best_gain = -1
                for c in coverers[w]:
                    if c in current_code:
                        continue
                    gain = 0
                    for w2 in sphere_lists[c]:
                        if temp_counts[w2] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_c = c
                if best_c is None:
                    break
                current_code.add(best_c)
                for w2 in sphere_lists[best_c]:
                    temp_counts[w2] += 1
        else:
            # Perturb best code
            if best_code is None:
                current_code = set(initial_indices)
            else:
                current_code = set(best_code)
                # Randomly swap some codewords
                swaps = rng.randint(3, 8)
                cur_counts = compute_counts(current_code)
                for _ in range(swaps):
                    if len(current_code) < 2:
                        break
                    c_remove = rng.choice(list(current_code))
                    replacements = find_replacements(c_remove, current_code, cur_counts)
                    if replacements:
                        c_add = rng.choice(replacements)
                        current_code.remove(c_remove)
                        current_code.add(c_add)
                        for w in sphere_lists[c_remove]:
                            cur_counts[w] -= 1
                        for w in sphere_lists[c_add]:
                            cur_counts[w] += 1
        
        current_counts = compute_counts(current_code)
        
        if not is_valid(current_code, current_counts):
            # Repair: add codewords until valid
            while not is_valid(current_code, current_counts) and time.time() < deadline - 1.0:
                uncovered = get_uncovered(current_code, current_counts)
                if not uncovered:
                    break
                w = uncovered[0]
                best_c = None
                best_gain = -1
                for c in coverers[w]:
                    if c in current_code:
                        continue
                    gain = 0
                    for w2 in sphere_lists[c]:
                        if current_counts[w2] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_c = c
                if best_c is None:
                    break
                current_code.add(best_c)
                for w2 in sphere_lists[best_c]:
                    current_counts[w2] += 1
        
        # Simulated Annealing with improved schedule
        T = 20.0
        T_min = 0.001
        cooling = 0.9985
        max_iter = 150000
        iter_count = 0
        
        cur_len = len(current_code)
        
        # Track recent best for plateau detection
        recent_best = cur_len
        plateau_count = 0
        
        while time.time() < deadline - 3.0 and iter_count < max_iter:
            iter_count += 1
            
            # Try random move
            move_type = rng.random()
            
            if move_type < 0.55:
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
                    if cur_len < recent_best:
                        recent_best = cur_len
                        plateau_count = 0
                    else:
                        plateau_count += 1
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
                    else:
                        # No valid swap; try adding a random codeword (worsening)
                        if len(current_code) < 100:
                            c_add = rng.randint(0, total_words - 1)
                            if c_add not in current_code:
                                current_code.add(c_add)
                                for w in sphere_lists[c_add]:
                                    current_counts[w] += 1
                                cur_len += 1
            elif move_type < 0.75:
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
                # Repair greedily
                repair_limit = 15
                while not is_valid(current_code, current_counts) and repair_limit > 0:
                    uncovered = get_uncovered(current_code, current_counts)
                    if not uncovered:
                        break
                    w = uncovered[0]
                    best_c = None
                    best_gain = -1
                    for c in coverers[w]:
                        if c in current_code:
                            continue
                        gain = 0
                        for w2 in sphere_lists[c]:
                            if current_counts[w2] == 0:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_c = c
                    if best_c is None:
                        break
                    current_code.add(best_c)
                    for w2 in sphere_lists[best_c]:
                        current_counts[w2] += 1
                    cur_len += 1
                    repair_limit -= 1
                    if cur_len < best_len:
                        best_len = cur_len
                        best_code = current_code.copy()
                        best_counts = current_counts.copy()
                    if cur_len < recent_best:
                        recent_best = cur_len
                        plateau_count = 0
                    else:
                        plateau_count += 1
            
            # Cool down with plateau-based reheating
            if plateau_count > 5000:
                T = max(T, 5.0)  # Reheat
                plateau_count = 0
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        # After SA, try iterative greedy reduction with multiple passes
        for pass_num in range(5):
            changed = True
            while changed and time.time() < deadline - 1.0:
                changed = False
                code_list = list(current_code)
                rng.shuffle(code_list)
                for c in code_list:
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
            if not changed:
                break

    # Final validation
    if best_code is None:
        best_code = set(initial_indices)
        best_counts = compute_counts(best_code)

    # Ensure validity
    if not is_valid(best_code, best_counts):
        # Fallback
        fallback = set(initial_indices)
        fb_counts = compute_counts(fallback)
        while not is_valid(fallback, fb_counts) and time.time() < deadline - 0.5:
            uncovered = get_uncovered(fallback, fb_counts)
            if not uncovered:
                break
            w = uncovered[0]
            best_c = None
            best_gain = -1
            for c in coverers[w]:
                if c in fallback:
                    continue
                gain = 0
                for w2 in sphere_lists[c]:
                    if fb_counts[w2] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            if best_c is None:
                break
            fallback.add(best_c)
            for w2 in sphere_lists[best_c]:
                fb_counts[w2] += 1
        best_code = fallback
        best_counts = fb_counts

    # Final polish: one more greedy reduction pass
    final_counts = compute_counts(best_code)
    changed = True
    while changed and time.time() < deadline - 0.1:
        changed = False
        code_list = list(best_code)
        rng.shuffle(code_list)
        for c in code_list:
            if can_remove(c, best_code, final_counts):
                best_code.remove(c)
                for w in sphere_lists[c]:
                    final_counts[w] -= 1
                changed = True

    return [list(idx_to_word(idx)) for idx in best_code]
