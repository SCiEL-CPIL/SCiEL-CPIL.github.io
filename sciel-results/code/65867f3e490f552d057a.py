import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 90s hard limit

    N = 17
    MIN_D = 6
    TOTAL = 1 << N

    # Precompute popcount lookup for all 17-bit values (fast vectorized index)
    popcount_lookup = np.array([bin(i).count('1') for i in range(TOTAL)], dtype=np.uint8)

    # ---- 1. Generate the known 256-word linear code [17,8,6] ----
    def generate_linear_code():
        basis = []
        current_code = [0]
        for g in range(1, TOTAL):
            if time.time() > deadline:
                break
            valid = True
            for c in current_code:
                if popcount_lookup[g ^ c] < MIN_D:
                    valid = False
                    break
            if valid:
                basis.append(g)
                new_code = []
                for c in current_code:
                    new_code.append(c)
                    new_code.append(c ^ g)
                current_code = new_code
                if len(basis) >= 8:
                    break
        return current_code

    best_code = generate_linear_code()
    best_size = len(best_code)
    best_set = set(best_code)

    # ---- Precompute all words array for vectorized operations ----
    all_words_np = np.arange(TOTAL, dtype=np.uint32)

    # ---- Weight groups for biased orderings ----
    weight_groups = {}
    for w in range(TOTAL):
        weight_groups.setdefault(int(popcount_lookup[w]), []).append(w)

    all_words_list = list(range(TOTAL))

    # ---- Vectorized feasibility check for a batch of words against code ----
    def feasible_batch(words_list, code_np):
        """Returns boolean array: True if word i is feasible against all of code_np."""
        if code_np.size == 0:
            return np.ones(len(words_list), dtype=bool)
        words_arr = np.array(words_list, dtype=np.uint32)
        xors = words_arr[:, None] ^ code_np[None, :]
        dists = popcount_lookup[xors]
        return dists.min(axis=1) >= MIN_D

    # ---- Fast greedy pass over a given order (optimized) ----
    def greedy_pass(order):
        code = []
        code_np = np.array([], dtype=np.uint32)
        for w in order:
            if time.time() > deadline - 5:
                break
            if code_np.size == 0:
                code.append(w)
                code_np = np.array([w], dtype=np.uint32)
            else:
                xors = code_np ^ w
                if popcount_lookup[xors].min() >= MIN_D:
                    code.append(w)
                    code_np = np.append(code_np, w)
        return code

    # ---- Fast extend: add feasible words from a candidate pool to a code ----
    def fast_extend(code, candidates_np, max_add=None):
        """Try to add words from candidates_np to code. Returns extended code list."""
        code_set = set(code)
        code_np = np.array(code, dtype=np.uint32)
        added = 0
        if max_add is None:
            max_add = len(candidates_np)
        batch_size = 1024
        for start in range(0, len(candidates_np), batch_size):
            if time.time() > deadline - 3:
                break
            if added >= max_add:
                break
            batch = candidates_np[start:start+batch_size]
            # Filter out words already in code
            mask = np.array([w not in code_set for w in batch], dtype=bool)
            if not mask.any():
                continue
            batch_filtered = batch[mask]
            if batch_filtered.size == 0:
                continue
            feas = feasible_batch(batch_filtered.tolist(), code_np)
            for i, f in enumerate(feas):
                if f:
                    w = int(batch_filtered[i])
                    code.append(w)
                    code_set.add(w)
                    code_np = np.append(code_np, w)
                    added += 1
                    if added >= max_add:
                        break
        return code

    # ---- Local search: drop one word, try to add multiple, keep if net gain ----
    def local_search(code, max_rounds=15):
        code_np = np.array(code, dtype=np.uint32)
        all_candidates = np.array([w for w in range(TOTAL) if w not in set(code)], dtype=np.uint32)
        for round_num in range(max_rounds):
            if time.time() > deadline - 10:
                break
            improved = False
            n = len(code)
            # Try dropping random positions
            try_positions = list(range(n))
            random.shuffle(try_positions)
            max_attempts = min(n, 30)
            for i in try_positions[:max_attempts]:
                if time.time() > deadline - 10:
                    break
                removed = code[i]
                temp_code = code[:i] + code[i+1:]
                temp_set = set(temp_code)
                temp_np = np.array(temp_code, dtype=np.uint32)
                # Find candidates not in temp_code
                temp_candidates = np.array([w for w in all_candidates if w not in temp_set and w != removed], dtype=np.uint32)
                # Also add the removed word as a candidate (it was in the original set)
                # Actually removed is already excluded from all_candidates since it was in code
                # But we should allow re-adding it? No, we dropped it to make room.
                # Let's try to add at least 2 words
                added = 0
                batch_size = 512
                for start in range(0, min(len(temp_candidates), 10000), batch_size):
                    if time.time() > deadline - 10:
                        break
                    if added >= 2:
                        break
                    batch = temp_candidates[start:start+batch_size]
                    feas = feasible_batch(batch.tolist(), temp_np)
                    for j, f in enumerate(feas):
                        if f:
                            w = int(batch[j])
                            temp_code.append(w)
                            temp_set.add(w)
                            temp_np = np.append(temp_np, w)
                            added += 1
                            if added >= 2:
                                break
                if len(temp_code) > len(code):
                    code = temp_code
                    code_np = temp_np
                    # Update all_candidates: remove newly added, add back removed
                    new_set = set(code)
                    all_candidates = np.array([w for w in range(TOTAL) if w not in new_set], dtype=np.uint32)
                    improved = True
                    break
            if not improved:
                break
        return code

    # ---- Main search loop ----
    pass_count = 0
    while time.time() < deadline - 10:
        pass_count += 1

        if pass_count % 4 == 0:
            # Weight-biased ordering
            order = []
            weight_order = sorted(weight_groups.keys())
            if random.random() < 0.5:
                random.shuffle(weight_order)
            for wt in weight_order:
                group = weight_groups[wt][:]
                random.shuffle(group)
                order.extend(group)
            current_code = greedy_pass(order)
        elif pass_count % 4 == 1:
            # Uniform random ordering
            random.shuffle(all_words_list)
            current_code = greedy_pass(all_words_list[:])
        elif pass_count % 4 == 2:
            # Extend current best with random candidates
            current_code = list(best_code)
            random_candidates = np.array(random.sample(range(TOTAL), min(50000, TOTAL)), dtype=np.uint32)
            current_code = fast_extend(current_code, random_candidates, max_add=50)
        else:
            # Local search on best code
            if time.time() < deadline - 15:
                current_code = local_search(list(best_code), max_rounds=10)

        if len(current_code) > best_size:
            best_size = len(current_code)
            best_code = current_code
            best_set = set(best_code)

    # Final safety polish on the best code found
    if time.time() < deadline - 3:
        random_candidates = np.array(random.sample(range(TOTAL), min(50000, TOTAL)), dtype=np.uint32)
        polished = fast_extend(list(best_code), random_candidates, max_add=20)
        if len(polished) > best_size:
            best_code = polished
            best_size = len(best_code)

    return best_code
