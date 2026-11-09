import time
import random
import numpy as np

def construct():
    t_start = time.time()
    DEADLINE = t_start + 85.0
    N = 4096

    def is_valid(vals):
        m = len(vals)
        seen = set()
        for i in range(m):
            a = vals[i]
            for j in range(i, m):
                b = vals[j]
                for k in range(j, m):
                    c = vals[k]
                    s = a + b + c
                    if s in seen:
                        return False
                    seen.add(s)
        return True

    def compute_triple_sums(vals):
        m = len(vals)
        sums = set()
        for i in range(m):
            for j in range(i, m):
                for k in range(j, m):
                    sums.add(vals[i] + vals[j] + vals[k])
        return sums

    def check_add_fast(v, chosen, sums):
        """Fast numpy-based check if v can be added to chosen with existing triple sums."""
        m = len(chosen)
        new_sums = set()
        new_sums.add(3 * v)

        if m == 0:
            if 3 * v in sums:
                return False, None
            return True, new_sums

        chosen_arr = np.array(chosen, dtype=np.int64)

        # (v, v, u) for each u
        new_sums.update((2 * v + chosen_arr).tolist())
        # (v, u, u) for each u
        new_sums.update((v + 2 * chosen_arr).tolist())
        # (v, u, w) for u <= w
        pair_sums = chosen_arr[:, None] + chosen_arr[None, :]
        mask = np.triu(np.ones((m, m), dtype=bool))
        valid_pairs = pair_sums[mask]
        new_sums.update((v + valid_pairs).tolist())

        if new_sums & sums:
            return False, None
        return True, new_sums

    def collision_density(v, chosen, sums):
        """
        Estimate how many triple sums involving v would collide with existing sums.
        Lower is better (0 means no collision).
        """
        m = len(chosen)
        if m == 0:
            return 0 if 3 * v not in sums else 1

        chosen_arr = np.array(chosen, dtype=np.int64)
        sums_np = set(sums)

        # Count collisions for each type of triple
        count = 0
        if 3 * v in sums_np:
            count += 1

        # (v, v, u)
        vv_u = 2 * v + chosen_arr
        for s in vv_u:
            if s in sums_np:
                count += 1

        # (v, u, u)
        v_uu = v + 2 * chosen_arr
        for s in v_uu:
            if s in sums_np:
                count += 1

        # (v, u, w) for u <= w - sample if m is large
        if m <= 50:
            pair_sums = chosen_arr[:, None] + chosen_arr[None, :]
            mask = np.triu(np.ones((m, m), dtype=bool))
            valid_pairs = pair_sums[mask]
            triple_sums = v + valid_pairs
            for s in triple_sums:
                if s in sums_np:
                    count += 1
        else:
            # Sample pairs
            idx = np.random.choice(m * (m + 1) // 2, size=min(500, m * (m + 1) // 2), replace=False)
            # Convert linear index to (i, j) with i <= j
            # For upper triangle of m x m, linear index k corresponds to
            # row = (sqrt(8*k+1)-1)/2 approximately
            rows = np.zeros(len(idx), dtype=int)
            cols = np.zeros(len(idx), dtype=int)
            for li, k in enumerate(idx):
                # Find i such that i*(i-1)/2 + i <= k < (i+1)*(i+1)/2
                # Actually for upper triangle including diagonal:
                # number of elements in rows 0..i-1 = i*(i-1)/2 + i = i*(i+1)/2
                # So k is in row i where i*(i+1)/2 <= k < (i+1)*(i+2)/2
                i_val = int((np.sqrt(8 * k + 1) - 1) / 2)
                # Adjust: row i has elements from col i to m-1, count = m - i
                # Cumulative: sum_{r=0}^{i-1} (m - r) = i*m - i*(i-1)/2
                # Let's just do it properly
                pass
            # Simpler: just iterate
            sampled = 0
            for _ in range(200):
                i = np.random.randint(0, m)
                j = np.random.randint(i, m)
                s = v + chosen_arr[i] + chosen_arr[j]
                if s in sums_np:
                    count += 1
                sampled += 1
            # Scale up
            total_pairs = m * (m + 1) // 2
            if sampled > 0:
                count += int(count / sampled * total_pairs)

        return count

    def greedy_build(candidate_order, deadline, chosen_init=None, sums_init=None):
        """Greedy add from candidate order, optionally starting from an initial set."""
        if chosen_init is not None:
            chosen = list(chosen_init)
            sums = set(sums_init) if sums_init else compute_triple_sums(chosen)
            chosen_set = set(chosen)
        else:
            chosen = []
            sums = set()
            chosen_set = set()

        for v in candidate_order:
            if time.time() > deadline:
                break
            if v in chosen_set:
                continue
            if v < 1 or v > N:
                continue
            ok, ns = check_add_fast(v, chosen, sums)
            if ok:
                sums |= ns
                chosen.append(v)
                chosen_set.add(v)

        return sorted(chosen), sums

    def simulated_annealing(chosen, sums, deadline, initial_temp=1.0, cooling=0.999):
        """
        Simulated annealing to escape local optima.
        Moves: remove one element, or swap one element with another.
        Objective: maximize |chosen|, with penalty for invalidity (shouldn't happen if we track sums).
        """
        current = list(chosen)
        current_sums = set(sums)
        current_set = set(current)
        best = list(current)
        best_sums = set(current_sums)
        best_len = len(current)
        temp = initial_temp

        # Precompute candidate pool: all values 1..N not in current
        all_vals = set(range(1, N + 1))
        
        while time.time() < deadline:
            # Choose a random move
            move_type = random.random()
            
            if move_type < 0.5 and len(current) > 5:
                # Remove a random element
                idx = random.randint(0, len(current) - 1)
                removed = current[idx]
                
                # New state without removed
                new_current = [x for x in current if x != removed]
                new_set = set(new_current)
                
                # Recompute sums (expensive but necessary)
                new_sums = compute_triple_sums(new_current)
                
                # Try to add back some elements greedily
                added_any = False
                for v in sorted(all_vals - new_set):
                    if time.time() > deadline:
                        break
                    ok, ns = check_add_fast(v, new_current, new_sums)
                    if ok:
                        new_sums |= ns
                        new_current.append(v)
                        new_set.add(v)
                        added_any = True
                        if len(new_current) >= len(current):
                            break
                
                new_len = len(new_current)
                delta_obj = new_len - len(current)
                
                # Accept if improving or with probability exp(delta/temp)
                if delta_obj >= 0 or random.random() < np.exp(delta_obj / max(temp, 0.01)):
                    current = new_current
                    current_sums = new_sums
                    current_set = new_set
                    if new_len > best_len:
                        best = list(current)
                        best_sums = set(current_sums)
                        best_len = new_len
                        
            elif move_type < 0.8:
                # Swap: remove one, try to add a different one
                if len(current) < 3:
                    temp *= cooling
                    continue
                idx = random.randint(0, len(current) - 1)
                removed = current[idx]
                
                new_current = [x for x in current if x != removed]
                new_set = set(new_current)
                new_sums = compute_triple_sums(new_current)
                
                # Try to add a random candidate
                candidates = list(all_vals - new_set)
                if not candidates:
                    temp *= cooling
                    continue
                    
                # Try a few random candidates
                added = False
                for _ in range(5):
                    v = random.choice(candidates)
                    ok, ns = check_add_fast(v, new_current, new_sums)
                    if ok:
                        new_sums |= ns
                        new_current.append(v)
                        new_set.add(v)
                        added = True
                        break
                
                new_len = len(new_current)
                delta_obj = new_len - len(current)
                
                if delta_obj >= 0 or random.random() < np.exp(delta_obj / max(temp, 0.01)):
                    current = new_current
                    current_sums = new_sums
                    current_set = new_set
                    if new_len > best_len:
                        best = list(current)
                        best_sums = set(current_sums)
                        best_len = new_len
                        
            else:
                # Pure addition attempt: try to add a random candidate
                candidates = list(all_vals - current_set)
                if not candidates:
                    break
                    
                # Use density to pick promising candidates
                if len(candidates) > 200:
                    # Sample a subset and pick the one with lowest density
                    sample = random.sample(candidates, min(200, len(candidates)))
                    best_v = min(sample, key=lambda v: collision_density(v, current, current_sums))
                else:
                    best_v = min(candidates, key=lambda v: collision_density(v, current, current_sums))
                
                ok, ns = check_add_fast(best_v, current, current_sums)
                if ok:
                    current_sums |= ns
                    current.append(best_v)
                    current_set.add(best_v)
                    if len(current) > best_len:
                        best = list(current)
                        best_sums = set(current_sums)
                        best_len = len(current)
            
            temp *= cooling
            if temp < 0.001:
                break

        return best, best_sums

    def local_improve(chosen, sums, deadline):
        """Try to add any missing values."""
        chosen_list = list(chosen)
        chosen_set = set(chosen_list)
        sums_set = set(sums)

        improved = True
        while improved and time.time() < deadline:
            improved = False
            for v in range(1, N + 1):
                if time.time() > deadline:
                    break
                if v in chosen_set:
                    continue
                ok, ns = check_add_fast(v, chosen_list, sums_set)
                if ok:
                    sums_set |= ns
                    chosen_list.append(v)
                    chosen_set.add(v)
                    improved = True
                    break

        return sorted(chosen_list), sums_set

    best = []
    best_sums = set()

    # Strategy 1: Base-3 sparse start
    base_candidates = []
    for i in range(8):
        for d in [1, 2]:
            val = d * (3 ** i)
            if val <= N:
                base_candidates.append(val)
    base_candidates.sort()
    base_result, base_sums = greedy_build(base_candidates, DEADLINE - 60.0)
    if is_valid(base_result):
        best = base_result
        best_sums = base_sums

    # Strategy 2: Greedy with ascending order from base
    if time.time() < DEADLINE - 50:
        asc_order = list(range(1, N + 1))
        res, sums = greedy_build(asc_order, DEADLINE - 50.0, best, best_sums)
        if is_valid(res) and len(res) > len(best):
            best = res
            best_sums = sums

    # Strategy 3: Greedy with descending order
    if time.time() < DEADLINE - 40:
        desc_order = list(range(N, 0, -1))
        res, sums = greedy_build(desc_order, DEADLINE - 40.0, best, best_sums)
        if is_valid(res) and len(res) > len(best):
            best = res
            best_sums = sums

    # Strategy 4: Randomized greedy restarts
    random.seed(42)
    for trial in range(5):
        if time.time() > DEADLINE - 30:
            break
        rand_order = list(range(1, N + 1))
        random.shuffle(rand_order)
        res, sums = greedy_build(rand_order, DEADLINE - 30.0, best, best_sums)
        if is_valid(res) and len(res) > len(best):
            best = res
            best_sums = sums

    # Strategy 5: Simulated annealing
    if time.time() < DEADLINE - 10 and best:
        sa_result, sa_sums = simulated_annealing(best, best_sums, DEADLINE - 5.0, initial_temp=2.0, cooling=0.9995)
        if is_valid(sa_result) and len(sa_result) > len(best):
            best = sa_result
            best_sums = sa_sums

    # Strategy 6: Final local improvement
    if time.time() < DEADLINE - 2:
        final_result, final_sums = local_improve(best, best_sums, DEADLINE - 1.0)
        if is_valid(final_result) and len(final_result) > len(best):
            best = final_result

    best = sorted(set(best))
    best = [v for v in best if 1 <= v <= N]
    if is_valid(best):
        return best

    # Fallback
    fallback = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    if is_valid(fallback):
        return fallback
    return [1, 2, 4, 8, 16]

