import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    N = 32
    MAX_VAL = 512
    MAX_D = 255  # max d such that a + 2*d <= 512 for some a >= 1

    def time_left():
        return deadline - time.time()

    # Bitmask helpers
    def list_to_mask(lst):
        m = 0
        for x in lst:
            m |= (1 << (x - 1))
        return m

    def mask_to_list(mask):
        return [i + 1 for i in range(MAX_VAL) if (mask >> i) & 1]

    def score_mask(mask):
        count = 0
        for d in range(1, MAX_D + 1):
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count

    def delta_score(mask, old_bit, new_bit):
        """Compute change in score when flipping old_bit to new_bit."""
        new_mask = mask ^ old_bit ^ new_bit
        delta = 0
        for d in range(1, MAX_D + 1):
            old_has = (mask & (mask >> d) & (mask >> (2 * d))) != 0
            new_has = (new_mask & (new_mask >> d) & (new_mask >> (2 * d))) != 0
            if old_has != new_has:
                delta += 1 if new_has else -1
        return delta, new_mask

    # Precompute for each position x, the list of d values where x participates
    # This helps us limit which d's we need to check in delta_score
    # But for simplicity, keep the full scan since MAX_D=255 is small

    # --- Parameterized construction ---
    # Model: set is approximately a union of arithmetic progressions or structured sets
    # We'll parameterize by choosing "clusters" and then optimize

    def gen_structured_seed(seed_idx):
        """Generate diverse structured initial solutions."""
        random.seed(seed_idx)
        solutions = []

        # 1. Pure arithmetic progressions
        for d in range(4, 17):
            if 1 + (N - 1) * d <= MAX_VAL:
                lst = [1 + i * d for i in range(N)]
                m = list_to_mask(lst)
                s = score_mask(m)
                solutions.append((lst, m, s))

        # 2. Sumset A + B
        for mult in [2, 3, 4, 5, 6]:
            for trial in range(50):
                sizeA = random.randint(2, 10)
                sizeB = random.randint(2, 10)
                A = set(random.sample(range(1, 80), sizeA))
                maxB = (MAX_VAL - 1) // mult
                B = set(random.sample(range(1, maxB + 1), sizeB))
                S = set()
                for a in A:
                    for b in B:
                        v = a + mult * b
                        if 1 <= v <= MAX_VAL:
                            S.add(v)
                if len(S) >= N:
                    lst = random.sample(sorted(S), N)
                    m = list_to_mask(lst)
                    s = score_mask(m)
                    solutions.append((lst, m, s))

        # 3. Residue classes mod m
        for mod in range(3, 12):
            for start in range(1, mod + 1):
                lst = []
                for v in range(start, MAX_VAL + 1, mod):
                    lst.append(v)
                    if len(lst) == N:
                        break
                if len(lst) == N:
                    m = list_to_mask(lst)
                    s = score_mask(m)
                    solutions.append((lst, m, s))

        # 4. Random
        for _ in range(30):
            lst = random.sample(range(1, MAX_VAL + 1), N)
            m = list_to_mask(lst)
            s = score_mask(m)
            solutions.append((lst, m, s))

        # 5. Greedy construction: build set element by element to maximize score
        for _ in range(20):
            m = 0
            s = 0
            used = set()
            order = list(range(1, MAX_VAL + 1))
            random.shuffle(order)
            for v in order:
                if len(used) == N:
                    break
                # Try adding v
                new_m = m | (1 << (v - 1))
                if new_m.bit_count() != m.bit_count() + 1:
                    continue
                new_s = score_mask(new_m)
                if new_s >= s:
                    m = new_m
                    s = new_s
                    used.add(v)
            if len(used) == N:
                lst = list(used)
                m = list_to_mask(lst)
                s = score_mask(m)
                solutions.append((lst, m, s))

        # 6. Cantor-like sets: base-b digit patterns
        # A number x = sum(d_i * b^i) where each d_i is chosen from a small set
        for base in [3, 4, 5, 6]:
            for digit_set in range(1, base):
                # Generate all numbers with N or fewer digits in this base
                # where each digit is in {0, digit_set} or similar small sets
                candidates = []
                max_power = 0
                while base ** max_power <= MAX_VAL:
                    max_power += 1
                # Use recursive generation
                def gen_cantor(pos, current):
                    if pos == max_power:
                        if 1 <= current <= MAX_VAL:
                            candidates.append(current)
                        return
                    for d in [0, digit_set]:
                        gen_cantor(pos + 1, current + d * (base ** pos))
                gen_cantor(0, 0)
                candidates = sorted(set(candidates))
                if len(candidates) >= N:
                    lst = random.sample(candidates, N)
                    m = list_to_mask(lst)
                    s = score_mask(m)
                    solutions.append((lst, m, s))

        # 7. Union of two disjoint APs with different steps
        for d1 in range(3, 12):
            for d2 in range(3, 12):
                if d1 == d2:
                    continue
                for a1 in range(1, 20):
                    for a2 in range(1, 20):
                        set1 = set(a1 + i * d1 for i in range(N // 2 + 1))
                        set2 = set(a2 + i * d2 for i in range(N // 2 + 1))
                        combined = set1 | set2
                        combined = {x for x in combined if 1 <= x <= MAX_VAL}
                        if len(combined) >= N:
                            lst = random.sample(sorted(combined), N)
                            m = list_to_mask(lst)
                            s = score_mask(m)
                            solutions.append((lst, m, s))

        # 8. Base-4 digit restriction: only use digits 0 and 2 (or 0 and 1)
        for allowed_digits in [[0, 1], [0, 2], [1, 3], [0, 3]]:
            candidates = []
            max_power = 0
            while 4 ** max_power <= MAX_VAL:
                max_power += 1
            def gen_base4(pos, current):
                if pos == max_power:
                    if 1 <= current <= MAX_VAL:
                        candidates.append(current)
                    return
                for d in allowed_digits:
                    gen_base4(pos + 1, current + d * (4 ** pos))
            gen_base4(0, 0)
            candidates = sorted(set(candidates))
            if len(candidates) >= N:
                lst = random.sample(candidates, N)
                m = list_to_mask(lst)
                s = score_mask(m)
                solutions.append((lst, m, s))

        # 9. Structured: take a base AP and perturb some elements
        for d in range(4, 14):
            base_ap = [1 + i * d for i in range(N)]
            for trial in range(15):
                lst = base_ap[:]
                # Perturb 2-4 random elements
                num_perturb = random.randint(2, 4)
                for _ in range(num_perturb):
                    idx = random.randint(0, N - 1)
                    lst[idx] = random.randint(1, MAX_VAL)
                # Ensure uniqueness
                seen = set()
                unique_lst = []
                for x in lst:
                    if x not in seen and 1 <= x <= MAX_VAL:
                        seen.add(x)
                        unique_lst.append(x)
                if len(unique_lst) < N:
                    # Fill remaining with random
                    while len(unique_lst) < N:
                        x = random.randint(1, MAX_VAL)
                        if x not in seen:
                            seen.add(x)
                            unique_lst.append(x)
                lst = unique_lst[:N]
                m = list_to_mask(lst)
                s = score_mask(m)
                solutions.append((lst, m, s))

        return solutions

    # --- Local search with delta scoring ---
    def local_search(individual, max_iter=100, time_limit=None):
        """Hill climbing using delta scoring for efficient evaluation, with AP-completing moves."""
        lst = individual[0][:]
        m = individual[1]
        s = individual[2]
        s_set = set(lst)

        if time_limit is None:
            time_limit = 0.5

        start_ls = time.time()
        for _ in range(max_iter):
            if time.time() - start_ls > time_limit:
                break
            improved = False

            # Strategy 1: Try to complete 3-term progressions
            # For each pair (a, a+d) in the set, try to add a+2d
            # Or for each (a, a+2d), try to add a+d
            for d in range(1, MAX_D + 1):
                if time.time() - start_ls > time_limit:
                    break
                # Find pairs that are close to forming a 3-AP
                for a in range(1, MAX_VAL - 2 * d + 1):
                    if a in s_set and (a + d) in s_set and (a + 2 * d) not in s_set:
                        # Replace a random element with a+2d if it improves
                        new_val = a + 2 * d
                        # Find an element to replace (prefer one that's not in many 3-APs)
                        best_replace_idx = -1
                        best_delta = 0
                        for idx in range(N):
                            if lst[idx] == new_val:
                                continue
                            old_bit = 1 << (lst[idx] - 1)
                            new_bit = 1 << (new_val - 1)
                            delta, _ = delta_score(m, old_bit, new_bit)
                            if delta > best_delta:
                                best_delta = delta
                                best_replace_idx = idx
                        if best_replace_idx >= 0:
                            old = lst[best_replace_idx]
                            s_set.discard(old)
                            s_set.add(new_val)
                            lst[best_replace_idx] = new_val
                            m ^= (1 << (old - 1)) ^ (1 << (new_val - 1))
                            s += best_delta
                            improved = True
                            break  # Only one replacement per d
                    elif a in s_set and (a + 2 * d) in s_set and (a + d) not in s_set:
                        new_val = a + d
                        best_replace_idx = -1
                        best_delta = 0
                        for idx in range(N):
                            if lst[idx] == new_val:
                                continue
                            old_bit = 1 << (lst[idx] - 1)
                            new_bit = 1 << (new_val - 1)
                            delta, _ = delta_score(m, old_bit, new_bit)
                            if delta > best_delta:
                                best_delta = delta
                                best_replace_idx = idx
                        if best_replace_idx >= 0:
                            old = lst[best_replace_idx]
                            s_set.discard(old)
                            s_set.add(new_val)
                            lst[best_replace_idx] = new_val
                            m ^= (1 << (old - 1)) ^ (1 << (new_val - 1))
                            s += best_delta
                            improved = True
                            break

            # Strategy 2: Standard random swap search
            if not improved:
                for i in range(N):
                    if time.time() - start_ls > time_limit:
                        break
                    old = lst[i]
                    old_bit = 1 << (old - 1)
                    best_new = old
                    best_delta = 0
                    # Try several random replacements
                    for _ in range(12):
                        new = random.randint(1, MAX_VAL)
                        if new == old or new in s_set:
                            continue
                        new_bit = 1 << (new - 1)
                        delta, _ = delta_score(m, old_bit, new_bit)
                        if delta > best_delta:
                            best_delta = delta
                            best_new = new
                    if best_new != old:
                        s_set.discard(old)
                        s_set.add(best_new)
                        lst[i] = best_new
                        m ^= (1 << (old - 1)) ^ (1 << (best_new - 1))
                        s += best_delta
                        improved = True

            if not improved:
                break

        return (lst, m, s)

    # --- Exhaustive single-swap local search ---
    def exhaustive_swap_search(individual, time_budget=10.0):
        """Try all possible single-element replacements with optimized delta scoring."""
        lst = individual[0][:]
        m = individual[1]
        s = individual[2]
        s_set = set(lst)
        start_es = time.time()

        improved = True
        while improved and time.time() - start_es < time_budget:
            improved = False
            for i in range(N):
                if time.time() - start_es >= time_budget:
                    break
                old = lst[i]
                old_bit = 1 << (old - 1)
                best_new = old
                best_delta = 0
                for new in range(1, MAX_VAL + 1):
                    if new == old or new in s_set:
                        continue
                    new_bit = 1 << (new - 1)
                    delta, _ = delta_score(m, old_bit, new_bit)
                    if delta > best_delta:
                        best_delta = delta
                        best_new = new
                if best_new != old:
                    s_set.discard(old)
                    s_set.add(best_new)
                    lst[i] = best_new
                    m ^= old_bit ^ (1 << (best_new - 1))
                    s += best_delta
                    improved = True

        return (lst, m, s)

    # --- Simulated annealing ---
    def simulated_annealing(individual, time_budget=15.0):
        """SA with delta scoring."""
        lst = individual[0][:]
        m = individual[1]
        s = individual[2]
        s_set = set(lst)
        best_lst = lst[:]
        best_m = m
        best_s = s

        start_sa = time.time()
        T = 10.0
        cooling = 0.98

        while time.time() - start_sa < time_budget:
            # Pick a random element to swap
            i = random.randint(0, N - 1)
            old = lst[i]
            old_bit = 1 << (old - 1)

            # Pick a random replacement
            new = random.randint(1, MAX_VAL)
            if new == old or new in s_set:
                continue

            new_bit = 1 << (new - 1)
            delta, new_m = delta_score(m, old_bit, new_bit)

            if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                # Accept
                s_set.discard(old)
                s_set.add(new)
                lst[i] = new
                m = new_m
                s += delta

                if s > best_s:
                    best_s = s
                    best_lst = lst[:]
                    best_m = m

            T *= cooling

        return (best_lst, best_m, best_s)

    # --- Two-swap local search ---
    def two_swap_search(individual, time_budget=5.0):
        """Try swapping two elements at a time."""
        lst = individual[0][:]
        m = individual[1]
        s = individual[2]
        s_set = set(lst)
        start_ts = time.time()
        improved = True

        while improved and time.time() - start_ts < time_budget:
            improved = False
            for i in range(N):
                if time.time() - start_ts >= time_budget:
                    break
                for j in range(i + 1, N):
                    if time.time() - start_ts >= time_budget:
                        break
                    old_i = lst[i]
                    old_j = lst[j]
                    old_bit_i = 1 << (old_i - 1)
                    old_bit_j = 1 << (old_j - 1)

                    best_new_i = old_i
                    best_new_j = old_j
                    best_delta = 0

                    # Try a few random pairs of replacements
                    for _ in range(6):
                        new_i = random.randint(1, MAX_VAL)
                        new_j = random.randint(1, MAX_VAL)
                        if new_i in s_set or new_j in s_set:
                            continue
                        if new_i == new_j:
                            continue
                        # Remove old bits, add new bits
                        temp_m = m ^ old_bit_i ^ old_bit_j
                        temp_m |= (1 << (new_i - 1)) | (1 << (new_j - 1))
                        # Compute delta
                        delta = 0
                        for d in range(1, MAX_D + 1):
                            old_has = (m & (m >> d) & (m >> (2 * d))) != 0
                            new_has = (temp_m & (temp_m >> d) & (temp_m >> (2 * d))) != 0
                            if old_has != new_has:
                                delta += 1 if new_has else -1
                        if delta > best_delta:
                            best_delta = delta
                            best_new_i = new_i
                            best_new_j = new_j

                    if best_new_i != old_i or best_new_j != old_j:
                        s_set.discard(old_i)
                        s_set.discard(old_j)
                        s_set.add(best_new_i)
                        s_set.add(best_new_j)
                        lst[i] = best_new_i
                        lst[j] = best_new_j
                        m ^= old_bit_i ^ old_bit_j ^ (1 << (best_new_i - 1)) ^ (1 << (best_new_j - 1))
                        s += best_delta
                        improved = True

        return (lst, m, s)

    # === Main algorithm ===

    # Step 1: Generate diverse initial population
    random.seed(42)
    population = []
    seen = set()

    # Add structured seeds from multiple seeds
    for seed_idx in [1, 2, 3, 4, 5, 6, 7, 8]:
        seeds = gen_structured_seed(seed_idx)
        for sol in seeds:
            key = tuple(sorted(sol[0]))
            if key not in seen:
                seen.add(key)
                population.append(sol)

    # Sort by fitness
    population.sort(key=lambda x: x[2], reverse=True)

    best_individual = population[0]
    best_score = best_individual[2]
    best_list = best_individual[0][:]
    best_mask = best_individual[1]

    print(f"Initial best: {best_score}", file=sys.stderr)

    # Step 2: Local search on top candidates
    pop_size = min(len(population), 80)
    top_candidates = population[:pop_size]

    for idx, ind in enumerate(top_candidates[:25]):
        if time_left() < 10:
            break
        t_budget = min(2.5, time_left() * 0.04)
        improved = local_search(ind, max_iter=100, time_limit=t_budget)
        if improved[2] > best_score:
            best_score = improved[2]
            best_list = improved[0][:]
            best_mask = improved[1]
        top_candidates[idx] = improved

    top_candidates.sort(key=lambda x: x[2], reverse=True)
    print(f"After local search: {best_score}", file=sys.stderr)

    # Step 3: Simulated annealing on best few
    for idx in range(min(6, len(top_candidates))):
        if time_left() < 15:
            break
        t_budget = min(15.0, time_left() * 0.15)
        sa_result = simulated_annealing(top_candidates[idx], time_budget=t_budget)
        if sa_result[2] > best_score:
            best_score = sa_result[2]
            best_list = sa_result[0][:]
            best_mask = sa_result[1]
            print(f"SA improved to: {best_score}", file=sys.stderr)
        top_candidates[idx] = sa_result

    top_candidates.sort(key=lambda x: x[2], reverse=True)

    # Step 4: Two-swap search on top
    for idx in range(min(12, len(top_candidates))):
        if time_left() < 8:
            break
        t_budget = min(4.0, time_left() * 0.07)
        ts_result = two_swap_search(top_candidates[idx], time_budget=t_budget)
        if ts_result[2] > best_score:
            best_score = ts_result[2]
            best_list = ts_result[0][:]
            best_mask = ts_result[1]
            print(f"Two-swap improved to: {best_score}", file=sys.stderr)
        top_candidates[idx] = ts_result

    top_candidates.sort(key=lambda x: x[2], reverse=True)

    # Step 5: Exhaustive single-swap on the best
    if time_left() > 5:
        t_budget = min(time_left() * 0.3, 20.0)
        es_result = exhaustive_swap_search((best_list, best_mask, best_score), time_budget=t_budget)
        if es_result[2] > best_score:
            best_score = es_result[2]
            best_list = es_result[0][:]
            best_mask = es_result[1]
            print(f"Exhaustive swap improved to: {best_score}", file=sys.stderr)

    # Step 6: Final local search polish
    if time_left() > 3:
        t_budget = min(time_left() * 0.4, 12.0)
        final_result = local_search((best_list, best_mask, best_score), max_iter=250, time_limit=t_budget)
        if final_result[2] > best_score:
            best_score = final_result[2]
            best_list = final_result[0][:]
            best_mask = final_result[1]

    # Step 7: One more SA pass if time allows
    if time_left() > 10:
        t_budget = min(time_left() * 0.5, 25.0)
        sa_result = simulated_annealing((best_list, best_mask, best_score), time_budget=t_budget)
        if sa_result[2] > best_score:
            best_score = sa_result[2]
            best_list = sa_result[0][:]
            best_mask = sa_result[1]

    print(f"Final score: {best_score}", file=sys.stderr)

    # Verify and return
    result = mask_to_list(best_mask)
    assert len(result) == N, f"Expected {N} elements, got {len(result)}"
    assert len(set(result)) == N, "Duplicate elements"
    assert all(1 <= x <= MAX_VAL for x in result), "Out of range"

    return result

# EVOLVE-BLOCK-END
