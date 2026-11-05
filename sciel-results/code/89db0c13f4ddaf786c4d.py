import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145
    N = 32
    RANGE = 512
    MAX_D = RANGE // 2  # max possible d is 255

    def score_set(s):
        """Count distinct d > 0 such that a, a+d, a+2d all in s for some a."""
        if not s:
            return 0
        s_set = set(s)
        distinct_d = set()
        for i in range(len(s)):
            xi = s[i]
            for j in range(i + 1, len(s)):
                xj = s[j]
                if (xi + xj) % 2 == 0:
                    m = (xi + xj) // 2
                    if m in s_set:
                        distinct_d.add(m - xi)
        return len(distinct_d)

    def score_set_fast(s):
        """Faster scoring using bitmask representation."""
        if not s:
            return 0
        mask = 0
        for x in s:
            mask |= (1 << x)
        distinct_d = 0
        for d in range(1, MAX_D + 1):
            # Check if there exists a such that a, a+d, a+2d all in mask
            # a ranges from 1 to RANGE-2d
            # (mask >> d) & (mask >> (2*d)) & mask should have a bit set at position a
            # where a is in [1, RANGE-2d]
            m1 = mask >> d
            m2 = mask >> (2 * d)
            common = m1 & m2 & mask
            if common & 0xFFFFFFFFFFFE:  # bits 1..511 (mask out bit 0)
                distinct_d += 1
        return distinct_d

    def score_and_diffs(s):
        """Return (score, set_of_d) for a set."""
        if not s:
            return 0, set()
        s_set = set(s)
        distinct_d = set()
        for i in range(len(s)):
            xi = s[i]
            for j in range(i + 1, len(s)):
                xj = s[j]
                if (xi + xj) % 2 == 0:
                    m = (xi + xj) // 2
                    if m in s_set:
                        distinct_d.add(m - xi)
        return len(distinct_d), distinct_d

    def local_search(s_init, time_limit):
        """Local search with targeted moves."""
        s = list(s_init)
        s_set = set(s)
        current_score, current_diffs = score_and_diffs(s)
        best_s = list(s)
        best_score = current_score
        best_diffs = set(current_diffs)
        
        no_improve_count = 0
        max_no_improve = 500
        
        while time.time() < time_limit and no_improve_count < max_no_improve:
            move_type = random.random()
            
            if move_type < 0.4:
                # Targeted move: try to add a new difference
                # Find a missing d and try to create it
                all_possible_d = set(range(1, MAX_D + 1))
                missing_d = all_possible_d - current_diffs
                if missing_d:
                    target_d = random.choice(list(missing_d))
                    # Find positions where we could place elements to create this d
                    # We need a, a+d, a+2d all in S
                    # Try removing one element and adding two new ones
                    i = random.randint(0, N - 1)
                    old_val = s[i]
                    # Try different a values
                    found = False
                    for a in range(1, RANGE - 2 * target_d + 1):
                        a1, a2, a3 = a, a + target_d, a + 2 * target_d
                        # Count how many of these are already in s
                        count_in = sum(1 for x in (a1, a2, a3) if x in s_set)
                        if count_in >= 1:
                            # We need to add the missing ones
                            missing = [x for x in (a1, a2, a3) if x not in s_set]
                            if len(missing) <= 1:
                                # Replace old_val with the missing one
                                new_val = missing[0] if missing else old_val
                                if new_val != old_val and new_val not in s_set:
                                    s[i] = new_val
                                    s_set.discard(old_val)
                                    s_set.add(new_val)
                                    new_score, new_diffs = score_and_diffs(s)
                                    delta = new_score - current_score
                                    if delta >= 0:
                                        current_score = new_score
                                        current_diffs = new_diffs
                                        if current_score > best_score:
                                            best_score = current_score
                                            best_s = list(s)
                                            best_diffs = set(new_diffs)
                                            no_improve_count = 0
                                        else:
                                            no_improve_count += 1
                                    else:
                                        s[i] = old_val
                                        s_set.discard(new_val)
                                        s_set.add(old_val)
                                        no_improve_count += 1
                                    found = True
                                    break
                    if found:
                        continue
                    # Revert if we changed
                    s[i] = old_val
                    s_set.discard(new_val)
                    s_set.add(old_val)
            
            elif move_type < 0.7:
                # 1-replace move: replace a random element
                i = random.randint(0, N - 1)
                old_val = s[i]
                new_val = random.randint(1, RANGE)
                if new_val == old_val or new_val in s_set:
                    no_improve_count += 1
                    continue
                s[i] = new_val
                s_set.discard(old_val)
                s_set.add(new_val)
                new_score, new_diffs = score_and_diffs(s)
                delta = new_score - current_score
                if delta >= 0:
                    current_score = new_score
                    current_diffs = new_diffs
                    if current_score > best_score:
                        best_score = current_score
                        best_s = list(s)
                        best_diffs = set(new_diffs)
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    # Accept with some probability for exploration
                    if random.random() < 0.1:
                        current_score = new_score
                        current_diffs = new_diffs
                        no_improve_count += 1
                    else:
                        s[i] = old_val
                        s_set.discard(new_val)
                        s_set.add(old_val)
                        no_improve_count += 1
            else:
                # 2-swap move
                i, j = random.sample(range(N), 2)
                old_i, old_j = s[i], s[j]
                new_i = random.randint(1, RANGE)
                new_j = random.randint(1, RANGE)
                if new_i == new_j:
                    no_improve_count += 1
                    continue
                if (new_i in s_set and new_i != old_i and new_i != old_j) or \
                   (new_j in s_set and new_j != old_i and new_j != old_j):
                    no_improve_count += 1
                    continue
                s[i] = new_i
                s[j] = new_j
                s_set.discard(old_i)
                s_set.discard(old_j)
                s_set.add(new_i)
                s_set.add(new_j)
                new_score, new_diffs = score_and_diffs(s)
                delta = new_score - current_score
                if delta >= 0:
                    current_score = new_score
                    current_diffs = new_diffs
                    if current_score > best_score:
                        best_score = current_score
                        best_s = list(s)
                        best_diffs = set(new_diffs)
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    if random.random() < 0.05:
                        current_score = new_score
                        current_diffs = new_diffs
                        no_improve_count += 1
                    else:
                        s[i] = old_i
                        s[j] = old_j
                        s_set.discard(new_i)
                        s_set.discard(new_j)
                        s_set.add(old_i)
                        s_set.add(old_j)
                        no_improve_count += 1
        
        return best_s, best_score

    def generate_initializations():
        """Generate diverse initializations."""
        inits = []
        
        # 1. Random
        inits.append(random.sample(range(1, RANGE + 1), N))
        
        # 2. Structured: spread out
        step = RANGE // N
        structured = [i * step + 1 for i in range(N)]
        inits.append(structured)
        
        # 3. Arithmetic progression base
        ap = [1 + i * 15 for i in range(N)]
        inits.append(ap)
        
        # 4. Mixed scales
        mixed = []
        for i in range(N):
            if i < N // 2:
                mixed.append(1 + i * 8)
            else:
                mixed.append(256 + (i - N // 2) * 8)
        inits.append(mixed)
        
        # 5. Powers of 2 based
        pow2 = []
        val = 1
        while len(pow2) < N:
            pow2.append(val)
            val = val * 2
            if val > RANGE:
                val = 1 + random.randint(0, RANGE // 2)
        inits.append(pow2[:N])
        
        # 6. Greedy construction
        s = []
        s_set = set()
        for _ in range(N):
            best_val = -1
            best_score = -1
            candidates = list(range(1, RANGE + 1))
            random.shuffle(candidates)
            for val in candidates:
                if val in s_set:
                    continue
                new_s = s + [val]
                sc = score_set(new_s)
                if sc > best_score:
                    best_score = sc
                    best_val = val
            if best_val != -1:
                s.append(best_val)
                s_set.add(best_val)
        inits.append(sorted(s))
        
        return inits

    best_overall = None
    best_overall_score = 0
    
    inits = generate_initializations()
    
    # Run local search on each initialization
    sa_time_per_run = 15
    for init in inits:
        if time.time() > deadline - 5:
            break
        sa_limit = min(time.time() + sa_time_per_run, deadline - 5)
        s_res, score_res = local_search(init, sa_limit)
        if score_res > best_overall_score:
            best_overall_score = score_res
            best_overall = s_res
    
    # If we have time left, do one more long local search on the best found
    if time.time() < deadline - 10 and best_overall is not None:
        remaining_time = deadline - 10 - time.time()
        s_res, score_res = local_search(best_overall, time.time() + remaining_time)
        if score_res > best_overall_score:
            best_overall_score = score_res
            best_overall = s_res

    if best_overall is None:
        best_overall = list(range(1, N + 1))
        
    result = sorted(best_overall)
    # Final validation
    if len(set(result)) != N or not all(1 <= x <= RANGE for x in result):
        result = list(range(1, N + 1))
        
    return result
