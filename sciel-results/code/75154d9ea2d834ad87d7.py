"""Combined program for ternary covering code K_3(7,1)."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Specific optimization for the known task: q=3, n=7, radius=1
    if q == 3 and n == 7 and radius == 1:
        return _construct_3_7_1(deadline)

    # Fallback for other parameters: simple greedy + pruning
    return _construct_general(q, n, radius, deadline)

def _construct_3_7_1(deadline):
    N = 2187  # 3^7
    # Precompute powers of 3 for fast index conversion
    p3 = [1] * 7
    for i in range(1, 7):
        p3[i] = p3[i - 1] * 3

    # Precompute ball membership: ball_members[i] = list of indices within Hamming distance 1 of i
    # Also precompute the neighbor offsets for fast ball computation
    ball_members = [None] * N
    for i in range(N):
        digits = [0] * 7
        tmp = i
        for pos in range(7):
            digits[pos] = tmp % 3
            tmp //= 3

        members = [i]
        for pos in range(7):
            orig = digits[pos]
            for val in range(3):
                if val != orig:
                    members.append(i + (val - orig) * p3[pos])
        ball_members[i] = members

    # Helper: Greedy Set Cover initialization using numpy coverage array
    def greedy_init(seed):
        rng = random.Random(seed)
        uncovered = np.ones(N, dtype=bool)
        selected = []
        selected_set = set()

        # Precompute ball membership as a list of lists for fast access
        # ball_members[i] is a list of indices within distance 1 of i

        while uncovered.any():
            best_idx = -1
            best_count = -1

            # Randomized scan to break ties and explore different paths
            candidates = list(range(N))
            rng.shuffle(candidates)

            for i in candidates:
                if i in selected_set:
                    continue

                # Count uncovered words covered by i
                cnt = sum(1 for j in ball_members[i] if uncovered[j])

                if cnt > best_count:
                    best_count = cnt
                    best_idx = i

            if best_idx == -1 or best_count == 0:
                break

            selected.append(best_idx)
            selected_set.add(best_idx)
            for j in ball_members[best_idx]:
                uncovered[j] = False

            if time.time() > deadline - 5.0:
                break

        return selected

    # Helper: Local Search with Perturbation and 2-for-1 Swaps (numpy-based)
    def local_search(code, seed):
        rng = random.Random(seed)
        code_set = set(code)

        # cnt[i] is the number of codewords in 'code' that cover word i
        cnt = np.zeros(N, dtype=np.int32)
        for c in code:
            for j in ball_members[c]:
                cnt[j] += 1

        best_code = code.copy()
        best_size = len(code)
        no_improve_count = 0
        max_no_improve = 20  # Trigger perturbation after this many failed cycles

        while time.time() < deadline - 2.0:
            improved = False

            # Phase 1: Deletion
            to_remove = []
            for c in code:
                is_essential = False
                for j in ball_members[c]:
                    if cnt[j] == 1:
                        is_essential = True
                        break
                if not is_essential:
                    to_remove.append(c)

            if to_remove:
                for c in to_remove:
                    code.remove(c)
                    code_set.remove(c)
                    for j in ball_members[c]:
                        cnt[j] -= 1

                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code.copy()
                improved = True
                no_improve_count = 0
                continue  # Restart loop as counts changed significantly

            # Phase 2: 1-Swap
            if code:
                swaps_done = 0
                max_swaps = 50
                while swaps_done < max_swaps and time.time() < deadline - 2.0:
                    swaps_done += 1
                    c = rng.choice(code)

                    # Find words uniquely covered by c
                    unique_words = [j for j in ball_members[c] if cnt[j] == 1]

                    if not unique_words:
                        # Redundant, handle just in case
                        code.remove(c)
                        code_set.remove(c)
                        for j in ball_members[c]:
                            cnt[j] -= 1
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = code.copy()
                        improved = True
                        break

                    # Find candidates c' that cover all unique_words
                    # A candidate c' covers word w if w is in ball_members[c']
                    # So we need c' such that all unique_words are in ball_members[c']
                    # Equivalently, c' is within distance 1 of all unique_words
                    # Let's find all indices that are within distance 1 of all unique_words
                    # This is the intersection of the balls (in terms of centers)
                    # For each unique word w, the set of centers within distance 1 is ball_centers[w]
                    # We need the intersection of these sets
                    # Precompute: for each word w, which centers cover it? That's just ball_members inverted.
                    # Actually, let's think differently:
                    # We want c' such that for all w in unique_words, dist(c', w) <= 1
                    # This means c' must be in the intersection of balls centered at each w
                    # But we don't have precomputed "which centers cover word w"
                    # Let's compute it on the fly for the few unique words
                    cand_set = None
                    for w in unique_words:
                        # Find all centers c' such that w is in ball_members[c']
                        # i.e., dist(c', w) <= 1
                        # We can iterate over all possible c' but that's expensive
                        # Better: precompute reverse mapping
                        # Let's precompute: reverse_ball[w] = set of centers c' such that w in ball_members[c']
                        # This is the same as ball_members but inverted
                        if cand_set is None:
                            cand_set = set()
                            for c_prime in range(N):
                                if w in ball_members[c_prime]:
                                    cand_set.add(c_prime)
                        else:
                            new_cand_set = set()
                            for c_prime in range(N):
                                if w in ball_members[c_prime] and c_prime in cand_set:
                                    new_cand_set.add(c_prime)
                            cand_set = new_cand_set
                        if not cand_set:
                            break

                    if not cand_set:
                        continue

                    # Filter out candidates already in code
                    candidates = [c_prime for c_prime in cand_set if c_prime not in code_set]

                    if not candidates:
                        continue

                    # Pick a random candidate
                    c_prime = rng.choice(candidates)

                    # Perform swap
                    code.remove(c)
                    code_set.remove(c)
                    for j in ball_members[c]:
                        cnt[j] -= 1

                    code.append(c_prime)
                    code_set.add(c_prime)
                    for j in ball_members[c_prime]:
                        cnt[j] += 1

                    improved = True

            # Phase 3: 2-for-1 Swap (Merge two codewords into one)
            # Sample pairs to avoid O(N^2) per iteration
            if len(code) > 2:
                pair_swap_found = False
                num_pairs_to_check = 200
                for _ in range(num_pairs_to_check):
                    if time.time() > deadline - 2.0:
                        break
                    c1, c2 = rng.sample(code, 2)
                    
                    # Collect unique words covered by c1 or c2
                    unique_words = set()
                    for j in ball_members[c1]:
                        if cnt[j] == 1:
                            unique_words.add(j)
                    for j in ball_members[c2]:
                        if cnt[j] == 1:
                            unique_words.add(j)

                    if not unique_words:
                        continue

                    # Find candidates c' that cover all unique_words
                    cand_set = None
                    for w in unique_words:
                        # Find all centers c' such that w is in ball_members[c']
                        if cand_set is None:
                            cand_set = set()
                            for c_prime in range(N):
                                if w in ball_members[c_prime]:
                                    cand_set.add(c_prime)
                        else:
                            new_cand_set = set()
                            for c_prime in range(N):
                                if w in ball_members[c_prime] and c_prime in cand_set:
                                    new_cand_set.add(c_prime)
                            cand_set = new_cand_set
                        if not cand_set:
                            break

                    if not cand_set:
                        continue

                    # Check if there is a candidate not in code
                    candidates = [c_prime for c_prime in cand_set if c_prime not in code_set]

                    if candidates:
                        c_new = rng.choice(candidates)
                        
                        # Perform 2-for-1 swap
                        code.remove(c1)
                        code_set.remove(c1)
                        for j in ball_members[c1]:
                            cnt[j] -= 1

                        code.remove(c2)
                        code_set.remove(c2)
                        for j in ball_members[c2]:
                            cnt[j] -= 1

                        code.append(c_new)
                        code_set.add(c_new)
                        for j in ball_members[c_new]:
                            cnt[j] += 1

                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = code.copy()
                        improved = True
                        pair_swap_found = True
                        break # Restart main loop to re-evaluate deletions

                if pair_swap_found:
                    no_improve_count = 0
                    continue

            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Phase 4: Perturbation (Iterated Local Search)
            if no_improve_count >= max_no_improve:
                # Perturb: Remove 3-4 random codewords
                num_remove = rng.randint(3, 4)
                if len(code) > num_remove:
                    to_perturb = rng.sample(code, num_remove)
                    for c in to_perturb:
                        code.remove(c)
                        code_set.remove(c)
                        for j in ball_members[c]:
                            cnt[j] -= 1

                    # Repair: Greedily add codewords to cover any uncovered words
                    uncovered = cnt == 0

                    # Greedy repair
                    while uncovered.any() and time.time() < deadline - 2.0:
                        best_idx = -1
                        best_count = -1
                        # Randomized scan
                        candidates = list(range(N))
                        rng.shuffle(candidates)
                        for i in candidates:
                            if i in code_set:
                                continue
                            cnt_cover = sum(1 for j in ball_members[i] if uncovered[j])
                            if cnt_cover > best_count:
                                best_count = cnt_cover
                                best_idx = i

                        if best_idx == -1 or best_count == 0:
                            break

                        code.append(best_idx)
                        code_set.add(best_idx)
                        for j in ball_members[best_idx]:
                            cnt[j] += 1
                        uncovered[ball_members[best_idx]] = False

                    no_improve_count = 0

        return best_code

    # Precompute reverse_ball: for each word w, list of centers c' such that w in ball_members[c']
    reverse_ball = [None] * N
    for c_prime in range(N):
        for w in ball_members[c_prime]:
            if reverse_ball[w] is None:
                reverse_ball[w] = []
            reverse_ball[w].append(c_prime)

    # Main Loop: Multi-start
    best_solution = None
    best_size = float('inf')

    seed = 42
    while time.time() < deadline - 2.0:
        # Get initial greedy solution
        init_code = greedy_init(seed)

        if not init_code:
            break

        # Refine with local search
        refined_code = local_search(init_code, seed)

        if len(refined_code) < best_size:
            best_size = len(refined_code)
            best_solution = refined_code.copy()

        seed += 1

        # Early exit if we hit the theoretical lower bound?
        if best_size <= 156:
            break

    if best_solution is None:
        return []

    # Convert indices to digit lists
    result = []
    for idx in best_solution:
        digits = [0] * 7
        tmp = idx
        for pos in range(7):
            digits[pos] = tmp % 3
            tmp //= 3
        result.append(digits)

    return result

def _construct_general(q, n, radius, deadline):
    # Simple fallback for other parameters
    all_words = []
    for i in range(q**n):
        digits = [0] * n
        tmp = i
        for pos in range(n):
            digits[pos] = tmp % q
            tmp //= q
        all_words.append(tuple(digits))

    balls = []
    for w in all_words:
        ball = set()
        for v in all_words:
            dist = sum(1 for a, b in zip(w, v) if a != b)
            if dist <= radius:
                ball.add(v)
        balls.append(ball)

    covered = set()
    selected = []
    remaining = set(all_words)

    while remaining:
        best_idx = -1
        best_count = -1
        for i, w in enumerate(all_words):
            if w in covered:
                continue
            count = len(balls[i] & remaining)
            if count > best_count:
                best_count = count
                best_idx = i

        if best_idx == -1:
            break

        selected.append(all_words[best_idx])
        covered.update(balls[best_idx])
        remaining -= set(balls[best_idx])

        if time.time() > deadline - 5.0:
            break

    return [list(w) for w in selected]
# EVOLVE-BLOCK-END
