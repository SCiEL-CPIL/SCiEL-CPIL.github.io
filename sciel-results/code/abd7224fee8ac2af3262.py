import itertools
import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin before 120s limit

    total_words = q ** n

    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Precompute balls (all words within radius of each word)
    balls = [[] for _ in range(total_words)]

    if radius == 1:
        for i, w in enumerate(all_words):
            balls[i] = [i]
            for pos in range(n):
                for new_val in range(q):
                    if new_val == w[pos]:
                        continue
                    new_word = list(w)
                    new_word[pos] = new_val
                    idx = word_to_idx[tuple(new_word)]
                    balls[i].append(idx)
    else:
        for i, w in enumerate(all_words):
            for j, v in enumerate(all_words):
                dist = sum(1 for a, b in zip(w, v) if a != b)
                if dist <= radius:
                    balls[i].append(j)

    # Precompute ball sizes for quick access
    ball_sizes = [len(b) for b in balls]

    # --- Fast coverage helpers ---

    def get_coverage(code_indices):
        cov = [0] * total_words
        for idx in code_indices:
            for nb in balls[idx]:
                cov[nb] += 1
        return cov

    def is_valid_cov(cov):
        # Check all covered
        for c in cov:
            if c == 0:
                return False
        return True

    def count_uncovered(cov):
        cnt = 0
        for c in cov:
            if c == 0:
                cnt += 1
        return cnt

    def find_first_uncovered(cov):
        for i in range(total_words):
            if cov[i] == 0:
                return i
        return -1

    # --- Fast minimize: avoid list copies, use index set ---

    def minimize(code_indices):
        if not code_indices:
            return code_indices
        # Work with a list and track removals
        current = list(code_indices)
        cov = get_coverage(current)
        # Use a set for O(1) membership
        current_set = set(current)
        # Repeatedly try to remove
        changed = True
        while changed:
            changed = False
            # Try to find any removable codeword
            for i in range(len(current)):
                idx = current[i]
                # Check if removing idx keeps coverage
                removable = True
                for nb in balls[idx]:
                    if cov[nb] <= 1:
                        removable = False
                        break
                if removable:
                    # Remove it
                    for nb in balls[idx]:
                        cov[nb] -= 1
                    current.pop(i)
                    current_set.discard(idx)
                    changed = True
                    break
        return current

    # --- Greedy set cover ---

    def greedy_cover():
        uncovered = bytearray([1]) * total_words
        code_indices = []
        remaining = total_words
        while remaining > 0:
            best_idx = -1
            best_count = 0
            for i in range(total_words):
                count = 0
                for nb in balls[i]:
                    if uncovered[nb]:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_idx = i
            if best_idx == -1 or best_count == 0:
                break
            code_indices.append(best_idx)
            for nb in balls[best_idx]:
                if uncovered[nb]:
                    uncovered[nb] = 0
                    remaining -= 1
        return code_indices

    # --- Randomized greedy ---

    def randomized_greedy():
        uncovered = bytearray([1]) * total_words
        code_indices = []
        remaining = total_words
        while remaining > 0:
            # Compute coverage counts for all candidates
            counts = [0] * total_words
            for i in range(total_words):
                cnt = 0
                for nb in balls[i]:
                    if uncovered[nb]:
                        cnt += 1
                counts[i] = cnt
            max_count = max(counts)
            if max_count == 0:
                break
            # Pick randomly among top candidates
            candidates = [i for i, c in enumerate(counts) if c == max_count]
            best_idx = random.choice(candidates)
            code_indices.append(best_idx)
            for nb in balls[best_idx]:
                if uncovered[nb]:
                    uncovered[nb] = 0
                    remaining -= 1
        return code_indices

    # --- Local search: swap-based with periodic minimize ---

    def local_search(initial_code, time_limit):
        current = minimize(initial_code[:])
        if not current:
            return current
        cov = get_coverage(current)
        current_set = set(current)
        best = current[:]
        best_size = len(best)

        step = 0
        while time.time() < time_limit:
            if len(current) == 0:
                break

            step += 1

            # Periodic minimize to keep code tight
            if step % 50 == 0:
                current = minimize(current)
                if not current:
                    break
                cov = get_coverage(current)
                current_set = set(current)
                if len(current) < best_size:
                    best = current[:]
                    best_size = len(best)
                continue

            i = random.randint(0, len(current) - 1)
            old = current[i]

            # Choose new codeword: with some probability target an uncovered area
            # by picking a random word and checking
            new = random.randint(0, total_words - 1)

            if new in current_set:
                continue

            # Compute tentative coverage
            test_cov = cov[:]
            for nb in balls[old]:
                test_cov[nb] -= 1
            for nb in balls[new]:
                test_cov[nb] += 1

            if not is_valid_cov(test_cov):
                continue

            # Accept the swap
            current[i] = new
            current_set.discard(old)
            current_set.add(new)
            cov = test_cov

            # After a few swaps, try minimize to see if we can shrink
            if step % 10 == 0:
                tent = minimize(current[:])
                if len(tent) < len(current):
                    current = tent
                    cov = get_coverage(current)
                    current_set = set(current)
                    if len(current) < best_size:
                        best = current[:]
                        best_size = len(best)
                else:
                    # Revert to pre-minimize state if it didn't help
                    pass

        # Final minimize
        current = minimize(current)
        if len(current) < best_size:
            best = current[:]
            best_size = len(best)

        return best

    # --- Local search with targeted replacement ---
    # Try replacing a codeword with one that covers more currently-undercovered areas

    def local_search_targeted(initial_code, time_limit):
        current = minimize(initial_code[:])
        if not current:
            return current
        cov = get_coverage(current)
        current_set = set(current)
        best = current[:]
        best_size = len(best)

        step = 0
        while time.time() < time_limit:
            if len(current) == 0:
                break

            step += 1

            if step % 50 == 0:
                current = minimize(current)
                if not current:
                    break
                cov = get_coverage(current)
                current_set = set(current)
                if len(current) < best_size:
                    best = current[:]
                    best_size = len(best)
                continue

            i = random.randint(0, len(current) - 1)
            old = current[i]

            # Find a good replacement: try a few random candidates
            accepted = False
            for attempt in range(3):
                new = random.randint(0, total_words - 1)
                if new in current_set:
                    continue

                test_cov = cov[:]
                for nb in balls[old]:
                    test_cov[nb] -= 1
                for nb in balls[new]:
                    test_cov[nb] += 1

                if not is_valid_cov(test_cov):
                    continue

                # Accept
                current[i] = new
                current_set.discard(old)
                current_set.add(new)
                cov = test_cov
                accepted = True
                break

            if not accepted:
                continue

            if step % 10 == 0:
                tent = minimize(current[:])
                if len(tent) < len(current):
                    current = tent
                    cov = get_coverage(current)
                    current_set = set(current)
                    if len(current) < best_size:
                        best = current[:]
                        best_size = len(best)

        current = minimize(current)
        if len(current) < best_size:
            best = current[:]
            best_size = len(best)

        return best

    # --- Main strategy ---

    # Strategy 1: Greedy
    code1 = greedy_cover()
    code1 = minimize(code1)
    best_code = code1
    best_size = len(best_code)

    # Strategy 2: Randomized greedy (multiple runs)
    remaining_time = deadline - time.time()
    num_runs = max(1, int(remaining_time / 3.0))

    for run in range(num_runs):
        if time.time() > deadline - 8.0:
            break
        code = randomized_greedy()
        code = minimize(code)
        if len(code) < best_size:
            best_code = code
            best_size = len(best_code)

    # Strategy 3: Local search from best
    remaining = deadline - time.time()
    if remaining > 5.0:
        ls_result = local_search(best_code, deadline - 2.0)
        if len(ls_result) < best_size:
            best_code = ls_result
            best_size = len(best_code)

    # Strategy 4: Targeted local search
    remaining = deadline - time.time()
    if remaining > 5.0:
        ls_result2 = local_search_targeted(best_code, deadline - 1.0)
        if len(ls_result2) < best_size:
            best_code = ls_result2
            best_size = len(best_code)

    # Final minimize
    best_code = minimize(best_code)

    # Convert to words
    return [list(all_words[idx]) for idx in best_code]
