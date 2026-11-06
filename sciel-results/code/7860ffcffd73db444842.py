import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 80.0  # Safety margin for 90s kill

    def validate_seq(seq):
        """Validate a sequence: strictly increasing, in [1,256], all consecutive block sums distinct"""
        n = len(seq)
        if n == 0:
            return True
        for i in range(n):
            if seq[i] < 1 or seq[i] > 256:
                return False
            if i > 0 and seq[i] <= seq[i-1]:
                return False
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + seq[i]
        sums = set()
        for i in range(n):
            for j in range(i+1, n+1):
                s = prefix[j] - prefix[i]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def greedy_smallest(start_val=1):
        """Greedy: always pick the smallest valid next element"""
        seq = []
        prefix_sums = [0]
        used_sums = set()
        last_val = 0
        for _ in range(256):
            found = False
            for x in range(start_val, 257):
                if x <= last_val or x > 256:
                    continue
                new_prefix = prefix_sums[-1] + x
                valid = True
                for p in prefix_sums:
                    s = new_prefix - p
                    if s in used_sums:
                        valid = False
                        break
                if not valid:
                    continue
                # Check new sums for internal collisions
                seen = set()
                for p in prefix_sums:
                    s = new_prefix - p
                    if s in seen:
                        valid = False
                        break
                    seen.add(s)
                if not valid:
                    continue
                seq.append(x)
                prefix_sums.append(new_prefix)
                used_sums.update(seen)
                last_val = x
                start_val = x + 1
                found = True
                break
            if not found:
                break
        return seq

    def greedy_randomized(start_val=1, seed_offset=0, bias=1.0):
        """Randomized greedy with weighted selection"""
        seq = []
        prefix_sums = [0]
        used_sums = set()
        last_val = 0
        rng = random.Random(seed_offset)
        for _ in range(256):
            valid_candidates = []
            for x in range(start_val, 257):
                if x <= last_val or x > 256:
                    continue
                new_prefix = prefix_sums[-1] + x
                valid = True
                for p in prefix_sums:
                    s = new_prefix - p
                    if s in used_sums:
                        valid = False
                        break
                if not valid:
                    continue
                seen = set()
                for p in prefix_sums:
                    s = new_prefix - p
                    if s in seen:
                        valid = False
                        break
                    seen.add(s)
                if not valid:
                    continue
                valid_candidates.append((x, new_prefix, seen))
            if not valid_candidates:
                break
            if len(valid_candidates) <= 2:
                x, new_prefix, new_sums = valid_candidates[0]
            else:
                weights = [1.0 / ((i + 1) ** bias) for i in range(len(valid_candidates))]
                total_w = sum(weights)
                r = rng.random() * total_w
                cum = 0
                chosen = 0
                for i, w in enumerate(weights):
                    cum += w
                    if r <= cum:
                        chosen = i
                        break
                x, new_prefix, new_sums = valid_candidates[chosen]
            seq.append(x)
            prefix_sums.append(new_prefix)
            used_sums.update(new_sums)
            last_val = x
            start_val = x + 1
        return seq

    def max_extension_estimate(last_val, current_len):
        """Upper bound on how many more elements can be added.
        Elements must be strictly increasing integers in (last_val, 256].
        So at most 256 - last_val more elements."""
        return 256 - last_val

    def iterative_deepening_dfs(target_len, max_time=5.0):
        """Iterative deepening DFS with pruning based on max extension heuristic."""
        seq = []
        prefix_sums = [0]
        used_sums = set()
        deadline = start_time + max_time
        found = [False]

        def dfs(last_val, depth):
            if found[0]:
                return
            if time.time() > deadline:
                return
            if depth == target_len:
                found[0] = True
                return

            # Pruning: if even with maximum possible extension we can't reach target
            remaining_needed = target_len - depth
            max_possible = max_extension_estimate(last_val, depth)
            if max_possible < remaining_needed:
                return

            for x in range(last_val + 1, 257):
                if found[0] or time.time() > deadline:
                    return

                new_prefix = prefix_sums[-1] + x
                valid = True
                for p in prefix_sums:
                    s = new_prefix - p
                    if s in used_sums:
                        valid = False
                        break
                if not valid:
                    continue

                seen = set()
                for p in prefix_sums:
                    s = new_prefix - p
                    if s in seen:
                        valid = False
                        break
                    seen.add(s)
                if not valid:
                    continue

                # Commit
                seq.append(x)
                prefix_sums.append(new_prefix)
                used_sums.update(seen)

                dfs(x, depth + 1)

                # Backtrack
                seq.pop()
                prefix_sums.pop()
                for s in seen:
                    used_sums.discard(s)

                if found[0]:
                    return

        dfs(0, 0)
        if found[0]:
            return list(seq)
        return None

    def csp_backtrack_with_ordering(target_len, max_time=5.0):
        """Backtracking with value ordering that tries to maximize future flexibility.
        Tries smaller values first (to leave room) but with some randomness."""
        seq = []
        prefix_sums = [0]
        used_sums = set()
        deadline = start_time + max_time
        found = [False]
        rng = random.Random(42)

        def dfs(last_val, depth):
            if found[0]:
                return
            if time.time() > deadline:
                return
            if depth == target_len:
                found[0] = True
                return

            remaining_needed = target_len - depth
            max_possible = max_extension_estimate(last_val, depth)
            if max_possible < remaining_needed:
                return

            # Generate candidates
            candidates = []
            for x in range(last_val + 1, 257):
                new_prefix = prefix_sums[-1] + x
                valid = True
                for p in prefix_sums:
                    s = new_prefix - p
                    if s in used_sums:
                        valid = False
                        break
                if not valid:
                    continue
                seen = set()
                for p in prefix_sums:
                    s = new_prefix - p
                    if s in seen:
                        valid = False
                        break
                    seen.add(s)
                if not valid:
                    continue
                candidates.append((x, new_prefix, seen))

            if not candidates:
                return

            # Order candidates: prefer smaller values (more room), but shuffle slightly
            # to avoid getting stuck in one pattern
            if depth % 3 == 0:
                # Slight randomization
                for i in range(len(candidates) - 1, 0, -1):
                    j = rng.randint(0, i)
                    candidates[i], candidates[j] = candidates[j], candidates[i]

            for x, new_prefix, new_sums in candidates:
                if found[0] or time.time() > deadline:
                    return

                seq.append(x)
                prefix_sums.append(new_prefix)
                used_sums.update(new_sums)

                dfs(x, depth + 1)

                seq.pop()
                prefix_sums.pop()
                for s in new_sums:
                    used_sums.discard(s)

                if found[0]:
                    return

        dfs(0, 0)
        if found[0]:
            return list(seq)
        return None

    # Strategy 1: Pure greedy smallest
    greedy_result = greedy_smallest()
    best_seq = list(greedy_result)
    best_len = len(best_seq)

    # Strategy 2: Try different starting points with greedy
    for start in range(1, 20):
        if time.time() - start_time > time_limit - 20:
            break
        seq = greedy_smallest(start)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = list(seq)

    # Strategy 3: Many randomized greedy attempts
    for attempt in range(100):
        if time.time() - start_time > time_limit - 30:
            break
        start_val = random.randint(1, 15)
        bias = random.uniform(0.5, 2.0)
        seq = greedy_randomized(start_val=start_val, seed_offset=attempt * 7 + 1, bias=bias)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = list(seq)

    # Strategy 4: Iterative deepening DFS to extend beyond greedy
    for target in range(best_len + 1, min(best_len + 10, 40)):
        if time.time() - start_time > time_limit - 10:
            break
        remaining_time = min(4.0, time_limit - 10 - (time.time() - start_time))
        if remaining_time < 0.5:
            break
        result = iterative_deepening_dfs(target, max_time=remaining_time)
        if result is not None and len(result) > best_len:
            best_len = len(result)
            best_seq = list(result)

    # Strategy 5: CSP backtracking with ordering
    for target in range(best_len + 1, min(best_len + 6, 40)):
        if time.time() - start_time > time_limit - 5:
            break
        remaining_time = min(3.0, time_limit - 5 - (time.time() - start_time))
        if remaining_time < 0.5:
            break
        result = csp_backtrack_with_ordering(target, max_time=remaining_time)
        if result is not None and len(result) > best_len:
            best_len = len(result)
            best_seq = list(result)

    # Strategy 6: More randomized attempts with varied parameters
    for attempt in range(50):
        if time.time() - start_time > time_limit - 3:
            break
        start_val = random.randint(1, 25)
        bias = random.uniform(0.3, 3.0)
        seq = greedy_randomized(start_val=start_val, seed_offset=5000 + attempt * 17, bias=bias)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = list(seq)

    # Final validation
    if not validate_seq(best_seq):
        best_seq = [1, 2, 4, 8, 16, 32, 64, 128]
        assert validate_seq(best_seq), "Fallback sequence invalid"

    return best_seq
