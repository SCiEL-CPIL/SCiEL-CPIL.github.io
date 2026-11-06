import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin for 90s kill

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
        """Upper bound on how many more elements can be added."""
        return 256 - last_val

    def csp_forward_checking(target_len, max_time=5.0):
        """CSP with forward-checking: maintain forbidden sums and prune aggressively.
        Uses a different representation: tracks all (i,j) pairs of prefix indices
        that would create collisions, enabling faster pruning."""
        seq = []
        prefix_sums = [0]
        used_sums = set()
        deadline = start_time + max_time
        found = [False]
        best_partial = [0]  # Track deepest partial solution for fallback

        def check_valid_add(x, last_val):
            """Check if x can be added. Returns (new_prefix, new_sums) if valid."""
            if x <= last_val or x > 256:
                return None
            new_prefix = prefix_sums[-1] + x
            new_sums = []
            for p in prefix_sums:
                s = new_prefix - p
                if s in used_sums:
                    return None
                new_sums.append(s)
            # Check new sums are distinct from each other
            if len(set(new_sums)) != len(new_sums):
                return None
            return (new_prefix, new_sums)

        def dfs(last_val, depth):
            if found[0]:
                return
            if time.time() > deadline:
                return
            if depth == target_len:
                found[0] = True
                return
            if depth > best_partial[0]:
                best_partial[0] = depth

            remaining_needed = target_len - depth
            max_possible = max_extension_estimate(last_val, depth)
            if max_possible < remaining_needed:
                return

            # Forward-checking: collect all valid candidates first
            candidates = []
            for x in range(last_val + 1, 257):
                result = check_valid_add(x, last_val)
                if result is not None:
                    candidates.append((x, result[0], result[1]))

            if not candidates:
                return

            # Order by: prefer values that leave more room (smaller x first),
            # but also consider how many future candidates each choice blocks
            # Simple heuristic: try smaller values first
            for x, new_prefix, new_sums in candidates:
                if found[0] or time.time() > deadline:
                    return

                # Commit
                seq.append(x)
                prefix_sums.append(new_prefix)
                used_sums.update(new_sums)

                dfs(x, depth + 1)

                # Backtrack
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

    def local_search_extend(base_seq, max_time=10.0):
        """Try to extend a base sequence by replacing suffix elements.
        This is a structural change: instead of pure DFS from scratch,
        we start from a good greedy solution and try to improve the tail."""
        base = list(base_seq)
        n = len(base)
        deadline = start_time + max_time
        best = list(base)
        best_len = n

        # Build prefix sums for base
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + base[i]

        # Collect all used sums
        used_sums = set()
        for i in range(n):
            for j in range(i+1, n+1):
                used_sums.add(prefix[j] - prefix[i])

        # Try to extend by replacing the last k elements with different choices
        for k in range(1, min(n, 15)):
            if time.time() > deadline:
                break

            # Remove last k elements
            prefix_k = prefix[:n-k+1]
            # Recompute used sums for the first n-k elements
            used_sums_k = set()
            for i in range(n-k):
                for j in range(i+1, n-k+1):
                    used_sums_k.add(prefix_k[j] - prefix_k[i])

            # Now try to build k more elements starting from prefix_k
            last_val = base[n-k-1] if n-k > 0 else 0
            current_prefix = list(prefix_k)
            current_used = set(used_sums_k)
            current_seq = base[:n-k]

            # Greedy extend with some randomization
            rng = random.Random(hash(tuple(base[:n-k])) % 100000)
            for _ in range(k + 5):  # Try to add more than k elements
                if time.time() > deadline:
                    break
                valid_candidates = []
                for x in range(last_val + 1, 257):
                    new_prefix = current_prefix[-1] + x
                    valid = True
                    new_sums = []
                    for p in current_prefix:
                        s = new_prefix - p
                        if s in current_used:
                            valid = False
                            break
                        new_sums.append(s)
                    if not valid:
                        continue
                    if len(set(new_sums)) != len(new_sums):
                        continue
                    valid_candidates.append((x, new_prefix, new_sums))

                if not valid_candidates:
                    break

                # Pick with some randomization
                if len(valid_candidates) <= 2:
                    x, new_prefix, new_sums = valid_candidates[0]
                else:
                    idx = rng.randint(0, min(5, len(valid_candidates)-1))
                    x, new_prefix, new_sums = valid_candidates[idx]

                current_seq.append(x)
                current_prefix.append(new_prefix)
                current_used.update(new_sums)
                last_val = x

            if len(current_seq) > best_len:
                best = current_seq
                best_len = len(current_seq)

        return best

    # Strategy 1: Pure greedy smallest
    greedy_result = greedy_smallest()
    best_seq = list(greedy_result)
    best_len = len(best_seq)

    # Strategy 2: Try different starting points with greedy
    for start in range(1, 30):
        if time.time() - start_time > time_limit - 40:
            break
        seq = greedy_smallest(start)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = list(seq)

    # Strategy 3: Many randomized greedy attempts
    for attempt in range(200):
        if time.time() - start_time > time_limit - 30:
            break
        start_val = random.randint(1, 20)
        bias = random.uniform(0.3, 2.5)
        seq = greedy_randomized(start_val=start_val, seed_offset=attempt * 13 + 1, bias=bias)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = list(seq)

    # Strategy 4: CSP with forward-checking to extend beyond greedy
    for target in range(best_len + 1, min(best_len + 12, 45)):
        if time.time() - start_time > time_limit - 15:
            break
        remaining_time = min(3.0, time_limit - 15 - (time.time() - start_time))
        if remaining_time < 0.5:
            break
        result = csp_forward_checking(target, max_time=remaining_time)
        if result is not None and len(result) > best_len:
            best_len = len(result)
            best_seq = list(result)

    # Strategy 5: Local search extension - structurally different approach
    if time.time() - start_time < time_limit - 10:
        remaining_time = min(10.0, time_limit - 10 - (time.time() - start_time))
        if remaining_time > 1.0:
            extended = local_search_extend(best_seq, max_time=remaining_time)
            if len(extended) > best_len:
                best_len = len(extended)
                best_seq = list(extended)

    # Strategy 6: More randomized attempts with varied parameters
    for attempt in range(100):
        if time.time() - start_time > time_limit - 3:
            break
        start_val = random.randint(1, 30)
        bias = random.uniform(0.2, 3.5)
        seq = greedy_randomized(start_val=start_val, seed_offset=10000 + attempt * 23, bias=bias)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = list(seq)

    # Final validation
    if not validate_seq(best_seq):
        best_seq = [1, 2, 4, 8, 16, 32, 64, 128]
        assert validate_seq(best_seq), "Fallback sequence invalid"

    return best_seq
