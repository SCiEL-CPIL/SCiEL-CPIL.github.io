import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 50.0  # Safety margin for 60s kill

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

    def can_add_fast(prefix_sums, used_sums, x, last_val):
        """Check if x can be appended given current prefix_sums and used_sums.
        Returns (new_prefix, new_sums) if valid, else None."""
        if x <= last_val or x > 256:
            return None
        new_prefix = prefix_sums[-1] + x
        new_sums = []
        for p in prefix_sums:
            s = new_prefix - p
            if s in used_sums:
                return None
            new_sums.append(s)
        # Check new_sums are all distinct
        seen = set()
        for s in new_sums:
            if s in seen:
                return None
            seen.add(s)
        return (new_prefix, new_sums)

    def greedy_smallest(start_val=1):
        """Greedy: always pick the smallest valid next element, using fast can_add"""
        seq = []
        prefix_sums = [0]
        used_sums = set()
        last_val = 0
        for _ in range(256):
            found = False
            for x in range(start_val, 257):
                result = can_add_fast(prefix_sums, used_sums, x, last_val)
                if result is not None:
                    new_prefix, new_sums = result
                    seq.append(x)
                    prefix_sums.append(new_prefix)
                    used_sums.update(new_sums)
                    last_val = x
                    start_val = x + 1
                    found = True
                    break
            if not found:
                break
        return seq

    def greedy_with_lookahead(start_val=1, lookahead_depth=2):
        """Greedy with lookahead: at each step, try candidates and check if they lead to dead-end soon."""
        seq = []
        prefix_sums = [0]
        used_sums = set()
        last_val = 0
        
        def try_extend(prefix_sums, used_sums, last_val, depth, max_depth):
            """Try to extend the sequence by max_depth more elements. Returns count of extensions possible."""
            if depth >= max_depth:
                return 0
            count = 0
            for x in range(last_val + 1, 257):
                result = can_add_fast(prefix_sums, used_sums, x, last_val)
                if result is not None:
                    new_prefix, new_sums = result
                    prefix_sums.append(new_prefix)
                    used_sums.update(new_sums)
                    count += 1 + try_extend(prefix_sums, used_sums, x, depth + 1, max_depth)
                    used_sums.difference_update(new_sums)
                    prefix_sums.pop()
                    if count >= 3:  # Early exit if enough options
                        return count
            return count
        
        for _ in range(256):
            best_x = None
            best_score = -1
            for x in range(start_val, 257):
                result = can_add_fast(prefix_sums, used_sums, x, last_val)
                if result is not None:
                    new_prefix, new_sums = result
                    # Lookahead: check how many options remain after picking x
                    prefix_sums.append(new_prefix)
                    used_sums.update(new_sums)
                    score = try_extend(prefix_sums, used_sums, x, 0, lookahead_depth)
                    used_sums.difference_update(new_sums)
                    prefix_sums.pop()
                    # Prefer smaller x but also consider future flexibility
                    combined = score * 10 - x * 0.1
                    if combined > best_score:
                        best_score = combined
                        best_x = x
                        best_result = result
            if best_x is None:
                break
            new_prefix, new_sums = best_result
            seq.append(best_x)
            prefix_sums.append(new_prefix)
            used_sums.update(new_sums)
            last_val = best_x
            start_val = best_x + 1
        return seq

    def greedy_randomized(start_val=1, seed_offset=0):
        """Randomized greedy: at each step, pick from valid candidates with bias"""
        seq = []
        prefix_sums = [0]
        used_sums = set()
        last_val = 0
        rng = random.Random(seed_offset)
        for _ in range(256):
            valid_candidates = []
            for x in range(start_val, 257):
                result = can_add_fast(prefix_sums, used_sums, x, last_val)
                if result is not None:
                    valid_candidates.append((x, result))
            if not valid_candidates:
                break
            if len(valid_candidates) <= 3:
                x, (new_prefix, new_sums) = valid_candidates[0]
            else:
                weights = [1.0 / (i + 1) for i in range(len(valid_candidates))]
                total_w = sum(weights)
                r = rng.random() * total_w
                cum = 0
                chosen = 0
                for i, w in enumerate(weights):
                    cum += w
                    if r <= cum:
                        chosen = i
                        break
                x, (new_prefix, new_sums) = valid_candidates[chosen]
            seq.append(x)
            prefix_sums.append(new_prefix)
            used_sums.update(new_sums)
            last_val = x
            start_val = x + 1
        return seq

    def csp_backtrack(target_len, max_time=5.0):
        """CSP-style backtracking with forward checking and MRV heuristic."""
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

            # Collect all valid candidates
            valid_candidates = []
            for x in range(last_val + 1, 257):
                new_prefix = prefix_sums[-1] + x
                new_sums = []
                valid = True
                for p in prefix_sums:
                    s = new_prefix - p
                    if s in used_sums:
                        valid = False
                        break
                    new_sums.append(s)
                if not valid:
                    continue
                seen = set()
                for s in new_sums:
                    if s in seen:
                        valid = False
                        break
                    seen.add(s)
                if not valid:
                    continue
                valid_candidates.append((x, new_prefix, new_sums))

            if not valid_candidates:
                return

            # MRV: try candidates that leave the fewest options next (most constraining first)
            # Approximate: prefer smaller x (they constrain less) but also check immediate next step
            # For speed, just try in order but with a slight preference for values that allow continuation
            for x, new_prefix, new_sums in valid_candidates:
                if found[0] or time.time() > deadline:
                    return

                # Quick forward check: is there at least one valid next value?
                has_next = False
                for nx in range(x + 1, min(x + 20, 257)):  # Check only a few
                    np = new_prefix + nx
                    valid_next = True
                    for p in prefix_sums + [new_prefix]:
                        s = np - p
                        if s in used_sums or s in new_sums:
                            valid_next = False
                            break
                    if valid_next:
                        has_next = True
                        break

                if not has_next and depth < target_len - 1:
                    # Dead end, skip
                    continue

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

    def csp_backtrack_from(seq, target_len, max_time=5.0):
        """CSP backtracking starting from an existing prefix sequence."""
        if len(seq) >= target_len:
            return list(seq)
        
        prefix_sums = [0]
        for v in seq:
            prefix_sums.append(prefix_sums[-1] + v)
        used_sums = set()
        for i in range(len(seq)):
            for j in range(i+1, len(seq)+1):
                used_sums.add(prefix_sums[j] - prefix_sums[i])
        
        deadline = start_time + max_time
        found = [False]
        last_val = seq[-1] if seq else 0
        current_seq = list(seq)

        def dfs(lv, depth):
            if found[0]:
                return
            if time.time() > deadline:
                return
            if depth == target_len:
                found[0] = True
                return

            for x in range(lv + 1, 257):
                if found[0] or time.time() > deadline:
                    return

                new_prefix = prefix_sums[-1] + x
                new_sums = []
                valid = True
                for p in prefix_sums:
                    s = new_prefix - p
                    if s in used_sums:
                        valid = False
                        break
                    new_sums.append(s)
                if not valid:
                    continue
                seen = set()
                for s in new_sums:
                    if s in seen:
                        valid = False
                        break
                    seen.add(s)
                if not valid:
                    continue

                current_seq.append(x)
                prefix_sums.append(new_prefix)
                used_sums.update(new_sums)

                dfs(x, depth + 1)

                current_seq.pop()
                prefix_sums.pop()
                for s in new_sums:
                    used_sums.discard(s)

                if found[0]:
                    return

        dfs(last_val, len(seq))
        if found[0]:
            return list(current_seq)
        return None

    # Strategy 1: Pure greedy smallest (fast version)
    greedy_result = greedy_smallest()
    best_seq = list(greedy_result)
    best_len = len(best_seq)

    # Strategy 2: Try different starting points with fast greedy
    for start in range(1, 30):
        if time.time() - start_time > time_limit - 20:
            break
        seq = greedy_smallest(start)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = list(seq)

    # Strategy 3: Greedy with lookahead
    for start in range(1, 15):
        if time.time() - start_time > time_limit - 15:
            break
        seq = greedy_with_lookahead(start_val=start, lookahead_depth=2)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = list(seq)

    # Strategy 4: Many randomized greedy attempts (fast)
    for attempt in range(40):
        if time.time() - start_time > time_limit - 12:
            break
        seq = greedy_randomized(seed_offset=attempt * 7 + 1)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = list(seq)

    # Strategy 5: CSP backtracking to extend beyond greedy
    # Try to find sequences of length best_len + 1, +2, etc.
    for target in range(best_len + 1, min(best_len + 10, 40)):
        if time.time() - start_time > time_limit - 3:
            break
        remaining_time = min(4.0, time_limit - 3 - (time.time() - start_time))
        if remaining_time < 0.5:
            break
        # Try extending from best_seq prefix
        result = csp_backtrack_from(best_seq, target, max_time=remaining_time)
        if result is not None and len(result) > best_len:
            best_len = len(result)
            best_seq = list(result)
            continue
        # Also try fresh backtracking
        remaining_time = min(3.0, time_limit - 3 - (time.time() - start_time))
        if remaining_time < 0.5:
            break
        result = csp_backtrack(target, max_time=remaining_time)
        if result is not None and len(result) > best_len:
            best_len = len(result)
            best_seq = list(result)

    # Strategy 6: More randomized attempts with different biases
    for attempt in range(20):
        if time.time() - start_time > time_limit - 2:
            break
        start_val = random.randint(1, 20)
        seq = greedy_randomized(start_val=start_val, seed_offset=1000 + attempt * 13)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = list(seq)

    # Final validation
    if not validate_seq(best_seq):
        best_seq = [1, 2, 4, 8, 16, 32, 64, 128]
        assert validate_seq(best_seq), "Fallback sequence invalid"

    return best_seq
