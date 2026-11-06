import time
import random

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin before 90s limit

    def check_valid(seq):
        """Check if all consecutive block sums are distinct"""
        n = len(seq)
        sums = set()
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.add(s)
                if len(sums) < (j - i + 1):
                    return False
        return len(sums) == n * (n + 1) // 2

    def get_all_sums(seq):
        n = len(seq)
        sums = set()
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.add(s)
        return sums

    def greedy_asc():
        vals = []
        sums_set = set()
        for x in range(1, 257):
            new_sums = set()
            total = x
            new_sums.add(total)
            for a in reversed(vals):
                total += a
                new_sums.add(total)
            if not sums_set.intersection(new_sums):
                vals.append(x)
                sums_set.update(new_sums)
        return vals

    def greedy_desc():
        vals = []
        sums_set = set()
        for x in range(256, 0, -1):
            new_sums = set()
            total = x
            new_sums.add(total)
            for a in reversed(vals):
                total += a
                new_sums.add(total)
            if not sums_set.intersection(new_sums):
                vals.append(x)
                sums_set.update(new_sums)
        vals.sort()
        return vals

    def greedy_desc_start(start):
        vals = []
        sums_set = set()
        for x in range(start, 0, -1):
            new_sums = set()
            total = x
            new_sums.add(total)
            for a in reversed(vals):
                total += a
                new_sums.add(total)
            if not sums_set.intersection(new_sums):
                vals.append(x)
                sums_set.update(new_sums)
        vals.sort()
        return vals

    # Try multiple greedy strategies
    candidates = []
    candidates.append(greedy_asc())
    candidates.append(greedy_desc())

    # Greedy descending with different starting points
    for start in range(256, 150, -1):
        candidates.append(greedy_desc_start(start))

    # Greedy ascending with different starting points
    for start in range(1, 30):
        vals = []
        sums_set = set()
        for x in range(start, 257):
            new_sums = set()
            total = x
            new_sums.add(total)
            for a in reversed(vals):
                total += a
                new_sums.add(total)
            if not sums_set.intersection(new_sums):
                vals.append(x)
                sums_set.update(new_sums)
        candidates.append(vals)

    # Also try some random greedy variants
    for _ in range(50):
        if time.time() - start_time > time_limit:
            break
        vals = []
        sums_set = set()
        remaining = list(range(1, 257))
        random.shuffle(remaining)
        for x in remaining:
            new_sums = set()
            total = x
            new_sums.add(total)
            for a in reversed(vals):
                total += a
                new_sums.add(total)
            if not sums_set.intersection(new_sums):
                vals.append(x)
                sums_set.update(new_sums)
        vals.sort()
        candidates.append(vals)

    best_seq = max(candidates, key=len)
    best_len = len(best_seq)

    # Try to extend the best sequence by inserting new elements
    def try_extend(seq):
        """Try to insert additional elements into the sequence"""
        current = list(seq)
        used = set(current)
        improved = True
        while improved:
            improved = False
            for pos in range(len(current) + 1):
                if pos == 0:
                    lo, hi = 1, current[0] - 1 if current else 256
                elif pos == len(current):
                    lo, hi = current[-1] + 1, 256
                else:
                    lo, hi = current[pos-1] + 1, current[pos] - 1

                if lo > hi:
                    continue

                # Try values from high to low (larger values create more distinct sums)
                for x in range(hi, lo - 1, -1):
                    if x in used:
                        continue
                    new_seq = current[:pos] + [x] + current[pos:]
                    new_sums = get_all_sums(new_seq)
                    if len(new_sums) == len(new_seq) * (len(new_seq) + 1) // 2:
                        current = new_seq
                        used.add(x)
                        improved = True
                        break
                if improved:
                    break
        return current

    # Try to extend
    extended = try_extend(best_seq)
    if len(extended) > best_len:
        best_seq = extended
        best_len = len(best_seq)

    # Local search: try swapping/removing elements to free up room
    def local_search(seq, max_iters=500):
        """Try random removals and re-insertions to find better configurations"""
        current = list(seq)
        current_len = len(current)

        for _ in range(max_iters):
            if time.time() - start_time > time_limit:
                break

            # Strategy 1: Remove a random element and try to reinsert elsewhere
            if len(current) < 3:
                break
            i = random.randint(0, len(current) - 1)
            val = current.pop(i)

            # Try to reinsert at a different position
            inserted = False
            positions = list(range(len(current) + 1))
            random.shuffle(positions)
            for pos in positions:
                if pos == i:
                    continue
                if pos == 0:
                    if current and current[0] <= val:
                        continue
                elif pos == len(current):
                    if current and current[-1] >= val:
                        continue
                else:
                    if current[pos-1] >= val or current[pos] <= val:
                        continue

                new_seq = current[:pos] + [val] + current[pos:]
                new_sums = get_all_sums(new_seq)
                if len(new_sums) == len(new_seq) * (len(new_seq) + 1) // 2:
                    current = new_seq
                    inserted = True
                    break

            if not inserted:
                current.insert(i, val)

            # Try to extend
            extended = try_extend(current)
            if len(extended) > current_len:
                return extended

            # Strategy 2: Remove two elements and try to reinsert
            if len(current) >= 4:
                i1 = random.randint(0, len(current) - 1)
                i2 = random.randint(0, len(current) - 1)
                if i1 == i2:
                    continue
                if i1 > i2:
                    i1, i2 = i2, i1

                v1 = current.pop(i1)
                v2 = current.pop(i2)

                inserted = False
                # Try inserting both back
                for pos1 in range(len(current) + 1):
                    if pos1 == 0:
                        if current and current[0] <= min(v1, v2):
                            continue
                    elif pos1 == len(current):
                        if current and current[-1] >= max(v1, v2):
                            continue
                    else:
                        if current[pos1-1] >= min(v1, v2) or current[pos1] <= max(v1, v2):
                            continue

                    for pos2 in range(pos1 + 1, len(current) + 2):
                        # Determine valid range for second insertion
                        temp = current[:pos1] + [v1] + current[pos1:]
                        if pos2 == 0:
                            continue
                        elif pos2 == len(temp):
                            if temp[-1] >= v2:
                                continue
                        else:
                            if temp[pos2-1] >= v2 or temp[pos2] <= v2:
                                continue

                        new_seq = temp[:pos2] + [v2] + temp[pos2:]
                        new_sums = get_all_sums(new_seq)
                        if len(new_sums) == len(new_seq) * (len(new_seq) + 1) // 2:
                            current = new_seq
                            inserted = True
                            break
                    if inserted:
                        break

                if not inserted:
                    # Restore
                    current.insert(i2, v2)
                    current.insert(i1, v1)

                # Try to extend
                extended = try_extend(current)
                if len(extended) > current_len:
                    return extended

        return current

    # Run local search
    searched = local_search(best_seq)
    if len(searched) > best_len:
        best_seq = searched
        best_len = len(best_seq)

    # DFS with better pruning - try to find even longer sequences
    def dfs(seq, sums_set, last_val, depth=0):
        nonlocal best_seq, best_len, start_time, time_limit

        current_len = len(seq)

        if current_len > best_len:
            best_len = current_len
            best_seq = list(seq)

        # Time check
        if time.time() - start_time > time_limit:
            return

        # Pruning: max possible extension
        remaining = 256 - last_val
        if current_len + remaining <= best_len:
            return

        # Try candidates from high to low (larger values create more distinct sums)
        for x in range(256, last_val, -1):
            if (256 - x) % 16 == 0 and time.time() - start_time > time_limit:
                return

            # Compute new sums
            new_sums = set()
            total = x
            new_sums.add(total)
            for a in reversed(seq):
                total += a
                new_sums.add(total)

            if sums_set.intersection(new_sums):
                continue

            dfs(seq + [x], sums_set | new_sums, x, depth + 1)

            if time.time() - start_time > time_limit:
                return

    # Start DFS from the best sequence found
    if best_seq:
        dfs(best_seq, get_all_sums(best_seq), best_seq[-1])

    # Additional local search after DFS
    if time.time() - start_time < time_limit - 5:
        searched2 = local_search(best_seq, max_iters=300)
        if len(searched2) > best_len:
            best_seq = searched2
            best_len = len(best_seq)

    return best_seq
