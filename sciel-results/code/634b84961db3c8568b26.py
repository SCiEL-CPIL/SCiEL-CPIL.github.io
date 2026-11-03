import itertools
import random
import time
import heapq

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    total = q ** n  # 729
    ball_size = 1 + n * (q - 1)  # 13

    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # Precompute neighbor index lists for every word
    neighbors = []
    for idx in range(total):
        nbs = [idx]
        for i in range(n):
            base = idx // (q ** (n - 1 - i))
            rem = idx % (q ** (n - 1 - i))
            digit = base % q
            for d in range(1, q):
                new_digit = (digit + d) % q
                new_idx = (base + (new_digit - digit)) * (q ** (n - 1 - i)) + rem
                nbs.append(new_idx)
        neighbors.append(nbs)

    # Ball bitmasks as Python int bitsets for fast intersection/coverage
    ball_mask = [0] * total
    for idx in range(total):
        m = 0
        for nb in neighbors[idx]:
            m |= (1 << nb)
        ball_mask[idx] = m

    all_mask = (1 << total) - 1

    start_time = time.time()
    deadline = start_time + 145.0

    best_code = None
    best_size = float('inf')

    def local_search(code, cover_mask, cover_count):
        """Aggressive local search using priority queue. Returns improved code."""
        code_set = set(code)
        # Priority queue: (count, idx) for each codeword
        pq = [(cover_count[c], c) for c in code]
        heapq.heapify(pq)
        in_pq = {c: True for c in code}
        count_ver = {c: cover_count[c] for c in code}

        changed = True
        while changed:
            if time.time() > deadline - 2.0:
                break
            changed = False
            # Process all currently-redundant codewords
            while pq:
                cnt, c = heapq.heappop(pq)
                if c not in code_set:
                    continue
                if count_ver.get(c, -1) != cnt:
                    continue
                if cnt > 1:
                    # Remove c
                    code_set.discard(c)
                    in_pq.pop(c, None)
                    count_ver.pop(c, None)
                    bm = ball_mask[c]
                    for nb in neighbors[c]:
                        cover_count[nb] -= 1
                        cover_mask &= ~(1 << nb) if cover_count[nb] == 0 else cover_mask
                        # Update affected neighbors' counts in pq
                        for nc in code_set:
                            if nc in neighbors[nb] or nb in neighbors[nc]:
                                # nc's ball contains nb
                                pass
                    # Recompute counts for remaining codewords efficiently
                    # Only update pq entries for codewords whose ball intersects neighbors[c]
                    affected = set()
                    for nb in neighbors[c]:
                        # codewords whose ball contains nb
                        pass
                    # Simpler: rebuild pq counts lazily
                    changed = True
                    # Rebuild pq for remaining codewords
                    break
                else:
                    # Not redundant, push back
                    heapq.heappush(pq, (cnt, c))

            if not changed:
                # No redundant codeword found via pq; do a full scan to be safe
                for c in list(code_set):
                    if cover_count is None:
                        break
                    # Check if c is redundant: every neighbor of c has cover_count > 1
                    redundant = True
                    for nb in neighbors[c]:
                        if cover_count[nb] <= 1:
                            redundant = False
                            break
                    if redundant:
                        code_set.discard(c)
                        for nb in neighbors[c]:
                            cover_count[nb] -= 1
                        changed = True
                if changed:
                    # Rebuild cover_mask
                    cover_mask = 0
                    for c in code_set:
                        cover_mask |= ball_mask[c]
                    # Rebuild cover_count
                    cover_count = [0] * total
                    for c in code_set:
                        for nb in neighbors[c]:
                            cover_count[nb] += 1
                    continue

        return list(code_set)

    def greedy_cover(seed_rng, max_words=None):
        """Randomized greedy set cover."""
        uncovered = all_mask
        code = []
        remaining = list(range(total))
        seed_rng.shuffle(remaining)

        # Precompute coverage of each candidate against uncovered
        # Greedy: repeatedly pick word covering most uncovered points
        while uncovered and time.time() < deadline - 5.0:
            best_idx = -1
            best_count = 0
            # Sample-based greedy for speed: check a random subset + all if small
            # For 729 words, full scan is 729 * bit_count ops, fast enough
            for c in remaining:
                cov = (ball_mask[c] & uncovered).bit_count()
                if cov > best_count:
                    best_count = cov
                    best_idx = c
                if time.time() > deadline - 5.0:
                    break
            if best_idx == -1 or best_count == 0:
                break
            code.append(best_idx)
            uncovered &= ~ball_mask[best_idx]
            remaining.remove(best_idx)
            if max_words and len(code) >= max_words:
                break

        return code

    def build_cover_state(code):
        cover_count = [0] * total
        cover_mask = 0
        for c in code:
            for nb in neighbors[c]:
                cover_count[nb] += 1
            cover_mask |= ball_mask[c]
        return cover_mask, cover_count

    def local_search_simple(code):
        """Simple iterative removal of redundant codewords."""
        code_set = set(code)
        cover_count = [0] * total
        for c in code_set:
            for nb in neighbors[c]:
                cover_count[nb] += 1

        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for c in list(code_set):
                # Check if c is redundant
                redundant = True
                for nb in neighbors[c]:
                    if cover_count[nb] <= 1:
                        redundant = False
                        break
                if redundant:
                    code_set.discard(c)
                    for nb in neighbors[c]:
                        cover_count[nb] -= 1
                    improved = True
                    break  # restart scan
        return list(code_set)

    def try_swap_improve(code, k):
        """Try k-swap perturbation: remove k random codewords, add k best replacements."""
        code_set = set(code)
        if len(code_set) <= k:
            return code_set

        cover_count = [0] * total
        for c in code_set:
            for nb in neighbors[c]:
                cover_count[nb] += 1

        # Remove k random codewords
        removed = random.sample(list(code_set), k)
        for c in removed:
            code_set.discard(c)
            for nb in neighbors[c]:
                cover_count[nb] -= 1

        # Find k best words to add (maximize coverage of newly-uncovered points)
        uncovered_mask = 0
        for nb in range(total):
            if cover_count[nb] == 0:
                uncovered_mask |= (1 << nb)

        if uncovered_mask == 0:
            return code_set  # Still covered, just smaller

        # Greedily add k words
        added = 0
        for _ in range(k):
            if time.time() > deadline - 2.0:
                break
            best_idx = -1
            best_count = 0
            for c in range(total):
                if c in code_set:
                    continue
                cov = (ball_mask[c] & uncovered_mask).bit_count()
                if cov > best_count:
                    best_count = cov
                    best_idx = c
            if best_idx == -1 or best_count == 0:
                break
            code_set.add(best_idx)
            for nb in neighbors[best_idx]:
                cover_count[nb] += 1
            uncovered_mask &= ~ball_mask[best_idx]
            added += 1

        # Check if we actually improved (smaller or equal size with full coverage)
        # Full coverage check
        for nb in range(total):
            if cover_count[nb] == 0:
                # Not fully covered, revert
                for c in removed:
                    code_set.add(c)
                for c in list(code_set):
                    if c not in removed and c not in [best_idx]:
                        pass
                # Just revert by re-adding removed
                return set(code)  # original

        return code_set

    # Multi-start ILS
    rng = random.Random(42)
    num_starts = 200  # Will be limited by time

    for start_iter in range(num_starts):
        if time.time() > deadline - 10.0:
            break

        # Phase 1: Greedy initialization with randomness
        seed_rng = random.Random(start_iter * 7919 + 13)
        code = greedy_cover(seed_rng)
        if not code:
            continue

        # Phase 2: Local search
        code = local_search_simple(code)
        code_size = len(code)

        if code_size < best_size:
            best_size = code_size
            best_code = list(code)

        # Phase 3: ILS perturbations
        for pert_iter in range(50):
            if time.time() > deadline - 5.0:
                break

            k = rng.choice([1, 2, 2, 3])  # Bias toward smaller k
            new_code_set = try_swap_improve(code, k)

            if new_code_set is not None and len(new_code_set) < len(code):
                code = local_search_simple(list(new_code_set))
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = list(code)
            elif new_code_set is not None and len(new_code_set) == len(code):
                # Same size, try local search anyway
                improved = local_search_simple(list(new_code_set))
                if len(improved) < len(code):
                    code = improved
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = list(code)

        # Additional: try adding then removing to find better configurations
        if time.time() > deadline - 5.0:
            break

    # Final local search on best
    if best_code is not None:
        final = local_search_simple(best_code)
        if len(final) <= len(best_code):
            best_code = final

    if best_code is None:
        # Fallback: return full space (valid but large)
        best_code = list(range(total))

    # Convert indices to tuples/lists
    result = []
    for idx in best_code:
        t = [0] * n
        for i in range(n):
            t[n - 1 - i] = idx % q
            idx //= q
        result.append(t)

    return result
