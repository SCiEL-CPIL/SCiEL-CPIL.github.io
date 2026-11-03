"""Covering code K_3(7,1) via greedy set cover + count-based local search with swaps + polish."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

_DEADLINE = 140.0  # seconds; evaluator kills at 150s


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + _DEADLINE

    total = q ** n
    powers = [q ** (n - 1 - i) for i in range(n)]

    # Precompute all words as tuples
    all_words = []
    for idx in range(total):
        digits = []
        x = idx
        for _ in range(n):
            digits.append(x % q)
            x //= q
        digits.reverse()
        all_words.append(tuple(digits))

    # Precompute ball masks and reverse_cover
    ball_masks = [0] * total
    reverse_cover = [[] for _ in range(total)]
    for center_idx in range(total):
        center = all_words[center_idx]
        mask = 1 << center_idx
        if radius >= 1:
            for pos in range(n):
                for alt in range(q):
                    if alt == center[pos]:
                        continue
                    new_idx = center_idx + (alt - center[pos]) * powers[pos]
                    mask |= (1 << new_idx)
        ball_masks[center_idx] = mask
        # Build reverse_cover: for each word in the ball, add center as a coverer
        # We iterate over set bits of the mask
        m = mask
        while m:
            low = m & (-m)
            j = low.bit_length() - 1
            reverse_cover[j].append(center_idx)
            m ^= low

    all_mask = (1 << total) - 1
    best_code = None
    best_size = total  # Upper bound

    # Greedy set cover with randomized tie-breaking
    def greedy_cover():
        uncovered = all_mask
        selected = []
        remaining = list(range(total))

        while uncovered != 0:
            if time.time() > deadline - 5:
                return None

            best_count = -1
            best_candidates = []

            for c in remaining:
                covered = ball_masks[c] & uncovered
                cnt = covered.bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_candidates = [c]
                elif cnt == best_count and cnt > 0:
                    best_candidates.append(c)

            if not best_candidates:
                break

            best_idx = random.choice(best_candidates)
            selected.append(best_idx)
            uncovered &= ~ball_masks[best_idx]
            remaining.remove(best_idx)

        if uncovered != 0:
            return None
        return selected

    # Count-based local search with removal and swap moves
    def local_search(code, deadline_ts):
        code = list(code)
        if not code:
            return code

        # Build counts: for each word, how many codewords cover it
        counts = [0] * total
        for c in code:
            m = ball_masks[c]
            while m:
                low = m & (-m)
                j = low.bit_length() - 1
                counts[j] += 1
                m ^= low

        improved = True
        while improved and time.time() < deadline_ts:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)

            for i in indices:
                if time.time() >= deadline_ts:
                    break

                c_remove = code[i]

                # Subtract counts for c_remove
                m = ball_masks[c_remove]
                while m:
                    low = m & (-m)
                    j = low.bit_length() - 1
                    counts[j] -= 1
                    m ^= low

                # Check for holes (words now uncovered)
                holes = []
                m = ball_masks[c_remove]
                while m:
                    low = m & (-m)
                    j = low.bit_length() - 1
                    if counts[j] == 0:
                        holes.append(j)
                    m ^= low

                if not holes:
                    # Successful removal
                    code.pop(i)
                    improved = True
                    break
                else:
                    # Restore counts
                    m = ball_masks[c_remove]
                    while m:
                        low = m & (-m)
                        j = low.bit_length() - 1
                        counts[j] += 1
                        m ^= low

                    # Try swap: find a codeword that covers all holes
                    # Start with the hole with smallest reverse_cover
                    holes.sort(key=lambda x: len(reverse_cover[x]))
                    candidates = set(reverse_cover[holes[0]])
                    for h in holes[1:]:
                        candidates &= set(reverse_cover[h])
                        if not candidates:
                            break

                    candidates.discard(c_remove)

                    if candidates:
                        # Pick a random candidate (or the one with best heuristic)
                        # Heuristic: prefer candidates that cover more words with low counts
                        cand_list = list(candidates)
                        # Score: sum of counts of words covered (lower is better, less redundant)
                        # Actually, we want to minimize the "waste" - prefer candidates
                        # that cover words with already high counts (so we don't add new redundancy)
                        # But any valid swap works; random is a good baseline.
                        # Let's pick the one that minimizes the max count increase
                        best_cand = None
                        best_score = None
                        for cand in cand_list:
                            # Compute score: sum of counts of words in ball[cand] not in ball[c_remove]
                            # Simpler: just random
                            pass
                        # For speed, just random pick
                        chosen = random.choice(cand_list)

                        # Perform swap: remove c_remove, add chosen
                        # Update counts
                        m_remove = ball_masks[c_remove]
                        m_add = ball_masks[chosen]

                        # Subtract c_remove
                        m = m_remove
                        while m:
                            low = m & (-m)
                            j = low.bit_length() - 1
                            counts[j] -= 1
                            m ^= low

                        # Add chosen
                        m = m_add
                        while m:
                            low = m & (-m)
                            j = low.bit_length() - 1
                            counts[j] += 1
                            m ^= low

                        code[i] = chosen
                        improved = True
                        break

        return code

    # High-precision polish: exhaustive removal pass + swap refinement
    def polish(code, deadline_ts):
        code = list(code)
        if not code:
            return code

        # Multiple passes of exhaustive removal
        for _ in range(10):
            if time.time() >= deadline_ts:
                break
            changed = False
            for i in range(len(code) - 1, -1, -1):
                if time.time() >= deadline_ts:
                    break
                c = code[i]
                # Check if c is redundant: its ball is covered by others
                others_mask = 0
                for j in range(len(code)):
                    if j != i:
                        others_mask |= ball_masks[code[j]]
                unique = ball_masks[c] & ~others_mask
                if unique == 0:
                    code.pop(i)
                    changed = True
            if not changed:
                break

        # Final swap refinement
        counts = [0] * total
        for c in code:
            m = ball_masks[c]
            while m:
                low = m & (-m)
                j = low.bit_length() - 1
                counts[j] += 1
                m ^= low

        for _ in range(5):
            if time.time() >= deadline_ts:
                break
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)
            for i in indices:
                if time.time() >= deadline_ts:
                    break
                c_remove = code[i]
                m = ball_masks[c_remove]
                while m:
                    low = m & (-m)
                    j = low.bit_length() - 1
                    counts[j] -= 1
                    m ^= low

                holes = []
                m = ball_masks[c_remove]
                while m:
                    low = m & (-m)
                    j = low.bit_length() - 1
                    if counts[j] == 0:
                        holes.append(j)
                    m ^= low

                if not holes:
                    code.pop(i)
                    improved = True
                    break
                else:
                    m = ball_masks[c_remove]
                    while m:
                        low = m & (-m)
                        j = low.bit_length() - 1
                        counts[j] += 1
                        m ^= low

                    holes.sort(key=lambda x: len(reverse_cover[x]))
                    candidates = set(reverse_cover[holes[0]])
                    for h in holes[1:]:
                        candidates &= set(reverse_cover[h])
                        if not candidates:
                            break
                    candidates.discard(c_remove)
                    if candidates:
                        chosen = random.choice(list(candidates))
                        m = ball_masks[c_remove]
                        while m:
                            low = m & (-m)
                            j = low.bit_length() - 1
                            counts[j] -= 1
                            m ^= low
                        m = ball_masks[chosen]
                        while m:
                            low = m & (-m)
                            j = low.bit_length() - 1
                            counts[j] += 1
                            m ^= low
                        code[i] = chosen
                        improved = True
                        break
            if not improved:
                break

        return code

    # Main loop: multiple restarts
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break

        code = greedy_cover()
        if code is None:
            continue

        # Local search
        code = local_search(code, deadline - 5)

        # Polish
        if time.time() < deadline - 2:
            code = polish(code, deadline - 1)

        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]

    # Convert best_code to list of lists
    if best_code is None:
        # Fallback: tetracode direct sum
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    result = []
    for idx in best_code:
        digits = []
        x = idx
        for _ in range(n):
            digits.append(x % q)
            x //= q
        digits.reverse()
        result.append(digits)

    return result
# EVOLVE-BLOCK-END
