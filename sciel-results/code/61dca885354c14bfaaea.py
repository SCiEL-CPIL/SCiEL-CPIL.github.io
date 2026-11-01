"""Ternary covering code via fast greedy initialization + local search + random-restart
with bitset-accelerated coverage checks (pure-Python int bitmasks)."""
# EVOLVE-BLOCK-START
import time
import random
from collections import Counter


def construct(q, n, radius):
    """
    Build a covering code of length n over alphabet size q, covering radius `radius`.

    Representation:
      - Words are indexed 0..N-1 in base-q (digit 0 = LSB).
      - Each word's radius-`radius` ball is precomputed as a Python int bitmask.
      - Coverage of a code is the OR of its codewords' ball bitmasks.
      - A code is valid iff coverage == all_mask.

    Algorithm:
      1. Greedy maximal cover (fast, O(N) per step with int bitmasks).
      2. Aggressive reduction (remove redundant codewords).
      3. Local search: repeatedly pick a random codeword, try replacing it with a
         random other word; accept if the new set is still valid; reduce after each
         improvement.
      4. Multiple random restarts of the greedy phase (with different tie-breaking
         orderings) to escape local minima, keeping the best code found.
    """
    start_time = time.time()
    deadline = start_time + 142.0  # 8s safety margin

    N = q ** n
    if N <= 0:
        return []

    # ---------- Precompute all words as digit lists (for final output) ----------
    all_words = []
    for idx in range(N):
        digits = []
        rem = idx
        for _ in range(n):
            digits.append(rem % q)
            rem //= q
        all_words.append(digits)

    # ---------- Precompute radius balls as int bitmasks ----------
    # ball[i] is a Python int with bit j set iff word j is within Hamming distance
    # `radius` of word i.
    # For efficiency, compute the distance on the fly per word.

    # For q=3, n=7, radius=1: ball size is 1 + n*(q-1) = 1 + 7*2 = 15 words.
    # In general: ball size = sum_{k=0}^{radius} C(n,k) * (q-1)^k
    # Precompute all words within radius of each word.

    def _neighbors_within_radius(idx, r):
        """Return list of indices within Hamming distance r of word idx."""
        # Get digits of idx
        digits = []
        rem = idx
        for _ in range(n):
            digits.append(rem % q)
            rem //= q
        # For radius 1 (the common case), enumerate neighbors directly
        if r == 1:
            result = [idx]
            for pos in range(n):
                orig = digits[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    new_digits = digits.copy()
                    new_digits[pos] = new_val
                    new_idx = 0
                    mult = 1
                    for d in new_digits:
                        new_idx += d * mult
                        mult *= q
                    result.append(new_idx)
            return result
        else:
            # General BFS (rarely needed for this task)
            from collections import deque
            visited = set()
            queue = deque([(idx, 0)])
            visited.add(idx)
            result = []
            while queue:
                cur, dist = queue.popleft()
                result.append(cur)
                if dist == r:
                    continue
                # Generate neighbors of cur
                cd = []
                rem = cur
                for _ in range(n):
                    cd.append(rem % q)
                    rem //= q
                for pos in range(n):
                    orig = cd[pos]
                    for new_val in range(q):
                        if new_val == orig:
                            continue
                        nd = cd.copy()
                        nd[pos] = new_val
                        ni = 0
                        mult = 1
                        for d in nd:
                            ni += d * mult
                            mult *= q
                        if ni not in visited:
                            visited.add(ni)
                            queue.append((ni, dist + 1))
            return result

    ball = []
    for i in range(N):
        nbrs = _neighbors_within_radius(i, radius)
        mask = 0
        for j in nbrs:
            mask |= (1 << j)
        ball.append(mask)

    all_mask = (1 << N) - 1

    # ---------- Greedy maximal cover ----------
    def greedy_cover():
        covered = 0
        code = []
        remaining = all_mask
        while remaining != 0 and time.time() < deadline - 10:
            best = -1
            best_count = -1
            # Find codeword covering most of `remaining`
            for c in range(N):
                cnt = (ball[c] & remaining).bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best = c
            if best_count == 0:
                break
            code.append(best)
            covered |= ball[best]
            remaining = all_mask & ~covered
        return code

    # ---------- Reduction: remove redundant codewords ----------
    def reduce_code(code):
        """Remove redundant codewords iteratively."""
        code = list(code)
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                if len(code) <= 1:
                    break
                # Check if code[i] is redundant
                # Coverage without code[i]:
                cov = 0
                for j, c in enumerate(code):
                    if j != i:
                        cov |= ball[c]
                if cov == all_mask:
                    code.pop(i)
                    changed = True
        return code

    # ---------- Fast validity check ----------
    def is_valid(code):
        cov = 0
        for c in code:
            cov |= ball[c]
            if cov == all_mask:
                return True
        return cov == all_mask

    # ---------- Local search ----------
    def local_search(code, time_limit_frac=0.7):
        """
        Hill-climbing local search: try 1-for-1 swaps, accept if still valid,
        then reduce. Stop when no improvement in a full pass or time runs out.
        """
        code = list(code)
        code_set = set(code)
        rng = random.Random(42)
        last_improvement_time = time.time()
        no_improve_limit = 3.0  # seconds

        while time.time() < deadline - 5:
            if time.time() - last_improvement_time > no_improve_limit:
                break

            improved_this_pass = False
            # Shuffle order of codewords to avoid bias
            order = list(range(len(code)))
            rng.shuffle(order)

            for idx in order:
                if time.time() > deadline - 5:
                    break
                old = code[idx]
                # Try several random candidates
                for _ in range(3):
                    new_c = rng.randrange(N)
                    if new_c == old or new_c in code_set:
                        continue
                    # Check validity of replacement
                    new_code = list(code)
                    new_code[idx] = new_c
                    if is_valid(new_code):
                        # Reduce
                        new_code = reduce_code(new_code)
                        if len(new_code) < len(code):
                            code = new_code
                            code_set = set(code)
                            improved_this_pass = True
                            last_improvement_time = time.time()
                            break
                        else:
                            # Even without size reduction, accept to escape local minima
                            # (the code is still valid and same size)
                            code = new_code
                            code_set = set(code)
                            break
                if improved_this_pass:
                    break

            if not improved_this_pass:
                # No improvement; try a perturbation: remove a random codeword and
                # re-greedy-fill to escape the local minimum
                if len(code) > 5 and time.time() < deadline - 10:
                    rem_idx = rng.randrange(len(code))
                    code.pop(rem_idx)
                    code_set = set(code)
                    # Greedy fill to restore validity
                    remaining = all_mask
                    for c in code:
                        remaining &= ~ball[c]
                    while remaining != 0 and time.time() < deadline - 5:
                        best = -1
                        best_count = -1
                        for c in range(N):
                            cnt = (ball[c] & remaining).bit_count()
                            if cnt > best_count:
                                best_count = cnt
                                best = c
                        if best_count == 0:
                            break
                        code.append(best)
                        code_set.add(best)
                        remaining &= ~ball[best]
                    code = reduce_code(code)
                    code_set = set(code)
                    last_improvement_time = time.time()

        return code

    # ---------- Main: multiple restarts ----------
    best_code = None
    best_size = N + 1

    # Number of restarts: use time to decide
    # First run: full greedy + local search
    for restart in range(20):
        if time.time() > deadline - 15:
            break

        # Randomize tie-breaking by shuffling the word order for greedy
        rng2 = random.Random(restart * 1000 + 7)
        word_order = list(range(N))
        rng2.shuffle(word_order)

        # Greedy cover using shuffled order for tie-breaking
        covered = 0
        code = []
        remaining = all_mask
        while remaining != 0 and time.time() < deadline - 10:
            best = -1
            best_count = -1
            for c in word_order:
                cnt = (ball[c] & remaining).bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best = c
            if best_count == 0:
                break
            code.append(best)
            covered |= ball[best]
            remaining = all_mask & ~covered

        code = reduce_code(code)

        if len(code) < best_size:
            best_code = code.copy()
            best_size = len(code)

        # Local search on this code (use remaining time wisely)
        if time.time() < deadline - 10:
            code = local_search(code)
            if len(code) < best_size:
                best_code = code.copy()
                best_size = len(code)

    # Final fallback
    if best_code is None:
        best_code = greedy_cover()
        best_code = reduce_code(best_code)

    # Final validity check
    if not is_valid(best_code):
        # Should not happen, but guard against it
        remaining = all_mask
        for c in best_code:
            remaining &= ~ball[c]
        while remaining != 0 and time.time() < deadline - 1:
            best = -1
            best_count = -1
            for c in range(N):
                cnt = (ball[c] & remaining).bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best = c
            if best_count == 0:
                break
            best_code.append(best)
            remaining &= ~ball[best]
        best_code = reduce_code(best_code)

    return [all_words[i] for i in best_code]
# EVOLVE-BLOCK-END
