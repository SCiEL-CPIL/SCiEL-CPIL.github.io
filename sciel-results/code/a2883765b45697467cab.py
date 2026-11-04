"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin before 150s kill

    N = q ** n  # 2187 for q=3, n=7
    ball_size = 1 + n * (q - 1)  # 15

    # Precompute all words as index -> digits
    def idx_to_word(idx):
        w = [0] * n
        for pos in range(n):
            w[pos] = idx % q
            idx //= q
        return w

    # Precompute balls (neighborhoods of radius 1)
    balls = []
    for i in range(N):
        digits = idx_to_word(i)
        nb = [i]
        for pos in range(n):
            orig = digits[pos]
            for val in range(q):
                if val == orig:
                    continue
                new_digits = digits[:]
                new_digits[pos] = val
                # Convert back to index
                j = 0
                for k in range(n):
                    j = j * q + new_digits[k]
                nb.append(j)
        balls.append(nb)

    # Build cover_map: for each word w, which codewords cover it
    cover_map = [[] for _ in range(N)]
    for c in range(N):
        for w in balls[c]:
            cover_map[w].append(c)

    def greedy_cover(seed):
        """Count-based greedy set cover."""
        rng = random.Random(seed)
        covered = [False] * N
        cnt = [ball_size] * N  # how many uncovered words each candidate covers
        code = []
        uncovered_count = N

        while uncovered_count > 0:
            # Find max count
            max_val = 0
            candidates = []
            for i in range(N):
                if cnt[i] > max_val:
                    max_val = cnt[i]
                    candidates = [i]
                elif cnt[i] == max_val and max_val > 0:
                    candidates.append(i)

            if not candidates:
                break

            c = rng.choice(candidates)
            code.append(c)

            # Update: mark covered, decrement counts
            for w in balls[c]:
                if not covered[w]:
                    covered[w] = True
                    uncovered_count -= 1
                    for c2 in cover_map[w]:
                        cnt[c2] -= 1

        return code

    def prune_redundant(code):
        """Remove redundant codewords using cover counts."""
        if not code:
            return code

        cover_count = [0] * N
        for c in code:
            for w in balls[c]:
                cover_count[w] += 1

        code_list = list(code)
        changed = True
        while changed:
            changed = False
            for i in range(len(code_list)):
                c = code_list[i]
                safe = True
                for w in balls[c]:
                    if cover_count[w] == 1:
                        safe = False
                        break
                if safe:
                    for w in balls[c]:
                        cover_count[w] -= 1
                    code_list.pop(i)
                    changed = True
                    break
        return code_list

    def local_search_swap(code, time_limit):
        """Try swapping: remove one codeword, find a single replacement that covers all gaps."""
        code_list = list(code)
        if not code_list:
            return code_list

        cover_count = [0] * N
        for c in code_list:
            for w in balls[c]:
                cover_count[w] += 1

        improved = True
        rng = random.Random(42)
        while improved and time.time() < time_limit:
            improved = False
            indices = list(range(len(code_list)))
            rng.shuffle(indices)
            for i in indices:
                if time.time() > time_limit:
                    break
                c_remove = code_list[i]
                # Tentatively remove
                for w in balls[c_remove]:
                    cover_count[w] -= 1

                # Find uncovered words
                uncovered = []
                for w in range(N):
                    if cover_count[w] == 0:
                        uncovered.append(w)

                if not uncovered:
                    # Can remove entirely
                    code_list.pop(i)
                    improved = True
                    break

                # Try to find a single codeword that covers all uncovered
                found = -1
                for c_add in range(N):
                    if c_add == c_remove:
                        continue
                    if c_add in code_list:
                        continue
                    ball_add = balls[c_add]
                    ok = True
                    for w in uncovered:
                        if w not in ball_add:
                            # Check if w is in ball_add by scanning
                            ok = False
                            break
                    if ok:
                        found = c_add
                        break

                if found >= 0:
                    # Do the swap
                    code_list[i] = found
                    for w in balls[found]:
                        cover_count[w] += 1
                    improved = True
                    break
                else:
                    # Restore
                    for w in balls[c_remove]:
                        cover_count[w] += 1

        return code_list

    def kick_and_repair(code, num_kicks, time_limit):
        """Remove random codewords and repair with greedy."""
        if len(code) <= num_kicks:
            return code

        code = list(code)
        removed = random.sample(range(len(code)), num_kicks)
        for idx in sorted(removed, reverse=True):
            code.pop(idx)

        # Find uncovered
        covered = [False] * N
        for c in code:
            for w in balls[c]:
                covered[w] = True

        uncovered = [w for w in range(N) if not covered[w]]
        if not uncovered:
            return code

        # Greedy repair: add codewords to cover uncovered
        cnt = [0] * N
        for c in range(N):
            for w in balls[c]:
                if not covered[w]:
                    cnt[c] += 1

        while uncovered and time.time() < time_limit:
            max_val = 0
            candidates = []
            for i in range(N):
                if cnt[i] > max_val:
                    max_val = cnt[i]
                    candidates = [i]
                elif cnt[i] == max_val and max_val > 0:
                    candidates.append(i)
            if not candidates:
                break
            c = random.choice(candidates)
            code.append(c)
            for w in balls[c]:
                if not covered[w]:
                    covered[w] = True
                    uncovered.remove(w)
                    for c2 in cover_map[w]:
                        cnt[c2] -= 1

        return code

    # Main search loop
    best_code = None
    best_size = float('inf')

    # Multiple greedy restarts with different seeds
    num_restarts = 30
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
        code = greedy_cover(i)
        code = prune_redundant(code)
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            if best_size <= 186:
                break

    # Local search on best solution
    if best_code is not None and time.time() < deadline - 5:
        ls_code = local_search_swap(best_code, deadline - 3)
        ls_code = prune_redundant(ls_code)
        if len(ls_code) < best_size:
            best_size = len(ls_code)
            best_code = ls_code[:]

    # Kick-and-repair iterations
    if best_code is not None and time.time() < deadline - 5:
        for num_kicks in [1, 2, 3, 4, 5, 6, 8, 10, 12]:
            if time.time() > deadline - 3:
                break
            kicked = kick_and_repair(best_code, num_kicks, deadline - 3)
            kicked = prune_redundant(kicked)
            if len(kicked) < best_size:
                best_size = len(kicked)
                best_code = kicked[:]
                # Try local search on improved solution
                if time.time() < deadline - 5:
                    ls2 = local_search_swap(best_code, deadline - 3)
                    ls2 = prune_redundant(ls2)
                    if len(ls2) < best_size:
                        best_size = len(ls2)
                        best_code = ls2[:]

    # Fallback
    if best_code is None:
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return [idx_to_word(c) for c in best_code]
# EVOLVE-BLOCK-END
