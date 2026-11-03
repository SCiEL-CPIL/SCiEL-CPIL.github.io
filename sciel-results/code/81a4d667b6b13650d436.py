import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    if not (q == 3 and n == 6 and radius == 1):
        return _generic_greedy(q, n, radius)

    start_time = time.time()
    deadline = 115.0

    N = q ** n

    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * q + val
        return idx

    def idx_to_word(idx):
        w = []
        for _ in range(n):
            w.append(idx % q)
            idx //= q
        return tuple(reversed(w))

    # Precompute all words
    all_words = [idx_to_word(i) for i in range(N)]

    # Precompute balls as lists (not sets) for faster iteration
    balls = [None] * N
    for i, w in enumerate(all_words):
        ball = [i]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    new_w = list(w)
                    new_w[pos] = new_val
                    ball.append(word_to_idx(tuple(new_w)))
        balls[i] = ball

    # Precompute coverers: for each word index, which ball-members cover it
    # coverers[i] = list of cw indices such that i in balls[cw]
    coverers = [set() for _ in range(N)]
    for cw in range(N):
        for neighbor in balls[cw]:
            coverers[neighbor].add(cw)

    # Convert coverers to lists for faster iteration
    coverers_lists = [list(s) for s in coverers]

    def tetracode_base():
        return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]

    def build_initial_code():
        tetracode = tetracode_base()
        initial_codewords = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                cw = h + list(tail)
                initial_codewords.append(word_to_idx(tuple(cw)))
        return set(initial_codewords)

    def greedy_reduce(current_code, cover_counts, balls):
        removed = True
        while removed:
            removed = False
            for cw in list(current_code):
                redundant = True
                for neighbor in balls[cw]:
                    if cover_counts[neighbor] <= 1:
                        redundant = False
                        break
                if redundant:
                    current_code.remove(cw)
                    for neighbor in balls[cw]:
                        cover_counts[neighbor] -= 1
                    removed = True

    def local_search(current_code, cover_counts, balls, coverers_lists, deadline):
        """Swap-based local search to reduce code size."""
        best_size = len(current_code)
        best_snapshot = list(current_code)
        best_cover = cover_counts.copy()

        temperature = 1.0
        cooling = 0.995
        stuck_count = 0

        while time.time() < deadline:
            greedy_reduce(current_code, cover_counts, balls)

            if len(current_code) < best_size:
                best_size = len(current_code)
                best_snapshot = list(current_code)
                best_cover = cover_counts.copy()
                stuck_count = 0

                if best_size <= 73:
                    break
            else:
                stuck_count += 1

            if stuck_count > 5000:
                # Restart with perturbation
                stuck_count = 0
                # Add a random word and remove a random one (perturbation)
                if len(current_code) > 1:
                    candidates = [i for i in range(N) if i not in current_code]
                    if candidates:
                        cw_add = random.choice(candidates)
                        cw_remove = random.choice(list(current_code))
                        # Try the swap
                        for neighbor in balls[cw_remove]:
                            cover_counts[neighbor] -= 1
                        current_code.remove(cw_remove)
                        for neighbor in balls[cw_add]:
                            cover_counts[neighbor] += 1
                        current_code.add(cw_add)
                        greedy_reduce(current_code, cover_counts, balls)

            # Pick a random non-redundant codeword to try removing
            cw_candidates = []
            for cw in current_code:
                is_red = True
                for neighbor in balls[cw]:
                    if cover_counts[neighbor] <= 1:
                        is_red = False
                        break
                if not is_red:
                    cw_candidates.append(cw)

            if not cw_candidates:
                temperature *= cooling
                continue

            cw_out = random.choice(cw_candidates)

            # Remove it temporarily
            for neighbor in balls[cw_out]:
                cover_counts[neighbor] -= 1
            current_code.remove(cw_out)

            # Find uncovered words
            uncovered = [neighbor for neighbor in balls[cw_out] if cover_counts[neighbor] == 0]

            if not uncovered:
                # It was actually redundant after all (shouldn't happen but safe)
                current_code.add(cw_out)
                for neighbor in balls[cw_out]:
                    cover_counts[neighbor] += 1
                continue

            # Find candidates that cover ALL uncovered words
            # Intersection of coverers for each uncovered word, minus current_code
            valid_cw_in = None
            if len(uncovered) == 1:
                u = uncovered[0]
                for cw in coverers_lists[u]:
                    if cw not in current_code and cw != cw_out:
                        valid_cw_in = cw
                        break
            elif len(uncovered) <= 5:
                # Small intersection
                candidate_set = None
                for u in uncovered:
                    if candidate_set is None:
                        candidate_set = set(coverers_lists[u])
                    else:
                        candidate_set &= set(coverers_lists[u])
                if candidate_set:
                    candidate_set -= current_code
                    candidate_set.discard(cw_out)
                    if candidate_set:
                        valid_cw_in = random.choice(list(candidate_set))

            if valid_cw_in is None:
                # Revert
                current_code.add(cw_out)
                for neighbor in balls[cw_out]:
                    cover_counts[neighbor] += 1
                temperature *= cooling
                continue

            # Perform the swap
            current_code.add(valid_cw_in)
            for neighbor in balls[valid_cw_in]:
                cover_counts[neighbor] += 1

            # Try to reduce further
            greedy_reduce(current_code, cover_counts, balls)

            # Check if this led to improvement or same size
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_snapshot = list(current_code)
                best_cover = cover_counts.copy()
                stuck_count = 0
            elif len(current_code) == best_size:
                # Accept same-size swap to explore (Metropolis-like)
                # Keep it; the greedy_reduce may have already optimized
                pass
            else:
                # Size increased (shouldn't happen since we removed 1 and added 1)
                # Revert to best state
                current_code = set(best_snapshot)
                cover_counts = best_cover.copy()
                greedy_reduce(current_code, cover_counts, balls)

            temperature *= cooling
            if temperature < 0.05:
                temperature = 1.0

        return best_snapshot, best_size

    # Multi-start local search
    best_overall = None
    best_overall_size = float('inf')

    # Start 1: tetracode-based
    current_code = build_initial_code()
    cover_counts = np.zeros(N, dtype=np.int32)
    for cw in current_code:
        for neighbor in balls[cw]:
            cover_counts[neighbor] += 1

    greedy_reduce(current_code, cover_counts, balls)
    best_snapshot, best_size = local_search(current_code, cover_counts, balls, coverers_lists, deadline)
    best_overall = best_snapshot
    best_overall_size = best_size

    if time.time() < deadline - 20 and best_overall_size > 73:
        # Start 2: Random initial code with greedy repair
        random.seed(42)
        # Start with a random set of ~80 words
        current_code = set(random.sample(range(N), 80))
        cover_counts = np.zeros(N, dtype=np.int32)
        for cw in current_code:
            for neighbor in balls[cw]:
                cover_counts[neighbor] += 1

        # Greedy repair: add words to cover uncovered
        while time.time() < deadline - 10:
            uncovered = np.where(cover_counts == 0)[0]
            if len(uncovered) == 0:
                break
            u = random.choice(uncovered)
            # Pick a random word that covers u
            candidates = [cw for cw in coverers_lists[u] if cw not in current_code]
            if not candidates:
                # Must add from coverers even if already in code (shouldn't happen)
                candidates = coverers_lists[u]
            cw_add = random.choice(candidates)
            current_code.add(cw_add)
            for neighbor in balls[cw_add]:
                cover_counts[neighbor] += 1

        if np.all(cover_counts > 0):
            greedy_reduce(current_code, cover_counts, balls)
            best_snapshot2, best_size2 = local_search(current_code, cover_counts, balls, coverers_lists, deadline)
            if best_size2 < best_overall_size:
                best_overall = best_snapshot2
                best_overall_size = best_size2

    # Final polish: try to remove any remaining redundant words
    if best_overall is not None:
        final_code = set(best_overall)
        final_cover = np.zeros(N, dtype=np.int32)
        for cw in final_code:
            for neighbor in balls[cw]:
                final_cover[neighbor] += 1
        greedy_reduce(final_code, final_cover, balls)

        if len(final_code) < best_overall_size:
            best_overall = list(final_code)
            best_overall_size = len(final_code)

    # Convert to output format
    best_words = []
    for idx in best_overall:
        best_words.append(list(idx_to_word(idx)))

    return best_words

def _generic_greedy(q, n, radius):
    # Fallback: simple greedy set cover
    if q ** n > 100000:
        raise ValueError("Space too large for generic greedy")

    N = q ** n

    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * q + val
        return idx

    def idx_to_word(idx):
        w = []
        for _ in range(n):
            w.append(idx % q)
            idx //= q
        return tuple(reversed(w))

    all_words = [idx_to_word(i) for i in range(N)]

    balls = [None] * N
    for i, w in enumerate(all_words):
        ball = set()
        ball.add(i)
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    new_w = list(w)
                    new_w[pos] = new_val
                    ball.add(word_to_idx(tuple(new_w)))
        balls[i] = ball

    uncovered = set(range(N))
    code = []

    while uncovered:
        # Find word covering most uncovered
        best_cw = None
        best_count = -1
        for cw in range(N):
            count = len(balls[cw] & uncovered)
            if count > best_count:
                best_count = count
                best_cw = cw
        code.append(best_cw)
        uncovered -= balls[best_cw]

    return [list(idx_to_word(idx)) for idx in code]
