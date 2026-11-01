"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools


def _tetracode_words(q=3, n=4):
    """Return the ternary tetracode: 9 words of length 4, a perfect covering code of radius 1."""
    return [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]


def construct(q, n, radius):
    """
    Construct a covering code for alphabet {0..q-1}^n with given covering radius.
    For the specific task (q=3, n=6, radius=1), uses a greedy set-cover approach
    initialized with the tetracode product construction, then tries to improve.
    """
    start_time = time.time()
    deadline = 110  # seconds, leaves margin before 120s limit

    # Handle the specific case first with specialized logic
    if q == 3 and n == 6 and radius == 1:
        return _construct_ternary_6_1(start_time, deadline)

    # General fallback: product construction or exhaustive if small enough
    total = q ** n
    if total <= 2000:
        # Try greedy set cover on the whole space
        all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
        # Build coverage sets
        balls = []
        for idx, word in enumerate(all_words):
            ball = set()
            # All words within Hamming distance <= radius
            for pos in itertools.product(range(n), repeat=radius):
                # Actually need to enumerate all words within distance
                pass
            # Simpler: for small spaces, just compute distance
            ball.add(idx)
            # Generate all words at distance 1
            for i in range(n):
                for v in range(q):
                    if v == word[i]:
                        continue
                    new_word = word[:i] + [v] + word[i+1:]
                    key = tuple(new_word)
                    # Find index - use a dict for lookup
                    pass
            balls.append(ball)
        # This is getting complicated; just use the tetracode-like approach
        # For general q, n, radius, use a simple greedy
        pass

    # Fallback: return a large trivial code
    return [list(w) for w in itertools.product(range(q), repeat=n)]


def _construct_ternary_6_1(start_time, deadline):
    """
    Specialized constructor for K_3(6,1).
    Strategy:
    1. Start with the tetracode product (81 codewords).
    2. Use greedy set-cover to find a minimal covering from a candidate pool.
    3. Try local search / perturbation to reduce size.
    """
    q, n, radius = 3, 6, 1
    total = q ** n  # 729

    # Precompute all words as tuples
    all_words = list(itertools.product(range(q), repeat=n))

    # Map from word tuple to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Precompute the "ball" of each word: all words within Hamming distance <= 1
    # Ball size = 1 + n*(q-1) = 1 + 6*2 = 13
    balls = []
    for i, w in enumerate(all_words):
        ball = set()
        ball.add(i)
        for pos in range(n):
            for v in range(q):
                if v == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = v
                idx = word_to_idx[tuple(new_w)]
                ball.add(idx)
        balls.append(frozenset(ball))

    # Start with tetracode product as initial seed
    tetracode = _tetracode_words(q, 4)
    # Product with all 2-tails
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            codeword = tuple(h + list(tail))
            idx = word_to_idx[codeword]
            initial_codewords.append(idx)

    # Greedy set cover: start with initial_codewords, try to remove redundant ones
    # First, find a minimal cover by greedy removal
    covered = set()
    for idx in initial_codewords:
        covered |= set(balls[idx])

    # Check if fully covered
    if len(covered) < total:
        # Add more codewords greedily
        uncovered = set(range(total)) - covered
        while uncovered:
            # Find codeword covering most uncovered
            best_idx = -1
            best_count = 0
            for idx in range(total):
                if idx in initial_codewords:
                    continue
                count = len(balls[idx] & uncovered)
                if count > best_count:
                    best_count = count
                    best_idx = idx
            if best_idx == -1 or best_count == 0:
                break
            initial_codewords.append(best_idx)
            covered |= set(balls[best_idx])
            uncovered -= set(balls[best_idx])
            if time.time() > deadline:
                break

    # Now try to minimize: iteratively remove a codeword and re-cover if needed
    best_codewords = initial_codewords[:]
    best_size = len(best_codewords)

    # Local search: try removing each codeword and see if we can re-cover with fewer
    rng = random.Random(42)
    for iteration in range(50):  # limited iterations
        if time.time() > deadline:
            break

        current = best_codewords[:]
        current_set = set(current)
        covered = set()
        for idx in current:
            covered |= set(balls[idx])

        if len(covered) < total:
            continue

        # Try to remove one codeword at a time
        removed = False
        for idx in current[:]:
            new_current = [x for x in current if x != idx]
            new_covered = set()
            for x in new_current:
                new_covered |= set(balls[x])

            if len(new_covered) == total:
                # Can remove this codeword
                best_codewords = new_current
                best_size = len(new_current)
                removed = True
                if time.time() > deadline:
                    break
                break

        if not removed:
            # Try adding a random codeword and then minimizing
            if best_size < total:
                random_idx = rng.randint(0, total - 1)
                if random_idx not in best_codewords:
                    trial = best_codewords + [random_idx]
                    # Try to minimize trial
                    trial_set = set(trial)
                    trial_covered = set()
                    for x in trial:
                        trial_covered |= set(balls[x])
                    if len(trial_covered) == total:
                        # Try removing from trial
                        for idx in trial[:]:
                            new_trial = [x for x in trial if x != idx]
                            new_covered = set()
                            for x in new_trial:
                                new_covered |= set(balls[x])
                            if len(new_covered) == total and len(new_trial) < best_size:
                                best_codewords = new_trial
                                best_size = len(new_trial)
                                break

        if best_size <= 71:
            break

    # Final attempt: greedy from scratch with different initialization
    if time.time() < deadline - 10:
        greedy_cover = _greedy_set_cover(balls, total, deadline)
        if len(greedy_cover) < best_size:
            best_codewords = greedy_cover
            best_size = len(best_codewords)

    # Convert back to list of lists
    result = [list(all_words[idx]) for idx in best_codewords]
    return result


def _greedy_set_cover(balls, total, deadline):
    """
    Greedy set cover to find a small covering code.
    """
    uncovered = set(range(total))
    selected = []
    start_time = time.time()

    while uncovered:
        if time.time() > deadline:
            break
        best_idx = -1
        best_count = 0
        for idx in range(total):
            if idx in selected:
                continue
            count = len(balls[idx] & uncovered)
            if count > best_count:
                best_count = count
                best_idx = idx
        if best_idx == -1 or best_count == 0:
            break
        selected.append(best_idx)
        uncovered -= set(balls[best_idx])

    return selected


# EVOLVE-BLOCK-END
