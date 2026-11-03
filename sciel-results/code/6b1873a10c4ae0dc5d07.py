import itertools
import random
import time


def _ball_indices(w, q, n, radius):
    """Yield the linear index of every word within Hamming distance <= radius of w."""
    idx = 0
    for d in range(radius + 1):
        for pos in itertools.combinations(range(n), d):
            if d == 0:
                yield idx
            else:
                base = list(w)
                for deltas in itertools.product(range(1, q), repeat=d):
                    for i, p in enumerate(pos):
                        base[p] = (w[p] + deltas[i]) % q
                    yield 0
                    for p in range(n):
                        yield base[p]
                    # compute index
                    s = 0
                    for p in range(n):
                        s = s * q + base[p]
                    yield s
                base = list(w)


def _word_to_indices(w, q, n, radius):
    """Return a set of linear indices covered by word w within radius."""
    result = set()
    for d in range(radius + 1):
        for pos in itertools.combinations(range(n), d):
            if d == 0:
                s = 0
                for p in range(n):
                    s = s * q + w[p]
                result.add(s)
            else:
                base = list(w)
                for deltas in itertools.product(range(1, q), repeat=d):
                    for i, p in enumerate(pos):
                        base[p] = (w[p] + deltas[i]) % q
                    s = 0
                    for p in range(n):
                        s = s * q + base[p]
                    result.add(s)
    return result


def _idx_to_word(idx, q, n):
    w = [0] * n
    for p in range(n - 1, -1, -1):
        w[p] = idx % q
        idx //= q
    return w


def _hamming_dist(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def _covered_by(codeword, q, n, radius):
    """Return set of indices covered by a codeword."""
    return _word_to_indices(codeword, q, n, radius)


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    total_words = q ** n
    target = 73  # best known

    # --- Construction: split into head (4 coords) and tail (n-4 coords) ---
    # Head: tetracode (9 perfect words, radius 1)
    # Tail: cover all 3^(n-4) tail words with radius 1
    # For n=6: tail is 2 coords, radius 1 covers all 9 with 9 words (each covers 5)
    # But we can do better: use fewer tail words and let head cover the rest.

    # Strategy: For each tail word t, assign a set of head words H_t.
    # A word (h, t) covers (h', t') if dist(h,h') <= radius_head and dist(t,t') <= radius_tail
    # where radius_head + radius_tail = radius.
    # We want to minimize sum_t |H_t| such that for every (h, t'):
    #   exists t with dist(t, t') <= radius_tail and h in H_t with dist(h, h') <= radius_head
    # Actually simpler: for each tail word t', the set of head words that need covering is
    #   { h : not exists t with dist(t,t')<=r_tail and h in H_t with dist(h,h')<=r_head }
    # This must be empty.

    # For n=6, radius=1: split as head=4, tail=2, r_head=1, r_tail=0? No.
    # r_head + r_tail = 1. Options: (1,0) or (0,1).
    # (1,0): tail must be exact match. So for each tail word t, H_t must cover all 9 head words
    #         with radius 1. But tetracode already does that with 9 words. So H_t = tetracode for all t.
    #         Total = 9 * 9 = 81. Same as before.
    # (0,1): head must be exact match. For each tail word t, H_t must be a set of head words such that
    #         for every head word h', there exists t' with dist(t,t')<=1 and h' in H_{t'}... 
    #         Wait, this doesn't work directly.

    # Let me think differently. The covering condition for (h, t'):
    #   exists (h_c, t_c) in code such that dist(h, h_c) + dist(t, t_c) <= 1
    # This means either:
    #   (a) dist(h, h_c) <= 1 and dist(t, t_c) = 0, i.e., t_c = t' and h_c within 1 of h
    #   (b) dist(h, h_c) = 0 and dist(t, t_c) <= 1, i.e., h_c = h and t_c within 1 of t'

    # So for each (h, t'), we need either:
    #   - some codeword (h_c, t') with dist(h, h_c) <= 1, OR
    #   - some codeword (h, t_c) with dist(t, t_c) <= 1

    # This is a 2D covering problem. Let's use a greedy approach.

    # For n=6: head is 4 coords, tail is 2 coords.
    # Tail space: 9 words. Head space: 9 words (tetracode).
    # Actually head space is all 3^4 = 81 words, but we only need to cover the 9 tetracode words?
    # No, we need to cover ALL 81 head words.

    # Let me just use a greedy set cover on the full 729-word space.

    # Precompute balls for all 729 words
    # Each word's ball (radius 1) has 1 + 6*2 = 13 words.

    # Greedy: repeatedly pick the word that covers the most uncovered words.

    uncovered = set(range(total_words))
    code = []

    # Precompute all balls
    all_balls = {}
    for idx in range(total_words):
        w = _idx_to_word(idx, q, n)
        all_balls[idx] = _word_to_indices(w, q, n, radius)

    # Greedy set cover
    while uncovered:
        if time.time() > deadline:
            break
        best_idx = -1
        best_count = -1
        for idx in range(total_words):
            if idx in code_set if 'code_set' in dir() else True:
                cnt = len(all_balls[idx] & uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_idx = idx
        if best_idx == -1 or best_count <= 0:
            break
        code.append(_idx_to_word(best_idx, q, n))
        uncovered -= all_balls[best_idx]

    if not uncovered:
        return code

    # Fallback: direct sum
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    if n >= 4:
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        return [list(w) for w in itertools.product(range(3), repeat=n)]


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0

    total_words = q ** n

    # For n=6, q=3, radius=1: use structured construction
    if q == 3 and n == 6 and radius == 1:
        return _construct_3_6_1(deadline)
    
    # General fallback: greedy
    return _greedy_cover(q, n, radius, deadline)


def _construct_3_6_1(deadline):
    """Specialized construction for q=3, n=6, radius=1."""
    # Head: 4 coords, Tail: 2 coords
    # Use tetracode for head (9 words, perfect radius 1)
    # For tail, we need to cover 9 tail words with radius 1
    # But we can share: a codeword (h, t) covers (h', t') if dist(h,h')<=1 and t=t', or h=h' and dist(t,t')<=1
    
    # Greedy on the 729-word space with structured initialization
    
    # Precompute balls
    all_balls = {}
    for idx in range(729):
        w = _idx_to_word(idx, 3, 6)
        all_balls[idx] = _word_to_indices(w, 3, 6, 1)

    uncovered = set(range(729))
    code = []
    code_set = set()

    # Initialize with tetracode x {0,0} to get a good start
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    for h in tetracode:
        w = h + [0, 0]
        idx = 0
        for p in range(6):
            idx = idx * 3 + w[p]
        code.append(w)
        code_set.add(idx)
        uncovered -= all_balls[idx]

    # Now greedily add words to cover remaining
    while uncovered and time.time() < deadline:
        best_idx = -1
        best_count = -1
        for idx in range(729):
            if idx not in code_set:
                cnt = len(all_balls[idx] & uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_idx = idx
        if best_idx == -1 or best_count <= 0:
            break
        w = _idx_to_word(best_idx, 3, 6)
        code.append(w)
        code_set.add(best_idx)
        uncovered -= all_balls[best_idx]

    if not uncovered:
        return code

    # Fallback
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]


def _greedy_cover(q, n, radius, deadline):
    total_words = q ** n
    all_balls = {}
    for idx in range(total_words):
        w = _idx_to_word(idx, q, n)
        all_balls[idx] = _word_to_indices(w, q, n, radius)

    uncovered = set(range(total_words))
    code = []
    code_set = set()

    while uncovered and time.time() < deadline:
        best_idx = -1
        best_count = -1
        for idx in range(total_words):
            if idx not in code_set:
                cnt = len(all_balls[idx] & uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_idx = idx
        if best_idx == -1 or best_count <= 0:
            break
        w = _idx_to_word(best_idx, q, n)
        code.append(w)
        code_set.add(best_idx)
        uncovered -= all_balls[best_idx]

    if not uncovered:
        return code

    # Fallback: direct sum
    if n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        return [list(w) for w in itertools.product(range(3), repeat=n)]
