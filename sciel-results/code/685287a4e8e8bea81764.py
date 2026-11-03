"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct(q, n, radius):
    if n == 0:
        return [[]]
    if radius < 0:
        return []
    if radius >= n:
        return [[0] * n]
    total = q ** n
    # Precompute digit powers for encoding/decoding
    pow_q = [1] * n
    for i in range(1, n):
        pow_q[i] = pow_q[i-1] * q

    def encode(word):
        v = 0
        for i, d in enumerate(word):
            v += d * pow_q[i]
        return v

    def decode(v):
        w = [0] * n
        for i in range(n):
            w[i] = v % q
            v //= q
        return w

    # Precompute ball (neighborhood) for each word as a frozenset of ints and a list
    balls = []
    for v in range(total):
        w = decode(v)
        # start with self
        ball = {v}
        if radius >= 1:
            for pos in range(n):
                orig = w[pos]
                for newd in range(q):
                    if newd == orig:
                        continue
                    w[pos] = newd
                    ball.add(encode(w))
                w[pos] = orig
        if radius >= 2:
            for p1 in range(n):
                for p2 in range(p1 + 1, n):
                    o1 = w[p1]
                    o2 = w[p2]
                    for d1 in range(q):
                        if d1 == o1:
                            continue
                        w[p1] = d1
                        for d2 in range(q):
                            if d2 == o2:
                                continue
                            w[p2] = d2
                            ball.add(encode(w))
                        w[p2] = o2
                    w[p1] = o1
        balls.append(frozenset(ball))

    ball_list = [list(b) for b in balls]

    # Greedy construction with bitset coverage for speed
    def greedy_init():
        # Use Python int as bitset for covered words
        covered_bits = 0
        uncovered = total
        codewords = []
        # We'll iterate over all words each time; for q=3,n=8,total=6561 it's OK.
        # To avoid O(total^2 * ballsize), we maintain a list of uncovered indices? 
        # Simpler: keep a set of uncovered words and sample.
        uncovered_set = set(range(total))
        while uncovered_set:
            best = -1
            best_gain = -1
            # Randomized greedy: evaluate a random subset first to get a good candidate, then full scan if time allows?
            # For correctness of feasibility, we need exact best or at least some word that covers at least one.
            # We'll do a randomized scan over a limited number of candidates to keep speed, but ensure progress.
            candidates = list(uncovered_set)
            random.shuffle(candidates)
            # Evaluate up to K candidates, choose best gain
            K = min(len(candidates), 200)
            for cand in candidates[:K]:
                b = balls[cand]
                gain = 0
                # Count uncovered in ball
                for u in b:
                    if u in uncovered_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best = cand
                if best_gain == len(uncovered_set):
                    break
            if best == -1 or best_gain <= 0:
                # Fallback: pick any uncovered word (its ball includes itself)
                best = next(iter(uncovered_set))
            codewords.append(best)
            # Remove covered words
            for u in balls[best]:
                if u in uncovered_set:
                    uncovered_set.remove(u)
            # If best was in uncovered_set, remove it (already removed if in ball)
            uncovered_set.discard(best)
        return codewords

    def optimize(codewords, time_limit):
        # Maintain coverage counts as array of ints for fast unique-coverage checks
        cov = [0] * total
        for cw in codewords:
            for u in ball_list[cw]:
                cov[u] += 1
        cw_set = set(codewords)

        def remove_redundant():
            changed = True
            removed = 0
            while changed:
                changed = False
                # Try to remove any codeword whose ball is fully covered by others
                # Sample indices to avoid worst-case
                idxs = list(range(len(codewords)))
                random.shuffle(idxs)
                for i in idxs:
                    cw = codewords[i]
                    # check if any word in its ball has coverage 1
                    unique = False
                    for u in ball_list[cw]:
                        if cov[u] == 1:
                            unique = True
                            break
                    if not unique:
                        # safe to remove
                        for u in ball_list[cw]:
                            cov[u] -= 1
                        codewords.pop(i)
                        cw_set.discard(cw)
                        changed = True
                        removed += 1
                        break
            return removed

        remove_redundant()

        start = time.time()
        # Local search: move/swap codewords to reduce size
        # We'll primarily do:
        # 1) Try removal (already done)
        # 2) Try replacing a codeword with another that covers its unique words and adds redundancy
        while time.time() - start < time_limit:
            remove_redundant()
            if not codewords:
                break
            i = random.randrange(len(codewords))
            old = codewords[i]
            # Find words uniquely covered by old
            unique_words = []
            for u in ball_list[old]:
                if cov[u] == 1:
                    unique_words.append(u)
            if not unique_words:
                # remove
                for u in ball_list[old]:
                    cov[u] -= 1
                codewords.pop(i)
                cw_set.discard(old)
                continue
            # Try to find a replacement that covers all unique_words and is not already present
            # Candidate pool: union of balls of unique_words, plus random samples
            cand_set = set()
            for u in unique_words[:10]:  # limit to first 10 for speed
                cand_set.update(ball_list[u])
            # Add some random candidates
            for _ in range(20):
                cand_set.add(random.randrange(total))
            best_new = -1
            best_score = -1
            for new in cand_set:
                if new == old or new in cw_set:
                    continue
                # Must cover all unique_words
                ok = True
                for u in unique_words:
                    if u not in balls[new]:
                        ok = False
                        break
                if not ok:
                    continue
                # Score: number of words in new's ball that are currently covered (redundancy gain)
                # and not in old's ball
                score = 0
                for u in ball_list[new]:
                    if u not in balls[old] and cov[u] > 0:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_new = new
            if best_new != -1:
                # Apply replacement
                for u in ball_list[old]:
                    cov[u] -= 1
                for u in ball_list[best_new]:
                    cov[u] += 1
                codewords[i] = best_new
                cw_set.discard(old)
                cw_set.add(best_new)

        remove_redundant()
        return codewords

    global_start = time.time()
    time_limit = 115
    best = None
    best_len = float('inf')

    while time.time() - global_start < time_limit - 2:
        # Greedy
        cw = greedy_init()
        # Optimize with remaining time minus safety
        opt_time = max(1.0, time_limit - (time.time() - global_start) - 2.0)
        cw = optimize(cw, opt_time)
        if len(cw) < best_len:
            best_len = len(cw)
            best = cw
        # If we hit a very good solution, we can stop early? Keep searching for better.
        if time.time() - global_start > time_limit - 2:
            break

    if best is None:
        # Fallback: all words
        best = list(range(total))

    return [decode(cw) for cw in best]
# EVOLVE-BLOCK-END
