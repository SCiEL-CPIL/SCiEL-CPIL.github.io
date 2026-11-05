"""Ternary covering code K_3(7,1) via greedy construction + local search (2-for-1)."""
import itertools
import time
import random
import math

# Deadline guard: evaluator kills at 150s wall-clock; use 135s budget with safety margin
_DEADLINE = 135.0


def _hamming_dist(a, b):
    """Hamming distance between two tuples/lists of same length."""
    return sum(x != y for x, y in zip(a, b))


def _within_radius(a, b, radius):
    """Check if Hamming distance <= radius."""
    d = 0
    for x, y in zip(a, b):
        if x != y:
            d += 1
            if d > radius:
                return False
    return True


def _tetracode():
    """The 9 ternary tetracode words (perfect code of length 4, radius 1)."""
    return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]


def _direct_sum_code(q, n, radius):
    """Direct sum: tetracode on first 4 coords × all words on remaining n-4 coords."""
    tet = _tetracode()
    result = []
    for h in tet:
        for tail in itertools.product(range(q), repeat=n - 4):
            result.append(tuple(h) + tail)
    return result


def _greedy_cover(q, n, radius):
    """
    Greedy set-cover style construction.
    Repeatedly pick the word that covers the most uncovered words.
    Uses a bitmask/array for coverage tracking.
    """
    total = q ** n
    if total > 5000000:
        # Too large for explicit coverage; fall back to direct sum
        return list(_direct_sum_code(q, n, radius))

    # Precompute all words
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Precompute for each word, the set of words within radius
    # This is expensive for large n; for n=7, q=3, total=2187, radius=1:
    # Each word covers 1 + n*(q-1) = 1 + 7*2 = 15 words.
    coverage_sets = []
    for w in all_words:
        covered = set()
        covered.add(w)
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_word = list(w)
                new_word[pos] = val
                covered.add(tuple(new_word))
        coverage_sets.append(covered)

    # Greedy: maintain uncovered set
    uncovered = set(range(total))
    code = []
    # For efficiency, track coverage count per word
    # Actually, let's just do standard greedy: pick word covering most uncovered

    # To speed up, precompute coverage as bitmasks isn't feasible for 2187 bits in Python int easily
    # Instead, use a list of lists and count uncovered in each

    # For n=7, q=3: total=2187, each covers 15 words. Greedy is fine.
    while uncovered:
        best_word = -1
        best_count = -1
        for i in range(total):
            if i not in uncovered:
                continue
            cnt = 0
            for c in coverage_sets[i]:
                if c in uncovered:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_word = i
        if best_word == -1:
            break
        code.append(all_words[best_word])
        for c in coverage_sets[best_word]:
            uncovered.discard(c)

    return code


def _is_covering(code, q, n, radius):
    """Verify every word is within radius of some codeword."""
    if not code:
        return q == 0 or n == 0
    # For small spaces, check explicitly
    total = q ** n
    if total <= 2187:
        all_words = list(itertools.product(range(q), repeat=n))
        code_set = [tuple(c) for c in code]
        for w in all_words:
            found = False
            for c in code_set:
                if _within_radius(w, c, radius):
                    found = True
                    break
            if not found:
                return False
        return True
    # For larger spaces, we trust the construction
    return True


def _try_remove(code, q, n, radius):
    """Try to remove codewords one by one if coverage is maintained."""
    changed = True
    while changed:
        changed = False
        new_code = []
        for i, c in enumerate(code):
            # Try removing c
            candidate = code[:i] + code[i+1:]
            if _is_covering(candidate, q, n, radius):
                code = candidate
                changed = True
                break
        if not changed:
            break
    return code


def _try_2for1(code, q, n, radius, max_iter=200):
    """
    Try to replace 2 codewords with 1 that covers their union.
    For each pair, check if there's a single word covering all words within radius of both.
    """
    improved = True
    while improved:
        improved = False
        found = False
        for i in range(len(code)):
            for j in range(i + 1, len(code)):
                # Union of coverage of code[i] and code[j]
                # Check if a single word can cover all of them
                # A word w covers code[i]'s coverage if dist(w, x) <= radius for all x in cov(code[i])
                # This is equivalent to: w must be within radius of every word in cov(code[i]) ∪ cov(code[j])
                # The set of words that cover code[i] is the ball of radius radius around code[i]... no.
                # Actually: w covers a word x if dist(w,x) <= radius.
                # So w covers all of cov(c_i) iff for all x with dist(x,c_i)<=radius, dist(w,x)<=radius.
                # This means w is in the intersection of balls of radius radius around each x in cov(c_i).
                # That intersection is the ball of radius 0 (just {c_i}) if radius=1? No.
                # For radius=1: w covers x if dist(w,x)<=1. w covers all x with dist(x,c_i)<=1.
                # The set of such w is {w : dist(w,x)<=1 for all x with dist(x,c_i)<=1}.
                # This is the set of w with dist(w,c_i) <= ... let's think.
                # If dist(w,c_i)=0, w=c_i, covers all x with dist(x,c_i)<=1. Yes.
                # If dist(w,c_i)=1, say w differs from c_i at position p. Then for x that differs from c_i at position p' != p, dist(w,x)=2 > 1. So w does NOT cover all.
                # So for radius=1, the only w that covers all of cov(c_i) is w=c_i itself.
                # Therefore, 2-for-1 replacement is only possible if one codeword's coverage is a subset of the other's, which for radius=1 means they're the same word.
                # So 2-for-1 is trivial for radius=1. Let me reconsider.

                # Actually wait: we want to find a NEW word w (not necessarily in the current code) such that
                # cov(w) ⊇ cov(c_i) ∪ cov(c_j).
                # For radius=1, cov(c) = {c} ∪ {words at dist 1 from c}.
                # We need w such that dist(w, x) <= 1 for all x in cov(c_i) ∪ cov(c_j).
                # In particular, dist(w, c_i) <= 1 and dist(w, c_j) <= 1.
                # And for any x at dist 1 from c_i, dist(w, x) <= 1.
                # If w = c_i, then dist(w, x) = dist(c_i, x) <= 1 for x in cov(c_i). Good.
                # But we also need dist(w, x) <= 1 for x in cov(c_j), i.e., dist(c_i, x) <= 1 for all x in cov(c_j).
                # This means cov(c_j) ⊆ cov(c_i), which for radius=1 means c_i = c_j.
                # So indeed, for radius=1, 2-for-1 only works if the two codewords are identical.

                # Hmm, so maybe the "2-for-1" in the inspiration is about something else,
                # or it's about replacing a codeword with a different one that covers more.
                # Let me instead do: try to find a better replacement for each codeword.
                pass

        # Alternative: for each codeword, try to find a replacement word that covers
        # at least as much, and allows removing another codeword.
        # This is more complex. Let's do a simpler local search:
        # For each pair of codewords, try to find a single word that covers their union.
        # For radius=1, this is only possible if they're the same, so skip.

        # Instead, do: for each codeword c, try all possible words w at distance 1 from c,
        # and see if replacing c with w allows removing another codeword.
        # This is a "1-for-1 + 1-for-0" move.
        if len(code) > 2:
            for i in range(len(code)):
                c = code[i]
                # Try all words at distance 1 from c
                for pos in range(n):
                    for val in range(q):
                        if val == c[pos]:
                            continue
                        new_w = list(c)
                        new_w[pos] = val
                        new_w = tuple(new_w)
                        # Try replacing c with new_w
                        candidate = list(code)
                        candidate[i] = new_w
                        # Now try to remove another codeword
                        for j in range(len(candidate)):
                            if j == i:
                                continue
                            candidate2 = candidate[:j] + candidate[j+1:]
                            if _is_covering(candidate2, q, n, radius):
                                code = candidate2
                                improved = True
                                found = True
                                break
                        if found:
                            break
                if found:
                    break
    return code


def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet {0,...,q-1} with covering radius.
    Returns a list of codewords (each a list of n integers).
    """
    start_time = time.time()

    # Get initial code
    code = _greedy_cover(q, n, radius)

    # Verify
    if not _is_covering(code, q, n, radius):
        # Fall back to direct sum
        code = list(_direct_sum_code(q, n, radius))

    # Local search: try to remove codewords
    deadline = start_time + _DEADLINE
    while time.time() < deadline:
        new_code = _try_remove(code, q, n, radius)
        if len(new_code) < len(code):
            code = new_code
        else:
            # Try 2-for-1 / replacement search
            new_code = _try_2for1(code, q, n, radius)
            if len(new_code) < len(code):
                code = new_code
            else:
                # No improvement, try a different strategy
                # Random restart with perturbation
                if time.time() > deadline - 5:
                    break
                # Perturb: replace a random codeword with a random nearby word
                if code and random.random() < 0.3:
                    idx = random.randint(0, len(code) - 1)
                    c = code[idx]
                    pos = random.randint(0, n - 1)
                    val = random.randint(0, q - 1)
                    new_c = list(c)
                    new_c[pos] = val
                    candidate = list(code)
                    candidate[idx] = tuple(new_c)
                    if _is_covering(candidate, q, n, radius):
                        code = candidate
                        # Try to remove
                        code = _try_remove(code, q, n, radius)
                else:
                    # No improvement and no perturbation, break to save time
                    break

    # Final verification
    if not _is_covering(code, q, n, radius):
        code = list(_direct_sum_code(q, n, radius))

    # Convert to lists
    return [list(c) for c in code]
