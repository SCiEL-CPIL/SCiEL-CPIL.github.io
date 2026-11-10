import math
import time


def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255,
    with gcd(a, b) > 1 for all distinct pairs.
    Maximizes cardinality.
    """
    deadline = time.time() + 50.0  # safety margin under 60s

    N = 255

    # The smallest prime factor of 255 is 3 (255 = 3 * 5 * 17).
    # The set of all multiples of 3 in [1, 255] is pairwise non-coprime
    # (any two share factor 3) and contains 255. Its size is floor(255/3) = 85.
    base_set = list(range(3, 256, 3))

    def is_valid(s):
        if not s:
            return False
        if 255 not in s:
            return False
        if any(x < 1 or x > 255 for x in s):
            return False
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                if math.gcd(s[i], s[j]) == 1:
                    return False
        return True

    best = base_set
    if not is_valid(best):
        best = [255]

    # Try to see if a larger set exists by attempting to add multiples of 5 or 17
    # that are not multiples of 3, while maintaining pairwise non-coprime property.
    # This is a greedy expansion attempt; we keep the best valid set found.
    candidates_to_add = [x for x in range(1, 256)
                         if x not in set(base_set) and (x % 5 == 0 or x % 17 == 0)]
    current = list(base_set)
    current_set = set(current)

    # Greedily try to add candidates if they remain non-coprime with all current members.
    for cand in candidates_to_add:
        if time.time() > deadline:
            break
        ok = all(math.gcd(cand, m) > 1 for m in current)
        if ok:
            current.append(cand)
            current_set.add(cand)

    if len(current) > len(best) and is_valid(current):
        best = current

    # Also try starting from multiples of 5 and 17 as baselines (smaller, but check).
    for p in (5, 17):
        if time.time() > deadline:
            break
        s = list(range(p, 256, p))
        if 255 not in s:
            s.append(255)
        if is_valid(s) and len(s) > len(best):
            best = s

    # Final safety: ensure validity.
    if not is_valid(best):
        best = [255]

    return best
