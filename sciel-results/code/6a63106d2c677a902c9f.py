"""New design: difference-space coverage via greedy construction + local search."""
import time
import random
import sys

def _score(s):
    """Count distinct d>0 with a 3-term AP in set s."""
    s = set(s)
    diffs = set()
    n = len(s)
    sorted_s = sorted(s)
    for i in range(n):
        for j in range(i+1, n):
            d = sorted_s[j] - sorted_s[i]
            if d > 0 and (sorted_s[i] + 2*d) in s:
                diffs.add(d)
    return len(diffs)

def _score_bitmask(s):
    """Faster scoring using a set for membership."""
    sset = set(s)
    diffs = set()
    for a in sset:
        for d in range(1, 256):
            if a + d in sset and a + 2*d in sset:
                diffs.add(d)
    return len(diffs)

def _score_fast(s):
    """Score using sorted list and two-pointer style."""
    sset = set(s)
    diffs = set()
    for a in sset:
        # For each a, check d values
        max_d = min((512 - a) // 2, 255)
        for d in range(1, max_d + 1):
            if (a + d) in sset and (a + 2*d) in sset:
                diffs.add(d)
    return len(diffs)

def construct():
    start = time.time()
    deadline = start + 130.0  # safety margin before 150s kill

    best_set = list(range(1, 33))
    best_score = _score(best_set)

    random.seed(42)

    # Precompute for each d, the valid midpoints m where m-d >= 1 and m+d <= 512
    # d is valid if there exists m with m-d >= 1 and m+d <= 512
    # i.e., d+1 <= m <= 512-d, so d <= 255

    # Greedy construction: build set element by element, choosing element that
    # maximizes new differences covered
    def greedy_construct():
        """Build a 32-element set greedily."""
        s = set()
        # Start with some anchor points
        # Try to cover many differences early
        # Heuristic: pick elements spread out
        candidates = list(range(1, 513))
        random.shuffle(candidates)

        # First few elements: spread them out
        # Try a few different starting strategies
        best_greedy = None
        best_greedy_score = 0

        for trial in range(5):
            s = set()
            # Pick first element
            first = random.choice(range(1, 513))
            s.add(first)

            for _ in range(31):
                best_elem = None
                best_gain = -1
                # Sample candidates (not all 512 for speed)
                sample_size = min(100, 512 - len(s))
                sample = random.sample(list(range(1, 513)), sample_size)
                sample = [x for x in sample if x not in s]
                if not sample:
                    break

                for elem in sample:
                    new_s = s | {elem}
                    # Quick score estimate
                    sc = _score_fast(new_s)
                    if sc > best_gain:
                        best_gain = sc
                        best_elem = elem

                if best_elem is not None:
                    s.add(best_elem)
                else:
                    # Add random
                    remaining = [x for x in range(1, 513) if x not in s]
                    if remaining:
                        s.add(random.choice(remaining))

            sc = _score(s)
            if sc > best_greedy_score:
                best_greedy_score = sc
                best_greedy = list(s)

        return best_greedy, best_greedy_score

    # Run a few greedy constructions
    for _ in range(3):
        if time.time() > deadline:
            break
        gs, gsc = greedy_construct()
        if gsc > best_score:
            best_score = gsc
            best_set = gs

    # Local search: swap elements to improve score
    def local_search(s, max_iters=5000):
        s = list(s)
        current_score = _score(s)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            random.shuffle(s)
            for i in range(len(s)):
                if time.time() > deadline:
                    return s, current_score
                for j in range(len(s)):
                    if i == j:
                        continue
                    # Try replacing s[i] with a random new element
                    new_elem = random.randint(1, 512)
                    if new_elem in s:
                        continue
                    old_elem = s[i]
                    s[i] = new_elem
                    new_score = _score(s)
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                    else:
                        s[i] = old_elem
        return s, current_score

    # Run local search from best
    if time.time() < deadline:
        ls_set, ls_score = local_search(best_set)
        if ls_score > best_score:
            best_score = ls_score
            best_set = ls_set

    # Try more random restarts with local search
    restarts = 0
    while time.time() < deadline - 5.0:
        restarts += 1
        # Random 32-element set
        s = random.sample(range(1, 513), 32)
        sc = _score(s)
        if sc > best_score:
            best_score = sc
            best_set = s
        # Local search from this
        if time.time() < deadline - 5.0:
            ls_set, ls_score = local_search(s, max_iters=2000)
            if ls_score > best_score:
                best_score = ls_score
                best_set = ls_set

    return best_set

# Quick self-test
if __name__ == "__main__":
    result = construct()
    print(f"Set size: {len(result)}")
    print(f"Distinct: {len(set(result))}")
    print(f"Min: {min(result)}, Max: {max(result)}")
    print(f"Score: {_score(result)}")
    print(f"Set: {sorted(result)}")
