"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

N = 128
DEADLINE = 80.0  # safety margin before 90s kill

def _has_bad_triple(S):
    """Return True if S contains three distinct members a,b,c with a+b,a+c,b+c all in S."""
    lst = sorted(S)
    L = len(lst)
    for i in range(L):
        for j in range(i + 1, L):
            for k in range(j + 1, L):
                a, b, c = lst[i], lst[j], lst[k]
                if (a + b) in S and (a + c) in S and (b + c) in S:
                    return True
    return False

def _validate(S):
    return not _has_bad_triple(S)

def _greedy_build(rng, seed_set=None):
    """Build a valid set by greedily adding random elements from 1..128."""
    if seed_set is not None:
        S = set(seed_set)
    else:
        S = set()
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    for x in candidates:
        if x in S:
            continue
        S.add(x)
        if not _validate(S):
            S.remove(x)
    return S

def _greedy_remove_and_add(S, rng):
    """Try removing one element to unblock, then greedily add more."""
    S = set(S)
    # Try each element as a potential removal
    for rem in list(S):
        S.remove(rem)
        # Now try to add back any element (including rem) in random order
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        improved = False
        for x in candidates:
            if x in S:
                continue
            S.add(x)
            if not _validate(S):
                S.remove(x)
            else:
                improved = True
        if len(S) > len(set(seed_for_compare)):
            return S
    return S

def construct():
    start = time.time()
    best = None
    best_len = 0
    rng = random.Random(42)

    # Initial: use the known-good half-interval as a strong starting point
    initial = list(range(N // 2 + 1, N + 1))
    if _validate(set(initial)):
        best = set(initial)
        best_len = len(best)

    # Greedy restarts with random seeds
    while time.time() < DEADLINE:
        # Random seed: pick a random subset to start from
        seed_size = rng.randint(0, 30)
        seed = set(rng.sample(range(1, N + 1), seed_size))
        S = _greedy_build(rng, seed)
        if len(S) > best_len:
            best = S
            best_len = len(S)
            if time.time() > DEADLINE:
                break

        # Also try extending the current best
        S2 = _greedy_build(rng, best)
        if len(S2) > best_len:
            best = S2
            best_len = len(S2)

        # Occasionally try local search: remove one, re-add greedily
        if best and time.time() < DEADLINE - 5:
            for _ in range(3):
                rem = rng.choice(list(best))
                S3 = set(best)
                S3.remove(rem)
                candidates = list(range(1, N + 1))
                rng.shuffle(candidates)
                for x in candidates:
                    if x in S3:
                        continue
                    S3.add(x)
                    if not _validate(S3):
                        S3.remove(x)
                if len(S3) > best_len:
                    best = S3
                    best_len = len(S3)

    # Final validation
    if best is None:
        return list(range(N // 2 + 1, N + 1))
    result = sorted(best)
    assert _validate(set(result)), "Final set is invalid!"
    return result

# EVOLVE-BLOCK-END
