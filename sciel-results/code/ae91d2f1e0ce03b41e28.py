"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    def is_valid_addition(s, candidate):
        """Check if adding candidate to s keeps the set valid."""
        s_with = s | {candidate}
        for a in s:
            if a == candidate:
                continue
            for b in s:
                if b <= a:
                    continue
                if (a + b) in s_with and (a + candidate) in s_with and (b + candidate) in s_with:
                    return False
        return True

    base_A = set(range(65, 129))
    base_B = set(range(17, 33))
    candidates = list(range(33, 65))

    best_set = base_A | base_B
    best_size = len(best_set)

    # Deterministic strategies
    strategies = [
        list(range(33, 65)),
        list(range(64, 32, -1)),
    ]

    # Random orderings
    rng = random.Random(42)
    for _ in range(20):
        c = candidates[:]
        rng.shuffle(c)
        strategies.append(c)

    for strat in strategies:
        if time.time() > deadline:
            break
        S = base_A | base_B
        for x in strat:
            if time.time() > deadline:
                break
            if is_valid_addition(S, x):
                S.add(x)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    result = [int(x) for x in sorted(best_set)]
    return result
# EVOLVE-BLOCK-END
