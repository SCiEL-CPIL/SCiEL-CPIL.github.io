"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import product, combinations

P = {'n': 40, 'maximum_size': 18}

# Primes up to 40
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
PRIME_TO_IDX = {p: i for i, p in enumerate(PRIMES)}
DIM = len(PRIMES)

def get_vector(n):
    """Get the prime exponent vector for n."""
    vec = [0] * DIM
    for i, p in enumerate(PRIMES):
        while n % p == 0:
            vec[i] += 1
            n //= p
    return tuple(vec)

# Precompute vectors for all candidates 2..40
CANDIDATES = list(range(2, P['n'] + 1))
VECTORS = [get_vector(x) for x in CANDIDATES]
ZERO_VEC = tuple([0] * DIM)

def expand_sums(indices):
    """Generate all {-1,0,1} linear combinations of vectors at given indices."""
    if not indices:
        return {ZERO_VEC}
    sums = {ZERO_VEC}
    for idx in indices:
        v = VECTORS[idx]
        new_sums = set()
        for s in sums:
            new_sums.add(s)
            new_sums.add(tuple(s[i] + v[i] for i in range(DIM)))
            new_sums.add(tuple(s[i] - v[i] for i in range(DIM)))
        sums = new_sums
    return sums

def is_independent_mitm(selected_indices, new_idx):
    """
    Check if the vector at new_idx is in the {-1, 0, 1}-span of the vectors at selected_indices.
    If yes, adding it creates a dependency (collision).
    If no, it is independent.
    """
    v_new = VECTORS[new_idx]
    if not selected_indices:
        return True

    k = len(selected_indices)
    mid = k // 2
    left_indices = selected_indices[:mid]
    right_indices = selected_indices[mid:]

    left_sums = expand_sums(left_indices)
    right_sums = expand_sums(right_indices)

    for r_sum in right_sums:
        target = tuple(v_new[i] - r_sum[i] for i in range(DIM))
        if target in left_sums:
            return False

    return True

def greedy_add(S, deadline):
    """Greedily add independent elements to set S."""
    added = True
    while added and time.time() < deadline:
        added = False
        candidates = [c for c in range(len(CANDIDATES)) if c not in S]
        random.shuffle(candidates)
        for c in candidates:
            if is_independent_mitm(list(S), c):
                S.add(c)
                added = True
                break
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    # Initial best set: Primes
    primes = [i for i, x in enumerate(CANDIDATES) if all(x % d for d in range(2, int(x**0.5)+1))]
    best_set = set(primes)
    best_size = len(best_set)

    for restart in range(200):
        if time.time() > deadline:
            break

        # Vary starting set for diversity
        if restart % 3 == 0:
            S = set(primes)
        elif restart % 3 == 1:
            # Start from primes minus one random prime
            S = set(primes)
            if S:
                S.discard(random.choice(list(S)))
        else:
            # Start from a random subset of primes
            S = set(random.sample(primes, max(1, len(primes) - random.randint(0, 3))))

        # Greedy Add
        S = greedy_add(S, deadline)

        if len(S) > best_size:
            best_set = S
            best_size = len(S)
            if best_size == P['maximum_size']:
                break

        # 1-swap Phase
        if len(S) < P['maximum_size'] and time.time() < deadline - 3:
            for _ in range(80):
                if time.time() > deadline:
                    break
                if not S:
                    break
                to_remove = random.choice(list(S))
                S.remove(to_remove)

                candidates = [c for c in range(len(CANDIDATES)) if c not in S]
                if not candidates:
                    S.add(to_remove)
                    break
                to_add = random.choice(candidates)

                if is_independent_mitm(list(S), to_add):
                    S.add(to_add)
                    S = greedy_add(S, deadline)
                    if len(S) > best_size:
                        best_set = S
                        best_size = len(S)
                        if best_size == P['maximum_size']:
                            break
                else:
                    S.add(to_remove)

        # 2-swap Phase (remove two, add two)
        if len(S) < P['maximum_size'] and time.time() < deadline - 2:
            for _ in range(30):
                if time.time() > deadline:
                    break
                if len(S) < 2:
                    break
                to_remove1, to_remove2 = random.sample(list(S), 2)
                S.discard(to_remove1)
                S.discard(to_remove2)

                candidates = [c for c in range(len(CANDIDATES)) if c not in S]
                if len(candidates) < 2:
                    S.add(to_remove1)
                    S.add(to_remove2)
                    break
                to_add1, to_add2 = random.sample(candidates, 2)

                # Check if both additions are valid
                if is_independent_mitm(list(S), to_add1) and is_independent_mitm(list(S | {to_add1}), to_add2):
                    S.add(to_add1)
                    S.add(to_add2)
                    S = greedy_add(S, deadline)
                    if len(S) > best_size:
                        best_set = S
                        best_size = len(S)
                        if best_size == P['maximum_size']:
                            break
                else:
                    S.add(to_remove1)
                    S.add(to_remove2)

        if best_size == P['maximum_size']:
            break

    # Convert indices back to values
    return [CANDIDATES[i] for i in best_set]

# EVOLVE-BLOCK-END
