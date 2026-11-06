"""Construct a 12-state binary synchronizing automaton maximizing the exact reset threshold."""
import time
import random
from collections import deque

N = 12
DEADLINE = 85.0  # safety margin below the 90s kill limit


def _reset_length(a, b):
    """Exact shortest reset-word length via BFS on the subset automaton, or -1 if not synchronizing."""
    full = frozenset(range(N))
    if len(full) == 1:
        return 0
    dist = {full: 0}
    q = deque([full])
    while q:
        s = q.popleft()
        d = dist[s]
        for m in (a, b):
            t = frozenset(m[i] for i in s)
            if len(t) == 1:
                return d + 1
            if t not in dist:
                dist[t] = d + 1
                q.append(t)
    return -1


def _perm_from_cycle(cycle):
    """Permutation mapping cycle[i] -> cycle[(i+1) % N]."""
    perm = [0] * N
    n = len(cycle)
    for i in range(n):
        perm[cycle[i]] = cycle[(i + 1) % n]
    return perm


def _random_defect(a):
    """Random defect map: collapse the cycle into a small subset (keeps a long tail)."""
    k = random.randint(2, 5)
    targets = random.sample(range(N), k)
    b = [random.choice(targets) for _ in range(N)]
    # bias: make most states map into a single target so the transient is long
    if random.random() < 0.6:
        b = [targets[0]] * N
        for _ in range(random.randint(1, k)):
            b[random.randrange(N)] = random.choice(targets[1:])
    return b


def _perturb(b):
    """Small random perturbation of a defect map."""
    b = list(b)
    for _ in range(random.randint(1, 3)):
        b[random.randrange(N)] = random.randrange(N)
    return b


def _greedy(a, b):
    """Coordinate descent: repeatedly try improving single entries of b."""
    best = _reset_length(a, b)
    if best < 0:
        return b, -1
    b = list(b)
    improved = True
    while improved:
        improved = False
        for i in range(N):
            old = b[i]
            for v in range(N):
                if v == old:
                    continue
                b[i] = v
                L = _reset_length(a, b)
                if L > best:
                    best = L
                    improved = True
                else:
                    b[i] = old
    return b, best


def construct():
    start = time.time()
    best_a = None
    best_b = None
    best_len = -1

    # enumerate all distinct cycle structures (permutations) up to relabeling
    # to give the search a strong structural prior
    cycles = []
    for n in range(1, N + 1):
        # one representative n-cycle over states 0..n-1, fix remaining as self-loops? No: must be a single cycle of length N for a permutation to be a full cycle.
        pass

    # We will instead randomize cycles directly (a full N-cycle permutation).
    while True:
        if time.time() > DEADLINE:
            break
        cycle = list(range(N))
        random.shuffle(cycle)
        a = _perm_from_cycle(cycle)
        b = _random_defect(a)
        L = _reset_length(a, b)
        if L > best_len:
            best_len = L
            best_a = a
            best_b = b
        # local refinement
        if time.time() < DEADLINE - 1.0:
            b2, L2 = _greedy(a, b)
            if L2 > best_len:
                best_len = L2
                best_a = a
                best_b = b2
            # occasional perturbation restart from the current best
            if random.random() < 0.3 and time.time() < DEADLINE - 1.0:
                b3 = _perturb(b2)
                L3 = _reset_length(a, b3)
                if L3 > best_len:
                    best_len = L3
                    best_b = b3

    if best_len < 0:
        # fallback: guaranteed synchronizing trivial automaton
        return [[(i + 1) % N for i in range(N)], [0] * N]

    return [list(best_a), list(best_b)]


if __name__ == "__main__":
    a, b = construct()
    print("reset length:", _reset_length(a, b))
    print("a:", a)
    print("b:", b)
