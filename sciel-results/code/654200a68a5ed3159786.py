import time
import random

def construct():
    N = 256
    start = time.time()
    DEADLINE = 80.0

    # squares up to 2N = 512
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1

    # conflict graph: edge a-b (a<b) iff a+b is a perfect square
    conflict = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for s in squares:
            b = s - a
            if a < b <= N:
                conflict[a].add(b)
                conflict[b].add(a)

    # diagonal constraint: 2a must not be a perfect square
    self_bad = [False] * (N + 1)
    for a in range(1, N + 1):
        if 2 * a in squares:
            self_bad[a] = True

    def compatible(S_bool, a):
        if self_bad[a]:
            return False
        for b in conflict[a]:
            if S_bool[b]:
                return False
        return True

    def add_pass(S_set, S_bool):
        changed = True
        while changed:
            changed = False
            for a in range(1, N + 1):
                if not S_bool[a] and compatible(S_bool, a):
                    S_bool[a] = True
                    S_set.add(a)
                    changed = True
        return S_set, S_bool

    def improve(S_set, S_bool):
        S_set, S_bool = add_pass(S_set, S_bool)
        improved = True
        while improved:
            improved = False
            for a in list(S_set):
                snap_set = set(S_set)
                snap_bool = S_bool[:]
                S_bool[a] = False
                S_set.remove(a)
                before = len(S_set)
                S_set, S_bool = add_pass(S_set, S_bool)
                after = len(S_set)
                if after > before:
                    improved = True
                else:
                    S_set = snap_set
                    S_bool = snap_bool
        return S_set, S_bool

    def greedy_from(order, S_set=None, S_bool=None):
        if S_set is None:
            S_set = set()
            S_bool = [False] * (N + 1)
        for a in order:
            if not S_bool[a] and compatible(S_bool, a):
                S_bool[a] = True
                S_set.add(a)
        return S_set, S_bool

    best_set = None
    best_size = 0

    def consider(S_set, S_bool):
        nonlocal best_set, best_size
        S_set, S_bool = improve(S_set, S_bool)
        if len(S_set) > best_size:
            best_size = len(S_set)
            best_set = set(S_set)
        return S_set, S_bool

    # deterministic starts
    for order in (list(range(1, N + 1)), list(range(N, 0, -1))):
        S_set, S_bool = greedy_from(order)
        consider(S_set, S_bool)

    rng = random.Random(20240607)
    while time.time() - start < DEADLINE:
        mode = rng.random()
        if mode < 0.5 or best_set is None:
            # fresh random greedy
            order = list(range(1, N + 1))
            rng.shuffle(order)
            S_set, S_bool = greedy_from(order)
        else:
            # kick from current best to escape local optima
            S_set = set(best_set)
            S_bool = [False] * (N + 1)
            for a in S_set:
                S_bool[a] = True
            kick = rng.randint(3, 12)
            for _ in range(kick):
                if S_set:
                    a = rng.choice(list(S_set))
                    S_set.remove(a)
                    S_bool[a] = False
            order = list(range(1, N + 1))
            rng.shuffle(order)
            S_set, S_bool = greedy_from(order, S_set, S_bool)
        consider(S_set, S_bool)

    if best_set is None:
        for a in range(1, N + 1):
            if not self_bad[a]:
                return [a]
        return [1]

    return sorted(best_set)
