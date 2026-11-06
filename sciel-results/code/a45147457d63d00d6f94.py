import time
import random
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1
SIZE = 1 << N


def _build_subset_trans(trans):
    size = SIZE
    nxt = [0] * size
    for mask in range(1, size):
        lsb = mask & (-mask)
        i = lsb.bit_length() - 1
        rest = mask ^ lsb
        nxt[mask] = nxt[rest] | (1 << trans[i])
    return nxt


def compute_threshold(trans_a, trans_b):
    """Exact shortest reset-word length; 0 if not synchronizing."""
    next_a = _build_subset_trans(trans_a)
    next_b = _build_subset_trans(trans_b)

    dist = [-1] * SIZE
    dist[FULL_MASK] = 0
    queue = deque([FULL_MASK])
    while queue:
        cur = queue.popleft()
        if cur & (cur - 1) == 0:
            return dist[cur]
        na = next_a[cur]
        if dist[na] == -1:
            dist[na] = dist[cur] + 1
            queue.append(na)
        nb = next_b[cur]
        if dist[nb] == -1:
            dist[nb] = dist[cur] + 1
            queue.append(nb)
    return 0


class _FastEval:
    """Incremental subset-transition + BFS threshold for single-edge mutations."""

    def __init__(self, a, b):
        self.a = a[:]
        self.b = b[:]
        self.na = _build_subset_trans(self.a)
        self.nb = _build_subset_trans(self.b)
        self.score = self._bfs()

    def _bfs(self):
        na, nb = self.na, self.nb
        dist = [-1] * SIZE
        dist[FULL_MASK] = 0
        q = deque([FULL_MASK])
        while q:
            cur = q.popleft()
            if cur & (cur - 1) == 0:
                return dist[cur]
            x = na[cur]
            if dist[x] == -1:
                dist[x] = dist[cur] + 1
                q.append(x)
            y = nb[cur]
            if dist[y] == -1:
                dist[y] = dist[cur] + 1
                q.append(y)
        return 0

    def _update(self, arr, old, new, i):
        # Update all subsets containing bit i (masks where bit i is set)
        # Iterate masks with bit i set: step through using a clever loop
        # Simple: for mask in range(SIZE): if mask & (1<<i): recompute
        # But that's 2^(N-1) per update. Use the DP relation:
        # For masks containing i, arr[mask] depends on arr[mask^(1<<i)] and trans[i].
        # We must update in increasing order of popcount? No — the DP is:
        #   arr[mask] = arr[mask ^ lsb(mask)] | (1 << trans[lsb_index])
        # where lsb is the lowest set bit. This only works if we remove the LOWEST bit.
        # So the DP is defined by removing the lowest set bit, not an arbitrary bit.
        # This means we can't easily update for an arbitrary i.
        # Fallback: rebuild from scratch (still fast for N=12).
        self.na if arr is self.a else self.nb
        # Just rebuild
        if arr is self.a:
            self.na = _build_subset_trans(self.a)
        else:
            self.nb = _build_subset_trans(self.b)

    def mutate(self, letter, state, new_dest):
        """Apply a single-edge mutation and recompute score incrementally."""
        if letter == 0:
            old = self.a[state]
            if old == new_dest:
                return self.score
            self.a[state] = new_dest
            self._update(self.a, old, new_dest, state)
        else:
            old = self.b[state]
            if old == new_dest:
                return self.score
            self.b[state] = new_dest
            self._update(self.b, old, new_dest, state)
        self.score = self._bfs()
        return self.score

    def revert(self, letter, state, old_dest):
        if letter == 0:
            self.a[state] = old_dest
            self.na = _build_subset_trans(self.a)
        else:
            self.b[state] = old_dest
            self.nb = _build_subset_trans(self.b)
        self.score = self._bfs()
        return self.score


def _seed_structures():
    """Generate diverse initial automata from known high-threshold families."""
    seeds = []

    # Family 1: Cycle + single defect (classic Černý variant)
    for sink in range(N):
        a = [(i + 1) % N for i in range(N)]
        b = [(i - 1) % N for i in range(N)]
        b[sink] = sink
        seeds.append((a[:], b[:]))

    # Family 2: Pinwheel (a = cycle, b = rotation by shift)
    for shift in range(1, N):
        a = [(i + 1) % N for i in range(N)]
        b = [(i + shift) % N for i in range(N)]
        seeds.append((a[:], b[:]))

    # Family 3: One-escape (b maps most to a small set, one state escapes)
    for target_size in [1, 2, 3]:
        for offset in range(N):
            a = [(i + 1) % N for i in range(N)]
            b = [i % target_size for i in range(N)]
            escape = (offset + N - 1) % N
            b[escape] = (escape + 1) % N
            seeds.append((a[:], b[:]))

    # Family 4: Two disjoint cycles (ping-pong)
    half = N // 2
    for perm_a in range(1, half + 1):
        a = [0] * N
        b = [0] * N
        for i in range(half):
            a[i] = (i + perm_a) % half
            a[half + i] = half + (i + perm_a) % half
            b[i] = (i + 1) % half
            b[half + i] = half + (i + 1) % half
        seeds.append((a[:], b[:]))

    # Family 5: Random with structure bias
    for _ in range(20):
        a = [(i + random.randint(1, 3)) % N for i in range(N)]
        b = [random.randint(0, N // 2) for _ in range(N)]
        seeds.append((a[:], b[:]))

    return seeds


def _local_search(ev, deadline, max_no_improve=30):
    """Hill-climb a single automaton using single-edge mutations."""
    best_a = ev.a[:]
    best_b = ev.b[:]
    best_score = ev.score
    no_improve = 0

    while time.time() < deadline and no_improve < max_no_improve:
        # Choose letter to mutate: bias toward b (usually the "reset" letter)
        letter = 0 if random.random() < 0.3 else 1
        state = random.randint(0, N - 1)
        old_dest = ev.a[state] if letter == 0 else ev.b[state]
        new_dest = random.randint(0, N - 1)
        if new_dest == old_dest:
            continue

        ev.mutate(letter, state, new_dest)
        if ev.score > best_score:
            best_score = ev.score
            best_a = ev.a[:]
            best_b = ev.b[:]
            no_improve = 0
        else:
            ev.revert(letter, state, old_dest)
            no_improve += 1

    return best_a, best_b, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    best_score = 0
    best_a = None
    best_b = None

    seeds = _seed_structures()
    random.shuffle(seeds)

    # Evaluate all seeds and keep the best
    for a, b in seeds:
        if time.time() >= deadline:
            break
        ev = _FastEval(a, b)
        if ev.score > best_score:
            best_score = ev.score
            best_a = ev.a[:]
            best_b = ev.b[:]

    # Local search on the best seed, then on other promising seeds
    # Sort seeds by initial score for prioritized local search
    seed_scores = []
    for a, b in seeds:
        if time.time() >= deadline:
            break
        ev = _FastEval(a, b)
        seed_scores.append((ev.score, a, b))

    seed_scores.sort(key=lambda x: -x[0])

    # Local search on top seeds
    for score, a, b in seed_scores:
        if time.time() >= deadline:
            break
        if score == 0:
            continue
        ev = _FastEval(a, b)
        la, lb, ls = _local_search(ev, deadline, max_no_improve=25)
        if ls > best_score:
            best_score = ls
            best_a = la[:]
            best_b = lb[:]

    # Additional multi-start: random restarts with local search
    restarts = 0
    while time.time() < deadline - 5.0 and restarts < 30:
        restarts += 1
        # Random seed with mild structure
        a = [(i + random.randint(1, 3)) % N for i in range(N)]
        b = [random.randint(0, N - 1) for _ in range(N)]
        ev = _FastEval(a, b)
        if ev.score == 0:
            continue
        la, lb, ls = _local_search(ev, deadline, max_no_improve=20)
        if ls > best_score:
            best_score = ls
            best_a = la[:]
            best_b = lb[:]

    # Final: if we have time, do one more round of local search on the global best
    if time.time() < deadline - 3.0 and best_a is not None:
        ev = _FastEval(best_a, best_b)
        la, lb, ls = _local_search(ev, deadline, max_no_improve=40)
        if ls > best_score:
            best_score = ls
            best_a = la[:]
            best_b = lb[:]

    # Fallback: ensure we return a valid synchronizing automaton
    if best_score == 0 or best_a is None:
        best_a = [(i + 1) % N for i in range(N)]
        best_b = [(i - 1) % N for i in range(N)]
        best_b[0] = 0
        best_score = compute_threshold(best_a, best_b)

    return [best_a, best_b]
