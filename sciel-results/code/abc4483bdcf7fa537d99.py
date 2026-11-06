import time
import random

N = 9
DEADLINE = 110.0  # safety margin before the 120s limit

def _time_left():
    return DEADLINE - time.monotonic()

def _validate(path):
    """Return True if path is a valid induced path (snake)."""
    n = len(path)
    if n < 2:
        return True
    seen = set(path)
    if len(seen) != n:
        return False
    for i in range(n - 1):
        if (path[i] ^ path[i + 1]).bit_count() != 1:
            return False
    for i in range(n):
        for j in range(i + 2, n):
            if (path[i] ^ path[j]).bit_count() == 1:
                return False
    return True

def _build_forbidden(path):
    """forbidden = all vertices adjacent to path[:-2] (the non-adjacent vertices)."""
    forbidden = 0
    for p in path[:-2]:
        for b in range(N):
            forbidden |= (1 << (p ^ (1 << b)))
    return forbidden

def _beam_search(seed, beam, max_len, time_limit, rng):
    """Beam search extending from a seed path, returning the longest valid path found."""
    start = time.monotonic()
    # states: list of (path, forbidden, used)
    states = [(list(seed), _build_forbidden(seed), set(seed))]
    best = list(seed)
    best_len = len(seed)
    bit_orders = [list(range(N)) for _ in range(4)]
    for bo in bit_orders:
        rng.shuffle(bo)
    while _time_left() > 0.5:
        if time.monotonic() - start > time_limit:
            break
        next_states = []
        for path, forbidden, used in states:
            cur = path[-1]
            candidates = []
            for b in bit_orders[len(path) % 4]:
                v = cur ^ (1 << b)
                if (1 << v) & used:
                    continue
                if (1 << v) & forbidden:
                    continue
                candidates.append((b, v))
            # score candidates: prefer those with more open moves
            scored = []
            for b, v in candidates:
                open_moves = 0
                nb_forbidden = 0
                for b2 in range(N):
                    w = v ^ (1 << b2)
                    if (1 << w) & used:
                        continue
                    if (1 << w) & forbidden:
                        nb_forbidden += 1
                        continue
                    open_moves += 1
                scored.append((open_moves, -nb_forbidden, b, v))
            scored.sort(reverse=True)
            for _, _, b, v in scored:
                if len(next_states) >= beam:
                    break
                new_path = path + [v]
                new_used = used | {(1 << v)}
                # update forbidden: remove neighbors of path[-2], add neighbors of path[-1]
                new_forbidden = forbidden
                p_prev = path[-2]
                for b2 in range(N):
                    new_forbidden &= ~(1 << (p_prev ^ (1 << b2)))
                p_last = path[-1]
                for b2 in range(N):
                    new_forbidden |= (1 << (p_last ^ (1 << b2)))
                next_states.append((new_path, new_forbidden, new_used))
                if len(new_path) > best_len:
                    best_len = len(new_path)
                    best = new_path
        if not next_states:
            break
        states = next_states[:beam]
    return best

def _local_improve(path, time_limit):
    """Try to improve the snake by reordering / extending. Returns improved path."""
    start = time.monotonic()
    best = list(path)
    best_len = len(path)
    rng = random.Random(hash(tuple(path)) & 0xFFFFFFFF)
    while _time_left() > 1.0 and (time.monotonic() - start) < time_limit:
        # try extending from the end with a random bit order
        cur = list(best)
        forbidden = _build_forbidden(cur)
        used = set(cur)
        cur_forbidden = forbidden
        improved = False
        while True:
            last = cur[-1]
            bits = list(range(N))
            rng.shuffle(bits)
            found = False
            for b in bits:
                v = last ^ (1 << b)
                if (1 << v) & used:
                    continue
                if (1 << v) & cur_forbidden:
                    continue
                cur.append(v)
                used.add(v)
                # update forbidden
                if len(cur) >= 3:
                    p_prev = cur[-3]
                    for b2 in range(N):
                        cur_forbidden &= ~(1 << (p_prev ^ (1 << b2)))
                    p_last = cur[-2]
                    for b2 in range(N):
                        cur_forbidden |= (1 << (p_last ^ (1 << b2)))
                found = True
                improved = True
                break
            if not found:
                break
        if len(cur) > best_len:
            best = cur
            best_len = len(cur)
        if not improved:
            break
    return best

def _greedy_seed(rng):
    """Greedy snake with random bit order."""
    path = [0]
    used = {0}
    forbidden = 0
    while True:
        cur = path[-1]
        bits = list(range(N))
        rng.shuffle(bits)
        nxt = None
        for b in bits:
            v = cur ^ (1 << b)
            if (1 << v) & used:
                continue
            if (1 << v) & forbidden:
                continue
            nxt = v
            break
        if nxt is None:
            break
        path.append(nxt)
        used.add(nxt)
        if len(path) >= 3:
            p_prev = path[-3]
            for b in range(N):
                forbidden &= ~(1 << (p_prev ^ (1 << b)))
            p_last = path[-2]
            for b in range(N):
                forbidden |= (1 << (p_last ^ (1 << b)))
    return path

def construct():
    global DEADLINE
    DEADLINE = time.monotonic() + 110.0
    rng = random.Random(42)
    best = None
    best_len = 0

    # Phase 1: greedy seeds
    for i in range(20):
        if _time_left() < 5.0:
            break
        seed = _greedy_seed(rng)
        if len(seed) > best_len:
            best = seed
            best_len = len(seed)

    # Phase 2: beam search from best seeds
    for i in range(15):
        if _time_left() < 15.0:
            break
        seed = _greedy_seed(rng)
        if len(seed) < best_len - 10:
            continue
        result = _beam_search(seed, beam=12, max_len=2**N, time_limit=4.0, rng=rng)
        if len(result) > best_len:
            best = result
            best_len = len(result)

    # Phase 3: local improvement on best
    if _time_left() > 10.0 and best is not None:
        improved = _local_improve(best, time_limit=30.0)
        if len(improved) > best_len:
            best = improved
            best_len = len(improved)

    # Phase 4: more beam searches with different seeds
    for i in range(20):
        if _time_left() < 10.0:
            break
        seed = _greedy_seed(rng)
        if len(seed) < best_len - 15:
            continue
        result = _beam_search(seed, beam=20, max_len=2**N, time_limit=3.0, rng=rng)
        if len(result) > best_len:
            best = result
            best_len = len(result)
            if _time_left() > 15.0:
                improved = _local_improve(best, time_limit=15.0)
                if len(improved) > best_len:
                    best = improved
                    best_len = len(improved)

    # Final local improvement
    if _time_left() > 5.0 and best is not None:
        improved = _local_improve(best, time_limit=min(20.0, max(0.0, _time_left() - 2.0)))
        if len(improved) > best_len:
            best = improved
            best_len = len(improved)

    # Safety: ensure we return a valid result
    if best is None or len(best) < 2:
        # fallback: simple greedy
        best = _greedy_seed(random.Random(0))

    if not _validate(best):
        # should not happen, but fallback
        best = _greedy_seed(random.Random(1))

    return best
