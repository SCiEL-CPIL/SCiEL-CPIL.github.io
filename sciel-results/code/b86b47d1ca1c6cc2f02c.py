"""Snake-in-the-box Q_9: bitmask beam search with heuristic pruning and restarts."""
import time
import random
import numpy as np

N = 9
V = 1 << N  # 512 vertices

# Precompute neighbors for each vertex as a bitmask
NEIGHBOR_MASKS = np.zeros(V, dtype=np.uint64)
for v in range(V):
    m = 0
    for b in range(N):
        m |= 1 << (v ^ (1 << b))
    NEIGHBOR_MASKS[v] = m

# Precompute popcount for all possible differences (0..511)
POP = np.array([i.bit_count() for i in range(V)], dtype=np.uint8)

# All-ones mask for V vertices
ALL_MASK = (1 << V) - 1


def _is_induced(path, used_mask):
    """Check path is a valid induced path: consecutive differ by 1 bit,
    non-consecutive differ by >= 2 bits."""
    for i in range(len(path) - 1):
        d = path[i] ^ path[i + 1]
        if d.bit_count() != 1:
            return False
    # Non-consecutive: for each vertex, its neighbors in the path must be
    # only its immediate predecessor and successor
    for i, v in enumerate(path):
        nbrs = NEIGHBOR_MASKS[v] & used_mask
        # Count how many path vertices are neighbors of v
        cnt = 0
        for j, p in enumerate(path):
            if i != j and (nbrs >> p) & 1:
                cnt += 1
        if cnt > 2:
            return False
        if cnt == 2:
            # Both neighbors must be immediate predecessor and successor
            prev_ok = (i > 0 and (nbrs >> path[i - 1]) & 1)
            next_ok = (i < len(path) - 1 and (nbrs >> path[i + 1]) & 1)
            if not (prev_ok and next_ok):
                return False
    return True


def _beam_search(start, beam_width, deadline, rng):
    """Beam search from a start vertex. Returns best path found."""
    best_path = [start]
    # State: (path_list, used_mask)
    # We'll keep a list of candidate states, each is (path, used_mask)
    # For efficiency, path is a list of ints, used_mask is int bitmask

    candidates = [([start], 1 << start)]

    while time.time() < deadline:
        # Expand all candidates
        next_candidates = []
        for path, used_mask in candidates:
            cur = path[-1]
            nbrs = NEIGHBOR_MASKS[cur]
            # Available neighbors: not used, and adding keeps induced property
            avail = nbrs & ~used_mask
            # For each available neighbor, check induced constraint
            # The new vertex v must not be adjacent to any path vertex
            # except cur (its predecessor)
            # Since v is a neighbor of cur, and v not in used,
            # we need: v is not adjacent to any path vertex other than cur
            # path vertices other than cur: path[:-1]
            # v adjacent to p means (1 << p) in NEIGHBOR_MASKS[v]
            # i.e., (NEIGHBOR_MASKS[v] >> p) & 1
            # We need NEIGHBOR_MASKS[v] & used_mask == (1 << cur)
            # i.e., the only used vertex adjacent to v is cur
            v_list = []
            while avail:
                lb = avail & (-avail)
                v = lb.bit_length() - 1
                avail ^= lb
                # Check: v's neighbors in used set must be exactly {cur}
                if (NEIGHBOR_MASKS[v] & used_mask) == (1 << cur):
                    v_list.append(v)
            if not v_list:
                continue
            # Heuristic: score each extension
            # Prefer vertices with more future options (degree in complement)
            # and that don't trap the path
            scored = []
            for v in v_list:
                new_used = used_mask | (1 << v)
                # Count available next steps from v
                next_avail = NEIGHBOR_MASKS[v] & ~new_used
                # Also check: from v, how many of those are valid extensions
                # (i.e., the next vertex w would have NEIGHBOR_MASKS[w] & new_used == (1<<v))
                # This is expensive; use a cheaper proxy: popcount of next_avail
                # weighted by whether v is a "dead end"
                # A better heuristic: prefer v where the number of valid
                # continuations is high. Use popcount as proxy.
                score = POP[next_avail & (ALL_MASK >> 0)]  # popcount of next_avail
                # Penalize if v is adjacent to many already-used vertices
                # (reduces flexibility)
                adj_used = POP[NEIGHBOR_MASKS[v] & used_mask]
                score = score - adj_used
                scored.append((score, v))
            scored.sort(key=lambda x: (-x[0], x[1]))
            # Take top extensions
            for score, v in scored[:beam_width]:
                new_path = path + [v]
                new_used = used_mask | (1 << v)
                next_candidates.append((new_path, new_used))
                if len(new_path) > len(best_path):
                    best_path = new_path

        if not next_candidates:
            break

        # Keep top beam_width candidates by length, then by heuristic
        # Score candidate: (len(path), -sum of future options at end)
        def cand_score(item):
            path, used_mask = item
            cur = path[-1]
            next_avail = NEIGHBOR_MASKS[cur] & ~used_mask
            # Count valid continuations
            cnt = 0
            na = next_avail
            while na:
                lb = na & (-na)
                w = lb.bit_length() - 1
                na ^= lb
                if (NEIGHBOR_MASKS[w] & used_mask) == (1 << cur):
                    cnt += 1
            return (len(path), -cnt)

        next_candidates.sort(key=cand_score, reverse=True)
        candidates = next_candidates[:beam_width]

        # Check time
        if time.time() >= deadline:
            break

    return best_path


def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s kill

    rng = random.Random(42)
    best_path = None
    best_len = 0

    # Strategy: multiple restarts with different beam widths and random starts
    # Allocate time across phases
    phase_deadline = start_time + 105

    # Phase 1: greedy baseline (fast, gives a good seed)
    greedy_path = _greedy_snake()
    if len(greedy_path) > best_len:
        best_len = len(greedy_path)
        best_path = greedy_path

    # Phase 2: beam search with increasing beam widths
    beam_widths = [4, 8, 16, 32]
    starts = [0]
    # Add some random starts
    for _ in range(6):
        starts.append(rng.randint(0, V - 1))
    # Deduplicate
    starts = list(dict.fromkeys(starts))

    # Time budget per (beam_width, start) combination
    total_combos = len(beam_widths) * len(starts)
    time_per_combo = (phase_deadline - start_time) / total_combos
    time_per_combo = max(time_per_combo, 2.0)  # minimum 2s per combo

    for bw in beam_widths:
        for s in starts:
            if time.time() >= phase_deadline:
                break
            combo_deadline = min(time.time() + time_per_combo, phase_deadline)
            path = _beam_search(s, bw, combo_deadline, rng)
            if len(path) > best_len:
                best_len = len(path)
                best_path = path

    # Phase 3: if time remains, do more random restarts with best beam width
    if time.time() < deadline - 5:
        for _ in range(20):
            if time.time() >= deadline - 5:
                break
            s = rng.randint(0, V - 1)
            combo_deadline = min(time.time() + 3.0, deadline - 5)
            path = _beam_search(s, 16, combo_deadline, rng)
            if len(path) > best_len:
                best_len = len(path)
                best_path = path

    if best_path is None:
        best_path = greedy_path

    return best_path


def _greedy_snake():
    """Fast greedy snake: always extend with lowest valid bit flip."""
    path = [0]
    used = {0}
    while True:
        cur = path[-1]
        nxt = None
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            # Check induced: v must not be adjacent to any path vertex except cur
            ok = True
            for p in path[:-1]:
                if (v ^ p).bit_count() == 1:
                    ok = False
                    break
            if ok:
                nxt = v
                break
        if nxt is None:
            return path
        path.append(nxt)
        used.add(nxt)
