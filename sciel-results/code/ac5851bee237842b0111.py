"""Snake-in-the-box Q_9: Combined beam search - fast 1-ply bulk search + 2-ply polish."""
import time
import random
import math

N = 9
SIZE = 1 << N
BIT_MASKS = [1 << b for b in range(N)]

# Precompute neighbor masks
NEIGHBOR_MASKS = [0] * SIZE
for v in range(SIZE):
    m = 0
    for b in range(N):
        m |= 1 << (v ^ BIT_MASKS[b])
    NEIGHBOR_MASKS[v] = m


def _greedy_extend(start, bit_order=None):
    """Greedy extension from a single start vertex using forbidden_mask."""
    if bit_order is None:
        bit_order = list(range(N))
    path = [start]
    used_mask = 1 << start
    forbidden_mask = 0

    while True:
        cur = path[-1]
        candidates = NEIGHBOR_MASKS[cur] & ~(used_mask | forbidden_mask)
        if candidates == 0:
            break

        best_v = None
        best_score = SIZE + 1
        tmp = candidates
        while tmp:
            lsb = tmp & (-tmp)
            v = lsb.bit_length() - 1
            tmp ^= lsb
            new_forbidden = forbidden_mask | NEIGHBOR_MASKS[cur]
            new_used = used_mask | (1 << v)
            next_opts = NEIGHBOR_MASKS[v] & ~(new_used | new_forbidden)
            score = next_opts.bit_count()
            if score < best_score:
                best_score = score
                best_v = v

        if best_v is None:
            break
        path.append(best_v)
        used_mask |= (1 << best_v)
        forbidden_mask |= NEIGHBOR_MASKS[cur]

    return path


def _beam_search(start, beam_width, max_time, rng, alpha=1.0, beta=0.5, use_lookahead=False, gamma=0.1):
    """
    Beam search with tunable scoring.
    If use_lookahead=False: fast 1-ply scoring (primary mode)
    If use_lookahead=True:  2-ply soft-min scoring (polish mode)
    """
    path = [start]
    used_mask = 1 << start
    forbidden_mask = 0
    beam = [(path, used_mask, forbidden_mask)]
    best_path = list(path)
    best_len = 1
    t0 = time.time()

    while True:
        if time.time() > max_time:
            break
        if not beam:
            break

        new_beam = []
        seen_suffixes = set()

        for cur_path, cur_used, cur_forb in beam:
            cur = cur_path[-1]
            candidates = NEIGHBOR_MASKS[cur] & ~(cur_used | cur_forb)

            if candidates == 0:
                if len(cur_path) > best_len:
                    best_len = len(cur_path)
                    best_path = list(cur_path)
                continue

            tmp = candidates
            while tmp:
                lsb = tmp & (-tmp)
                v = lsb.bit_length() - 1
                tmp ^= lsb

                new_path = cur_path + [v]
                new_used = cur_used | (1 << v)
                new_forb = cur_forb | NEIGHBOR_MASKS[cur]

                key = tuple(new_path[-4:])
                if key in seen_suffixes:
                    continue
                seen_suffixes.add(key)

                new_beam.append((new_path, new_used, new_forb))
                if len(new_path) > best_len:
                    best_len = len(new_path)
                    best_path = list(new_path)

        if not new_beam:
            break

        if use_lookahead:
            def _score(entry):
                p, used, forb = entry
                cur = p[-1]
                next_opts_mask = NEIGHBOR_MASKS[cur] & ~(used | forb)
                opt_count = next_opts_mask.bit_count()
                
                # Soft-min via log-sum-exp over next-step options
                # Lower score = better (we want to minimize)
                # Soft-min approximates min but is smooth
                if opt_count == 0:
                    return (-len(p) * alpha)
                
                # Collect v_opts for each candidate next step
                log_terms = []
                tmp2 = next_opts_mask
                while tmp2:
                    lsb2 = tmp2 & (-tmp2)
                    v2 = lsb2.bit_length() - 1
                    tmp2 ^= lsb2
                    v_opts_mask = NEIGHBOR_MASKS[v2] & ~(used | (1 << v2) | forb)
                    v_opts_count = v_opts_mask.bit_count()
                    # Use v_opts_count + 1 to avoid log(0)
                    log_terms.append(math.log(v_opts_count + 1))
                
                # Soft-min: -log(sum(exp(-x_i))) approximates min(x_i) smoothly
                # We want to reward paths where the worst next-step option is still good
                max_log = max(log_terms)
                sum_exp = sum(math.exp(x - max_log) for x in log_terms)
                soft_min_val = max_log - math.log(sum_exp)
                
                # Convert to a score: higher opt_count and higher soft_min_val is better
                # We negate because sort is ascending
                return (-len(p) * alpha - opt_count * beta - soft_min_val * gamma)
        else:
            def _score(entry):
                p, used, forb = entry
                cur = p[-1]
                next_opts = (NEIGHBOR_MASKS[cur] & ~(used | forb)).bit_count()
                return (-len(p) * alpha + next_opts * beta)

        new_beam.sort(key=_score)
        beam = new_beam[:beam_width]

    return best_path


def _beam_search_from_path(prefix, beam_width, max_time, rng, alpha=1.0, beta=0.5, use_lookahead=False, gamma=0.1):
    """Beam search continuing from an existing path prefix."""
    if not prefix:
        return _beam_search(0, beam_width, max_time, rng, alpha, beta, use_lookahead, gamma)

    start = prefix[-1]
    used_mask = 0
    for v in prefix:
        used_mask |= 1 << v
    forbidden_mask = 0
    for v in prefix[:-1]:
        forbidden_mask |= NEIGHBOR_MASKS[v]

    path = list(prefix)
    beam = [(path, used_mask, forbidden_mask)]
    best_path = list(path)
    best_len = len(path)
    t0 = time.time()

    while True:
        if time.time() > max_time:
            break
        if not beam:
            break

        new_beam = []
        seen_suffixes = set()

        for cur_path, cur_used, cur_forb in beam:
            cur = cur_path[-1]
            candidates = NEIGHBOR_MASKS[cur] & ~(cur_used | cur_forb)

            if candidates == 0:
                if len(cur_path) > best_len:
                    best_len = len(cur_path)
                    best_path = list(cur_path)
                continue

            tmp = candidates
            while tmp:
                lsb = tmp & (-tmp)
                v = lsb.bit_length() - 1
                tmp ^= lsb

                new_path = cur_path + [v]
                new_used = cur_used | (1 << v)
                new_forb = cur_forb | NEIGHBOR_MASKS[cur]

                key = tuple(new_path[-4:])
                if key in seen_suffixes:
                    continue
                seen_suffixes.add(key)

                new_beam.append((new_path, new_used, new_forb))
                if len(new_path) > best_len:
                    best_len = len(new_path)
                    best_path = list(new_path)

        if not new_beam:
            break

        if use_lookahead:
            def _score(entry):
                p, used, forb = entry
                cur = p[-1]
                next_opts_mask = NEIGHBOR_MASKS[cur] & ~(used | forb)
                opt_count = next_opts_mask.bit_count()
                
                if opt_count == 0:
                    return (-len(p) * alpha)
                
                log_terms = []
                tmp2 = next_opts_mask
                while tmp2:
                    lsb2 = tmp2 & (-tmp2)
                    v2 = lsb2.bit_length() - 1
                    tmp2 ^= lsb2
                    v_opts_mask = NEIGHBOR_MASKS[v2] & ~(used | (1 << v2) | forb)
                    v_opts_count = v_opts_mask.bit_count()
                    log_terms.append(math.log(v_opts_count + 1))
                
                max_log = max(log_terms)
                sum_exp = sum(math.exp(x - max_log) for x in log_terms)
                soft_min_val = max_log - math.log(sum_exp)
                
                return (-len(p) * alpha - opt_count * beta - soft_min_val * gamma)
        else:
            def _score(entry):
                p, used, forb = entry
                cur = p[-1]
                next_opts = (NEIGHBOR_MASKS[cur] & ~(used | forb)).bit_count()
                return (-len(p) * alpha + next_opts * beta)

        new_beam.sort(key=_score)
        beam = new_beam[:beam_width]

    return best_path


def _local_improve(path, deadline, rng, alpha=1.0, beta=0.5):
    """Try to improve path by re-extending from various cut points."""
    best_path = list(path)
    best_len = len(path)
    n = len(path)

    if n < 4:
        return best_path

    positions = []
    for i in range(max(1, n // 2), n - 1):
        positions.append(i)
    for i in range(1, max(1, n // 4)):
        positions.append(i)
    rng.shuffle(positions)

    for pos in positions[:20]:
        if time.time() > deadline - 3:
            break
        prefix = path[:pos + 1]
        remaining = deadline - time.time()
        if remaining < 2:
            break
        dur = min(3.0, remaining - 1)
        improved = _beam_search_from_path(prefix, 40, time.time() + dur, rng, alpha, beta)
        if len(improved) > best_len:
            best_len = len(improved)
            best_path = improved

    return best_path


def _polish_with_lookahead(path, deadline, rng):
    """Final polish pass using 2-ply soft-min lookahead scoring with wider beam."""
    best_path = list(path)
    best_len = len(path)
    
    # Try extending from the end with lookahead scoring
    remaining = deadline - time.time()
    if remaining > 5:
        dur = min(10.0, remaining - 2)
        result = _beam_search_from_path(path, 80, time.time() + dur, rng, 1.5, 0.3, use_lookahead=True, gamma=0.2)
        if len(result) > best_len:
            best_len = len(result)
            best_path = result
    
    # Try local improvement with lookahead
    remaining = deadline - time.time()
    if remaining > 5 and best_len < 200:
        # Re-extend from a few cut points with lookahead
        n = len(best_path)
        cut_positions = [n - 5, n - 10, n // 2]
        rng.shuffle(cut_positions)
        for pos in cut_positions[:3]:
            remaining = deadline - time.time()
            if remaining < 3:
                break
            if pos < 1 or pos >= n - 1:
                continue
            prefix = best_path[:pos + 1]
            dur = min(4.0, remaining - 1)
            result = _beam_search_from_path(prefix, 60, time.time() + dur, rng, 1.2, 0.4, use_lookahead=True, gamma=0.15)
            if len(result) > best_len:
                best_len = len(result)
                best_path = result
    
    return best_path


def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    rng = random.Random(42)

    best_path = None
    best_len = 0

    # Phase 1: Greedy seed
    greedy = _greedy_extend(0)
    best_path = greedy
    best_len = len(greedy)

    # Phase 2: Fast 1-ply beam search with diverse configurations
    # More configs, each shorter, to maximize diversity
    param_configs = [
        (50, 8.0, 1.0, 0.5),
        (100, 10.0, 1.0, 0.3),
        (150, 12.0, 1.5, 0.2),
        (200, 14.0, 1.0, 0.4),
        (300, 16.0, 2.0, 0.1),
        (80, 8.0, 1.2, 0.35),
        (120, 10.0, 1.8, 0.15),
    ]

    for bw, duration, alpha, beta in param_configs:
        if time.time() > deadline - 20:
            break
        remaining = deadline - time.time()
        if remaining < duration + 2:
            duration = max(2.0, remaining - 2)
        t0 = time.time()
        result = _beam_search(0, bw, t0 + duration, rng, alpha, beta)
        if len(result) > best_len:
            best_len = len(result)
            best_path = result

    # Phase 3: Extend the best path found (fast 1-ply)
    if time.time() < deadline - 15 and best_len < 200:
        remaining = deadline - time.time()
        if remaining > 10:
            dur = min(15.0, remaining - 5)
            result = _beam_search_from_path(best_path, 100, time.time() + dur, rng, 1.0, 0.3)
            if len(result) > best_len:
                best_len = len(result)
                best_path = result

    # Phase 4: Multi-restart with varied parameters (fast 1-ply)
    if time.time() < deadline - 15:
        starts = list(range(SIZE))
        rng.shuffle(starts)
        restart_configs = [
            (60, 3.0, 1.0, 0.5),
            (100, 2.5, 1.5, 0.2),
            (80, 2.5, 1.2, 0.35),
            (150, 2.0, 2.0, 0.1),
        ]
        for start in starts[:25]:
            if time.time() > deadline - 10:
                break
            remaining = deadline - time.time()
            if remaining < 3:
                break
            for bw, dur, alpha, beta in restart_configs:
                if time.time() > deadline - 5:
                    break
                if remaining < dur:
                    dur = max(1.5, remaining - 1)
                result = _beam_search(start, bw, time.time() + dur, rng, alpha, beta)
                if len(result) > best_len:
                    best_len = len(result)
                    best_path = result
                if time.time() > deadline - 8:
                    break

    # Phase 5: Local improvement (fast 1-ply)
    if time.time() < deadline - 10:
        improved = _local_improve(best_path, deadline, rng, 1.0, 0.5)
        if len(improved) > best_len:
            best_len = len(improved)
            best_path = improved

    # Phase 6: Final polish with 2-ply soft-min lookahead
    if time.time() < deadline - 5:
        polished = _polish_with_lookahead(best_path, deadline, rng)
        if len(polished) > best_len:
            best_len = len(polished)
            best_path = polished

    # Phase 7: One more local improvement after polish
    if time.time() < deadline - 3:
        improved = _local_improve(best_path, deadline, rng, 1.2, 0.3)
        if len(improved) > best_len:
            best_len = len(improved)
            best_path = improved

    return best_path
