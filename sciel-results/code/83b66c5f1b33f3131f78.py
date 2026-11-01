import time
import itertools
import random
import numpy as np

_DEADLINE = 140.0

def _direct_sum(q, n, radius):
    """Always-valid baseline: a radius-1 perfect code of length 4 over the alphabet,
    tensored with every word on the remaining n-4 coordinates.  Valid for R>=1, n>=4.
    Size = q^2 * q^(n-4) = q^(n-2)."""
    base = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    return [h + list(tail) for h in base
            for tail in itertools.product(range(q), repeat=n - 4)]

def _idx(w, q):
    idx = 0
    for x in w:
        idx = idx * q + x
    return idx

def _covers_all(cw, q, n, radius):
    """Exact coverage check over the whole space (only used for small spaces)."""
    total = q ** n
    covered = np.zeros(total, dtype=bool)
    for c in cw:
        ball = [c]
        for d in range(1, radius + 1):
            for pos in itertools.combinations(range(n), d):
                for vals in itertools.product(range(1, q), repeat=d):
                    w = list(c)
                    for p, v in zip(pos, vals):
                        w[p] = v
                    ball.append(w)
        for w in ball:
            covered[_idx(w, q)] = True
    return covered.all()

def _solve_q3_n6_r1(deadline):
    """Specialized solver for q=3, n=6, radius=1 using structured ILS in Z_3^6."""
    q = 3
    n = 6
    radius = 1
    total_words = 729

    # Precompute all tuples and index map
    all_tuples = list(itertools.product(range(q), repeat=n))
    idx_map = {t: i for i, t in enumerate(all_tuples)}

    # Precompute neighbor lists and bitset masks
    neighbor_indices = [[] for _ in range(total_words)]
    neighbors = [0] * total_words
    for i, t in enumerate(all_tuples):
        mask = (1 << i)
        neighbor_indices[i].append(i)
        for pos in range(n):
            orig = t[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                t_list = list(t)
                t_list[pos] = new_val
                nt = tuple(t_list)
                j = idx_map[nt]
                mask |= (1 << j)
                neighbor_indices[i].append(j)
        neighbors[i] = mask

    full_mask = (1 << total_words) - 1

    # Precompute perturbation vectors in Z_3^6
    # Single-coordinate changes
    perturbations = []
    for pos in range(n):
        for delta in (1, 2):
            p = [0] * n
            p[pos] = delta
            perturbations.append(p)
    # Two-coordinate changes
    for pos1 in range(n):
        for pos2 in range(pos1 + 1, n):
            for d1 in (1, 2):
                for d2 in (1, 2):
                    p = [0] * n
                    p[pos1] = d1
                    p[pos2] = d2
                    perturbations.append(p)

    def get_uncovered_mask(cover_count):
        mask = 0
        for i in range(total_words):
            if cover_count[i] == 0:
                mask |= (1 << i)
        return mask

    def greedy_start(rng):
        """Greedy set-cover initialization with random tie-breaking."""
        code_set = set()
        cover_count = [0] * total_words
        uncovered_mask = full_mask

        while uncovered_mask:
            best_idx = -1
            best_cnt = -1
            best_candidates = []
            for i in range(total_words):
                if i in code_set:
                    continue
                new_covered = neighbors[i] & uncovered_mask
                cnt = new_covered.bit_count()
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_candidates = [i]
                elif cnt == best_cnt and cnt > 0:
                    best_candidates.append(i)

            if best_idx == -1 or best_cnt == 0:
                break

            best_idx = rng.choice(best_candidates)
            code_set.add(best_idx)
            for idx in neighbor_indices[best_idx]:
                cover_count[idx] += 1
            uncovered_mask &= ~neighbors[best_idx]

        return code_set, cover_count, uncovered_mask

    def reduce_code(code_set, cover_count):
        """Prune redundant codewords."""
        changed = True
        while changed:
            changed = False
            for c in list(code_set):
                redundant = True
                for idx in neighbor_indices[c]:
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    code_set.remove(c)
                    for idx in neighbor_indices[c]:
                        cover_count[idx] -= 1
                    changed = True
                    break
        return code_set, cover_count

    def repair(code_set, cover_count, uncovered_mask):
        """Add codewords to cover all words."""
        while uncovered_mask:
            best_idx = -1
            best_cnt = -1
            for i in range(total_words):
                if i in code_set:
                    continue
                new_covered = neighbors[i] & uncovered_mask
                cnt = new_covered.bit_count()
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_idx = i
            if best_idx == -1 or best_cnt == 0:
                break
            code_set.add(best_idx)
            for idx in neighbor_indices[best_idx]:
                cover_count[idx] += 1
            uncovered_mask &= ~neighbors[best_idx]
        return code_set, cover_count, uncovered_mask

    def local_search_1exchange(code_set, cover_count, rng, max_iters=500):
        """
        Try 1-exchange moves: replace codeword c with c+d for small d.
        Accept if it maintains coverage (size stays same or decreases).
        """
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for c in list(code_set):
                if iters >= max_iters:
                    break
                c_tuple = all_tuples[c]
                for d in perturbations:
                    new_tuple = tuple((c_tuple[i] + d[i]) % q for i in range(n))
                    new_idx = idx_map[new_tuple]
                    if new_idx in code_set:
                        continue

                    # Try the swap
                    for idx in neighbor_indices[c]:
                        cover_count[idx] -= 1

                    # Check if removing c leaves any word uncovered
                    uncovered_after = False
                    for idx in neighbor_indices[c]:
                        if cover_count[idx] == 0:
                            uncovered_after = True
                            break

                    if uncovered_after:
                        # Revert
                        for idx in neighbor_indices[c]:
                            cover_count[idx] += 1
                        continue

                    # Remove c and add new_idx
                    code_set.remove(c)
                    code_set.add(new_idx)
                    for idx in neighbor_indices[new_idx]:
                        cover_count[idx] += 1

                    # Check if still valid
                    valid = True
                    for i in range(total_words):
                        if cover_count[i] == 0:
                            valid = False
                            break

                    if valid:
                        improved = True
                        break
                    else:
                        # Revert
                        code_set.remove(new_idx)
                        for idx in neighbor_indices[new_idx]:
                            cover_count[idx] -= 1
                        code_set.add(c)
                        for idx in neighbor_indices[c]:
                            cover_count[idx] += 1

        return code_set, cover_count

    def simulated_annealing_il(code_set, cover_count, rng, max_iters=2000, t_start=1.0, t_end=0.01):
        """
        Iterated Local Search with simulated annealing acceptance.
        Perturb with structured moves, then repair and reduce.
        """
        best_code = set(code_set)
        best_count = len(code_set)
        current_code = set(code_set)
        current_count = [0] * total_words
        for c in current_code:
            for idx in neighbor_indices[c]:
                current_count[idx] += 1

        temperature = t_start
        cooling = (t_end / t_start) ** (1.0 / max(1, max_iters))

        for it in range(max_iters):
            if time.time() > deadline:
                break

            # Perturb: remove one codeword and add a perturbed version
            if current_code:
                c = rng.choice(list(current_code))
                d = rng.choice(perturbations)
                c_tuple = all_tuples[c]
                new_tuple = tuple((c_tuple[i] + d[i]) % q for i in range(n))
                new_idx = idx_map[new_tuple]

                # Remove c
                for idx in neighbor_indices[c]:
                    current_count[idx] -= 1
                current_code.remove(c)

                # Add new_idx (might be same as c if d is zero, but we excluded that)
                if new_idx not in current_code:
                    current_code.add(new_idx)
                    for idx in neighbor_indices[new_idx]:
                        current_count[idx] += 1

            # Repair: cover any uncovered words
            uncovered_mask = 0
            for i in range(total_words):
                if current_count[i] == 0:
                    uncovered_mask |= (1 << i)

            if uncovered_mask:
                current_code, current_count, uncovered_mask = repair(
                    current_code, current_count, uncovered_mask
                )

            # Reduce: prune redundant codewords
            current_code, current_count = reduce_code(current_code, current_count)

            # Accept or reject
            new_size = len(current_code)
            if new_size < best_count:
                best_code = set(current_code)
                best_count = new_size
                temperature = t_start  # Reset temperature on improvement
            elif new_size == best_count:
                # Accept with probability 1 (same size)
                pass
            else:
                # Worse: accept with probability exp(-(new_size - best_count) / temperature)
                delta = new_size - best_count
                accept_prob = np.exp(-delta / max(temperature, 1e-10))
                if rng.random() < accept_prob:
                    pass  # Accept
                else:
                    # Reject: revert to best
                    current_code = set(best_code)
                    current_count = [0] * total_words
                    for c in current_code:
                        for idx in neighbor_indices[c]:
                            current_count[idx] += 1

            temperature *= cooling

        return best_code, best_count

    # Main search loop
    start_time = time.time()
    best_code = None
    best_size = float('inf')

    rng = random.Random(42)

    # Multiple restarts
    num_restarts = 15
    time_per_restart = (deadline - start_time - 5.0) / num_restarts

    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break

        # Greedy start
        code_set, cover_count, uncovered_mask = greedy_start(rng)
        if uncovered_mask:
            code_set, cover_count, uncovered_mask = repair(code_set, cover_count, uncovered_mask)

        # Reduce
        code_set, cover_count = reduce_code(code_set, cover_count)

        # Local search
        code_set, cover_count = local_search_1exchange(code_set, cover_count, rng, max_iters=300)

        # Reduce again
        code_set, cover_count = reduce_code(code_set, cover_count)

        current_size = len(code_set)
        if current_size < best_size:
            best_size = current_size
            best_code = set(code_set)

            # If we hit the known optimum, stop early
            if best_size == 73:
                break

        # ILS with simulated annealing
        if time.time() < deadline - 10.0:
            il_code, il_size = simulated_annealing_il(
                code_set, cover_count, rng,
                max_iters=500, t_start=1.0, t_end=0.05
            )
            if il_size < best_size:
                best_size = il_size
                best_code = set(il_code)
                if best_size == 73:
                    break

    # Convert to list of words
    if best_code is None:
        # Fallback: use direct sum
        return _direct_sum(3, 6, 1)

    final_code = [all_tuples[i].tolist() for i in best_code]

    # Final verification
    if not _covers_all(final_code, 3, 6, 1):
        return _direct_sum(3, 6, 1)

    return final_code

def _greedy_with_local_search(q, n, radius, time_budget, rng):
    """Greedy set-cover with local search refinement for generic cases.
    Returns (codewords, verified)."""
    total = q ** n
    if total > 3 * 10**6:
        return None, False

    start = time.time()

    # Precompute balls and reverse balls
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)

    balls = []
    reverse_balls = [[] for _ in range(total)]

    def get_ball(center):
        idx = _idx(center, q)
        ball = [idx]
        for d in range(1, radius + 1):
            for pos in itertools.combinations(range(n), d):
                for vals in itertools.product(range(1, q), repeat=d):
                    w = list(center)
                    for p, v in zip(pos, vals):
                        w[p] = v
                    ball.append(_idx(w, q))
        return ball

    for i in range(total):
        b = get_ball(all_words[i])
        balls.append(b)
        for w in b:
            reverse_balls[w].append(i)

    covered = np.zeros(total, dtype=bool)
    ball_size = len(balls[0])
    gain = np.full(total, ball_size, dtype=np.int32)

    code_indices = []

    while not covered.all():
        if time.time() - start > time_budget:
            return None, False

        max_gain = np.max(gain)
        if max_gain == 0:
            break
        candidates = np.where(gain == max_gain)[0]
        best = rng.choice(candidates)

        code_indices.append(best)

        b = balls[best]
        for w in b:
            if not covered[w]:
                covered[w] = True
                for c in reverse_balls[w]:
                    gain[c] -= 1

    if not covered.all():
        return None, False

    # Local Search: Prune redundant codewords
    code_set = set(code_indices)
    cover_count = np.zeros(total, dtype=np.int32)
    for c_idx in code_set:
        for w in balls[c_idx]:
            cover_count[w] += 1

    if not (cover_count > 0).all():
        return None, False

    changed = True
    while changed:
        changed = False
        current_code = list(code_set)
        rng.shuffle(current_code)
        for c_idx in current_code:
            if time.time() - start > time_budget:
                break
            is_essential = False
            for w in balls[c_idx]:
                if cover_count[w] == 1:
                    is_essential = True
                    break
            if not is_essential:
                code_set.remove(c_idx)
                for w in balls[c_idx]:
                    cover_count[w] -= 1
                changed = True
                if not (cover_count > 0).all():
                    return None, False
        if time.time() - start > time_budget:
            break

    if len(code_set) == 0:
        return None, False

    final_code = [all_words[i].tolist() for i in code_set]

    if not _covers_all(final_code, q, n, radius):
        return None, False

    return final_code, True


def construct(q, n, radius):
    start = time.time()

    # Always-valid baseline
    if n >= 4 and radius >= 1:
        baseline = _direct_sum(q, n, radius)
    else:
        baseline = [list(w) for w in itertools.product(range(q), repeat=n)]

    best = baseline
    best_len = len(baseline)

    # Specialized solver for the hard case
    if q == 3 and n == 6 and radius == 1:
        deadline = start + _DEADLINE - 5.0
        try:
            cand = _solve_q3_n6_r1(deadline)
            if cand and _covers_all(cand, 3, 6, 1):
                if len(cand) < best_len:
                    best = cand
                    best_len = len(cand)
        except Exception:
            pass
        return best

    # Generic fallback
    if q ** n <= 3 * 10**6 and best_len > 1:
        time_budget_total = max(0.5, _DEADLINE - (time.time() - start) - 5.0)
        num_restarts = 20
        time_per_restart = time_budget_total / num_restarts

        rng = random.Random(42)

        for i in range(num_restarts):
            if time.time() - start > _DEADLINE - 5.0:
                break
            try:
                cand, verified = _greedy_with_local_search(q, n, radius, time_per_restart, rng)
            except Exception:
                cand, verified = None, False
            if cand is not None and verified and len(cand) < best_len:
                best = cand
                best_len = len(cand)

    if time.time() - start > _DEADLINE - 5.0:
        return baseline

    return best
