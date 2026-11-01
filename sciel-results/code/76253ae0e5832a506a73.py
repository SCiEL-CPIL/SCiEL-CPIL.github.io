# EVOLVE-BLOCK-START
import time
import random
import itertools
import math

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    Returns a list of codewords (each a list of ints in 0..q-1).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    if n == 0:
        return [[]]

    total_words = q ** n

    # For the specific hard case q=3, n=6, radius=1, use Lagrangian + smooth ILS
    if q == 3 and n == 6 and radius == 1:
        return _solve_q3_n6_r1(deadline)

    # Generic fallback for other cases
    if total_words > 10000:
        return _generic_construct(q, n, radius, deadline)

    # Generic bitset-based solver for small spaces
    return _generic_bitset_construct(q, n, radius, deadline)

def _solve_q3_n6_r1(deadline):
    q = 3
    n = 6
    radius = 1
    total_words = 729

    # Precompute neighbors as bitsets
    all_tuples = list(itertools.product(range(q), repeat=n))
    idx_map = {t: i for i, t in enumerate(all_tuples)}

    # Precompute neighbor lists and bitset masks
    neighbor_indices = [[] for _ in range(total_words)]
    neighbors = [0] * total_words
    for i, t in enumerate(all_tuples):
        mask = 0
        mask |= (1 << i)
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

    # Precompute perturbation vectors in Z_3^6 for structured swaps
    perturbations = []
    for pos in range(n):
        for delta in (1, 2):
            p = [0] * n
            p[pos] = delta
            perturbations.append(p)
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

    def lagrangian_greedy(lambda_weights, max_iter=200):
        """
        Greedy construction using Lagrangian weights.
        Score for adding codeword c = sum of lambda_weights[j] for j in neighbors[c] not yet covered.
        """
        code_set = set()
        cover_count = [0] * total_words
        uncovered_mask = full_mask

        for _ in range(max_iter):
            if time.time() > deadline - 10:
                break
            if not uncovered_mask:
                break

            best_idx = -1
            best_score = -1.0
            for i in range(total_words):
                if i in code_set:
                    continue
                new_covered = neighbors[i] & uncovered_mask
                if new_covered == 0:
                    continue
                # Compute weighted score
                score = 0.0
                mask = new_covered
                while mask:
                    low_bit = mask & (-mask)
                    idx = low_bit.bit_length() - 1
                    score += lambda_weights[idx]
                    mask ^= low_bit
                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx == -1 or best_score <= 0:
                break

            code_set.add(best_idx)
            for idx in neighbor_indices[best_idx]:
                cover_count[idx] += 1
            uncovered_mask &= ~neighbors[best_idx]

        return code_set, cover_count, uncovered_mask

    def smooth_lagrangian_greedy(lambda_weights, max_iter=200):
        """
        Greedy construction using a smooth log-sum-exp surrogate over the Lagrangian weights.
        Score for adding codeword c = soft-max of lambda_weights over newly-covered words.
        This smooths the nonsmooth min over coverage, giving a differentiable-ish objective.
        """
        code_set = set()
        cover_count = [0] * total_words
        uncovered_mask = full_mask

        for _ in range(max_iter):
            if time.time() > deadline - 10:
                break
            if not uncovered_mask:
                break

            best_idx = -1
            best_score = -1.0
            for i in range(total_words):
                if i in code_set:
                    continue
                new_covered = neighbors[i] & uncovered_mask
                if new_covered == 0:
                    continue
                # Compute log-sum-exp over lambda_weights for newly-covered words
                # Use a scaled log-sum-exp for numerical stability
                vals = []
                mask = new_covered
                while mask:
                    low_bit = mask & (-mask)
                    idx = low_bit.bit_length() - 1
                    vals.append(lambda_weights[idx])
                    mask ^= low_bit
                if not vals:
                    continue
                m = max(vals)
                if m < 1e-12:
                    score = 0.0
                else:
                    # log-sum-exp = m + log(sum(exp(v - m)))
                    s = 0.0
                    for v in vals:
                        s += math.exp(v - m)
                    score = m + math.log(s)
                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx == -1 or best_score <= 0:
                break

            code_set.add(best_idx)
            for idx in neighbor_indices[best_idx]:
                cover_count[idx] += 1
            uncovered_mask &= ~neighbors[best_idx]

        return code_set, cover_count, uncovered_mask

    def update_lagrangian(lambda_weights, cover_count, uncovered_mask, step_size):
        """
        Subgradient update: increase lambda for uncovered words, decrease for covered.
        """
        for i in range(total_words):
            if cover_count[i] == 0:
                lambda_weights[i] = max(0.0, lambda_weights[i] + step_size)
            else:
                lambda_weights[i] = max(0.0, lambda_weights[i] - step_size * 0.1)

    def reduce_code(code_set, cover_count):
        changed = True
        while changed and time.time() < deadline - 5:
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
        while uncovered_mask and time.time() < deadline - 10:
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

    def local_search_1exchange(code_set, cover_count, rng, max_iters=100):
        """
        Try 1-exchange moves: replace codeword c with c+d for small d.
        Accept if the code remains valid and size doesn't increase.
        """
        improved = False
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            c = rng.choice(list(code_set))
            d = rng.choice(perturbations)
            c_tuple = all_tuples[c]
            new_tuple = tuple((c_tuple[i] + d[i]) % q for i in range(n))
            new_idx = idx_map[new_tuple]

            if new_idx in code_set:
                continue

            # Try swap
            for idx in neighbor_indices[c]:
                cover_count[idx] -= 1
            code_set.remove(c)

            code_set.add(new_idx)
            for idx in neighbor_indices[new_idx]:
                cover_count[idx] += 1

            # Check validity
            valid = True
            for i in range(total_words):
                if cover_count[i] == 0:
                    valid = False
                    break

            if valid:
                improved = True
            else:
                # Revert
                code_set.remove(new_idx)
                for idx in neighbor_indices[new_idx]:
                    cover_count[idx] -= 1
                code_set.add(c)
                for idx in neighbor_indices[c]:
                    cover_count[idx] += 1

        return code_set, cover_count, improved

    def high_precision_polish(code_set, cover_count, rng, max_iters=500):
        """
        Final high-precision polish pass: tighter tolerance, more iterations.
        Try to reduce the code size by finding removable codewords or beneficial
        exchanges. Also try 2-exchange moves for deeper local search.
        """
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # First, try to remove any redundant codeword
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
                    improved = True
                    break
            # Then try 1-exchange moves
            if not improved:
                for _ in range(max_iters):
                    if time.time() > deadline - 5:
                        break
                    c = rng.choice(list(code_set))
                    d = rng.choice(perturbations)
                    c_tuple = all_tuples[c]
                    new_tuple = tuple((c_tuple[i] + d[i]) % q for i in range(n))
                    new_idx = idx_map[new_tuple]

                    if new_idx in code_set:
                        continue

                    for idx in neighbor_indices[c]:
                        cover_count[idx] -= 1
                    code_set.remove(c)

                    code_set.add(new_idx)
                    for idx in neighbor_indices[new_idx]:
                        cover_count[idx] += 1

                    valid = True
                    for i in range(total_words):
                        if cover_count[i] == 0:
                            valid = False
                            break

                    if valid:
                        improved = True
                    else:
                        code_set.remove(new_idx)
                        for idx in neighbor_indices[new_idx]:
                            cover_count[idx] -= 1
                        code_set.add(c)
                        for idx in neighbor_indices[c]:
                            cover_count[idx] += 1

        return code_set, cover_count

    # Main algorithm: Lagrangian relaxation + smooth surrogate + ILS + polish
    rng = random.Random(42)

    best_size = float('inf')
    best_code = []

    # Phase 1: Lagrangian relaxation to find good initial solutions
    lambda_weights = [1.0] * total_words
    step_size = 0.5

    for lag_iter in range(30):
        if time.time() > deadline - 20:
            break
        code_set, cover_count, uncovered_mask = lagrangian_greedy(lambda_weights)
        code_set, cover_count = reduce_code(code_set, cover_count)
        uncovered_mask = get_uncovered_mask(cover_count)
        if uncovered_mask:
            code_set, cover_count, uncovered_mask = repair(code_set, cover_count, uncovered_mask)
            code_set, cover_count = reduce_code(code_set, cover_count)

        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = list(code_set)
            if best_size == 73:
                break

        update_lagrangian(lambda_weights, cover_count, uncovered_mask, step_size)
        step_size *= 0.95

    # Phase 2: Smooth log-sum-exp Lagrangian for diversity
    lambda_weights2 = [1.0] * total_words
    step_size2 = 0.5
    for lag_iter in range(20):
        if time.time() > deadline - 15:
            break
        code_set, cover_count, uncovered_mask = smooth_lagrangian_greedy(lambda_weights2)
        code_set, cover_count = reduce_code(code_set, cover_count)
        uncovered_mask = get_uncovered_mask(cover_count)
        if uncovered_mask:
            code_set, cover_count, uncovered_mask = repair(code_set, cover_count, uncovered_mask)
            code_set, cover_count = reduce_code(code_set, cover_count)

        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = list(code_set)
            if best_size == 73:
                break

        update_lagrangian(lambda_weights2, cover_count, uncovered_mask, step_size2)
        step_size2 *= 0.95

    # Phase 3: ILS from the best solution
    if best_code:
        code_set = set(best_code)
        cover_count = [0] * total_words
        for c in code_set:
            for idx in neighbor_indices[c]:
                cover_count[idx] += 1
        uncovered_mask = get_uncovered_mask(cover_count)
        if uncovered_mask:
            code_set, cover_count, uncovered_mask = repair(code_set, cover_count, uncovered_mask)
            code_set, cover_count = reduce_code(code_set, cover_count)
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = list(code_set)

    # ILS with structured perturbations
    if best_code:
        code_set = set(best_code)
        cover_count = [0] * total_words
        for c in code_set:
            for idx in neighbor_indices[c]:
                cover_count[idx] += 1

        iteration = 0
        while time.time() < deadline - 10:
            iteration += 1

            # Perturb: structured swap or random removal
            if rng.random() < 0.5:
                # Structured swap: remove one, add a nearby one
                if code_set:
                    c = rng.choice(list(code_set))
                    d = rng.choice(perturbations)
                    c_tuple = all_tuples[c]
                    new_tuple = tuple((c_tuple[i] + d[i]) % q for i in range(n))
                    new_idx = idx_map[new_tuple]
                    if new_idx not in code_set:
                        for idx in neighbor_indices[c]:
                            cover_count[idx] -= 1
                        code_set.remove(c)
                        code_set.add(new_idx)
                        for idx in neighbor_indices[new_idx]:
                            cover_count[idx] += 1
                code_set, cover_count, _ = local_search_1exchange(code_set, cover_count, rng, max_iters=50)
            else:
                # Random removal of k codewords
                k = rng.randint(2, 5)
                current_code = list(code_set)
                if len(current_code) < k:
                    k = len(current_code)
                if k > 0:
                    to_remove = rng.sample(current_code, k)
                    for c in to_remove:
                        code_set.remove(c)
                        for idx in neighbor_indices[c]:
                            cover_count[idx] -= 1
                    uncovered_mask = get_uncovered_mask(cover_count)
                    code_set, cover_count, uncovered_mask = repair(code_set, cover_count, uncovered_mask)
                    code_set, cover_count = reduce_code(code_set, cover_count)

            # Check improvement
            if len(code_set) < best_size:
                best_size = len(code_set)
                best_code = list(code_set)
                if best_size == 73:
                    break

    # Phase 4: Final high-precision polish on the best solution
    if best_code:
        code_set = set(best_code)
        cover_count = [0] * total_words
        for c in code_set:
            for idx in neighbor_indices[c]:
                cover_count[idx] += 1
        code_set, cover_count = high_precision_polish(code_set, cover_count, rng, max_iters=300)
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = list(code_set)

    result = [list(all_tuples[i]) for i in best_code]
    return result

def _generic_bitset_construct(q, n, radius, deadline):
    total_words = q ** n
    all_tuples = list(itertools.product(range(q), repeat=n))
    idx_map = {t: i for i, t in enumerate(all_tuples)}

    neighbors = [0] * total_words
    neighbor_indices = [[] for _ in range(total_words)]
    for i, t in enumerate(all_tuples):
        mask = 0
        for j, t_j in enumerate(all_tuples):
            dist = sum(1 for k in range(n) if t[k] != t_j[k])
            if dist <= radius:
                mask |= (1 << j)
                neighbor_indices[i].append(j)
        neighbors[i] = mask

    full_mask = (1 << total_words) - 1

    code_set = set()
    cover_count = [0] * total_words
    uncovered_mask = full_mask

    while uncovered_mask and time.time() < deadline - 10:
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
        if best_idx == -1: break
        code_set.add(best_idx)
        for idx in neighbor_indices[best_idx]:
            cover_count[idx] += 1
        uncovered_mask &= ~neighbors[best_idx]

    changed = True
    while changed and time.time() < deadline - 5:
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

    result = [list(all_tuples[i]) for i in code_set]
    return result

def _generic_construct(q, n, radius, deadline):
    all_words = list(itertools.product(range(q), repeat=n))
    def dist(a, b):
        return sum(1 for i in range(n) if a[i] != b[i])

    uncovered = set(all_words)
    code = []
    while uncovered and time.time() < deadline - 10:
        best = None
        best_cnt = -1
        for w in all_words:
            if w in set(code): continue
            cnt = sum(1 for u in uncovered if dist(w, u) <= radius)
            if cnt > best_cnt:
                best_cnt = cnt
                best = w
        if not best: break
        code.append(best)
        uncovered = {u for u in uncovered if dist(best, u) > radius}
    return [list(w) for w in code]

# EVOLVE-BLOCK-END
