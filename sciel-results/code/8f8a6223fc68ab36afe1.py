"""
MaxSAT-style stochastic local search for maximizing 3-APs in a 4-AP-free subset of [1,256].
Uses bitmask-based AP checking for speed, with a WalkSAT-inspired solver that
directly optimizes the trade-off between 4-AP avoidance (hard) and 3-AP count (soft).
"""
import time
import random
import math

N = 256
MAX_TIME = 82  # safety margin before 90s kill

# Precompute 3-APs and 4-APs as lists of (a, b, c) / (a, b, c, d)
_three_aps = []
for d in range(1, N // 2 + 1):
    for a in range(1, N - 2 * d + 1):
        _three_aps.append((a, a + d, a + 2 * d))

_four_aps = []
for d in range(1, N // 3 + 1):
    for a in range(1, N - 3 * d + 1):
        _four_aps.append((a, a + d, a + 2 * d, a + 3 * d))

# Map each element to indices of 3-APs and 4-APs it belongs to
_elem_to_3aps = [[] for _ in range(N + 1)]
for i, ap in enumerate(_three_aps):
    for x in ap:
        _elem_to_3aps[x].append(i)

_elem_to_4aps = [[] for _ in range(N + 1)]
for i, ap in enumerate(_four_aps):
    for x in ap:
        _elem_to_4aps[x].append(i)


def count_3aps(S):
    """Count distinct 3-term arithmetic progressions in set S."""
    S = set(S)
    count = 0
    for ap in _three_aps:
        if ap[0] in S and ap[1] in S and ap[2] in S:
            count += 1
    return count


def has_4ap(S):
    """Check if set S contains any nonconstant 4-term arithmetic progression."""
    S = set(S)
    for ap in _four_aps:
        if ap[0] in S and ap[1] in S and ap[2] in S and ap[3] in S:
            return True
    return False


def is_valid_add(S, x):
    """Check if adding x to S creates a 4-AP."""
    for ap_idx in _elem_to_4aps[x]:
        a, b, c, d = _four_aps[ap_idx]
        if x == a:
            if b in S and c in S and d in S:
                return False
        elif x == b:
            if a in S and c in S and d in S:
                return False
        elif x == c:
            if a in S and b in S and d in S:
                return False
        else:
            if a in S and b in S and c in S:
                return False
    return True


def delta_add(S, x):
    """Change in 3-AP count if we add x to S."""
    delta = 0
    for ap_idx in _elem_to_3aps[x]:
        a, b, c = _three_aps[ap_idx]
        if x == a:
            if b in S and c in S:
                delta += 1
        elif x == b:
            if a in S and c in S:
                delta += 1
        else:
            if a in S and b in S:
                delta += 1
    return delta


def delta_remove(S, x):
    """Change in 3-AP count if we remove x from S (negative = loss)."""
    delta = 0
    for ap_idx in _elem_to_3aps[x]:
        a, b, c = _three_aps[ap_idx]
        if x == a:
            if b in S and c in S:
                delta += 1
        elif x == b:
            if a in S and c in S:
                delta += 1
        else:
            if a in S and b in S:
                delta += 1
    return -delta


def is_valid_swap(S, x_out, x_in):
    """Check if swapping x_out for x_in is valid (no 4-APs in new set)."""
    for ap_idx in _elem_to_4aps[x_in]:
        a, b, c, d = _four_aps[ap_idx]
        if x_in == a:
            others = (b, c, d)
        elif x_in == b:
            others = (a, c, d)
        elif x_in == c:
            others = (a, b, d)
        else:
            others = (a, b, c)
        all_in = True
        for y in others:
            if y not in S or y == x_out:
                all_in = False
                break
        if all_in:
            return False
    return True


def swap_gain(S, x_out, x_in):
    """Net change in 3-AP count for swapping x_out out and x_in in."""
    loss = 0
    for ap_idx in _elem_to_3aps[x_out]:
        a, b, c = _three_aps[ap_idx]
        if x_out == a:
            if b in S and c in S:
                loss += 1
        elif x_out == b:
            if a in S and c in S:
                loss += 1
        else:
            if a in S and b in S:
                loss += 1

    gain = 0
    for ap_idx in _elem_to_3aps[x_in]:
        a, b, c = _three_aps[ap_idx]
        if x_in == a:
            if b in S and c in S and b != x_out and c != x_out:
                gain += 1
        elif x_in == b:
            if a in S and c in S and a != x_out and c != x_out:
                gain += 1
        else:
            if a in S and b in S and a != x_out and b != x_out:
                gain += 1

    return gain - loss


def greedy_construct(seed=0):
    """
    Greedy construction: add elements in order of highest marginal 3-AP gain
    while maintaining no-4-AP constraint.
    """
    random.seed(seed)
    S = set()
    available = list(range(1, N + 1))

    while available:
        best_gain = -1
        best_elem = None
        candidates = []
        for x in available:
            if not is_valid_add(S, x):
                continue
            gain = delta_add(S, x)
            candidates.append((gain, x))

        if not candidates:
            break

        candidates.sort(key=lambda item: item[0], reverse=True)

        top_k = min(3, len(candidates))
        if random.random() < 0.7:
            best_elem = candidates[0][1]
        else:
            best_elem = random.choice([c[1] for c in candidates[:top_k]])

        S.add(best_elem)
        available.remove(best_elem)

    return S


def local_search_1move(S, max_iters=5000):
    """
    Local search with single-add and single-remove moves.
    Returns improved set.
    """
    S = set(S)
    current_score = count_3aps(S)
    best_S = set(S)
    best_score = current_score

    for _ in range(max_iters):
        improved = False

        for x in range(1, N + 1):
            if x not in S:
                if not is_valid_add(S, x):
                    continue
                gain = delta_add(S, x)
                if gain > 0:
                    S.add(x)
                    current_score += gain
                    improved = True
                    break

        if improved:
            if current_score > best_score:
                best_score = current_score
                best_S = set(S)
            continue

        for x in list(S):
            loss = -delta_remove(S, x)
            if loss < 0:
                S.discard(x)
                current_score += loss
                improved = True
                break

        if not improved:
            break

    return best_S


def local_search_swap(S, max_iters=3000):
    """
    Local search with swap moves (remove one, add another).
    Returns improved set.
    """
    S = set(S)
    current_score = count_3aps(S)
    best_S = set(S)
    best_score = current_score
    not_in_S = set(range(1, N + 1)) - S

    for _ in range(max_iters):
        improved = False
        best_swap = None
        best_gain = 0

        for x_out in list(S):
            for x_in in not_in_S:
                if not is_valid_swap(S, x_out, x_in):
                    continue
                g = swap_gain(S, x_out, x_in)
                if g > best_gain:
                    best_gain = g
                    best_swap = (x_out, x_in)

        if best_swap is not None and best_gain > 0:
            x_out, x_in = best_swap
            S.discard(x_out)
            S.add(x_in)
            not_in_S.discard(x_in)
            not_in_S.add(x_out)
            current_score += best_gain
            improved = True

            if current_score > best_score:
                best_score = current_score
                best_S = set(S)

        if not improved:
            break

    return best_S


def _exp(x):
    """Safe exponent for acceptance probability."""
    if x < -50:
        return 0.0
    if x > 50:
        return 1.0
    return math.exp(x)


def walksat_solve(S, max_iters=20000, p_flip=0.3, time_limit=60):
    """
    WalkSAT-inspired solver: treats 4-APs as hard constraints (must be 0)
    and 3-APs as soft objectives (maximize). Uses random walks to escape local optima.
    """
    S = set(S)
    current_score = count_3aps(S)
    best_S = set(S)
    best_score = current_score
    start_time = time.time()

    # Track 4-AP violations
    def count_4ap_violations(S):
        S = set(S)
        count = 0
        for ap in _four_aps:
            if ap[0] in S and ap[1] in S and ap[2] in S and ap[3] in S:
                count += 1
        return count

    current_violations = count_4ap_violations(S)

    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break

        # If we have violations, try to fix them
        if current_violations > 0:
            # Find a 4-AP violation and randomly remove one element from it
            for ap in _four_aps:
                if all(x in S for x in ap):
                    x = random.choice(ap)
                    S.discard(x)
                    current_violations -= 1
                    current_score = count_3aps(S)
                    break
        else:
            # No violations: try to improve 3-AP count
            move_type = random.random()

            if move_type < 0.4:
                # Add move
                candidates = [x for x in range(1, N + 1) if x not in S and is_valid_add(S, x)]
                if candidates:
                    x = random.choice(candidates)
                    gain = delta_add(S, x)
                    if gain > 0 or random.random() < 0.1:
                        S.add(x)
                        current_score += gain
            elif move_type < 0.6:
                # Remove move
                if S:
                    x = random.choice(list(S))
                    loss = -delta_remove(S, x)
                    if loss < 0 or random.random() < 0.05:
                        S.discard(x)
                        current_score += loss
            else:
                # Swap move
                in_S = list(S)
                not_in_S = [x for x in range(1, N + 1) if x not in S]
                if in_S and not_in_S:
                    x_out = random.choice(in_S)
                    x_in = random.choice(not_in_S)
                    if is_valid_swap(S, x_out, x_in):
                        g = swap_gain(S, x_out, x_in)
                        if g > 0 or random.random() < 0.1:
                            S.discard(x_out)
                            S.add(x_in)
                            current_score += g

            # Random walk: flip a random element with small probability
            if random.random() < p_flip:
                x = random.randint(1, N)
                if x in S:
                    S.discard(x)
                else:
                    if is_valid_add(S, x):
                        S.add(x)
                current_score = count_3aps(S)
                current_violations = count_4ap_violations(S)

        if current_score > best_score and current_violations == 0:
            best_score = current_score
            best_S = set(S)

    return best_S


def construct():
    start_time = time.time()

    # Phase 1: Multiple greedy constructions with different seeds
    best_S = set()
    best_score = -1

    for seed in range(8):
        if time.time() - start_time > MAX_TIME * 0.12:
            break
        S = greedy_construct(seed)
        score = count_3aps(S)
        if score > best_score:
            best_score = score
            best_S = set(S)

    # Phase 2: Local search refinement
    S = local_search_1move(best_S)
    score = count_3aps(S)
    if score > best_score:
        best_score = score
        best_S = set(S)

    S = local_search_swap(best_S)
    score = count_3aps(S)
    if score > best_score:
        best_score = score
        best_S = set(S)

    S = local_search_1move(best_S)
    score = count_3aps(S)
    if score > best_score:
        best_score = score
        best_S = set(S)

    elapsed = time.time() - start_time
    if elapsed > MAX_TIME * 0.35:
        if has_4ap(best_S):
            best_S = greedy_construct(0)
            best_S = local_search_1move(best_S)
        return sorted(best_S)

    # Phase 3: WalkSAT-inspired global search with restarts
    remaining_time = MAX_TIME - elapsed
    n_restarts = max(2, int(remaining_time / 10))

    for r in range(n_restarts):
        if time.time() - start_time > MAX_TIME - 8:
            break

        # Perturb the best solution
        perturbed = set(best_S)
        for _ in range(random.randint(3, 10)):
            if random.random() < 0.5:
                candidates = [x for x in range(1, N + 1) if x not in perturbed]
                if candidates:
                    perturbed.add(random.choice(candidates))
            else:
                if perturbed:
                    perturbed.discard(random.choice(list(perturbed)))

        if has_4ap(perturbed):
            for x in list(perturbed):
                if not has_4ap(perturbed - {x}):
                    perturbed.discard(x)
                    break

        # Run WalkSAT solver
        ws_S = walksat_solve(perturbed, max_iters=10000, p_flip=0.25,
                              time_limit=min(15, MAX_TIME - (time.time() - start_time) - 5))
        ws_score = count_3aps(ws_S)

        if ws_score > best_score:
            best_score = ws_score
            best_S = set(ws_S)

        # Local search polish
        ls_S = local_search_1move(ws_S, max_iters=1500)
        ls_score = count_3aps(ls_S)
        if ls_score > best_score:
            best_score = ls_score
            best_S = set(ls_S)

        ls_S = local_search_swap(best_S, max_iters=800)
        ls_score = count_3aps(ls_S)
        if ls_score > best_score:
            best_score = ls_score
            best_S = set(ls_S)

    # Final polish
    final_S = local_search_1move(best_S, max_iters=2000)
    final_score = count_3aps(final_S)
    if final_score > best_score:
        best_score = final_score
        best_S = set(final_S)

    final_S = local_search_swap(best_S, max_iters=1000)
    final_score = count_3aps(final_S)
    if final_score > best_score:
        best_score = final_score
        best_S = set(final_S)

    # Final verification
    if has_4ap(best_S):
        best_S = greedy_construct(0)
        best_S = local_search_1move(best_S)

    return sorted(best_S)


if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}, 3-APs: {count_3aps(result)}")
    print(f"Has 4-AP: {has_4ap(result)}")
    print(f"First 20: {result[:20]}")
