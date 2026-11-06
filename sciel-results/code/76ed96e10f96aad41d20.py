import time
import math
import random
import bisect

N = 32
MAX_VAL = 512
MAX_D = 256

def _score_mask(mask):
    """Count distinct d > 0 such that a, a+d, a+2d all in set."""
    count = 0
    for d in range(1, MAX_D + 1):
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count

def _list_to_mask(lst):
    mask = 0
    for v in lst:
        mask |= (1 << (v - 1))
    return mask

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    best_score = -1
    best_list = list(range(1, 33))
    best_mask = _list_to_mask(best_list)
    best_score = _score_mask(best_mask)

    rng = random.Random(42)

    def _try_improve(current_list, current_set, current_mask, current_score, time_limit):
        """Run SA from a starting point, return best found."""
        nonlocal best_score, best_list, best_mask

        T = 50.0
        T_min = 0.01
        cooling = 0.99995  # Slow cooling
        steps = 0

        while time.time() < time_limit:
            # Choose a random element to swap
            x_idx = rng.randint(0, N - 1)
            x = current_list[x_idx]

            # Generate candidate y using heuristic
            if rng.random() < 0.15:
                # Pure random
                y = rng.randint(1, MAX_VAL)
                if y == x or y in current_set:
                    continue
            else:
                # Heuristic: try to create a new progression
                # Pick two random elements u, v from current set (not x)
                others = [v for v in current_set if v != x]
                if len(others) < 2:
                    y = rng.randint(1, MAX_VAL)
                    if y == x or y in current_set:
                        continue
                    return current_list, current_set, current_mask, current_score
                else:
                    u = others[rng.randint(0, len(others) - 1)]
                    v = others[rng.randint(0, len(others) - 1)]
                    if u == v:
                        v = others[rng.randint(0, len(others) - 1)]
                        if u == v:
                            y = rng.randint(1, MAX_VAL)
                            if y == x or y in current_set:
                                continue
                            return current_list, current_set, current_mask, current_score

                    # Three candidates to complete AP with u and v
                    candidates = []
                    # u, v, 2v-u
                    c = 2 * v - u
                    if 1 <= c <= MAX_VAL and c != x and c not in current_set:
                        candidates.append(c)
                    # v, u, 2u-v
                    c = 2 * u - v
                    if 1 <= c <= MAX_VAL and c != x and c not in current_set:
                        candidates.append(c)
                    # u, (u+v)/2, v if u+v even
                    if (u + v) % 2 == 0:
                        c = (u + v) // 2
                        if 1 <= c <= MAX_VAL and c != x and c not in current_set:
                            candidates.append(c)

                    if candidates:
                        y = rng.choice(candidates)
                    else:
                        y = rng.randint(1, MAX_VAL)
                        if y == x or y in current_set:
                            continue

            # Compute new score
            new_mask = current_mask ^ (1 << (x - 1)) ^ (1 << (y - 1))
            new_score = _score_mask(new_mask)
            delta = new_score - current_score

            # Accept if improvement or with SA probability
            if delta > 0 or (T > T_min and rng.random() < math.exp(delta / T)):
                current_score = new_score
                current_mask = new_mask
                current_set.remove(x)
                current_set.add(y)
                current_list.pop(x_idx)
                ins_idx = bisect.bisect_left(current_list, y)
                current_list.insert(ins_idx, y)

                if current_score > best_score:
                    best_score = current_score
                    best_list = current_list.copy()
                    best_mask = current_mask

            T *= cooling
            steps += 1

        return current_list, current_set, current_mask, current_score

    # Multiple diverse initializations
    inits = []

    # 1. Consecutive
    inits.append(list(range(1, 33)))

    # 2. Powers of 2 plus fill
    inits.append(sorted([1, 2, 4, 8, 16, 32, 64, 128, 256] + list(range(3, 26))))

    # 3. Arithmetic progression with large step
    inits.append(sorted([1 + i * 16 for i in range(32)]))

    # 4. Two interleaved APs
    inits.append(sorted([1 + i * 8 for i in range(16)] + [5 + i * 8 for i in range(16)]))

    # 5-10. Random samples
    for _ in range(6):
        inits.append(sorted(random.sample(range(1, MAX_VAL + 1), N)))

    # 11. Greedy construction
    greedy_set = set()
    greedy_list = []
    for _ in range(N):
        best_gain = -1
        best_val = None
        # Try all values, pick the one that maximizes score gain
        current_mask = _list_to_mask(greedy_list) if greedy_list else 0
        for v in range(1, MAX_VAL + 1):
            if v in greedy_set:
                continue
            new_mask = current_mask | (1 << (v - 1))
            s = _score_mask(new_mask)
            if s > best_gain:
                best_gain = s
                best_val = v
        if best_val is not None:
            greedy_set.add(best_val)
            greedy_list.append(best_val)
    inits.append(sorted(greedy_list))

    # Run SA from each initialization
    for idx, init in enumerate(inits):
        if time.time() > deadline:
            break

        current_list = sorted(init[:])
        if len(current_list) != N or any(x < 1 or x > MAX_VAL for x in current_list):
            current_list = sorted(random.sample(range(1, MAX_VAL + 1), N))

        current_set = set(current_list)
        current_mask = _list_to_mask(current_list)
        current_score = _score_mask(current_mask)

        if current_score > best_score:
            best_score = current_score
            best_list = current_list.copy()
            best_mask = current_mask

        # Allocate time for this run
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        # Give each run a portion of remaining time
        runs_left = len(inits) - idx
        time_per_run = remaining / runs_left * 0.9  # 90% of allocated
        local_deadline = time.time() + time_per_run

        current_list, current_set, current_mask, current_score = _try_improve(
            current_list, current_set, current_mask, current_score, local_deadline
        )

    return sorted(best_list)

construct()
