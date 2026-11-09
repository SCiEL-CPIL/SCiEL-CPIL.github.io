import time
import random
import math

TIME_LIMIT = 45.0
N = 256


def can_add(bitset, x):
    """
    Return True if adding x to the set represented by bitset creates no
    non-constant 3-term arithmetic progression.
    Only APs involving x need to be checked:
      - x is the middle term: (x-d, x, x+d)
      - x is the first term:  (x, x+d, x+2d)
      - x is the last term:   (x-2d, x-d, x)
    """
    # Case 2: x is middle. m = x-d, n = x+d
    max_d = min(x - 1, N - x)
    for d in range(1, max_d + 1):
        m = x - d
        n = x + d
        if (bitset & (1 << (m - 1))) and (bitset & (1 << (n - 1))):
            return False

    # Case 1: x is first. m = x+d, n = x+2d
    max_d = (N - x) // 2
    for d in range(1, max_d + 1):
        m = x + d
        n = x + 2 * d
        if (bitset & (1 << (m - 1))) and (bitset & (1 << (n - 1))):
            return False

    # Case 3: x is last. m = x-2d, n = x-d
    max_d = x // 2
    for d in range(1, max_d + 1):
        m = x - 2 * d
        n = x - d
        if (bitset & (1 << (m - 1))) and (bitset & (1 << (n - 1))):
            return False

    return True


def construct():
    start_time = time.time()

    recip = [0.0] * (N + 1)
    for i in range(1, N + 1):
        recip[i] = 1.0 / i

    # Initial solution: greedy add in increasing order (largest reciprocals first)
    current_bitset = 0
    current_score = 0.0
    for x in range(1, N + 1):
        if can_add(current_bitset, x):
            current_bitset |= (1 << (x - 1))
            current_score += recip[x]

    # Maintain in_set list
    in_set = [x for x in range(1, N + 1) if current_bitset & (1 << (x - 1))]

    # Maintain available (addable) list
    def rebuild_available():
        avail = []
        for x in range(1, N + 1):
            if not (current_bitset & (1 << (x - 1))):
                if can_add(current_bitset, x):
                    avail.append(x)
        return avail

    available = rebuild_available()

    best_bitset = current_bitset
    best_score = current_score

    # Greedy saturate: add everything currently addable
    def saturate():
        nonlocal current_bitset, current_score, in_set
        changed = True
        while changed:
            changed = False
            for x in range(1, N + 1):
                if time.time() - start_time > TIME_LIMIT:
                    return
                if not (current_bitset & (1 << (x - 1))):
                    if can_add(current_bitset, x):
                        current_bitset |= (1 << (x - 1))
                        current_score += recip[x]
                        in_set.append(x)
                        changed = True
        # refresh available
        nonlocal available
        available = []
        for x in range(1, N + 1):
            if not (current_bitset & (1 << (x - 1))):
                if can_add(current_bitset, x):
                    available.append(x)

    saturate()
    if current_score > best_score:
        best_score = current_score
        best_bitset = current_bitset

    current_temp = 0.02
    cooling_rate = 0.9995
    min_temp = 1e-7
    step_count = 0

    while time.time() - start_time < TIME_LIMIT:
        if current_temp < min_temp:
            # Restart from best with a modest temperature
            current_bitset = best_bitset
            current_score = best_score
            in_set = [x for x in range(1, N + 1) if current_bitset & (1 << (x - 1))]
            available = []
            for x in range(1, N + 1):
                if not (current_bitset & (1 << (x - 1))):
                    if can_add(current_bitset, x):
                        available.append(x)
            current_temp = 0.01

        step_count += 1
        move_type = random.random()

        if move_type < 0.45:
            # Try adding a random addable number
            if available:
                idx = random.randint(0, len(available) - 1)
                x_add = available[idx]
                # Re-validate (bitset may have changed since last refresh)
                if can_add(current_bitset, x_add):
                    current_bitset |= (1 << (x_add - 1))
                    current_score += recip[x_add]
                    in_set.append(x_add)
                    available.pop(idx)
                    if current_score > best_score:
                        best_score = current_score
                        best_bitset = current_bitset
                # else: stale entry, drop it
        elif move_type < 0.7:
            # Try removing a random number (always valid, may enable adds)
            if in_set:
                idx = random.randint(0, len(in_set) - 1)
                x_rem = in_set[idx]
                delta = -recip[x_rem]
                if delta > 0 or random.random() < math.exp(delta / current_temp):
                    current_bitset &= ~(1 << (x_rem - 1))
                    current_score += delta
                    in_set.pop(idx)
                    available.append(x_rem)
        else:
            # Swap: remove one, add another (requires the add to be valid)
            if in_set and len(in_set) < N:
                idx_rem = random.randint(0, len(in_set) - 1)
                x_rem = in_set[idx_rem]
                trial_bitset = current_bitset & ~(1 << (x_rem - 1))
                # Pick a candidate to add: prefer from available, else any not-in-set
                candidates = None
                if available:
                    # available may be stale; filter to those valid under trial_bitset
                    valid_cands = []
                    for x in available:
                        if not (trial_bitset & (1 << (x - 1))):
                            if can_add(trial_bitset, x):
                                valid_cands.append(x)
                    if valid_cands:
                        candidates = valid_cands
                if candidates is None:
                    # build fresh list of addable under trial_bitset
                    candidates = []
                    for x in range(1, N + 1):
                        if not (trial_bitset & (1 << (x - 1))):
                            if can_add(trial_bitset, x):
                                candidates.append(x)
                if candidates:
                    x_add = random.choice(candidates)
                    trial_bitset |= (1 << (x_add - 1))
                    delta = recip[x_add] - recip[x_rem]
                    if delta > 0 or random.random() < math.exp(delta / current_temp):
                        current_bitset = trial_bitset
                        current_score += delta
                        in_set.pop(idx_rem)
                        in_set.append(x_add)
                        # rebuild available (cheap enough, and keeps it consistent)
                        available = []
                        for x in range(1, N + 1):
                            if not (current_bitset & (1 << (x - 1))):
                                if can_add(current_bitset, x):
                                    available.append(x)
                        if current_score > best_score:
                            best_score = current_score
                            best_bitset = current_bitset

        current_temp *= cooling_rate

        # Periodic saturate to grab any now-addable elements
        if step_count % 300 == 0:
            saturate()
            if current_score > best_score:
                best_score = current_score
                best_bitset = current_bitset

    # Return the best valid set found
    return [x for x in range(1, N + 1) if best_bitset & (1 << (x - 1))]


if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(sum(1.0 / x for x in res))
