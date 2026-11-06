import time
import random
import math

def _score_bitmask(mask):
    """Fast score: count distinct d>0 where a, a+d, a+2d all in set."""
    count = 0
    for d in range(1, 256):
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count

def _set_to_mask(s):
    m = 0
    for x in s:
        m |= (1 << (x - 1))
    return m

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    rng = random.Random(12345)

    # --- Initializations: diverse structured starts ---
    candidates = []

    # 1. Dense core [1..32]
    s1 = list(range(1, 33))
    candidates.append(s1)

    # 2. Dense core [1..31] + outlier 47 (from cell findings: 23 diffs)
    s2 = list(range(1, 32)) + [47]
    candidates.append(s2)

    # 3. Two interleaved APs with small step
    s3 = list(range(1, 33, 2)) + list(range(2, 34, 2))  # just 1..32 again
    # Better: two interleaved with larger gaps
    s3 = [1 + i * 16 for i in range(16)] + [2 + i * 16 for i in range(16)]
    s3 = sorted(set(s3))
    if len(s3) == 32:
        candidates.append(s3)

    # 4. Dense core [1..28] + 4 spread outliers
    s4 = list(range(1, 29)) + [40, 60, 80, 100]
    candidates.append(s4)

    # 5. Random
    s5 = rng.sample(range(1, 513), 32)
    candidates.append(s5)

    # 6. Dense core [1..24] + 8 moderately spread
    s6 = list(range(1, 25)) + [30, 40, 50, 60, 70, 80, 90, 100]
    candidates.append(s6)

    best_mask = 0
    best_score = -1
    best_set = None

    for s in candidates:
        if len(set(s)) != 32:
            continue
        m = _set_to_mask(s)
        sc = _score_bitmask(m)
        if sc > best_score:
            best_score = sc
            best_mask = m
            best_set = list(s)

    # --- Multi-restart Simulated Annealing with polish ---
    num_restarts = 8
    # Each restart gets roughly equal budget, leave 10s for final polish
    restart_budget = (deadline - start_time - 10.0) / num_restarts

    for restart in range(num_restarts):
        if time.time() >= deadline - 12:
            break

        # Start from best, perturb for diversity
        if restart == 0:
            current_set = list(best_set)
        else:
            current_set = list(best_set)
            # Perturb: swap k random elements with random values
            k = min(5, 32)
            for _ in range(k):
                idx = rng.randint(0, 31)
                new_val = rng.randint(1, 512)
                if new_val not in current_set:
                    current_set[idx] = new_val
            # Deduplicate: if collision, pick another
            seen = set()
            deduped = []
            for v in current_set:
                if v not in seen:
                    seen.add(v)
                    deduped.append(v)
            while len(deduped) < 32:
                v = rng.randint(1, 512)
                if v not in seen:
                    seen.add(v)
                    deduped.append(v)
            current_set = deduped

        current_mask = _set_to_mask(current_set)
        current_score = _score_bitmask(current_mask)

        # SA parameters
        T = 12.0
        T_min = 0.01
        cooling_rate = 0.99999
        stall_count = 0
        stall_threshold = 1500

        while time.time() < deadline - 12 and T > T_min:
            # Generate neighbor: swap one element
            idx = rng.randint(0, 31)
            old_val = current_set[idx]
            new_val = rng.randint(1, 512)

            if new_val == old_val or new_val in current_set:
                continue

            # Update mask
            new_mask = current_mask
            new_mask &= ~(1 << (old_val - 1))
            new_mask |= (1 << (new_val - 1))

            new_score = _score_bitmask(new_mask)
            delta = new_score - current_score

            # Acceptance: always accept improvements, probabilistic for worsening
            if delta > 0 or (T > 0 and rng.random() < math.exp(delta / T)):
                current_set[idx] = new_val
                current_mask = new_mask
                current_score = new_score
                stall_count = 0

                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
                    best_set = list(current_set)
            else:
                stall_count += 1

            # Accelerate cooling when stalled
            if stall_count > stall_threshold:
                T *= 0.92
                stall_count = 0

            T *= cooling_rate

    # --- Final deterministic local search polish ---
    if time.time() < deadline - 3 and best_set is not None:
        current_set = list(best_set)
        current_mask = _set_to_mask(current_set)
        current_score = _score_bitmask(current_mask)

        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for i in range(32):
                if time.time() >= deadline - 3:
                    break
                old_val = current_set[i]
                best_new_val = None
                best_new_score = current_score

                # Try all possible replacements
                for new_val in range(1, 513):
                    if new_val == old_val or new_val in current_set:
                        continue
                    new_mask = current_mask
                    new_mask &= ~(1 << (old_val - 1))
                    new_mask |= (1 << (new_val - 1))
                    new_score = _score_bitmask(new_mask)
                    if new_score > best_new_score:
                        best_new_score = new_score
                        best_new_val = new_val

                if best_new_val is not None:
                    current_set[i] = best_new_val
                    current_mask = _set_to_mask(current_set)
                    current_score = best_new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
                        best_set = list(current_set)

    if best_set is None:
        best_set = list(range(1, 33))

    return list(best_set)
