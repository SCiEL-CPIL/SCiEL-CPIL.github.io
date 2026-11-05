"""Simulated annealing for ternary covering code K_3(7,1)."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_DEADLINE_MARGIN = 5.0  # seconds to spare before the 150 s kill


def _energy_vectorized(codes, q, n, radius):
    """Return (num_uncovered, covered_mask) for the given codes.
    A word is covered if its Hamming distance to some code is <= radius.
    """
    codes = np.asarray(codes, dtype=np.int32)
    m = len(codes)
    if m == 0:
        return q ** n, np.zeros(q ** n, dtype=bool)

    # Build the full space as a 2D array of shape (Q, n)
    Q = q ** n
    space = np.zeros((Q, n), dtype=np.int32)
    for i in range(n):
        space[:, i] = (np.arange(Q) // (q ** (n - 1 - i))) % q

    # Hamming distances: shape (Q, m)
    diff = space[:, None, :] != codes[None, :, :]  # (Q, m, n)
    dist = diff.sum(axis=2)                        # (Q, m)
    covered_mask = (dist <= radius).any(axis=1)    # (Q,)
    num_uncovered = int((~covered_mask).sum())
    return num_uncovered, covered_mask


def _greedy_init(q, n, radius, rng, max_iter=2000):
    """Greedy: repeatedly add the word that covers the most currently-uncovered words."""
    Q = q ** n
    # Precompute the space
    space = np.zeros((Q, n), dtype=np.int32)
    for i in range(n):
        space[:, i] = (np.arange(Q) // (q ** (n - 1 - i))) % q

    uncovered = np.ones(Q, dtype=bool)
    codes = []
    for _ in range(max_iter):
        if not uncovered.any():
            break
        # For each candidate, count how many uncovered words it covers
        # dist (Q, Q) is too big; do it in chunks over candidates
        best_gain = -1
        best_idx = -1
        chunk = 200
        for start in range(0, Q, chunk):
            end = min(start + chunk, Q)
            cand = space[start:end]  # (chunk, n)
            # distance from each candidate to all uncovered words
            # only compute against uncovered to save time
            unc_idx = np.where(uncovered)[0]
            if len(unc_idx) == 0:
                break
            d = (cand[:, None, :] != space[unc_idx][None, :, :]).sum(axis=2)  # (chunk, len(unc))
            gains = (d <= radius).sum(axis=1)
            local_best = int(gains.max())
            if local_best > best_gain:
                best_gain = local_best
                best_idx = start + int(gains.argmax())
        if best_idx < 0:
            break
        codes.append(space[best_idx].tolist())
        # update uncovered
        d = (space != np.asarray(codes[-1])[None, :]).sum(axis=1)
        uncovered[d > radius] = False
    return codes


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 150.0 - _DEADLINE_MARGIN
    rng = random.Random(12345)

    # Greedy initialization
    codes = _greedy_init(q, n, radius, rng, max_iter=300)
    codes = list(dict.fromkeys(codes))  # dedupe, preserve order

    num_uncovered, covered_mask = _energy_vectorized(codes, q, n, radius)

    if num_uncovered == 0:
        return codes

    # Precompute space
    Q = q ** n
    space = np.zeros((Q, n), dtype=np.int32)
    for i in range(n):
        space[:, i] = (np.arange(Q) // (q ** (n - 1 - i))) % q

    # Simulated annealing
    T = 10.0
    T_min = 0.01
    cooling = 0.995
    best_codes = list(codes)
    best_energy = len(codes) + num_uncovered * 10000
    current_energy = best_energy

    step = 0
    while time.time() < deadline:
        step += 1
        if T < T_min:
            T = 1.0  # restart temperature occasionally
            cooling = 0.995

        # Choose a random move
        move_type = rng.random()
        if move_type < 0.4 and len(codes) > 1:
            # Remove a random code
            idx = rng.randrange(len(codes))
            new_codes = codes[:idx] + codes[idx + 1:]
            new_num_unc, _ = _energy_vectorized(new_codes, q, n, radius)
            new_energy = len(new_codes) + new_num_unc * 10000
            delta = new_energy - current_energy
            if delta <= 0 or rng.random() < np.exp(-delta / T):
                codes = new_codes
                current_energy = new_energy
            T *= cooling
        elif move_type < 0.8:
            # Replace a random code with a random word
            idx = rng.randrange(len(codes))
            new_word = space[rng.randrange(Q)].tolist()
            new_codes = codes[:idx] + [new_word] + codes[idx + 1:]
            # dedupe
            new_codes = list(dict.fromkeys(new_codes))
            new_num_unc, _ = _energy_vectorized(new_codes, q, n, radius)
            new_energy = len(new_codes) + new_num_unc * 10000
            delta = new_energy - current_energy
            if delta <= 0 or rng.random() < np.exp(-delta / T):
                codes = new_codes
                current_energy = new_energy
            T *= cooling
        else:
            # Add a random word
            new_word = space[rng.randrange(Q)].tolist()
            new_codes = codes + [new_word]
            new_codes = list(dict.fromkeys(new_codes))
            new_num_unc, _ = _energy_vectorized(new_codes, q, n, radius)
            new_energy = len(new_codes) + new_num_unc * 10000
            delta = new_energy - current_energy
            if delta <= 0 or rng.random() < np.exp(-delta / T):
                codes = new_codes
                current_energy = new_energy
            T *= cooling

        # Track best
        if current_energy < best_energy:
            best_energy = current_energy
            best_codes = list(codes)

        # Periodic local search: try to remove redundant codes
        if step % 50 == 0:
            i = 0
            while i < len(best_codes):
                trial = best_codes[:i] + best_codes[i + 1:]
                if len(trial) > 0:
                    nu, _ = _energy_vectorized(trial, q, n, radius)
                    if nu == 0:
                        best_codes = trial
                        best_energy = len(trial)
                        codes = list(trial)
                        current_energy = best_energy
                        continue
                i += 1

    # Final: return the best valid solution
    # If best_codes is valid, use it; otherwise fall back to a valid construction
    final_num_unc, _ = _energy_vectorized(best_codes, q, n, radius)
    if final_num_unc == 0:
        return best_codes

    # Fallback: direct sum construction (always valid)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    import itertools
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
# EVOLVE-BLOCK-END
