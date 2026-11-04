"""Ternary covering code K_q(n, radius): greedy set cover with vectorized scoring."""
import time
import itertools
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before the 150s kill

    # ---- Build the full space (N = q^n) and the radius-1 "ball" offsets ----
    offsets = [(0,) * n]
    for pos in range(n):
        for d in range(1, q):
            e = [0] * n
            e[pos] = d
            offsets.append(tuple(e))
    n_off = len(offsets)

    # ---- Precompute coverage index arrays for every codeword (memory-bounded) ----
    # cover_idx[c] = flat indices of all words within Hamming distance <= radius of c
    max_mem_words = 8_000_000
    if n_off * q ** n <= max_mem_words:
        # offsets are unique tuples; build a shared lookup table
        off_arr = np.array(offsets, dtype=np.int64)
        off_to_idx = {tuple(row): i for i, row in enumerate(off_arr)}
        cover_idx = []
        for c in itertools.product(range(q), repeat=n):
            c_arr = np.array(c, dtype=np.int64)
            idx = np.array([off_to_idx[tuple(r)] for r in (c_arr + off_arr) % q], dtype=np.int64)
            cover_idx.append(idx)
    else:
        cover_idx = None  # fall back to per-candidate distance computation

    # ---- Greedy set cover over the whole space ----
    chosen = []
    chosen_set = set()
    if cover_idx is not None:
        cover = np.zeros(q ** n, dtype=np.int64)
        uncovered = q ** n
    else:
        cover = None
        uncovered = q ** n

    while True:
        if time.time() >= deadline:
            break
        if uncovered == 0:
            break

        # Build the candidate list, skipping words already chosen.
        if len(chosen_set) < q ** n:
            candidates = [c for c in itertools.product(range(q), repeat=n) if c not in chosen_set]
        else:
            candidates = []
        if not candidates:
            break

        if cover_idx is not None:
            # Vectorized: count newly-covered words per candidate.
            cand_idx = np.array(cover_idx, dtype=np.int64)
            cand_cover = cover[cand_idx]
            new_cover = (cand_cover == 0).sum(axis=1)
            best = int(np.argmax(new_cover))
            if new_cover[best] <= 0:
                # No candidate can reduce the uncovered count -> space not fully coverable
                break
            chosen = [c for c in candidates if c not in chosen_set]
            c = candidates[best]
            chosen_set.add(c)
            chosen.append(list(c))
            idx = cover_idx[best]
            cover[idx] += 1
            uncovered -= int(new_cover[best])
        else:
            # Fallback (large space): per-candidate distance computation.
            cands = np.array(candidates, dtype=np.int64)
            best_score = -1
            best_row = 0
            for r in range(cands.shape[0]):
                dist = np.abs(cands - cands[r]).sum(axis=1)
                score = int((dist <= radius).sum())
                if score > best_score:
                    best_score = score
                    best_row = r
            c = tuple(cands[best_row].tolist())
            chosen_set.add(c)
            chosen.append(list(c))
            dist = np.abs(cands - np.array(c, dtype=np.int64)).sum(axis=1)
            uncovered = int((dist > radius).sum())

    return chosen
