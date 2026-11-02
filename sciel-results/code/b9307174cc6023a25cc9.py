"""Kissing number in dimension 11: maximize count of integer vectors with min pairwise dist >= max norm."""
import time
import numpy as np

D = 11
DEADLINE = 130.0  # safety margin before 150s kill


def _build_candidates():
    """Build candidate integer vectors of weight 2 (two nonzero entries, each ±1).
    All have norm^2 = 2. Constraint for equal-norm vectors: dot(v_i,v_j) <= 1."""
    cands = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    cands.append(v)
    return np.array(cands, dtype=np.int64)  # shape (N, D), N = 4*C(11,2)=220


def construct():
    t0 = time.time()
    cands = _build_candidates()
    N = len(cands)
    # Precompute dot product matrix (N x N). dot <= 1 means compatible.
    dots = cands @ cands.T  # integer
    compat = (dots <= 1)  # boolean N x N
    np.fill_diagonal(compat, True)

    # Greedy: repeatedly pick candidate with max number of compatible remaining.
    remaining = set(range(N))
    chosen = []
    # degree = number of compatible in remaining
    # Use a simple greedy with recompute
    while remaining:
        # pick candidate maximizing compat count within remaining
        best = -1
        best_deg = -1
        for c in remaining:
            deg = sum(1 for r in remaining if compat[c, r])
            if deg > best_deg:
                best_deg = deg
                best = c
        chosen.append(best)
        remaining.discard(best)
        # remove incompatible
        to_remove = [r for r in remaining if not compat[best, r]]
        for r in to_remove:
            remaining.discard(r)
        if time.time() - t0 > DEADLINE:
            break

    chosen_arr = np.array(chosen, dtype=np.int64)
    vecs = cands[chosen_arr]

    # Local search: try to add candidates not in set, and swap to improve.
    # Represent chosen as a set for fast lookup.
    chosen_set = set(int(x) for x in chosen_arr)

    def is_compatible_with_all(c):
        for r in chosen_set:
            if not compat[c, r]:
                return False
        return True

    # Try to add any candidate compatible with all chosen.
    added = True
    while added and (time.time() - t0) < DEADLINE:
        added = False
        for c in range(N):
            if c in chosen_set:
                continue
            if is_compatible_with_all(c):
                chosen_set.add(c)
                added = True
                break

    # Local search: swap a chosen candidate with a non-chosen one to allow more additions.
    # Simulated-annealing style: try removing one chosen, then greedily re-add best.
    improved = True
    while improved and (time.time() - t0) < DEADLINE:
        improved = False
        base_size = len(chosen_set)
        best_size = base_size
        best_state = None
        # Try removing each chosen candidate
        for rem in list(chosen_set):
            trial = set(chosen_set)
            trial.discard(rem)
            # greedy fill from trial
            added_any = True
            while added_any and (time.time() - t0) < DEADLINE:
                added_any = False
                for c in range(N):
                    if c in trial:
                        continue
                    ok = True
                    for r in trial:
                        if not compat[c, r]:
                            ok = False
                            break
                    if ok:
                        trial.add(c)
                        added_any = True
                        break
            if len(trial) > best_size:
                best_size = len(trial)
                best_state = trial
        if best_state is not None and best_size > base_size:
            chosen_set = best_state
            improved = True

    result = cands[list(chosen_set)]
    return result


if __name__ == "__main__":
    arr = construct()
    print("count:", len(arr))
