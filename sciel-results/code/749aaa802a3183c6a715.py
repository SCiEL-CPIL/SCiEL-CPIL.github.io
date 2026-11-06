import random
import time
import numpy as np

def construct():
    N = 1000
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin

    rng = random.Random(42)

    # ---------- fast helpers ----------
    def add_x(mask, cnt, x):
        cnt[2 * x] += 1
        idx = np.nonzero(mask)[0]
        if idx.size:
            cnt[idx + x] += 1
        mask[x] = True

    def remove_x(mask, cnt, x):
        cnt[2 * x] -= 1
        idx = np.nonzero(mask)[0]
        if idx.size:
            other = idx[idx != x]
            if other.size:
                cnt[other + x] -= 1
        mask[x] = False

    def can_add(mask, cnt, x):
        if cnt[2 * x] >= 2:
            return False
        idx = np.nonzero(mask)[0]
        if idx.size:
            if np.any(cnt[idx + x] >= 2):
                return False
        return True

    # ---------- randomized greedy ----------
    def greedy_randomized(seed_skip=0.0, seed_order=None):
        mask = np.zeros(N + 1, dtype=bool)
        cnt = np.zeros(2 * N + 1, dtype=np.int32)
        order = list(range(1, N + 1))
        if seed_order is not None:
            rng.shuffle(order)
        for x in order:
            if mask[x]:
                continue
            if rng.random() < seed_skip:
                continue
            if can_add(mask, cnt, x):
                add_x(mask, cnt, x)
        return mask, cnt

    # ---------- local search ----------
    def local_search(mask, cnt, time_limit, lambda_pen=6.0):
        cur_size = int(mask.sum())
        best_size = cur_size
        best_mask = mask.copy()

        T = 1.5
        T_min = 0.05
        cooling = 0.99998

        def viol():
            return int(np.maximum(0, cnt - 2).sum())

        cur_v = viol()

        while time.time() < time_limit:
            r = rng.random()
            if r < 0.5:
                # INSERT
                feas = []
                for x in range(1, N + 1):
                    if not mask[x] and can_add(mask, cnt, x):
                        feas.append(x)
                if feas:
                    x = feas[rng.randrange(len(feas))]
                    add_x(mask, cnt, x)
                    cur_size += 1
                    cur_v = viol()
                    if cur_v == 0 and cur_size > best_size:
                        best_size = cur_size
                        best_mask = mask.copy()
            elif r < 0.85:
                # SWAP
                in_idx = np.nonzero(mask)[0]
                if not in_idx.size:
                    continue
                x = in_idx[rng.randrange(in_idx.size)]
                out_idx = [i for i in range(1, N + 1) if not mask[i]]
                if not out_idx:
                    continue
                y = out_idx[rng.randrange(len(out_idx))]
                remove_x(mask, cnt, x)
                add_x(mask, cnt, y)
                new_v = viol()
                dE = lambda_pen * (new_v - cur_v)
                if dE < 0 or rng.random() < np.exp(-dE / max(T, 1e-9)):
                    cur_v = new_v
                    if cur_v == 0 and cur_size > best_size:
                        best_size = cur_size
                        best_mask = mask.copy()
                else:
                    remove_x(mask, cnt, y)
                    add_x(mask, cnt, x)
                    cur_v = viol()
            else:
                # REMOVE (to free capacity)
                in_idx = np.nonzero(mask)[0]
                if not in_idx.size:
                    continue
                x = in_idx[rng.randrange(in_idx.size)]
                remove_x(mask, cnt, x)
                new_v = viol()
                dE = -1.0 + lambda_pen * (new_v - cur_v)
                if dE < 0 or rng.random() < np.exp(-dE / max(T, 1e-9)):
                    cur_size -= 1
                    cur_v = new_v
                else:
                    add_x(mask, cnt, x)
                    cur_size += 1
                    cur_v = viol()
            T = max(T * cooling, T_min)

        return best_mask, best_size

    # ---------- Phase 1: diverse greedy seeds ----------
    best_mask = None
    best_size = 0

    # Try several skip probabilities and orderings
    skip_probs = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
    for sp in skip_probs:
        if time.time() > deadline - 80:
            break
        for _ in range(3):
            if time.time() > deadline - 80:
                break
            mask, cnt = greedy_randomized(seed_skip=sp)
            s = int(mask.sum())
            if s > best_size:
                best_size = s
                best_mask = mask.copy()
                best_cnt = cnt.copy()

    # Also try a few random-order greedy
    for _ in range(5):
        if time.time() > deadline - 80:
            break
        mask, cnt = greedy_randomized(seed_skip=0.1, seed_order=True)
        s = int(mask.sum())
        if s > best_size:
            best_size = s
            best_mask = mask.copy()
            best_cnt = cnt.copy()

    if best_mask is None:
        best_mask, best_cnt = greedy_randomized(seed_skip=0.1)
        best_size = int(best_mask.sum())

    # ---------- Phase 2: local search ----------
    if time.time() < deadline - 10:
        best_mask, best_size = local_search(
            best_mask.copy(), best_cnt.copy(),
            deadline - 5.0, lambda_pen=6.0
        )

    # ---------- Phase 3: final polish (deterministic insert pass) ----------
    if time.time() < deadline - 2:
        cnt = np.zeros(2 * N + 1, dtype=np.int32)
        idx = np.nonzero(best_mask)[0]
        for a in idx:
            for b in idx[idx >= a]:
                cnt[a + b] += 1
        # Try to add any remaining feasible elements
        for x in range(1, N + 1):
            if best_mask[x]:
                continue
            if can_add(best_mask, cnt, x):
                add_x(best_mask, cnt, x)
        best_size = int(best_mask.sum())

    result = [int(i) for i in np.nonzero(best_mask)[0]]
    return result
