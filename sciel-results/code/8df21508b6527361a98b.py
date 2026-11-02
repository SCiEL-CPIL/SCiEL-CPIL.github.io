"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before the 150 s wall-clock kill

    # ---- Candidate pool: all integer vectors with squared norm 4 ----
    #  +/-2 e_i          : D * 2
    #  +/-e_i +/-e_j     : C(D,2) * 4
    #  +/-e_i +/-e_j +/-e_k : C(D,3) * 8
    def _comb2(n):
        return n * (n - 1) // 2
    def _comb3(n):
        return n * (n - 1) * (n - 2) // 6

    n2 = D * 2
    n12 = _comb2(D) * 4
    n123 = _comb3(D) * 8
    N = n2 + n12 + n123

    # ---- Build candidate value vectors directly (D,)-shaped, values in {-2,-1,0,1,2} ----
    cand_vals = []
    # type 0: +/-2 e_i
    for i in range(D):
        v = np.zeros(D, dtype=np.int8); v[i] = -2; cand_vals.append(v)
        v = np.zeros(D, dtype=np.int8); v[i] = 2;  cand_vals.append(v)
    # type 1: +/-e_i +/-e_j
    for i in range(D):
        for j in range(i + 1, D):
            for si in (-1, 1):
                for sj in (-1, 1):
                    v = np.zeros(D, dtype=np.int8); v[i] = si; v[j] = sj; cand_vals.append(v)
    # type 2: +/-e_i +/-e_j +/-e_k
    for i in range(D):
        for j in range(i + 1, D):
            for k in range(j + 1, D):
                for si in (-1, 1):
                    for sj in (-1, 1):
                        for sk in (-1, 1):
                            v = np.zeros(D, dtype=np.int8); v[i] = si; v[j] = sj; v[k] = sk
                            cand_vals.append(v)
    assert len(cand_vals) == N
    vals = np.stack(cand_vals, axis=0).astype(np.int32)  # (N, D)

    # ---- Conflict: squared distance < 4  <=>  dot > 1 (all have norm^2 = 4) ----
    dot = vals @ vals.T
    conflict = dot > 1
    np.fill_diagonal(conflict, False)
    degree = conflict.sum(axis=1).astype(np.int32)

    # ---- Conflict masks as 64-bit chunks ----
    nch = (N + 63) // 64
    masks = np.zeros((N, nch), dtype=np.uint64)
    for ci in range(N):
        row = conflict[ci]
        for b in range(nch):
            start = b * 64
            end = min(start + 64, N)
            seg = row[start:end]
            m = 0
            for bit in range(end - start):
                if seg[bit]:
                    m |= (1 << bit)
            masks[ci, b] = m

    def popcount(x):
        return bin(int(x)).count("1")

    def conflict_count_with(selmask, ci):
        c = 0
        for b in range(nch):
            c += popcount(masks[ci, b] & selmask[b])
        return c

    def update_counts(selmask, ci, sign):
        # add/subtract 1 to cnt of selected neighbours of ci
        for b in range(nch):
            x = int(masks[ci, b]) & int(selmask[b])
            while x:
                lb = x & (-x)
                bitpos = (lb.bit_length() - 1)
                nb = (b << 6) | bitpos
                if nb < N:
                    cnt[nb] += sign
                x ^= lb

    # ---- Greedy min-degree seed ----
    def greedy_seed():
        selected = np.zeros(N, dtype=bool)
        cur_deg = degree.copy()
        for _ in range(N):
            if time.time() > deadline:
                break
            cand = np.where(~selected)[0]
            if len(cand) == 0:
                break
            best = cand[np.argmin(cur_deg[cand])]
            selected[best] = True
            nb = np.where(conflict[best] & ~selected)[0]
            if len(nb):
                cur_deg[nb] -= 1
        return selected

    # ---- Polish: repeatedly remove the min-conflict selected vertex until zero conflict ----
    def polish(sel):
        selmask = np.zeros(nch, dtype=np.uint64)
        for ci in range(N):
            if sel[ci]:
                selmask[ci >> 6] |= (1 << (ci & 63))
        cnt = np.zeros(N, dtype=np.int32)
        for ci in range(N):
            if sel[ci]:
                cnt[ci] = conflict_count_with(selmask, ci)
        while True:
            if time.time() > deadline:
                break
            active = np.where(sel)[0]
            if len(active) == 0:
                break
            mc = int(cnt[active].min())
            if mc == 0:
                break
            to_rm = active[cnt[active] == mc][0]
            sel[to_rm] = False
            selmask[to_rm >> 6] &= ~(1 << (to_rm & 63))
            update_counts(selmask, to_rm, -1)
            cnt[to_rm] = 0
        return sel

    # ---- Simulated annealing over the selection set ----
    def sa_from(start_sel, iters, t0, t1):
        sel = start_sel.copy()
        selmask = np.zeros(nch, dtype=np.uint64)
        cnt = np.zeros(N, dtype=np.int32)
        for ci in range(N):
            if sel[ci]:
                selmask[ci >> 6] |= (1 << (ci & 63))
        for ci in range(N):
            if sel[ci]:
                cnt[ci] = conflict_count_with(selmask, ci)
        size = int(sel.sum())
        best_size = size
        best_sel = sel.copy()
        rng = np.random.default_rng()
        for it in range(iters):
            if time.time() > deadline:
                break
            frac = it / max(1, iters)
            temp = t0 * (t1 / t0) ** frac
            if rng.random() < 0.5:
                # ---- add move ----
                un = np.where(~sel)[0]
                if len(un) == 0:
                    continue
                v = int(un[rng.integers(len(un))])
                c = conflict_count_with(selmask, v)
                if c == 0 or (temp > 0 and rng.random() < np.exp(-c / temp)):
                    sel[v] = True
                    selmask[v >> 6] |= (1 << (v & 63))
                    cnt[v] = c
                    update_counts(selmask, v, +1)
                    size += 1
                    if size > best_size:
                        best_size = size
                        best_sel = sel.copy()
            else:
                # ---- swap move ----
                ssel = np.where(sel)[0]
                un = np.where(~sel)[0]
                if len(ssel) == 0 or len(un) == 0:
                    continue
                v_rm = int(ssel[rng.integers(len(ssel))])
                v_add = int(un[rng.integers(len(un))])
                if v_rm == v_add:
                    continue
                c_add = conflict_count_with(selmask, v_add)
                c_rm = int(cnt[v_rm])
                conflict_rm_add = bool(conflict[v_rm, v_add])
                c_add_excl = c_add - (1 if conflict_rm_add else 0)
                delta_conf = c_add_excl - c_rm
                if delta_conf < 0 or (temp > 0 and rng.random() < np.exp(delta_conf / temp)):
                    # remove v_rm
                    sel[v_rm] = False
                    selmask[v_rm >> 6] &= ~(1 << (v_rm & 63))
                    update_counts(selmask, v_rm, -1)
                    cnt[v_rm] = 0
                    # add v_add
                    sel[v_add] = True
                    selmask[v_add >> 6] |= (1 << (v_add & 63))
                    c = conflict_count_with(selmask, v_add)
                    cnt[v_add] = c
                    update_counts(selmask, v_add, +1)
                    if size > best_size:
                        best_size = size
                        best_sel = sel.copy()
        return best_sel, best_size

    # ---- Main: multi-restart SA + polish ----
    best_sel = greedy_seed()
    best_sel = polish(best_sel)
    best_size = int(best_sel.sum())

    restart = 0
    while time.time() < deadline - 3.0:
        restart += 1
        start = best_sel.copy()
        size0 = int(start.sum())
        if size0:
            ssel = np.where(start)[0]
            n_rm = min(3, size0)
            if n_rm:
                for v in ssel[np.random.default_rng(restart).choice(size0, size=n_rm, replace=False)]:
                    start[v] = False
        cand_sel, _ = sa_from(start, 4000, 3.0, 0.01)
        cand_sel = polish(cand_sel)
        cand_size = int(cand_sel.sum())
        if cand_size > best_size:
            best_size = cand_size
            best_sel = cand_sel

    # ---- Convert best_sel to integer vectors ----
    out = []
    for ci in range(N):
        if best_sel[ci]:
            out.append(vals[ci].astype(np.int64))
    return np.array(out, dtype=np.int64)

# EVOLVE-BLOCK-END
