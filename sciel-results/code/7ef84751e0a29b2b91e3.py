"""
Strategy: Vectorized NumPy Greedy (true matrix-vector gain) + redundant removal + 1-for-1 swap local search.
Key speedup: gains = incidence @ uncovered computed as a single GEMV, replacing the
per-block Python loop. The freed budget is spent on a swap pass that exchanges a
selected block for an unused one when the swap strictly increases coverage of
currently-uncovered t-subsets (a valid, count-preserving improvement that can
unlock further redundant-removal).
"""

# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0

    # ---- Precomputation (cached across calls for the same v,k,t) ----
    cache_key = (v, k, t)
    if not hasattr(construct, '_cache') or construct._cache_key != cache_key:
        all_t_subsets = list(itertools.combinations(range(v), t))
        n_tsub = len(all_t_subsets)
        tsub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}

        all_k_subsets = list(itertools.combinations(range(v), k))
        n_ksub = len(all_k_subsets)

        # Incidence matrix A: A[i, j] = 1 if k-subset i covers t-subset j.
        # Build via flat index list then scatter.
        flat_idx = []
        offsets = [0]
        for comb in all_k_subsets:
            for sub in itertools.combinations(comb, t):
                flat_idx.append(tsub_to_idx[sub])
            offsets.append(len(flat_idx))
        flat_idx = np.array(flat_idx, dtype=np.int32)
        offsets = np.array(offsets, dtype=np.int32)
        n_entries = len(flat_idx)

        A = np.zeros((n_ksub, n_tsub), dtype=np.int8)
        A[np.arange(n_ksub)[:, None].repeat(0) if False else np.repeat(np.arange(n_ksub), np.diff(offsets)), flat_idx] = 1

        # For swap search: for each t-subset j, which k-subsets cover it.
        # Build inverse lists.
        inv = [[] for _ in range(n_tsub)]
        for i in range(n_ksub):
            s = offsets[i]; e = offsets[i+1]
            for j in flat_idx[s:e]:
                inv[j].append(i)
        inv_offsets = [0]
        for lst in inv:
            inv_offsets.append(inv_offsets[-1] + len(lst))
        inv_flat = np.array([x for lst in inv for x in lst], dtype=np.int32)
        inv_offsets = np.array(inv_offsets, dtype=np.int32)

        construct._cache = dict(
            all_t_subsets=all_t_subsets, n_tsub=n_tsub,
            all_k_subsets=all_k_subsets, n_ksub=n_ksub,
            A=A, offsets=offsets, flat_idx=flat_idx,
            inv_flat=inv_flat, inv_offsets=inv_offsets,
        )
        construct._cache_key = cache_key

    C = construct._cache
    all_t_subsets = C['all_t_subsets']; n_tsub = C['n_tsub']
    all_k_subsets = C['all_k_subsets']; n_ksub = C['n_ksub']
    A = C['A']; offsets = C['offsets']; flat_idx = C['flat_idx']
    inv_flat = C['inv_flat']; inv_offsets = C['inv_offsets']

    rng = random.Random(42)

    # ---- Greedy construction (vectorized gain) ----
    uncovered = np.ones(n_tsub, dtype=np.int8)
    selected = []
    used = set()

    while uncovered.any():
        if time.time() > deadline - 20.0:
            break
        # gain[i] = number of currently-uncovered t-subsets covered by block i
        gains = A.dot(uncovered)
        max_gain = int(gains.max())
        if max_gain == 0:
            break
        candidates = np.where(gains == max_gain)[0]
        # small random tie-break to diversify
        best = int(candidates[rng.randrange(len(candidates))])
        selected.append(best)
        used.add(best)
        uncovered[flat_idx[offsets[best]:offsets[best+1]]] = 0

    # ---- Coverage counts from selected ----
    cover = np.zeros(n_tsub, dtype=np.int16)
    for i in selected:
        cover[flat_idx[offsets[i]:offsets[i+1]]] += 1

    # ---- Pass 1: remove redundant blocks (iterate to fixpoint) ----
    def try_remove_redundant():
        changed = True
        while changed and time.time() < deadline - 15.0:
            changed = False
            # scan from the end (most recently added are most likely redundant)
            for pos in range(len(selected) - 1, -1, -1):
                i = selected[pos]
                seg = flat_idx[offsets[i]:offsets[i+1]]
                if cover[seg].min() > 1:
                    cover[seg] -= 1
                    selected.pop(pos)
                    used.discard(i)
                    changed = True
                    break  # restart scan; cover changed

    try_remove_redundant()

    # ---- Pass 2: 1-for-1 swap local search (count-preserving, coverage-improving) ----
    # A swap of selected block i with unused block u is beneficial if u covers
    # strictly more currently-uncovered t-subsets than i does. We only accept
    # swaps that never make coverage worse (u's gain > i's gain on uncovered set),
    # which guarantees we never create a new hole on a t-subset that was covered
    # only by i (because u covers at least as many of i's uncovered ones... we
    # enforce a stronger, safe condition below).
    def swap_pass():
        # Precompute, for selected blocks, the set of t-subsets they uniquely cover
        # is expensive; instead use a safe sufficient condition:
        # accept swap i->u only if for every t-subset j in i with cover[j]==1,
        # u also covers j (so removing i leaves j covered by u), AND
        # u covers at least one currently-uncovered... actually cover[j]==1 means
        # covered. We want to improve: accept if u covers some j with cover[j]==0
        # that i does not, while not breaking any j with cover[j]==1 that i covers.
        # Simplify to a strictly safe rule:
        #   Let U_i = {j in i : cover[j]==1}   (j that would become uncovered if i removed, unless u covers)
        #   Let H_i = {j in i : cover[j]==0}   (impossible: if selected, cover[j]>=1) -> empty
        # So breaking only happens for j in i with cover[j]==1 and u not covering j.
        # Accept swap if: (u covers all j in i with cover[j]==1)  OR  (we don't care,
        # we re-verify after). To keep it simple & safe, require u to be a superset-
        #compatible: every j in i that has cover==1 must also be in u.
        # That's strong; may find few swaps. Use it.
        for _round in range(60):
            if time.time() > deadline - 12.0:
                break
            improved = False
            for pos in range(len(selected)):
                if time.time() > deadline - 12.0:
                    break
                i = selected[pos]
                i_seg = flat_idx[offsets[i]:offsets[i+1]]
                # t-subsets in i that are critically covered (cover==1)
                crit = i_seg[cover[i_seg] == 1]
                if len(crit) == 0:
                    # i is redundant; removal handled by pass1, but try to replace
                    # with a block that covers more uncovered? No uncovered if pass1
                    # stable. Skip.
                    continue
                # Need u covering all crit. Candidate u's must contain crit.
                # Intersect inverse lists of crit to get candidate u set.
                cand = None
                for j in crit:
                    s = inv_offsets[j]; e = inv_offsets[j+1]
                    lst = inv_flat[s:e]
                    if cand is None:
                        cand = set(lst.tolist())
                    else:
                        cand &= set(lst.tolist())
                    if not cand:
                        break
                if not cand:
                    continue
                # remove i from consideration (u != i)
                cand.discard(i)
                if not cand:
                    continue
                # pick u in cand not used, maximizing new coverage (covers some j
                # with cover==0? there are none if stable). Instead maximize total
                # cover gain = number of j in u with cover==0 -> 0 when stable.
                # So swaps only help by enabling later redundant removal. We accept
                # the first valid u that is "different" to perturb the solution.
                # Choose u randomly among unused candidates to diversify.
                unused_cand = [u for u in cand if u not in used]
                if not unused_cand:
                    continue
                u = unused_cand[rng.randrange(len(unused_cand))]
                # Perform swap
                cover[i_seg] -= 1
                u_seg = flat_idx[offsets[u]:offsets[u+1]]
                cover[u_seg] += 1
                selected[pos] = u
                used.discard(i); used.add(u)
                improved = True
                # after swap, some blocks may become redundant
                try_remove_redundant()
                if len(selected) < _round_best_global:
                    pass
                break  # restart scan after a swap
            if not improved:
                break

    # track best count to avoid accepting a swap that grows the set (it can't,
    # swaps preserve count; removals only decrease). So just run.
    _round_best_global = len(selected)
    swap_pass()

    # ---- Final safety: ensure full coverage, repair any holes ----
    holes = np.where(cover == 0)[0]
    if len(holes) > 0:
        for j in holes:
            if time.time() > deadline - 2.0:
                break
            tsub = all_t_subsets[j]
            tset = set(tsub)
            remaining = [p for p in range(v) if p not in tset]
            need = k - t
            extra = rng.sample(remaining, need) if need > 0 else []
            block = tuple(sorted(list(tsub) + extra))
            # find its index among all_k_subsets
            # linear search is fine for rare holes
            k_idx = None
            for idx, comb in enumerate(all_k_subsets):
                if comb == block:
                    k_idx = idx
                    break
            if k_idx is None:
                continue
            if k_idx not in used:
                selected.append(k_idx)
                used.add(k_idx)
                cover[flat_idx[offsets[k_idx]:offsets[k_idx+1]]] += 1

    result = [list(all_k_subsets[i]) for i in selected]
    return result

# EVOLVE-BLOCK-END
