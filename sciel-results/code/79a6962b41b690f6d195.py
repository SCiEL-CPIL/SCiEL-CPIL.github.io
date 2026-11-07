# EVOLVE-BLOCK-START
import time
from functools import lru_cache

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _build_candidates():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    # All cliques using Bron-Kerbosch style with pivot for efficiency
    cliques = []
    def gen_cliques(cand, cur):
        if cur:
            cliques.append(frozenset(cur))
        for i in range(len(cand)):
            v = cand[i]
            new_cand = [x for x in cand[i+1:] if v in adj[x]]
            gen_cliques(new_cand, cur + [v])
    gen_cliques(list(range(n)), [])

    # All independent sets (cliques of complement)
    non_adj = [set() for _ in range(n)]
    for i in range(n):
        non_adj[i] = set(range(n)) - adj[i] - {i}
    indep_sets = []
    def gen_is(cand, cur):
        if cur:
            indep_sets.append(frozenset(cur))
        for i in range(len(cand)):
            v = cand[i]
            new_cand = [x for x in cand[i+1:] if v in non_adj[x]]
            gen_is(new_cand, cur + [v])
    gen_is(list(range(n)), [])

    # Union of all cliques and all independent sets
    all_cands = list(set(cliques) | set(indep_sets))
    # Filter: keep only those with at least 1 vertex
    all_cands = [c for c in all_cands if len(c) >= 1]
    # Remove dominated candidates: if candidate A is a subset of candidate B, 
    # then A is never needed (B covers at least as much with same cost)
    all_cands.sort(key=len, reverse=True)
    non_dominated = []
    for c in all_cands:
        dominated = False
        for d in non_dominated:
            if c <= d:  # c is subset of d
                dominated = True
                break
        if not dominated:
            non_dominated.append(c)
    all_cands = non_dominated
    all_cands.sort(key=len, reverse=True)
    return all_cands

def _solve(candidates, n, deadline):
    full_mask = (1 << n) - 1
    # candidate masks
    cand_masks = [0] * len(candidates)
    for i, c in enumerate(candidates):
        m = 0
        for v in c:
            m |= (1 << v)
        cand_masks[i] = m

    timed_out = [False]
    
    # Precompute for each mask, which candidate indices are subsets
    num_cands = len(cand_masks)
    subset_cands = [None] * (1 << n)
    for mask in range(1 << n):
        valid = []
        for i, cm in enumerate(cand_masks):
            if (cm & mask) == cm:
                valid.append(i)
        subset_cands[mask] = valid

    # Memoized lower bound: minimum parts to cover rem_mask with any candidates
    # (same as min_parts but we only need a lower bound for pruning).
    # We use a greedy-style lower bound: the size of a maximal set of
    # vertices in rem_mask such that no single candidate covers two of them.
    # This is a valid lower bound on the number of parts needed.
    # For speed, we compute a simple lower bound: ceil(rem_mask.bit_count() / max_cand_size_in_rem)
    # where max_cand_size_in_rem is the largest candidate subset of rem_mask.
    # This is cheap and often tight enough for pruning.
    max_cand_size = [0] * (1 << n)
    for mask in range(1 << n):
        ms = 0
        for i in subset_cands[mask]:
            sz = cand_masks[i].bit_count()
            if sz > ms:
                ms = sz
        max_cand_size[mask] = ms

    def lower_bound(rem_mask):
        if rem_mask == 0:
            return 0
        mcs = max_cand_size[rem_mask]
        if mcs == 0:
            return 100
        return (rem_mask.bit_count() + mcs - 1) // mcs

    @lru_cache(maxsize=None)
    def min_parts(rem_mask):
        """Returns minimum parts needed to cover rem_mask."""
        if rem_mask == 0:
            return 0
        if timed_out[0]:
            return 100
        if time.time() > deadline:
            timed_out[0] = True
            return 100

        lb = lower_bound(rem_mask)
        best_add = 100
        # Try larger candidates first for better pruning
        for i in subset_cands[rem_mask]:
            cm = cand_masks[i]
            r = rem_mask ^ cm
            # Prune: even optimally covering r, total >= 1 + lower_bound(r)
            if 1 + lower_bound(r) >= best_add:
                continue
            add = 1 + min_parts(r)
            if add < best_add:
                best_add = add
                if best_add == lb:
                    break
        return best_add

    result = min_parts(full_mask)

    if timed_out[0]:
        return None

    # Now reconstruct: find the actual partition
    def reconstruct(rem_mask):
        if rem_mask == 0:
            return []
        target = min_parts(rem_mask)
        for i in subset_cands[rem_mask]:
            cm = cand_masks[i]
            r = rem_mask ^ cm
            if 1 + min_parts(r) == target:
                rest = reconstruct(r)
                part = [v for v in range(n) if (cm >> v) & 1]
                part.sort()
                return [part] + rest
        return []

    parts = reconstruct(full_mask)
    return parts

def construct():
    t0 = time.time()
    deadline = t0 + 80.0  # safety margin before 90s kill

    candidates = _build_candidates()
    parts = _solve(candidates, P['n'], deadline)

    if parts is None or not parts:
        # Fallback: greedy
        parts = _greedy_fallback()

    # If we have time, verify the result is valid; if not, fall back.
    # The lower-bound pruning only prunes branches that cannot improve, so
    # the result is still optimal. But let's double-check validity.
    if time.time() < deadline - 5:
        n = P['n']
        es = set()
        for u, v in P['edges']:
            es.add((u, v))
            es.add((v, u))
        valid = True
        all_used = []
        for part in parts:
            all_used.extend(part)
            adj_pairs = 0
            total_pairs = len(part) * (len(part) - 1) // 2
            for i in range(len(part)):
                for j in range(i+1, len(part)):
                    if (part[i], part[j]) in es:
                        adj_pairs += 1
            if not (adj_pairs == 0 or adj_pairs == total_pairs):
                valid = False
                break
        if not valid or len(all_used) != n or len(set(all_used)) != n:
            parts = _greedy_fallback()

    # Convert to list of lists of Python ints
    result = [list(sorted([int(v) for v in part])) for part in parts]
    return result

def _greedy_fallback():
    n = P['n']
    es = set()
    for u, v in P['edges']:
        es.add((u, v))
        es.add((v, u))
    parts = []
    for v in range(n):
        for part in parts:
            trial = part + [v]
            hits = [(u, w) in es for i, u in enumerate(trial) for w in trial[i+1:]]
            if all(hits) or not any(hits):
                part.append(v)
                break
        else:
            parts.append([v])
    return [[int(v) for v in p] for p in parts]

# EVOLVE-BLOCK-END
