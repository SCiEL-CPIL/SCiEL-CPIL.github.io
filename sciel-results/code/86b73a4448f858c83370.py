"""Rewritten cochromatic partition solver; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}


def construct():
    n = P['n']
    deadline = time.time() + 52.0  # safety margin before 60s limit

    # Build adjacency bitmask
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    full_mask = (1 << n) - 1

    # Precompute which subsets are cliques / independent sets
    is_clique = [False] * (1 << n)
    is_independent = [False] * (1 << n)
    is_valid = [False] * (1 << n)
    is_clique[0] = True
    is_independent[0] = True
    is_valid[0] = True

    for mask in range(1, 1 << n):
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        if is_clique[rest] and (rest & ~adj[v]) == 0:
            is_clique[mask] = True
        if is_independent[rest] and (rest & adj[v]) == 0:
            is_independent[mask] = True
        is_valid[mask] = is_clique[mask] or is_independent[mask]

    # ---- Lower bound: greedy clique cover of the induced subgraph on mask ----
    # (clique cover number is an admissible lower bound on cochromatic partition size)
    def clique_cover_lb(mask):
        count = 0
        m = mask
        while m:
            # greedy maximal clique starting from lowest vertex
            v = (m & (-m)).bit_length() - 1
            c = 1 << v
            cands = m & adj[v]
            while cands:
                u = (cands & (-cands)).bit_length() - 1
                c |= 1 << u
                cands &= adj[u]
            count += 1
            m ^= c
        return count

    # ---- Enumerate maximal valid subsets (maximal cliques / maximal indep sets) ----
    # of the induced subgraph on 'mask', returning a list of their bitmasks.
    def maximal_valid_subsets(mask):
        res = []
        seen = set()

        # maximal cliques (Bron–Kerbosch)
        def bronk(R, Pset, Xset):
            if Pset == 0 and Xset == 0:
                if R and R not in seen:
                    seen.add(R)
                    res.append(R)
                return
            # pivot = vertex in (Pset | Xset) with max degree in Pset
            piv_mask = Pset | Xset
            if piv_mask:
                pv = (piv_mask & (-piv_mask)).bit_length() - 1
                cand = Pset & adj[pv]
            else:
                cand = Pset
            while cand:
                vb = cand & (-cand)
                v = vb.bit_length() - 1
                bronk(R | vb, Pset & adj[v], Xset & adj[v])
                Pset ^= vb
                Xset |= vb
                cand ^= vb

        bronk(0, mask, 0)

        # maximal independent sets = maximal cliques of the complement graph
        for v in range(n):
            if (mask >> v) & 1:
                adj[v] = full_mask ^ adj[v]  # complement (no self-loop)
        # temporarily flip adj for the complement
        comp_adj = adj[:]
        for v in range(n):
            if (mask >> v) & 1:
                adj[v] = full_mask ^ adj[v]  # restore original
        # run bronk with comp_adj
        def bronk_comp(R, Pset, Xset):
            if Pset == 0 and Xset == 0:
                if R and R not in seen:
                    seen.add(R)
                    res.append(R)
                return
            piv_mask = Pset | Xset
            if piv_mask:
                pv = (piv_mask & (-piv_mask)).bit_length() - 1
                cand = Pset & comp_adj[pv]
            else:
                cand = Pset
            while cand:
                vb = cand & (-cand)
                v = vb.bit_length() - 1
                bronk_comp(R | vb, Pset & comp_adj[v], Xset & comp_adj[v])
                Pset ^= vb
                Xset |= vb
                cand ^= vb

        bronk_comp(0, mask, 0)
        return res

    # ---- Precompute exact minimum-parts DP for ALL masks (small n) ----
    # dp[mask] = minimum number of valid parts to partition 'mask'
    dp = [n + 1] * (1 << n)
    dp[0] = 0
    for mask in range(1, 1 << n):
        best = n + 1
        sub = mask
        while sub:
            if is_valid[sub]:
                c = 1 + dp[mask ^ sub]
                if c < best:
                    best = c
                    if best == 1:
                        break
            sub = (sub - 1) & mask
        dp[mask] = best

    exact_opt = dp[full_mask]

    # ---- Branch-and-bound search to find an actual partition of size exact_opt ----
    best_parts = None

    def search(mask, depth, current):
        nonlocal best_parts
        if time.time() > deadline:
            return
        if mask == 0:
            if best_parts is None or len(current) < len(best_parts):
                best_parts = list(current)
            return
        # admissible lower bound
        if depth + clique_cover_lb(mask) > exact_opt:
            return
        # if exact count already achieved, record
        if depth == exact_opt:
            if mask == 0:
                if best_parts is None or len(current) < len(best_parts):
                    best_parts = list(current)
            return
        # candidate parts: maximal valid subsets of mask (prefer large ones)
        cands = maximal_valid_subsets(mask)
        cands.sort(key=lambda m: -bin(m).count('1'))
        for part in cands:
            if part & mask != part:
                continue
            if not is_valid[part]:
                continue
            current.append(part)
            search(mask ^ part, depth + 1, current)
            current.pop()
            if best_parts is not None and len(best_parts) <= depth + 1:
                # can't beat with more parts; prune
                break

    search(full_mask, 0, [])

    # ---- Reconstruct from the exact DP (guaranteed valid) ----
    def reconstruct(mask):
        parts = []
        while mask:
            sub = mask
            chosen = None
            while sub:
                if is_valid[sub] and 1 + dp[mask ^ sub] == dp[mask]:
                    chosen = sub
                    break
                sub = (sub - 1) & mask
            if chosen is None:
                # fallback: single vertex
                chosen = mask & (-mask)
            parts.append([i for i in range(n) if (chosen >> i) & 1])
            mask ^= chosen
        return parts

    partition = reconstruct(full_mask)

    # ---- Validate & fix types ----
    seen = [False] * n
    ok = True
    for part in partition:
        if not part:
            ok = False
            break
        for v in part:
            if seen[v]:
                ok = False
                break
            seen[v] = True
        if not ok:
            break
        pm = 0
        for v in part:
            pm |= 1 << v
        if not is_valid[pm]:
            ok = False
            break
    if not ok or any(not seen[i] for i in range(n)):
        partition = [[i] for i in range(n)]

    result = [[int(v) for v in part] for part in partition]
    return result

# EVOLVE-BLOCK-END
