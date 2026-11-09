"""Phylogenetic topology recovery: FDQ quartet consensus with NJ fallback."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

# ---- Distance matrix (Jukes-Cantor correction), vectorized -----------------
def _jc_distance_matrix(sequences, labels):
    n = len(labels)
    idx = {lab: i for i, lab in enumerate(labels)}
    # Map each sequence to 4 one-hot integer arrays for fast hamming
    codes = np.zeros((n, len(sequences[labels[0]]), 4), dtype=np.int8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        for j, ch in enumerate(s):
            codes[i, j, {'A': 0, 'C': 1, 'G': 2, 'T': 3}[ch]] = 1
    L = codes.shape[1]
    # Hamming distance matrix
    # For each pair, count positions where they differ = 4 - sum over bases of (both==base)
    # Use matrix multiplication per base
    D = np.zeros((n, n), dtype=np.float64)
    for b in range(4):
        M = codes[:, :, b]  # (n, L)
        agree = M @ M.T      # (n, n) counts of both having base b
        D += agree
    D = 4.0 * L - D          # hamming distances (each base contributes 1 per position)
    # diagonal is 0
    np.fill_diagonal(D, 0.0)
    p = D / float(L)
    # JC correction: -0.75 * ln(1 - 4p/3), guard p>=0.75
    p = np.clip(p, 0.0, 0.7499)
    D = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    np.fill_diagonal(D, 0.0)
    return D


def _neighbor_joining(D, labels):
    """Neighbor joining on distance matrix D (n x n). Returns Newick string."""
    n = len(labels)
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"
    if n == 3:
        return "(" + labels[0] + ",(" + labels[1] + "," + labels[2] + "));"

    nodes = list(range(n))
    tree = {i: labels[i] for i in range(n)}
    dist = D.copy()
    active = [True] * n
    active_count = n
    while active_count > 3:
        # row sums over active
        total = np.zeros(n, dtype=np.float64)
        for a in nodes:
            if not active[a]:
                continue
            s = 0.0
            for b in nodes:
                if b != a and active[b]:
                    s += dist[a, b]
            total[a] = s
        # find min criterion
        best = None
        for ia, a in enumerate(nodes):
            if not active[a]:
                continue
            for b in nodes[ia + 1:]:
                if not active[b]:
                    continue
                criterion = (active_count - 2) * dist[a, b] - total[a] - total[b]
                if best is None or criterion < best[0]:
                    best = (criterion, a, b)
        _, a, b = best
        # new node index
        new = n
        n += 1
        dist = np.zeros((n, n), dtype=np.float64)
        for i in range(new):
            for j in range(new):
                dist[i, j] = D[i, j]
        # compute new distances
        rest = [x for x in range(new) if x != a and x != b]
        for x in rest:
            val = 0.5 * (dist[a, x] + dist[b, x] - dist[a, b])
            dist[new, x] = val
            dist[x, new] = val
        # build new tree string
        joined = "(" + tree[a] + "," + tree[b] + ")"
        tree[new] = joined
        # deactivate a, b
        active[a] = False
        active[b] = False
        active.append(True)
        nodes.append(new)
        active_count -= 1
    # remaining 3 active nodes
    rem = [x for x in range(n) if active[x]]
    return "(" + ",".join(tree[x] for x in rem) + ");"


def _splits_from_newick(newick, labels):
    """Return set of non-trivial splits as frozensets of label indices."""
    lab_to_idx = {lab: i for i, lab in enumerate(labels)}
    # Parse to get bipartitions
    splits = set()
    stack = []
    # tokenize
    i = 0
    s = newick
    n = len(s)
    # We'll do a recursive descent using a function
    def parse(pos):
        # returns (child_set, new_pos)
        if s[pos] == '(':
            pos += 1
            children = []
            while True:
                child, pos = parse(pos)
                children.append(child)
                if s[pos] == ',':
                    pos += 1
                elif s[pos] == ')':
                    pos += 1
                    break
            merged = set()
            for c in children:
                merged |= c
            return merged, pos
        else:
            # leaf name until , ) ;
            start = pos
            while pos < n and s[pos] not in ',);':
                pos += 1
            name = s[start:pos].strip()
            return {lab_to_idx[name]}, pos
    # handle trailing semicolon
    if s.endswith(';'):
        s = s[:-1]
    allset = set(range(len(labels)))
    def parse2(pos):
        if s[pos] == '(':
            pos += 1
            children = []
            while True:
                child, pos = parse2(pos)
                children.append(child)
                if s[pos] == ',':
                    pos += 1
                elif s[pos] == ')':
                    pos += 1
                    break
            merged = set()
            for c in children:
                merged |= c
            return merged, pos
        else:
            start = pos
            while pos < len(s) and s[pos] not in ',);':
                pos += 1
            name = s[start:pos].strip()
            return {lab_to_idx[name]}, pos
    rootset, _ = parse2(0)
    # Need all internal nodes. Re-parse collecting internal sets.
    internals = []
    def parse3(pos):
        if s[pos] == '(':
            pos += 1
            children = []
            while True:
                child, pos = parse3(pos)
                children.append(child)
                if s[pos] == ',':
                    pos += 1
                elif s[pos] == ')':
                    pos += 1
                    break
            merged = set()
            for c in children:
                merged |= c
            internals.append(merged)
            return merged, pos
        else:
            start = pos
            while pos < len(s) and s[pos] not in ',);':
                pos += 1
            name = s[start:pos].strip()
            return {lab_to_idx[name]}, pos
    parse3(0)
    for m in internals:
        comp = allset - m
        if 0 < len(m) < len(allset):
            a, b = frozenset(m), frozenset(comp)
            splits.add(min(a, b, key=len))
    return splits


def _quartet_consensus_tree(D, labels, deadline):
    """FDQ: score quartets, build compatible-split tree."""
    n = len(labels)
    if n <= 3:
        return "(" + ",".join(labels) + ");"
    lab_to_idx = {lab: i for i, lab in enumerate(labels)}
    allset = frozenset(range(n))

    # Score all quartets from distance matrix
    # For each quartet {a,b,c,d}, compare sums:
    # s_ab_cd = D[a,b]+D[c,d], s_ac_bd = D[a,c]+D[b,d], s_ad_bc = D[a,d]+D[b,c]
    # The smallest sum corresponds to the true split.
    # Accumulate votes per split (frozenset of the smaller side).
    votes = {}
    t_start = time.time()
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    sab_cd = D[a, b] + D[c, d]
                    sac_bd = D[a, c] + D[b, d]
                    sad_bc = D[a, d] + D[b, c]
                    m = min(sab_cd, sac_bd, sad_bc)
                    if m == sab_cd:
                        split = frozenset((a, b))
                    elif m == sac_bd:
                        split = frozenset((a, c))
                    else:
                        split = frozenset((a, d))
                    # normalize to smaller side
                    comp = allset - split
                    if len(comp) < len(split):
                        split = comp
                    votes[split] = votes.get(split, 0) + 1
                    if time.time() - t_start > deadline * 0.5:
                        break
                if time.time() - t_start > deadline * 0.5:
                    break
            if time.time() - t_start > deadline * 0.5:
                break
        if time.time() - t_start > deadline * 0.5:
            break

    # Sort splits by vote count descending
    if not votes:
        return _neighbor_joining(D, labels)
    ranked = sorted(votes.items(), key=lambda kv: -kv[1])

    # Greedily add compatible splits
    # A split is compatible with an existing split if neither separates the other.
    # Two splits A/B and C/D are incompatible if there exist a in A∩C, b in A∩D, c in B∩C, d in B∩D
    chosen = []
    for split, v in ranked:
        if time.time() > deadline:
            break
        comp = allset - split
        ok = True
        for existing in chosen:
            ecomp = allset - existing
            # check incompatibility
            if (len(split & existing) > 0 and
                len(split & ecomp) > 0 and
                len(comp & existing) > 0 and
                len(comp & ecomp) > 0):
                ok = False
                break
        if ok:
            chosen.append(split)
        if len(chosen) >= n - 3:
            break

    # Build tree from chosen splits using a leaf-addition / Buneman-like method
    # Simpler: build a tree by starting with a star and refining via splits,
    # but easiest robust method: use the chosen splits to build a tree via
    # a recursive decomposition.
    # We'll build the tree by repeatedly finding a split that partitions current
    # clusters, but to keep it simple and robust, we fall back to NJ if the
    # split set is not a full tree.
    # Construct a tree from splits using a "split to tree" via a stack of clusters.
    # Use a recursive builder: given a set of taxa S and the list of splits restricted
    # to S, find a split that partitions S, split, recurse.
    def build_tree(taxa_set):
        # taxa_set: frozenset of indices
        if len(taxa_set) <= 1:
            return labels[list(taxa_set)[0]] if len(taxa_set) == 1 else ""
        if len(taxa_set) == 2:
            a, b = list(taxa_set)
            return "(" + labels[a] + "," + labels[b] + ")"
        # find a split in chosen that partitions taxa_set
        best_split = None
        for sp in chosen:
            part = sp & taxa_set
            if 0 < len(part) < len(taxa_set):
                best_split = part
                break
        if best_split is None:
            # no internal split: return a star (or NJ on these)
            # use NJ on the distance submatrix for these taxa
            idx_list = list(taxa_set)
            sub = D[np.ix_(idx_list, idx_list)]
            sub_labels = [labels[i] for i in idx_list]
            return _neighbor_joining(sub, sub_labels)
        other = taxa_set - best_split
        left = build_tree(best_split)
        right = build_tree(other)
        return "(" + left + "," + right + ")"
    newick = build_tree(allset)
    return newick


def reconstruct(sequences):
    # Deadline guard: keep a safety margin before 270s wall clock.
    t0 = time.time()
    deadline = 240.0  # safe margin under 270s

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n <= 3:
        return "(" + ",".join(labels) + ");"

    # Compute distance matrix
    D = _jc_distance_matrix(sequences, labels)

    # Primary: quartet consensus
    try:
        if time.time() - t0 < deadline:
            newick = _quartet_consensus_tree(D, labels, deadline - (time.time() - t0))
            # Validate: all labels present exactly once
            from collections import Counter
            import re
            found = re.findall(r't\d+', newick)
            cnt = Counter(found)
            if len(cnt) == n and all(cnt[l] == 1 for l in labels):
                return newick
    except Exception:
        pass

    # Fallback: neighbor joining
    try:
        if time.time() - t0 < deadline:
            newick = _neighbor_joining(D, labels)
            from collections import Counter
            import re
            found = re.findall(r't\d+', newick)
            cnt = Counter(found)
            if len(cnt) == n and all(cnt[l] == 1 for l in labels):
                return newick
    except Exception:
        pass

    # Last resort: star tree
    return "(" + ",".join(labels) + ");"

# EVOLVE-BLOCK-END
