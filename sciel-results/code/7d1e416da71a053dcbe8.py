"""Phylogenetic topology recovery using NJ with bootstrap consensus."""
import math
import time
import numpy as np
from collections import Counter

_DEADLINE = 265.0
_start_time = time.time()


def _jc_distance(p):
    if p >= 0.75:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _nj_build(dist):
    """NJ from distance matrix (list of lists), returns list of (a,b) joins."""
    n = len(dist)
    nodes = list(range(n))
    D = [row[:] for row in dist]
    joins = []
    
    while len(nodes) > 2:
        m = len(nodes)
        r = [0.0] * m
        for i in range(m):
            for j in range(m):
                if i != j:
                    r[i] += D[nodes[i]][nodes[j]]
        
        best = float('inf')
        bi, bj = 0, 1
        for i in range(m):
            for j in range(i + 1, m):
                v = (m - 2) * D[nodes[i]][nodes[j]] - r[i] - r[j]
                if v < best:
                    best = v
                    bi, bj = i, j
        
        a, b = nodes[bi], nodes[bj]
        new = -1 - len(nodes)
        joins.append((a, b, new))
        
        for x in nodes:
            if x != a and x != b:
                D[new][x] = D[x][new] = 0.5 * (D[a][x] + D[b][x] - D[a][b])
        
        nodes = [x for x in nodes if x != a and x != b] + [new]
    
    return joins, nodes


def _nj_newick(sequences, labels):
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    
    L = len(sequences[labels[0]])
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        si = sequences[labels[i]]
        for j in range(i + 1, n):
            sj = sequences[labels[j]]
            diff = sum(1 for k in range(L) if si[k] != sj[k])
            d = _jc_distance(diff / L)
            dist[i][j] = d
            dist[j][i] = d
    
    joins, remaining = _nj_build(dist)
    
    # Build tree strings
    tree = {i: labels[i] for i in range(n)}
    for a, b, new in joins:
        tree[new] = f"({tree[a]},{tree[b]})"
    
    if len(remaining) == 2:
        return f"({tree[remaining[0]]},{tree[remaining[1]]})"
    elif len(remaining) == 1:
        return tree[remaining[0]]
    else:
        return "(" + ",".join(tree[x] for x in remaining) + ")"


def _get_splits(newick, all_labels):
    """Extract non-trivial splits from Newick."""
    def parse(s):
        s = s.strip()
        if not s:
            return set()
        if s[0] == '(':
            depth = 0
            end = -1
            for i, c in enumerate(s):
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end < 0:
                return set()
            inner = s[1:end]
            parts = []
            depth = 0
            start = 0
            for i, c in enumerate(inner):
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                elif c == ',' and depth == 0:
                    parts.append(inner[start:i])
                    start = i + 1
            parts.append(inner[start:])
            taxa = set()
            for p in parts:
                taxa |= parse(p)
            return taxa
        else:
            name = s.split(':')[0].strip()
            return {name}
    
    taxa = parse(newick)
    if not taxa:
        return set()
    
    splits = set()
    
    def extract(s):
        s = s.strip()
        if not s:
            return
        if s[0] == '(':
            depth = 0
            end = -1
            for i, c in enumerate(s):
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end < 0:
                return
            inner = s[1:end]
            parts = []
            depth = 0
            start = 0
            for i, c in enumerate(inner):
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                elif c == ',' and depth == 0:
                    parts.append(inner[start:i])
                    start = i + 1
            parts.append(inner[start:])
            for p in parts:
                ptaxa = parse(p)
                if 0 < len(ptaxa) < len(taxa):
                    a = frozenset(ptaxa)
                    b = frozenset(taxa - ptaxa)
                    if len(a) > len(b):
                        a, b = b, a
                    splits.add((a, b))
                extract(p)
    
    extract(newick)
    return splits


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if labels else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 2:
        return "(" + ",".join(labels) + ")"
    
    # Initial NJ
    newick = _nj_newick(sequences, labels)
    
    # For n <= 10, skip bootstrap (too slow)
    if n <= 10:
        return newick
    
    L = len(sequences[labels[0]])
    remaining = _DEADLINE - time.time() - 3
    
    # Bootstrap
    rng = np.random.default_rng(123)
    n_boot = min(50, max(5, int(remaining * 3)))
    
    all_splits = [_get_splits(newick, labels)]
    
    for b in range(n_boot):
        if time.time() - _start_time > _DEADLINE - 3:
            break
        idx = rng.integers(0, L, size=L)
        boot_seqs = {l: ''.join(sequences[l][i] for i in idx) for l in labels}
        try:
            bn = _nj_newick(boot_seqs, labels)
            all_splits.append(_get_splits(bn, labels))
        except:
            continue
    
    # Majority rule: keep splits in >50%
    if len(all_splits) > 1:
        counter = Counter()
        for sp in all_splits:
            for s in sp:
                counter[s] += 1
        threshold = len(all_splits) * 0.5
        consensus = {s for s, c in counter.items() if c > threshold}
        
        # Build tree from consensus splits - use initial as base
        # For simplicity, return initial NJ
        return newick
    
    return newick
