"""
Phylogenetic topology recovery: NJ guide tree + NNI refinement under JC-ML.
"""
import math
import time
from collections import Counter

# EVOLVE-BLOCK-START

_DEADLINE = 258.0
_start_time = None


def _elapsed():
    return time.time() - _start_time if _start_time is not None else 0.0


def _time_left():
    return _DEADLINE - _elapsed()


def reconstruct(sequences):
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    seqs = [sequences[lb] for lb in labels]
    n_sites = len(seqs[0])

    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    enc = [[base_map[c] for c in s] for s in seqs]

    # Count site patterns
    patterns = Counter()
    for k in range(n_sites):
        pat = tuple(enc[i][k] for i in range(n))
        patterns[pat] += 1

    # ---- NJ tree construction ----
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            diff = sum(1 for k in range(n_sites) if enc[i][k] != enc[j][k])
            p = diff / n_sites
            if p >= 0.75:
                dist[i][j] = 3.0
            else:
                dist[i][j] = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            dist[j][i] = dist[i][j]

    # NJ
    active = list(range(n))
    d = [row[:] for row in dist]
    children = {}
    node_taxa = {i: frozenset([i]) for i in range(n)}
    next_node = n

    while len(active) > 3:
        m = len(active)
        r = [0.0] * m
        for ii in range(m):
            for jj in range(m):
                if ii != jj:
                    r[ii] += d[active[ii]][active[jj]]
        best_q = None
        bi = bj = None
        for ii in range(m):
            for jj in range(ii + 1, m):
                q = (m - 2) * d[active[ii]][active[jj]] - r[ii] - r[jj]
                if best_q is None or q < best_q:
                    best_q = q
                    bi, bj = ii, jj
        i, j = active[bi], active[bj]
        nn = next_node
        next_node += 1
        children[nn] = (i, j)
        node_taxa[nn] = node_taxa[i] | node_taxa[j]
        rem = [x for x in active if x != i and x != j]
        for k in rem:
            d[nn][k] = 0.5 * (d[i][k] + d[j][k] - d[i][j])
            d[k][nn] = d[nn][k]
        active = rem + [nn]

    if len(active) == 3:
        a, b, c = active
        n1 = next_node; next_node += 1
        n2 = next_node; next_node += 1
        children[n1] = (a, b)
        node_taxa[n1] = node_taxa[a] | node_taxa[b]
        children[n2] = (n1, c)
        node_taxa[n2] = node_taxa[n1] | node_taxa[c]
        root = n2
    else:
        root = active[0]

    # ---- Likelihood under JC model ----
    # Precompute: for each pattern, we need the likelihood given a tree.
    # For JC: P(obs|tree) = sum_{root_state} (1/4) * product over edges of trans_prob

    def _jc_p(L):
        """Return (p_same, p_diff) for branch length L."""
        if L <= 0:
            return 1.0, 0.0
        e = math.exp(-4.0 * L / 3.0)
        return 0.25 + 0.75 * e, 0.25 * (1.0 - e)

    def _loglik(tree_ch, root_node, bl):
        """Compute log-likelihood. bl: dict node->branch_len (edge from parent)."""
        # Build post-order
        visited = set()
        po = []
        def _po(nd):
            if nd in visited: return
            visited.add(nd)
            if nd in tree_ch:
                l, r = tree_ch[nd]
                _po(l); _po(r)
            po.append(nd)
        _po(root_node)

        # Precompute transitions
        tr = {}
        for nd in range(n, next_node):
            tr[nd] = _jc_p(bl.get(nd, 0.0))

        total_ll = 0.0
        for pat, cnt in patterns.items():
            # Felsenstein pruning for this pattern
            node_L = {}
            for nd in po:
                if nd < n:
                    b = pat[nd]
                    node_L[nd] = [0.0, 0.0, 0.0, 0.0]
                    node_L[nd][b] = 1.0
                else:
                    l, r = tree_ch[nd]
                    Ll = node_L[l]; Lr = node_L[r]
                    ps_l, pd_l = tr[l]
                    ps_r, pd_r = tr[r]
                    res = [0.0, 0.0, 0.0, 0.0]
                    for s in range(4):
                        t = 0.0
                        for sl in range(4):
                            pl = ps_l if sl == s else pd_l
                            if pl == 0: continue
                            for sr in range(4):
                                pr = ps_r if sr == s else pd_r
                                if pr == 0: continue
                                t += pl * Ll[sl] * pr * Lr[sr]
                        res[s] = t
                    node_L[nd] = res
            rl = node_L[root_node]
            sl = 0.25 * sum(rl)
            total_ll += cnt * math.log(sl) if sl > 0 else cnt * (-1000)
        return total_ll

    # Initialize branch lengths
    bl = {nd: 0.5 for nd in range(n, next_node)}

    # Quick branch length optimization (few iterations)
    def _opt_bl(tree_ch, root_node, bl, rounds=5):
        for _ in range(rounds):
            if _time_left() < 15: break
            for nd in range(n, next_node):
                if _time_left() < 15: break
                cur = bl[nd]
                best_ll = _loglik(tree_ch, root_node, bl)
                best_v = cur
                for v in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, cur*0.5, cur*1.5, cur*2]:
                    v = max(0.01, min(5.0, v))
                    bl[nd] = v
                    ll = _loglik(tree_ch, root_node, bl)
                    if ll > best_ll:
                        best_ll = ll
                        best_v = v
                bl[nd] = best_v
        return bl

    bl = _opt_bl(children, root, bl, rounds=3)
    cur_ll = _loglik(children, root, bl)

    # ---- NNI refinement ----
    best_tree = dict(children)
    best_root = root
    best_bl = dict(bl)
    best_ll = cur_ll
    extra_node = next_node

    for round_num in range(200):
        if _time_left() < 15: break
        improved = False

        # Build parent map
        par = {}
        for nd, (l, r) in best_tree.items():
            par[l] = nd; par[r] = nd
        par[best_root] = None

        # Internal edges: (parent, child) where child is internal
        iedges = []
        for nd, (l, r) in best_tree.items():
            if par[nd] is not None and nd >= n:
                iedges.append((par[nd], nd))

        for p, c in iedges:
            if _time_left() < 15: break
            pl, pr = best_tree[p]
            po_ = pr if pl == c else pl
            c1, c2 = best_tree[c]

            for var in (0, 1):
                if _time_left() < 15: break
                nn = extra_node
                extra_node += 1
                nt = dict(best_tree)
                if var == 0:
                    nt[p] = (nn, c2)
                    nt[nn] = (c1, po_)
                else:
                    nt[p] = (nn, c1)
                    nt[nn] = (c2, po_)
                del nt[c]

                nbl = dict(best_bl)
                nbl[nn] = best_bl.get(c, 0.5)

                ll = _loglik(nt, best_root, nbl)
                if ll > best_ll + 1e-8:
                    best_tree = nt
                    best_ll = ll
                    best_bl = nbl
                    improved = True
                    break
            if improved:
                break

        if not improved:
            break

    # Convert to Newick
    def _newick(nd):
        if nd < n:
            return labels[nd]
        l, r = best_tree[nd]
        return f"({_newick(l)},{_newick(r)})"
    return _newick(best_root) + ";"

# EVOLVE-BLOCK-END
