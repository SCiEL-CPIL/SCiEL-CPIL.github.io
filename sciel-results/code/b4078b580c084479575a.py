"""Phylogenetic topology recovery: ML via Felsenstein + NNI on Jukes-Cantor."""
import math
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 250.0


def _check_deadline():
    return time.time() < _DEADLINE


def _jc_distance(a, b):
    n = len(a)
    diff = sum(1 for x, y in zip(a, b) if x != y)
    p = diff / n
    if p >= 0.75:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"

    # Distance matrix
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
            dist[i][j] = d
            dist[j][i] = d

    # NJ guide tree: tree maps internal_node -> (left, right)
    active = list(range(n))
    tree = {}
    next_internal = n

    while len(active) > 2:
        m = len(active)
        if m == 3:
            a, b, c = active
            n1, n2 = next_internal, next_internal + 1
            next_internal += 2
            tree[n1] = (a, b)
            tree[n2] = (n1, c)
            break

        r = [0.0] * m
        for i in range(m):
            for j in range(m):
                if i != j:
                    r[i] += dist[active[i]][active[j]]

        best_val = float('inf')
        best_i, best_j = 0, 1
        for i in range(m):
            for j in range(i + 1, m):
                val = (m - 2) * dist[active[i]][active[j]] - r[i] - r[j]
                if val < best_val:
                    best_val = val
                    best_i, best_j = i, j

        a, b = active[best_i], active[best_j]
        new_node = next_internal
        next_internal += 1
        tree[new_node] = (a, b)

        remaining = [x for x in active if x not in (a, b)]
        for x in remaining:
            dist[new_node][x] = max(0.0, 0.5 * (dist[a][x] + dist[b][x] - dist[a][b]))
            dist[x][new_node] = dist[new_node][x]
        active = remaining + [new_node]

    internal_nodes = sorted(tree.keys())
    state_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    num_sites = len(sequences[labels[0]])

    def get_postorder(t):
        processed = set(range(n))
        order = []
        remaining = sorted(t.keys())
        while remaining:
            progress = False
            nr = []
            for node in remaining:
                left, right = t[node]
                if left in processed and right in processed:
                    order.append(node)
                    processed.add(node)
                    progress = True
                else:
                    nr.append(node)
            if not progress:
                order.extend(nr)
                break
            remaining = nr
        return order

    def compute_ll(t, bls):
        total_ll = 0.0
        po = get_postorder(t)
        for site in range(num_sites):
            alpha = {}
            for taxon in range(n):
                s = state_map[sequences[labels[taxon]][site]]
                alpha[taxon] = [0.0, 0.0, 0.0, 0.0]
                alpha[taxon][s] = 1.0
            for node in po:
                left, right = t[node]
                bl_l = bls.get((node, left), 0.1)
                bl_r = bls.get((node, right), 0.1)
                ps_l = (1.0 + 2.0 * math.exp(-3.0 * bl_l)) / 4.0
                pd_l = (1.0 - math.exp(-3.0 * bl_l)) / 4.0
                ps_r = (1.0 + 2.0 * math.exp(-3.0 * bl_r)) / 4.0
                pd_r = (1.0 - math.exp(-3.0 * bl_r)) / 4.0
                na = [0.0, 0.0, 0.0, 0.0]
                for s in range(4):
                    ls = sum((ps_l if sl == s else pd_l) * alpha[left][sl] for sl in range(4))
                    rs = sum((ps_r if sr == s else pd_r) * alpha[right][sr] for sr in range(4))
                    na[s] = ls * rs
                alpha[node] = na
            root = po[-1]
            ll_site = sum(0.25 * alpha[root][s] for s in range(4))
            total_ll += math.log(ll_site) if ll_site > 1e-300 else -1000.0
        return total_ll

    # Initialize branch lengths
    bls = {}
    for node in internal_nodes:
        left, right = tree[node]
        bls[(node, left)] = 0.1
        bls[(node, right)] = 0.1

    current_ll = compute_ll(tree, bls)

    # NNI refinement
    improved = True
    it = 0
    while improved and it < 30 and _check_deadline():
        improved = False
        it += 1
        best_ll = current_ll
        best_tree = None
        best_bls = None

        for node in internal_nodes:
            if not _check_deadline():
                break
            left, right = tree[node]
            # Check if left is internal
            if left >= n and left in tree:
                ll, lr = tree[left]
                # NNI: ((ll, lr), right) -> (ll, (lr, right))
                # New topology: node has (ll, new), left has (lr, right)
                # But we need to reuse node IDs. Let's create a valid new tree.
                nt = {k: v for k, v in tree.items()}
                nt[node] = (ll, right)
                nt[left] = (lr, node)  # This creates a cycle! Bad.
                # Proper NNI:
                # Original: node=(left, right), left=(ll, lr)
                # NNI1: node=(ll, right), left=(lr, X) where X is new - not allowed
                # The correct NNI swaps: the three subtrees are ll, lr, right
                # Original grouping: (ll,lr) then with right
                # Alt1: (lr,right) then with ll
                # Alt2: (ll,right) then with lr
                # For Alt1: node should have children (ll, left_new), left_new has (lr, right)
                # But we don't have left_new. We can reuse: make node=(ll, left), left=(lr, right)
                # Wait, that's the same as original if we just swap. Let me think...
                # Original: node -> (left, right), left -> (ll, lr)
                # The quartet is {ll, lr, right, [sibling of node]}
                # NNI changes the internal edge. The three subtrees attached to the internal edge
                # are: subtree(ll), subtree(lr), subtree(right).
                # Original: (ll,lr) joined, then joined with right.
                # NNI1: (lr,right) joined, then joined with ll.
                # NNI2: (ll,right) joined, then joined with lr.
                # For NNI1: node gets (ll, left'), left' gets (lr, right). But left' must be left.
                # So: node=(ll, left), left=(lr, right). But originally left=(ll,lr).
                # This means we're changing both node and left's children.
                nt1 = {k: v for k, v in tree.items()}
                nt1[node] = (ll, left)
                nt1[left] = (lr, right)
                bls1 = {k: v for k, v in bls.items()}
                bls1[(node, ll)] = bls.get((node, left), 0.1)
                bls1[(node, left)] = bls.get((left, ll), 0.1)
                bls1[(left, lr)] = bls.get((left, lr), 0.1)
                bls1[(left, right)] = bls.get((node, right), 0.1)
                ll1 = compute_ll(nt1, bls1)
                if ll1 > best_ll + 1e-10:
                    best_ll = ll1
                    best_tree = nt1
                    best_bls = bls1

                nt2 = {k: v for k, v in tree.items()}
                nt2[node] = (lr, left)
                nt2[left] = (ll, right)
                bls2 = {k: v for k, v in bls.items()}
                bls2[(node, lr)] = bls.get((node, left), 0.1)
                bls2[(node, left)] = bls.get((left, lr), 0.1)
                bls2[(left, ll)] = bls.get((left, ll), 0.1)
                bls2[(left, right)] = bls.get((node, right), 0.1)
                ll2 = compute_ll(nt2, bls2)
                if ll2 > best_ll + 1e-10:
                    best_ll = ll2
                    best_tree = nt2
                    best_bls = bls2

            if right >= n and right in tree:
                rl, rr = tree[right]
                nt1 = {k: v for k, v in tree.items()}
                nt1[node] = (left, rl)
                nt1[right] = (rr, left)
                bls1 = {k: v for k, v in bls.items()}
                bls1[(node, left)] = bls.get((node, left), 0.1)
                bls1[(node, rl)] = bls.get((node, right), 0.1)
                bls1[(right, rr)] = bls.get((right, rr), 0.1)
                bls1[(right, left)] = bls.get((right, rl), 0.1)
                ll1 = compute_ll(nt1, bls1)
                if ll1 > best_ll + 1e-10:
                    best_ll = ll1
                    best_tree = nt1
                    best_bls = bls1

                nt2 = {k: v for k, v in tree.items()}
                nt2[node] = (left, rr)
                nt2[right] = (rl, left)
                bls2 = {k: v for k, v in bls.items()}
                bls2[(node, left)] = bls.get((node, left), 0.1)
                bls2[(node, rr)] = bls.get((node, right), 0.1)
                bls2[(right, rl)] = bls.get((right, rl), 0.1)
                bls2[(right, left)] = bls.get((right, rr), 0.1)
                ll2 = compute_ll(nt2, bls2)
                if ll2 > best_ll + 1e-10:
                    best_ll = ll2
                    best_tree = nt2
                    best_bls = bls2

        if best_tree is not None:
            tree = best_tree
            bls = best_bls
            current_ll = best_ll
            improved = True

    def to_newick(node):
        if node < n:
            return labels[node]
        left, right = tree[node]
        return f"({to_newick(left)},{to_newick(right)})"

    root = internal_nodes[-1] if internal_nodes else 0
    return to_newick(root) + ";"
