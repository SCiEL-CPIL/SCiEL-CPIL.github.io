"""MLE-based phylogenetic topology recovery with time guard."""
import math
import time
import itertools
import numpy as np

_START_TIME = time.time()
_WALL_LIMIT = 270.0
_SAFETY = 5.0

def _time_left():
    return _WALL_LIMIT - _SAFETY - (time.time() - _START_TIME)

def _encode_seq(seq):
    m = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    return np.array([m[c] for c in seq], dtype=np.int8)

def _jc_transition(t):
    """Return (p_aa, p_ab) for JC transition at time t."""
    e = math.exp(-4.0 * t / 3.0)
    return 0.25 + 0.75 * e, 0.25 - 0.25 * e

def _jc_likelihood(tree_dict, taxa, data, bl):
    """
    Compute log-likelihood under JC.
    tree_dict: {internal_node: (child1, child2)}
    bl: {(parent, child): length}
    """
    # Post-order traversal
    children = {}
    for node, (l, r) in tree_dict.items():
        children[node] = (l, r)
    for l, r in tree_dict.values():
        if l not in children:
            children[l] = None
        if r not in children:
            children[r] = None
    
    # Find root
    all_children = set()
    for l, r in tree_dict.values():
        all_children.add(l)
        all_children.add(r)
    all_nodes = set(children.keys())
    root = list(all_nodes - all_children)[0]
    
    # Post-order
    order = []
    visited = set()
    def visit(node):
        if node in visited:
            return
        visited.add(node)
        if children[node] is None:
            order.append(node)
        else:
            l, r = children[node]
            visit(l)
            visit(r)
            order.append(node)
    visit(root)
    
    n_sites = len(next(iter(data.values())))
    node_ll = {}
    
    for node in order:
        if node in taxa:
            seq = data[node]
            ll = np.zeros((n_sites, 4), dtype=np.float64)
            for s in range(n_sites):
                ll[s, seq[s]] = 1.0
            node_ll[node] = ll
        else:
            l, r = tree_dict[node]
            bl_l = bl.get((node, l), 0.0)
            bl_r = bl.get((node, r), 0.0)
            p_aa_l, p_ab_l = _jc_transition(bl_l)
            p_aa_r, p_ab_r = _jc_transition(bl_r)
            
            T_l = np.array([[p_aa_l, p_ab_l, p_ab_l, p_ab_l],
                            [p_ab_l, p_aa_l, p_ab_l, p_ab_l],
                            [p_ab_l, p_ab_l, p_aa_l, p_ab_l],
                            [p_ab_l, p_ab_l, p_ab_l, p_aa_l]])
            T_r = np.array([[p_aa_r, p_ab_r, p_ab_r, p_ab_r],
                            [p_ab_r, p_aa_r, p_ab_r, p_ab_r],
                            [p_ab_r, p_ab_r, p_aa_r, p_ab_r],
                            [p_ab_r, p_ab_r, p_ab_r, p_aa_r]])
            
            L_l = node_ll[l]
            L_r = node_ll[r]
            
            # L_parent[a] = (T_l @ L_l[s])[a] * (T_r @ L_r[s])[a]
            term_l = L_l @ T_l.T
            term_r = L_r @ T_r.T
            node_ll[node] = term_l * term_r
    
    root_ll = node_ll[root]
    site_ll = root_ll.sum(axis=1)
    site_ll = np.maximum(site_ll, 1e-300)
    return float(np.sum(np.log(site_ll)))

def _optimize_bl(tree_dict, taxa, data, initial_bl, max_iter=40):
    """Optimize branch lengths via gradient ascent."""
    edges = []
    for node, (l, r) in tree_dict.items():
        edges.append((node, l))
        edges.append((node, r))
    
    bl = {e: max(0.01, initial_bl.get(e, 0.1)) for e in edges}
    ll = _jc_likelihood(tree_dict, taxa, data, bl)
    
    for _ in range(max_iter):
        if _time_left() < 5.0:
            break
        # Numerical gradients
        grads = {}
        for e in edges:
            h = 1e-4 * max(1.0, abs(bl[e]))
            bl_p = dict(bl); bl_p[e] = bl[e] + h
            bl_m = dict(bl); bl_m[e] = max(0.001, bl[e] - h)
            ll_p = _jc_likelihood(tree_dict, taxa, data, bl_p)
            ll_m = _jc_likelihood(tree_dict, taxa, data, bl_m)
            grads[e] = (ll_p - ll_m) / (bl_p[e] - bl_m[e])
        
        # Gradient ascent with line search
        step = 0.1
        improved = False
        for _ in range(15):
            new_bl = {e: max(0.001, bl[e] + step * grads[e]) for e in edges}
            new_ll = _jc_likelihood(tree_dict, taxa, data, new_bl)
            if new_ll > ll:
                bl = new_bl
                ll = new_ll
                improved = True
                break
            step *= 0.5
        if not improved:
            break
    
    return bl, ll

def _all_topologies(n):
    """Enumerate all unrooted binary tree topologies for n labeled taxa (0..n-1)."""
    if n == 3:
        return [{"r": (0, 1), "r1": ("r", 2)}]
    
    result = []
    seen = set()
    
    def find_root(td):
        ac = set()
        for l, r in td.values():
            ac.add(l); ac.add(r)
        return list(set(td.keys()) - ac)[0]
    
    def canon(td):
        def c(node):
            if isinstance(node, int):
                return str(node)
            l, r = td[node]
            return "(" + ",".join(sorted([c(l), c(r)])) + ")"
        return c(find_root(td))
    
    def gen(taxa_set):
        if len(taxa_set) == 3:
            a, b, c = sorted(taxa_set)
            return [{"r": (a, b), "r1": ("r", c)}]
        
        out = []
        ss = sorted(taxa_set)
        smallest = ss[0]
        rest = ss[1:]
        
        for sz in range(1, len(taxa_set)):
            for sub_rest in itertools.combinations(rest, sz - 1):
                A = frozenset([smallest] + list(sub_rest))
                B = frozenset(taxa_set - A)
                if len(B) < 1:
                    continue
                if min(A) >= min(B):
                    continue
                
                for tA in gen(A):
                    for tB in gen(B):
                        rA = find_root(tA)
                        rB = find_root(tB)
                        
                        # Combine
                        new_td = {}
                        for k, v in tA.items():
                            new_td[k] = v
                        ren = {}
                        for k in tB:
                            nn = f"b{k}"
                            ren[k] = nn
                        for k, v in tB.items():
                            new_td[ren[k]] = (ren.get(v[0], v[0]), ren.get(v[1], v[1]))
                        new_td["root"] = (rA, ren[rB])
                        
                        cstr = canon(new_td)
                        if cstr not in seen:
                            seen.add(cstr)
                            out.append(new_td)
        return out
    
    return gen(frozenset(range(n)))

def _tree_to_newick(td, taxa):
    ac = set()
    for l, r in td.values():
        ac.add(l); ac.add(r)
    root = list(set(td.keys()) - ac)[0]
    def to_nw(node):
        if node in taxa:
            return node
        l, r = td[node]
        return f"({to_nw(l)},{to_nw(r)})"
    return to_nw(root) + ";"

def _jc_dist(a, b):
    d = sum(1 for x, y in zip(a, b) if x != y)
    p = d / len(a)
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def _nj(sequences, labels):
    nodes = list(labels)
    trees = {name: name for name in labels}
    matrix = {(a, b): _jc_dist(sequences[a], sequences[b]) for a in labels for b in labels if a != b}
    while len(nodes) > 3:
        n = len(nodes)
        total = {a: sum(matrix[(a, b)] for b in nodes if b != a) for a in nodes}
        best = None
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                crit = (n - 2) * matrix[(a, b)] - total[a] - total[b]
                if best is None or crit < best[0]:
                    best = (crit, a, b)
        _, a, b = best
        joined = f"({trees[a]},{trees[b]})"
        new = f"n{len(nodes)}_{a}_{b}"
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            matrix[(new, x)] = matrix[(x, new)] = 0.5 * (matrix[(a, x)] + matrix[(b, x)] - matrix[(a, b)])
        nodes = rest + [new]
        trees[new] = joined
    return "(" + ",".join(trees[x] for x in nodes) + ");"

def _newick_to_tree_dict(newick):
    s = newick.strip().rstrip(';')
    pos = [0]
    counter = [0]
    
    def parse():
        if pos[0] >= len(s):
            return None
        if s[pos[0]] == '(':
            pos[0] += 1
            left = parse()
            if pos[0] < len(s) and s[pos[0]] == ',':
                pos[0] += 1
            right = parse()
            if pos[0] < len(s) and s[pos[0]] == ')':
                pos[0] += 1
            while pos[0] < len(s) and s[pos[0]] not in (',', ';', ')'):
                pos[0] += 1
            nid = f"i{counter[0]}"
            counter[0] += 1
            return (nid, left, right)
        else:
            start = pos[0]
            while pos[0] < len(s) and s[pos[0]] not in (',', ')', ';'):
                pos[0] += 1
            return ('leaf', s[start:pos[0]], None)
    
    result = parse()
    td = {}
    taxa = []
    
    def build(node):
        if node is None:
            return None
        if node[0] == 'leaf':
            taxa.append(node[1])
            return node[1]
        nid, l, r = node
        lid = build(l)
        rid = build(r)
        td[nid] = (lid, rid)
        return nid
    
    build(result)
    return td, taxa

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    data = {label: _encode_seq(sequences[label]) for label in labels}
    
    if _time_left() < 10.0:
        return _nj(sequences, labels)
    
    if n <= 7:
        topologies = _all_topologies(n)
        best_ll = -float('inf')
        best_td = None
        
        for td in topologies:
            if _time_left() < 10.0:
                break
            edges = []
            for node, (l, r) in td.items():
                edges.append((node, l))
                edges.append((node, r))
            init_bl = {e: 0.1 for e in edges}
            bl, ll = _optimize_bl(td, labels, data, init_bl, max_iter=30)
            if ll > best_ll:
                best_ll = ll
                best_td = td
        
        if best_td is not None:
            return _tree_to_newick(best_td, labels)
        return _nj(sequences, labels)
    
    elif n <= 10:
        newick = _nj(sequences, labels)
        td, taxa = _newick_to_tree_dict(newick)
        edges = []
        for node, (l, r) in td.items():
            edges.append((node, l))
            edges.append((node, r))
        init_bl = {e: 0.1 for e in edges}
        bl, ll = _optimize_bl(td, labels, data, init_bl, max_iter=50)
        return _tree_to_newick(td, labels)
    
    else:
        return _nj(sequences, labels)
