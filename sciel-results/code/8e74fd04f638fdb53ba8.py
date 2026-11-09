"""
Phylogenetic topology recovery using Maximum Likelihood with Jukes-Cantor model
and NNI hill-climbing for topology optimization.

Strategy:
1. Build initial tree via Neighbor Joining on JC-corrected distances.
2. Optimize branch lengths on the fixed topology via gradient ascent.
3. Improve topology via NNI moves, accepting moves that improve log-likelihood.
4. Use a time deadline guard to return the best result before the 270s wall-clock limit.
"""
import math
import time
import random

import numpy as np

_DEADLINE = 265.0
_START_TIME = None


def _jc_distance(a, b):
    n = len(a)
    if n == 0:
        return 0.0
    diffs = sum(1 for x, y in zip(a, b) if x != y)
    p = diffs / n
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _neighbor_joining(sequences, labels):
    n = len(labels)
    if n <= 2:
        return tuple(labels)
    if n == 3:
        return (labels[0], (labels[1], labels[2]))

    dist = {}
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i != j:
                dist[(a, b)] = _jc_distance(sequences[a], sequences[b])

    nodes = list(labels)
    tree = {name: name for name in labels}

    while len(nodes) > 3:
        n = len(nodes)
        total = {}
        for a in nodes:
            s = 0.0
            for b in nodes:
                if b != a:
                    s += dist[(a, b)]
            total[a] = s

        best_criterion = float('inf')
        best_a, best_b = None, None
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                criterion = (n - 2) * dist[(a, b)] - total[a] - total[b]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_a, best_b = a, b

        a, b = best_a, best_b
        d_ab = dist[(a, b)]
        r_a = 0.5 * (d_ab + (total[a] - total[b]) / (n - 2))
        r_b = d_ab - r_a

        new_node = f"nj_{a}_{b}"
        tree[new_node] = (tree[a], tree[b])

        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            d_ax = dist[(a, x)]
            d_bx = dist[(b, x)]
            d_new_x = 0.5 * (d_ax + d_bx - d_ab)
            dist[(new_node, x)] = d_new_x
            dist[(x, new_node)] = d_new_x

        nodes = rest + [new_node]

    return (tree[nodes[0]], (tree[nodes[1]], tree[nodes[2]]))


def _tree_to_edges(tree):
    edges = []
    leaf_set = set()
    counter = [0]
    node_id = {}

    def _assign_id(node):
        if isinstance(node, str):
            return node
        if node not in node_id:
            node_id[node] = counter[0]
            counter[0] += 1
        return node_id[node]

    def _walk(node, parent=None):
        if isinstance(node, str):
            leaf_set.add(node)
            if parent is not None:
                edges.append((parent, node))
            return node
        nid = _assign_id(node)
        if parent is not None:
            edges.append((parent, nid))
        for child in node:
            _walk(child, nid)
        return nid

    _walk(tree)
    return edges, leaf_set, counter[0]


def _build_adjacency(edges):
    adj = {}
    for i, (u, v) in enumerate(edges):
        if u not in adj:
            adj[u] = []
        if v not in adj:
            adj[v] = []
        adj[u].append((v, i))
        adj[v].append((u, i))
    return adj


def _ml_loglik(data, labels, tree, blens):
    """Fast JC log-likelihood via pruning."""
    edges, leaf_set, _ = _tree_to_edges(tree)
    adj = _build_adjacency(edges)
    
    internal_nodes = [n for n in adj if n not in leaf_set]
    root = internal_nodes[0] if internal_nodes else edges[0][0]
    
    parent = {}
    children = {}
    order = []
    visited = set()
    
    def dfs(node, par):
        visited.add(node)
        parent[node] = par
        children[node] = []
        for nb, _ in adj[node]:
            if nb != par and nb not in visited:
                children[node].append(nb)
                dfs(nb, node)
        order.append(node)
    
    dfs(root, None)
    
    edge_map = {}
    for i, (u, v) in enumerate(edges):
        if parent.get(v) == u:
            edge_map[(u, v)] = i
        elif parent.get(u) == v:
            edge_map[(v, u)] = i
    
    # Precompute label indices
    label_idx = {lab: i for i, lab in enumerate(labels)}
    
    LL = {}
    for node in order:
        if node in leaf_set:
            li = label_idx[node]
            ll = np.zeros(4)
            ll[data[li]] = 1.0
            LL[node] = ll
        else:
            ll = np.ones(4)
            for child in children[node]:
                eidx = edge_map.get((node, child))
                if eidx is None:
                    eidx = edge_map.get((child, node))
                blen = blens[eidx] if eidx is not None else 0.0
                if blen < 0:
                    blen = 0.0
                p_same = 0.25 + 0.75 * math.exp(-3.0 * blen)
                p_diff = (1.0 - p_same) / 3.0
                child_ll = LL[child]
                s = child_ll.sum()
                contrib = child_ll * p_same + (s - child_ll) * p_diff
                ll = ll * contrib
            LL[node] = ll
    
    log_ll = 0.0
    for i in range(4):
        val = LL[root][i]
        if val > 0:
            log_ll += math.log(val)
        else:
            log_ll += -1000.0
    return log_ll


def _optimize_bl(data, labels, tree, max_iter=40, tol=1e-5):
    edges, _, _ = _tree_to_edges(tree)
    n_edges = len(edges)
    blens = [0.1] * n_edges
    cur_ll = _ml_loglik(data, labels, tree, blens)
    
    lr = 0.05
    for it in range(max_iter):
        if time.time() - _START_TIME > _DEADLINE - 5:
            break
        grad = np.zeros(n_edges)
        eps = 0.001
        for i in range(n_edges):
            orig = blens[i]
            blens[i] = orig + eps
            lp = _ml_loglik(data, labels, tree, blens)
            blens[i] = orig - eps
            lm = _ml_loglik(data, labels, tree, blens)
            blens[i] = orig
            grad[i] = (lp - lm) / (2 * eps)
        
        gn = np.linalg.norm(grad)
        if gn < tol:
            break
        for i in range(n_edges):
            blens[i] = max(0.0, blens[i] + lr * grad[i])
        
        new_ll = _ml_loglik(data, labels, tree, blens)
        if new_ll > cur_ll:
            cur_ll = new_ll
        else:
            for i in range(n_edges):
                blens[i] = max(0.0, blens[i] - lr * grad[i])
            lr *= 0.5
            if lr < 1e-7:
                break
    
    return blens, cur_ll


def _edges_to_tree(edges, leaf_set, labels):
    if len(edges) == 0:
        return labels[0]
    if len(edges) == 1:
        return (edges[0][0], edges[0][1])
    
    adj = _build_adjacency(edges)
    
    start = None
    for lab in labels:
        if lab in leaf_set:
            start = lab
            break
    if start is None:
        return tuple(labels)
    
    parent = {}
    children = {}
    visited = set()
    
    def dfs(node, par):
        visited.add(node)
        parent[node] = par
        children[node] = []
        for nb, _ in adj[node]:
            if nb != par and nb not in visited:
                children[node].append(nb)
                dfs(nb, node)
    
    dfs(start, None)
    
    def build_node(node):
        if node in leaf_set:
            return node
        kids = children[node]
        if len(kids) == 1:
            return build_node(kids[0])
        return tuple(build_node(k) for k in kids)
    
    return build_node(start)


def _nni_improve(data, labels, tree, blens, max_rounds=100):
    """NNI hill-climbing."""
    best_tree = tree
    best_bl = list(blens)
    best_ll = _ml_loglik(data, labels, tree, best_bl)
    
    for rnd in range(max_rounds):
        if time.time() - _START_TIME > _DEADLINE - 10:
            break
        
        edges, leaf_set, node_count = _tree_to_edges(best_tree)
        n_edges = len(edges)
        adj = _build_adjacency(edges)
        
        # Find internal edges
        int_eis = [i for i, (u, v) in enumerate(edges)
                  if u not in leaf_set and v not in leaf_set]
        
        if not int_eis:
            break
        
        best_move_ll = best_ll
        best_new_edges = None
        
        for ei in int_eis:
            if time.time() - _START_TIME > _DEADLINE - 10:
                break
            
            u, v = edges[ei]
            u_neigh = [nb for nb, eidx in adj[u] if eidx != ei]
            v_neigh = [nb for nb, eidx in adj[v] if eidx != ei]
            
            if len(u_neigh) != 2 or len(v_neigh) != 2:
                continue
            
            a1, a2 = u_neigh
            c1, c2 = v_neigh
            
            # Try 3 NNI alternatives
            # New internal nodes
            new_id1 = node_count + 1000 + rnd
            new_id2 = new_id1 + 1
            
            for alt in range(3):
                # Build new edge list
                new_edges = []
                for i, (a, b) in enumerate(edges):
                    if i == ei:
                        continue
                    new_edges.append((a, b))
                
                # Remove edges from u and v to their neighbors (except the ones we're rewiring)
                # u was connected to a1, a2, v -> remove u-a1, u-a2
                # v was connected to c1, c2, u -> remove v-c1, v-c2
                # Then add new connections based on NNI alternative
                
                # Filter out edges incident to u or v (we'll re-add them)
                new_edges = [(a, b) for (a, b) in new_edges if a not in (u, v) and b not in (u, v)]
                
                # NNI alternatives:
                # Alt 0: (a1, new_id1, c1) and (a2, new_id2, c2), connect new_id1-new_id2
                # Alt 1: (a1, new_id1, c2) and (a2, new_id2, c1), connect new_id1-new_id2
                # Alt 2: (a2, new_id1, c1) and (a1, new_id2, c2), connect new_id1-new_id2
                
                if alt == 0:
                    new_edges.append((a1, new_id1))
                    new_edges.append((c1, new_id1))
                    new_edges.append((a2, new_id2))
                    new_edges.append((c2, new_id2))
                elif alt == 1:
                    new_edges.append((a1, new_id1))
                    new_edges.append((c2, new_id1))
                    new_edges.append((a2, new_id2))
                    new_edges.append((c1, new_id2))
                else:
                    new_edges.append((a2, new_id1))
                    new_edges.append((c1, new_id1))
                    new_edges.append((a1, new_id2))
                    new_edges.append((c2, new_id2))
                
                new_edges.append((new_id1, new_id2))
                
                # New leaf set and node count
                new_leaf_set = set(leaf_set)
                new_nc = node_count + 2
                
                # Build tree from new edges
                new_tree = _edges_to_tree(new_edges, new_leaf_set, labels)
                
                # Optimize branch lengths on new tree
                new_bl, new_ll = _optimize_bl(data, labels, new_tree, max_iter=20)
                
                if new_ll > best_move_ll + 1e-10:
                    best_move_ll = new_ll
                    best_new_edges = new_edges
                    best_new_tree = new_tree
                    best_new_bl = new_bl
        
        if best_new_edges is not None and best_move_ll > best_ll + 1e-10:
            best_tree = best_new_tree
            best_bl = best_new_bl
            best_ll = best_move_ll
        else:
            break
    
    return best_tree, best_bl, best_ll


def _tree_to_newick(tree, labels):
    if isinstance(tree, str):
        return tree
    parts = []
    for child in tree:
        parts.append(_tree_to_newick(child, labels))
    return "(" + ",".join(parts) + ")"


def _validate_newick(newick, labels):
    for lab in labels:
        if newick.count(lab) != 1:
            return False
    return True


def reconstruct(sequences):
    global _START_TIME
    _START_TIME = time.time()

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n <= 2:
        return "(" + ",".join(labels) + ");"
    if n == 3:
        return "(" + ",".join(labels) + ");"

    # Pre-encode sequences
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    n_sites = len(sequences[labels[0]])
    data = np.zeros((n, n_sites), dtype=np.int8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        for j, c in enumerate(s):
            data[i, j] = seq_map[c]

    # Step 1: NJ
    initial_tree = _neighbor_joining(sequences, labels)

    # Step 2: Optimize BLs
    bl, ll = _optimize_bl(data, labels, initial_tree, max_iter=50)

    best_tree = initial_tree
    best_ll = ll
    best_bl = bl

    # Step 3: NNI hill-climbing
    best_tree, best_bl, best_ll = _nni_improve(data, labels, best_tree, best_bl, max_rounds=100)

    # Step 4: Final BL optimization
    best_bl, best_ll = _optimize_bl(data, labels, best_tree, max_iter=30)

    newick = _tree_to_newick(best_tree, labels)

    if not _validate_newick(newick, labels):
        newick = "(" + ",".join(labels) + ")"

    return newick + ";"


if __name__ == "__main__":
    seqs = {
        't0': 'ACGTACGTACGTACGT',
        't1': 'ACGTACGTACGTACGT',
        't2': 'ACGTACGTACGTACGA',
        't3': 'ACGTACGTACGTACGT',
    }
    result = reconstruct(seqs)
    print(result)
