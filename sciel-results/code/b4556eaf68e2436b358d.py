"""Parsimony NNI search initialized with UPGMA for phylogenetic topology recovery."""
# EVOLVE-BLOCK-START
import math
import numpy as np

def _jc_distances(sequences, labels):
    """Compute Jukes-Cantor corrected distance matrix using numpy."""
    n = len(labels)
    seqs = [sequences[lab] for lab in labels]
    L = len(seqs[0])
    
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    arrs = []
    for s in seqs:
        arr = np.array([char_map[c] for c in s], dtype=np.uint8)
        arrs.append(arr)
    
    stacked = np.stack(arrs, axis=0)  # shape (n, L)
    
    dist_mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            diffs = np.sum(stacked[i] != stacked[j])
            p = diffs / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            dist_mat[i, j] = d
            dist_mat[j, i] = d
    
    return dist_mat, stacked

def _build_upgma(dist, labels):
    """
    Build a UPGMA tree.
    Returns:
    - tree: dict {node_id: (left, right)} for internal nodes, {node_id: label} for leaves
    - root_id: int
    """
    n = len(labels)
    if n == 1:
        return {0: labels[0]}, 0
    if n == 2:
        return {0: (labels[0], labels[1])}, 0
        
    active = list(range(n))
    clusters = {i: [i] for i in range(n)}
    d = {}
    for i in range(n):
        for j in range(i+1, n):
            d[(i, j)] = dist[i, j]
            d[(j, i)] = dist[i, j]
            
    tree = {i: labels[i] for i in range(n)}
    next_id = n
    
    while len(active) > 1:
        min_d = float('inf')
        i, j = -1, -1
        for a in range(len(active)):
            for b in range(a+1, len(active)):
                u, v = active[a], active[b]
                if d[(u, v)] < min_d:
                    min_d = d[(u, v)]
                    i, j = u, v
        
        new_id = next_id
        next_id += 1
        
        tree[new_id] = (i, j)
        clusters[new_id] = clusters[i] + clusters[j]
        
        active.remove(i)
        active.remove(j)
        active.append(new_id)
        
        for k in active:
            if k == new_id: continue
            size_i = len(clusters[i])
            size_j = len(clusters[j])
            total = size_i + size_j
            new_d = (size_i * d[(i, k)] + size_j * d[(j, k)]) / total
            d[(new_id, k)] = new_d
            d[(k, new_id)] = new_d
            del d[(i, k)]
            del d[(j, k)]
            
        del d[(i, j)]
        del d[(j, i)]
        
    return tree, active[0]

def _score_parsimony(tree, root_id, aln, label_to_idx):
    """
    Score the tree using Fitch's algorithm.
    """
    L = aln.shape[1]
    total_score = 0
    
    # Pre-compute post-order traversal
    post_order = []
    def dfs(u):
        if isinstance(tree[u], str):
            post_order.append(u)
        else:
            l, r = tree[u]
            dfs(l)
            dfs(r)
            post_order.append(u)
            
    dfs(root_id)
    
    for k in range(L):
        sets = {}
        cost = 0
        for u in post_order:
            if isinstance(tree[u], str):
                lab = tree[u]
                idx = label_to_idx[lab]
                sets[u] = {aln[idx, k]}
            else:
                l, r = tree[u]
                sl = sets[l]
                sr = sets[r]
                inter = sl & sr
                if inter:
                    sets[u] = inter
                else:
                    sets[u] = sl | sr
                    cost += 1
        total_score += cost
        
    return total_score

def _nni_search(tree, root_id, aln, label_to_idx, max_iterations=100):
    """
    Perform NNI search to improve parsimony score.
    """
    current_tree = {k: v for k, v in tree.items()}
    current_root = root_id
    current_score = _score_parsimony(current_tree, current_root, aln, label_to_idx)
    
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        nodes = list(current_tree.keys())
        
        parent = {}
        def build_parent_map(u, p):
            parent[u] = p
            if not isinstance(current_tree[u], str):
                l, r = current_tree[u]
                build_parent_map(l, u)
                build_parent_map(r, u)
                
        build_parent_map(current_root, -1)
        
        candidates = []
        for u in nodes:
            if isinstance(current_tree[u], str):
                continue
            l, r = current_tree[u]
            if not isinstance(current_tree[l], str):
                candidates.append((u, l))
            if not isinstance(current_tree[r], str):
                candidates.append((u, r))
                
        for u, v in candidates:
            l_u, r_u = current_tree[u]
            if l_u == v:
                w = r_u
            else:
                w = l_u
                
            x, y = current_tree[v]
            
            p_u = parent[u]
            
            for alt in [1, 2]:
                new_tree = {k: v for k, v in current_tree.items()}
                
                id_v1 = max(new_tree.keys()) + 1
                id_u1 = id_v1 + 1
                
                if alt == 1:
                    new_tree[id_v1] = (w, x)
                    new_tree[id_u1] = (id_v1, y)
                else:
                    new_tree[id_v1] = (w, y)
                    new_tree[id_u1] = (id_v1, x)
                    
                if p_u != -1:
                    l_p, r_p = new_tree[p_u]
                    if l_p == u:
                        new_tree[p_u] = (id_u1, r_p)
                    else:
                        new_tree[p_u] = (l_p, id_u1)
                    new_root = current_root
                else:
                    new_root = id_u1
                    
                new_score = _score_parsimony(new_tree, new_root, aln, label_to_idx)
                
                if new_score < current_score:
                    current_tree = new_tree
                    current_root = new_root
                    current_score = new_score
                    improved = True
                    break
            
            if improved:
                break
                
    def to_newick(u):
        if isinstance(current_tree[u], str):
            return current_tree[u]
        else:
            l, r = current_tree[u]
            return f"({to_newick(l)},{to_newick(r)})"
            
    return to_newick(current_root) + ";"

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 3:
        if n == 1:
            return labels[0] + ";"
        if n == 2:
            return f"({labels[0]},{labels[1]});"
        if n == 3:
            return f"({labels[0]},{labels[1]},{labels[2]});"
            
    dist, aln = _jc_distances(sequences, labels)
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    
    tree, root_id = _build_upgma(dist, labels)
    
    newick = _nni_search(tree, root_id, aln, label_to_idx)
    
    return newick
# EVOLVE-BLOCK-END
