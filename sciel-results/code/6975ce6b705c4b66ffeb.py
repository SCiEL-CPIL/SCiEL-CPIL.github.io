# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_START = None
_DEADLINE_FRACTION = 0.85

def _now():
    return time.time()

def _init_time():
    global _START
    if _START is None:
        _START = time.time()

def _time_left():
    _init_time()
    return max(0.0, 270.0 * _DEADLINE_FRACTION - (time.time() - _START))

class TreeNode:
    def __init__(self, name=None, length=0.0):
        self.name = name
        self.length = length
        self.children = []
        self.parent = None
        self.id = None

    def add_child(self, child):
        child.parent = self
        self.children.append(child)
        return child

    def is_leaf(self):
        return len(self.children) == 0

    def to_newick(self):
        if self.is_leaf():
            s = self.name
        else:
            s = "(" + ",".join(c.to_newick() for c in self.children) + ")"
        if self.length > 0:
            s += f":{self.length:.4f}"
        return s

def _seqs_to_array(sequences):
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    seqs = [sequences[l] for l in labels]
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    arr = np.array([[mapping[c] for c in s] for s in seqs], dtype=np.int8)
    return labels, arr

def _jc_matrix(arr):
    n, L = arr.shape
    diff = (arr[:, None, :] != arr[None, :, :])
    counts = diff.sum(axis=2)
    p = counts / L
    p_safe = np.clip(p, 0, 0.7499)
    d = -0.75 * np.log(1 - (4.0/3.0) * p_safe)
    np.fill_diagonal(d, 0)
    return d

def _nj(dist, labels):
    n = len(labels)
    if n == 1:
        return TreeNode(labels[0])
    if n == 2:
        root = TreeNode()
        root.add_child(TreeNode(labels[0]))
        root.add_child(TreeNode(labels[1]))
        return root

    nodes = list(range(n))
    trees = {i: TreeNode(labels[i]) for i in range(n)}
    d = {}
    for i in range(n):
        d[i] = {}
        for j in range(n):
            d[i][j] = dist[i][j]
    
    while len(nodes) > 3:
        n_curr = len(nodes)
        total = {a: sum(d[a][b] for b in nodes if b != a) for a in nodes}
        
        best_criterion = float('inf')
        best_a, best_b = -1, -1
        
        for i, a in enumerate(nodes):
            for b in nodes[i+1:]:
                criterion = (n_curr - 2) * d[a][b] - total[a] - total[b]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_a, best_b = a, b
        
        a, b = best_a, best_b
        
        new_id = -1 - len(nodes)
        new_node = TreeNode()
        
        d_ab = d[a][b]
        sum_a = total[a]
        sum_b = total[b]
        
        l_a = 0.5 * (d_ab + (sum_a - sum_b) / (n_curr - 2))
        l_b = 0.5 * (d_ab - (sum_a - sum_b) / (n_curr - 2))
        
        if l_a < 0: l_a = 0.0
        if l_b < 0: l_b = 0.0
        
        trees[a].length = l_a
        trees[b].length = l_b
        new_node.add_child(trees[a])
        new_node.add_child(trees[b])
        trees[new_id] = new_node
        
        d[new_id] = {}
        for x in nodes:
            if x == a or x == b:
                continue
            d[new_id][x] = 0.5 * (d[a][x] + d[b][x] - d[a][b])
            d[x][new_id] = d[new_id][x]
            
        nodes.remove(a)
        nodes.remove(b)
        nodes.append(new_id)
        
    n1, n2, n3 = nodes
    d12 = d[n1][n2]
    d13 = d[n1][n3]
    d23 = d[n2][n3]
    
    l1 = 0.5 * (d12 + d13 - d23)
    l2 = 0.5 * (d12 + d23 - d13)
    l3 = 0.5 * (d13 + d23 - d12)
    
    if l1 < 0: l1 = 0.0
    if l2 < 0: l2 = 0.0
    if l3 < 0: l3 = 0.0
    
    center = TreeNode()
    trees[n1].length = l1
    trees[n2].length = l2
    trees[n3].length = l3
    center.add_child(trees[n1])
    center.add_child(trees[n2])
    center.add_child(trees[n3])
    
    return center

def _build_leaf_index(root):
    """Build a mapping from leaf name to index, and store leaf order."""
    leaf_names = []
    def collect(node):
        if node.is_leaf():
            leaf_names.append(node.name)
        else:
            for c in node.children:
                collect(c)
    collect(root)
    name_to_idx = {name: i for i, name in enumerate(leaf_names)}
    return leaf_names, name_to_idx

def _calc_likelihood(root, data, name_to_idx=None):
    """
    Calculate log-likelihood of the tree given the data under JC model.
    Vectorized over sites for speed.
    """
    if name_to_idx is None:
        _, name_to_idx = _build_leaf_index(root)
    
    L = data.shape[1]
    
    # Precompute JC transition probabilities for each branch length
    # We'll compute alpha bottom-up
    
    alpha = {}
    
    def traverse(node):
        if node.is_leaf():
            idx = name_to_idx[node.name]
            vec = np.zeros((4, L), dtype=np.float64)
            vec[data[idx], np.arange(L)] = 1.0
            alpha[node] = vec
        else:
            for c in node.children:
                traverse(c)
            
            length = node.children[0].length if node.children else 0.0
            # Actually each child has its own length
            sum_alpha = np.zeros((4, L), dtype=np.float64)
            for c in node.children:
                length = c.length
                if length <= 0:
                    sum_alpha += alpha[c]
                else:
                    u = math.exp(-4.0 * length / 3.0)
                    p_off = 0.25 * (1.0 - u)
                    p_on = 0.25 * (1.0 + 3.0 * u)
                    # Transition matrix P where P[i,j] = P(j->i)
                    # We need: sum_alpha[i,s] += sum_j alpha[c][j,s] * P[j,i]
                    # = sum_j alpha[c][j,s] * (p_on if j==i else p_off)
                    # = p_on * alpha[c][i,s] + p_off * sum_{j!=i} alpha[c][j,s]
                    # = p_on * alpha[c][i,s] + p_off * (sum_j alpha[c][j,s] - alpha[c][i,s])
                    # = (p_on - p_off) * alpha[c][i,s] + p_off * sum_j alpha[c][j,s]
                    # Note: p_on - p_off = 0.25*(1+3u) - 0.25*(1-u) = 0.25*4u = u
                    # So: u * alpha[c][i,s] + p_off * sum_j alpha[c][j,s]
                    child_sum = np.sum(alpha[c], axis=0)  # (L,)
                    sum_alpha += u * alpha[c] + p_off * child_sum[np.newaxis, :]
            
            alpha[node] = sum_alpha
        
    traverse(root)
    
    root_alpha = alpha[root]
    probs = np.sum(root_alpha, axis=0)
    probs = np.clip(probs, 1e-300, 1.0)
    log_lik = np.sum(np.log(probs))
    
    return log_lik

def _get_internal_edges(root):
    """Get all internal edges (edges where both endpoints are internal nodes)."""
    edges = []
    def traverse(node):
        if node.is_leaf():
            return
        for c in node.children:
            if not c.is_leaf():
                edges.append((node, c))
            traverse(c)
    traverse(root)
    return edges

def _nni_refine(root, data, name_to_idx=None, max_rounds=50):
    """
    Refine tree topology using NNI moves with likelihood evaluation.
    Returns the refined tree.
    """
    if name_to_idx is None:
        _, name_to_idx = _build_leaf_index(root)
    
    current_lik = _calc_likelihood(root, data, name_to_idx)
    
    for round_num in range(max_rounds):
        if _time_left() < 5.0:
            break
            
        improved = False
        edges = _get_internal_edges(root)
        
        for (parent, child) in edges:
            if _time_left() < 3.0:
                break
            
            # Find the other child of parent
            other_children = [c for c in parent.children if c is not child]
            if len(other_children) != 1:
                continue
            other = other_children[0]
            
            # Get the two children of child
            if len(child.children) != 2:
                continue
            c1, c2 = child.children
            
            # Save original lengths
            len_child = child.length
            len_c1 = c1.length
            len_c2 = c2.length
            len_other = other.length
            
            # NNI alternative 1: ((c1, other), c2) under parent
            # Structure:
            #   parent
            #   /    \
            #  child  c2
            #  /  \
            # c1   other
            
            # Save original state
            orig_parent_children = list(parent.children)
            orig_child_children = list(child.children)
            
            # Apply NNI 1
            child.children = [c1, other]
            c1.parent = child
            other.parent = child
            parent.children = [child, c2]
            c2.parent = parent
            # Keep lengths the same for now
            
            lik1 = _calc_likelihood(root, data, name_to_idx)
            
            if lik1 > current_lik + 1e-10:
                current_lik = lik1
                improved = True
                # Keep this topology
            else:
                # Revert
                child.children = [c1, c2]
                c1.parent = child
                c2.parent = child
                parent.children = [child, other]
                other.parent = parent
                # Revert lengths
                child.length = len_child
                c1.length = len_c1
                c2.length = len_c2
                other.length = len_other
            
            # NNI alternative 2: ((c2, other), c1) under parent
            # Structure:
            #   parent
            #   /    \
            #  child  c1
            #  /  \
            # c2   other
            
            child.children = [c2, other]
            c2.parent = child
            other.parent = child
            parent.children = [child, c1]
            c1.parent = parent
            
            lik2 = _calc_likelihood(root, data, name_to_idx)
            
            if lik2 > current_lik + 1e-10:
                current_lik = lik2
                improved = True
                # Keep this topology
            else:
                # Revert to original
                child.children = [c1, c2]
                c1.parent = child
                c2.parent = child
                parent.children = [child, other]
                other.parent = parent
                child.length = len_child
                c1.length = len_c1
                c2.length = len_c2
                other.length = len_other
        
        if not improved:
            break
    
    return root

def reconstruct(sequences):
    global _START
    _START = time.time()
    
    labels, arr = _seqs_to_array(sequences)
    n = len(labels)
    L = arr.shape[1]
    
    d_full = _jc_matrix(arr)
    
    # Generate candidates
    candidates = []
    
    # 1. Full data NJ
    tree_full = _nj(d_full, labels)
    candidates.append(tree_full)
    
    # 2. Bootstrap NJs
    num_bootstraps = 0
    max_bootstraps = 30  # Increased with faster likelihood
    
    while num_bootstraps < max_bootstraps and _time_left() > 15.0:
        indices = np.random.randint(0, L, L)
        arr_boot = arr[:, indices]
        d_boot = _jc_matrix(arr_boot)
        tree_boot = _nj(d_boot, labels)
        candidates.append(tree_boot)
        num_bootstraps += 1
    
    # Select best by likelihood, then apply NNI refinement
    best_tree = candidates[0]
    best_lik = -float('inf')
    
    for tree in candidates:
        if _time_left() < 10.0:
            break
        try:
            lik = _calc_likelihood(tree, arr)
            if lik > best_lik:
                best_lik = lik
                best_tree = tree
        except Exception:
            continue
    
    # Apply NNI refinement to the best tree
    try:
        best_tree = _nni_refine(best_tree, arr, max_rounds=30)
    except Exception:
        pass
    
    return best_tree.to_newick() + ";"

# EVOLVE-BLOCK-END
