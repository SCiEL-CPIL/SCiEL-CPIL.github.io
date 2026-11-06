import numpy as np
import scipy.optimize as opt
import time
import math

def _jc_distance(a, b):
    """Calculate Jukes-Cantor corrected distance."""
    diff = np.sum(a != b)
    p = diff / len(a)
    if p >= 0.75:
        return 3.0
    return -0.75 * math.log(1 - 4 * p / 3)

def _nj_with_lengths(sequences, labels):
    """Perform Neighbor Joining and return Newick string with branch lengths."""
    n = len(labels)
    if n == 1:
        return f"{labels[0]}:0.1;"
    if n == 2:
        return f"({labels[0]}:0.1,{labels[1]}:0.1);"
        
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
            D[i, j] = d
            D[j, i] = d
            
    nodes = list(range(n))
    tree_parts = {i: f"{labels[i]}:0.1" for i in range(n)}
    
    while len(nodes) > 2:
        n_nodes = len(nodes)
        r = np.zeros(n_nodes)
        for idx, i in enumerate(nodes):
            for j in nodes:
                if i != j:
                    r[idx] += D[i, j]
            
        best_criterion = float('inf')
        best_i, best_j = -1, -1
        for idx_i, i in enumerate(nodes):
            for idx_j in range(idx_i + 1, n_nodes):
                j = nodes[idx_j]
                q = (n_nodes - 2) * D[i, j] - r[idx_i] - r[idx_j]
                if q < best_criterion:
                    best_criterion = q
                    best_i, best_j = i, j
                    
        i, j = best_i, best_j
        
        # Calculate branch lengths
        l_i = 0.5 * D[i, j] + (r[idx_i] - r[idx_j]) / (2 * (n_nodes - 2))
        l_j = 0.5 * D[i, j] - (r[idx_i] - r[idx_j]) / (2 * (n_nodes - 2))
        
        if l_i < 0: l_i = 0.01
        if l_j < 0: l_j = 0.01
        
        k = max(nodes) + 1
        D[k, k] = 0
        for idx, x in enumerate(nodes):
            if x == i or x == j:
                continue
            D[k, x] = 0.5 * (D[i, x] + D[j, x] - D[i, j])
            D[x, k] = D[k, x]
            
        tree_parts[k] = f"({tree_parts[i]}:{l_i:.4f},{tree_parts[j]}:{l_j:.4f})"
        
        nodes = [x for x in nodes if x != i and x != j]
        nodes.append(k)
        
    if len(nodes) == 2:
        return f"({tree_parts[nodes[0]]},{tree_parts[nodes[1]]});"
    else:
        return tree_parts[nodes[0]] + ";"

def _parse_newick(newick):
    """
    Parses Newick string into a tree structure.
    Returns: nodes (list of dicts), root_id, label_map
    """
    newick = newick.strip()
    if newick.endswith(';'):
        newick = newick[:-1]
        
    nodes = []
    node_id_counter = 0
    stack = []
    label_map = {}
    
    def create_node(name, branch_len, parent=None):
        nonlocal node_id_counter
        nid = node_id_counter
        node_id_counter += 1
        node = {
            'id': nid,
            'name': name,
            'children': [],
            'parent': parent,
            'branch_len': branch_len
        }
        nodes.append(node)
        return nid

    i = 0
    while i < len(newick):
        char = newick[i]
        if char == '(':
            stack.append('OPEN')
            i += 1
        elif char == ',':
            i += 1
        elif char == ')':
            stack.append('CLOSE')
            i += 1
        elif char == ':':
            j = i + 1
            while j < len(newick) and newick[j] not in ',);':
                j += 1
            blen = float(newick[i+1:j])
            i = j
            if nodes:
                nodes[-1]['branch_len'] = blen
        else:
            j = i
            while j < len(newick) and newick[j] not in '(),:;':
                j += 1
            name = newick[i:j]
            i = j
            
            if stack and stack[-1] == 'CLOSE':
                stack.pop()
                # The last two nodes are children
                if len(nodes) >= 2:
                    child1 = nodes[-2]
                    child2 = nodes[-1]
                    parent_id = create_node(name, 0.1)
                    nodes[parent_id]['children'] = [child1['id'], child2['id']]
                    child1['parent'] = parent_id
                    child2['parent'] = parent_id
            else:
                nid = create_node(name, 0.1)
                label_map[name] = nid
                
    root_id = None
    for node in nodes:
        if node['parent'] is None:
            root_id = node['id']
            break
            
    return nodes, root_id, label_map

def _calc_likelihood(nodes, root_id, label_map, sequences):
    """Calculate log-likelihood for the tree using JC model."""
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    L = len(next(iter(sequences.values())))
    n_sites = L
    
    likelihoods = {}
    
    for label, nid in label_map.items():
        seq = sequences[label]
        vec = np.zeros((n_sites, 4))
        for s in range(n_sites):
            vec[s, seq_map[seq[s]]] = 1.0
        likelihoods[nid] = vec
        
    def process(node_id):
        node = nodes[node_id]
        if not node['children']:
            return
        
        child1 = node['children'][0]
        child2 = node['children'][1]
        
        process(child1)
        process(child2)
        
        L1 = likelihoods[child1]
        L2 = likelihoods[child2]
        
        l1 = nodes[child1]['branch_len']
        l2 = nodes[child2]['branch_len']
        
        def get_trans(l):
            exp_term = math.exp(-4 * l / 3)
            p_same = 0.25 + 0.75 * exp_term
            p_diff = 0.25 * (1 - exp_term)
            M = np.full((4, 4), p_diff)
            np.fill_diagonal(M, p_same)
            return M
            
        M1 = get_trans(l1)
        M2 = get_trans(l2)
        
        V1 = L1 @ M1.T
        V2 = L2 @ M2.T
        L_parent = V1 * V2
        
        likelihoods[node_id] = L_parent
        
    process(root_id)
    
    L_root = likelihoods[root_id]
    log_likelihood = np.sum(np.log(L_root.sum(axis=1) + 1e-300))
    
    return log_likelihood

def _optimize_branch_lengths(nodes, root_id, label_map, sequences):
    """Optimize branch lengths to maximize likelihood."""
    branch_ids = [n['id'] for n in nodes if n['id'] != root_id]
    
    def get_blens(x):
        for i, bid in enumerate(branch_ids):
            nodes[bid]['branch_len'] = x[i]
            
    def neg_log_likelihood(x):
        get_blens(x)
        ll = _calc_likelihood(nodes, root_id, label_map, sequences)
        return -ll
        
    x0 = np.array([nodes[bid]['branch_len'] for bid in branch_ids])
    
    try:
        res = opt.minimize(neg_log_likelihood, x0, method='L-BFGS-B', 
                           options={'maxiter': 50, 'ftol': 1e-4})
        get_blens(res.x)
        return -res.fun
    except Exception:
        return 0.0

def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 240
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    # 1. NJ
    newick = _nj_with_lengths(sequences, labels)
    
    # 2. Parse
    nodes, root_id, label_map = _parse_newick(newick)
    
    # 3. Optimize
    if time.time() < deadline - 10:
        _optimize_branch_lengths(nodes, root_id, label_map, sequences)
    
    return newick
