import math
import time
import numpy as np
import re

_GLOBAL_START = None
_DEADLINE = 260.0  # Total budget for all instances (270s - margin)

def _elapsed():
    global _GLOBAL_START
    if _GLOBAL_START is None:
        _GLOBAL_START = time.time()
    return time.time() - _GLOBAL_START

def _time_exceeded():
    return _elapsed() > _DEADLINE

def _is_valid_newick(newick, labels):
    if not newick:
        return False
    found_labels = re.findall(r't\d+', newick)
    return set(found_labels) == set(labels)

def _parse_newick_to_tree(newick):
    """
    Parse Newick string into an internal tree representation.
    Returns:
      - adj: dict {node_id: [neighbors]}
      - labels: list where labels[i] is the taxon label for leaf nodes, None for internal
      - branch_lengths: dict {(u,v): length}
    """
    s = newick.strip()
    if s.endswith(';'):
        s = s[:-1]
    
    node_counter = [0]
    adj = {}
    labels = []
    branch_lengths = {}
    
    def create_node(label=None):
        nid = node_counter[0]
        node_counter[0] += 1
        adj[nid] = []
        labels.append(label)
        return nid
    
    def parse(pos):
        if s[pos] == '(':
            pos += 1
            internal_id = create_node(None)
            first_child = None
            while True:
                child_id, pos = parse(pos)
                if first_child is None:
                    first_child = child_id
                else:
                    adj[internal_id].append(child_id)
                    adj[child_id].append(internal_id)
                
                if pos < len(s) and s[pos] == ',':
                    pos += 1
                    continue
                elif pos < len(s) and s[pos] == ')':
                    pos += 1
                    break
                else:
                    break
            
            if first_child is not None:
                adj[internal_id].append(first_child)
                adj[first_child].append(internal_id)
            
            # Skip branch length
            if pos < len(s) and s[pos] == ':':
                pos += 1
                start_len = pos
                while pos < len(s) and s[pos] not in ',;)':
                    pos += 1
                len_str = s[start_len:pos]
                try:
                    bl = float(len_str)
                except:
                    bl = 0.0
                # Store branch length for the edge connecting this internal node to its parent
                # We need to know the parent. 
                # The caller will add the edge.
                # So we return the branch length as well?
                # Let's modify return to include branch length to parent.
                return internal_id, pos, bl
            return internal_id, pos, 0.0
        else:
            end = pos
            while end < len(s) and s[end] not in ',:;)':
                end += 1
            label = s[pos:end]
            leaf_id = create_node(label)
            pos = end
            # Skip branch length
            if pos < len(s) and s[pos] == ':':
                pos += 1
                start_len = pos
                while pos < len(s) and s[pos] not in ',;)':
                    pos += 1
                len_str = s[start_len:pos]
                try:
                    bl = float(len_str)
                except:
                    bl = 0.0
                return leaf_id, pos, bl
            return leaf_id, pos, 0.0
    
    # We need to handle the root branch length if any, but unrooted trees usually don't have a root branch.
    # The parse function returns (id, pos, bl_to_parent).
    # The top-level call returns the root ID and its branch length (which should be ignored or 0).
    root_id, _, _ = parse(0)
    
    # Now we need to assign branch lengths to edges.
    # The `bl` returned by `parse` is the length of the edge from the node to its parent.
    # We didn't store the parent-child relationship with lengths directly in `adj`.
    # We need to re-parse or modify `parse` to store edges with lengths.
    
    # Let's re-implement parse to store edges.
    # This is getting messy. Let's use a simpler representation.
    # We'll store edges as a list of (u, v, length).
    
    pass # This function needs to be rewritten properly.

def _jc_distance_matrix(sequences, labels):
    n = len(labels)
    L = len(sequences[labels[0]])
    char_to_idx = {c: i for i, c in enumerate('ACGT')}
    seqs = np.zeros((n, L), dtype=np.int8)
    for i, l in enumerate(labels):
        s = sequences[l]
        for j, c in enumerate(s):
            seqs[i, j] = char_to_idx[c]
    
    diff = (seqs[:, None, :] != seqs[None, :, :]).sum(axis=2)
    p = diff.astype(np.float64) / L
    p_safe = np.minimum(p, 0.7499)
    d = -0.75 * np.log(1.0 - 4.0 * p_safe / 3.0)
    np.fill_diagonal(d, 0.0)
    d = (d + d.T) * 0.5
    return d

def _nj_tree(labels, dist):
    n = len(labels)
    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    d = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                d[(i, j)] = dist[i][j]
    
    next_id = n
    while len(nodes) > 3:
        m = len(nodes)
        r = {}
        for a in nodes:
            r[a] = sum(d[(a, b)] for b in nodes if b != a)
        
        best_crit = None
        best_pair = None
        for i in range(m):
            for j in range(i + 1, m):
                a, b = nodes[i], nodes[j]
                crit = (m - 2) * d[(a, b)] - r[a] - r[b]
                if best_crit is None or crit < best_crit:
                    best_crit = crit
                    best_pair = (a, b)
        
        a, b = best_pair
        da = 0.5 * d[(a, b)]
        db = 0.5 * d[(a, b)]
        for x in nodes:
            if x != a and x != b:
                da += 0.5 * (d[(a, x)] - d[(b, x)])
                db += 0.5 * (d[(b, x)] - d[(a, x)])
        
        da = max(da, 0.0)
        db = max(db, 0.0)
        
        new = next_id
        next_id += 1
        rest = [x for x in nodes if x not in (a, b)]
        
        for x in rest:
            d[(new, x)] = d[(x, new)] = 0.5 * (d[(a, x)] + d[(b, x)] - d[(a, b)])
            
        for x in rest:
            del d[(a, x)]
            del d[(b, x)]
            del d[(x, a)]
            del d[(x, b)]
        
        del d[(a, b)]
        del d[(b, a)]
        
        nodes = rest + [new]
        trees[new] = f"({trees[a]}:{da},{trees[b]}:{db})"
    
    return "(" + ",".join(trees[x] for x in nodes) + ");"

def _parse_newick_with_lengths(newick):
    """
    Parses Newick into adj, node_labels, and edge_lengths.
    edge_lengths is a dict {(u,v): length}
    """
    s = newick.strip()
    if s.endswith(';'):
        s = s[:-1]
    
    node_counter = [0]
    adj = {}
    node_labels = []
    edge_lengths = {}
    
    def create_node(label=None):
        nid = node_counter[0]
        node_counter[0] += 1
        adj[nid] = []
        node_labels.append(label)
        return nid
    
    def parse(pos, parent_id=None):
        if s[pos] == '(':
            pos += 1
            internal_id = create_node(None)
            if parent_id is not None:
                # We don't know the length yet, it comes after the closing paren
                pass
            first_child = None
            while True:
                child_id, child_pos, child_bl = parse(pos, internal_id)
                if first_child is None:
                    first_child = child_id
                else:
                    adj[internal_id].append(child_id)
                    adj[child_id].append(internal_id)
                
                # Store length
                if child_bl is not None:
                    edge = (internal_id, child_id) if internal_id < child_id else (child_id, internal_id)
                    edge_lengths[edge] = child_bl
                
                if child_pos < len(s) and s[child_pos] == ',':
                    pos = child_pos + 1
                    continue
                elif child_pos < len(s) and s[child_pos] == ')':
                    pos = child_pos + 1
                    break
                else:
                    pos = child_pos
                    break
            
            if first_child is not None:
                adj[internal_id].append(first_child)
                adj[first_child].append(internal_id)
            
            # Skip branch length of this internal node to its parent
            if pos < len(s) and s[pos] == ':':
                pos += 1
                start_len = pos
                while pos < len(s) and s[pos] not in ',;)':
                    pos += 1
                len_str = s[start_len:pos]
                try:
                    bl = float(len_str)
                except:
                    bl = 0.0
                return internal_id, pos, bl
            return internal_id, pos, 0.0
        else:
            end = pos
            while end < len(s) and s[end] not in ',:;)':
                end += 1
            label = s[pos:end]
            leaf_id = create_node(label)
            pos = end
            # Skip branch length
            if pos < len(s) and s[pos] == ':':
                pos += 1
                start_len = pos
                while pos < len(s) and s[pos] not in ',;)':
                    pos += 1
                len_str = s[start_len:pos]
                try:
                    bl = float(len_str)
                except:
                    bl = 0.0
                return leaf_id, pos, bl
            return leaf_id, pos, 0.0
    
    root_id, _, root_bl = parse(0)
    return adj, node_labels, edge_lengths

def _jc_log_likelihood(adj, node_labels, edge_lengths, sequences, labels):
    """
    Calculate log-likelihood of the tree under JC model.
    """
    n_nodes = len(node_labels)
    if n_nodes == 0:
        return 0.0
    
    # Find a leaf to root the tree
    root = None
    for i in range(n_nodes):
        if node_labels[i] is not None:
            root = i
            break
    if root is None:
        root = 0
        
    # Build parent and order
    parent = {root: -1}
    order = []
    stack = [root]
    visited = {root}
    while stack:
        u = stack.pop()
        order.append(u)
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                parent[v] = u
                stack.append(v)
    
    # Reverse order for post-order traversal
    post_order = order[::-1]
    
    L = len(sequences[labels[0]])
    total_ll = 0.0
    
    # Precompute transition probabilities
    # We need P(d) for each edge.
    # P(d) = 1/4 + 3/4 * exp(-4d/3)
    # Q(d) = 1/4 * (1 - exp(-4d/3))
    
    # For each site
    for l in range(L):
        # lp[node][state]
        # state: 0=A, 1=C, 2=G, 3=T
        lp = {}
        
        for u in post_order:
            if node_labels[u] is not None:
                # Leaf
                char = sequences[node_labels[u]][l]
                state_idx = {'A':0, 'C':1, 'G':2, 'T':3}[char]
                lp[u] = [0.0, 0.0, 0.0, 0.0]
                lp[u][state_idx] = 1.0
            else:
                # Internal
                # Get children
                children = [v for v in adj[u] if parent[v] == u]
                
                # Initialize lp[u] to 0
                lp_u = [0.0, 0.0, 0.0, 0.0]
                
                for v in children:
                    # Get branch length
                    edge = (u, v) if u < v else (v, u)
                    bl = edge_lengths.get(edge, 0.0)
                    
                    # Calculate P and Q
                    exp_term = math.exp(-4.0 * bl / 3.0)
                    P = 0.25 + 0.75 * exp_term
                    Q = 0.25 * (1.0 - exp_term)
                    
                    lp_v = lp[v]
                    
                    # New lp_u[s] += sum_t lp_v[t] * Trans(s, t)
                    # Trans(s, s) = P, Trans(s, t!=s) = Q
                    
                    # Optimization:
                    # lp_u[s] = sum_t lp_v[t] * (P if t==s else Q)
                    # = lp_v[s]*P + sum_{t!=s} lp_v[t]*Q
                    # = lp_v[s]*P + (sum_t lp_v[t] - lp_v[s])*Q
                    # = lp_v[s]*(P-Q) + sum_t lp_v[t]*Q
                    
                    sum_lp_v = sum(lp_v)
                    p_minus_q = P - Q
                    q = Q
                    
                    for s in range(4):
                        lp_u[s] += lp_v[s] * p_minus_q + sum_lp_v * q
                
                lp[u] = lp_u
        
        # Root likelihood
        lp_root = lp[root]
        total_ll += math.log(sum(lp_root))
        
    return total_ll

def _nni_refine(newick, sequences, labels):
    adj, node_labels, edge_lengths = _parse_newick_with_lengths(newick)
    
    current_ll = _jc_log_likelihood(adj, node_labels, edge_lengths, sequences, labels)
    
    improved = True
    max_rounds = 50
    round_num = 0
    
    while improved and round_num < max_rounds:
        improved = False
        round_num += 1
        if _time_exceeded():
            break
            
        best_ll = current_ll
        best_adj = None
        best_node_labels = None
        best_edge_lengths = None
        
        # Generate NNI moves
        # Identify internal edges
        # An edge (u,v) is internal if both u and v are internal nodes (or one is internal and the other is internal? No, unrooted tree internal nodes have degree >= 3 usually, but in a binary tree, internal nodes have degree 3).
        # In a binary unrooted tree, all internal nodes have degree 3.
        # So we look for edges where both endpoints are internal nodes.
        
        n_nodes = len(node_labels)
        internal_nodes = [i for i in range(n_nodes) if node_labels[i] is None]
        
        for u in internal_nodes:
            for v in adj[u]:
                if node_labels[v] is None and u < v:
                    # Edge (u,v) is an internal edge
                    # Get other neighbors
                    # u's neighbors: v, a, b
                    # v's neighbors: u, c, d
                    
                    u_neighbors = [x for x in adj[u] if x != v]
                    v_neighbors = [x for x in adj[v] if x != u]
                    
                    if len(u_neighbors) != 2 or len(v_neighbors) != 2:
                        continue # Not a binary tree or not an NNI candidate
                    
                    a, b = u_neighbors
                    c, d = v_neighbors
                    
                    # Current topology: (a, b) at u, (c, d) at v, connected by (u,v)
                    # NNI 1: (a, c) at u, (b, d) at v
                    # NNI 2: (a, d) at u, (b, c) at v
                    
                    for swap_type in [1, 2]:
                        # Create new adj
                        new_adj = {x: list(y) for x, y in adj.items()}
                        new_edge_lengths = dict(edge_lengths)
                        
                        # Remove u-v, u-a, u-b, v-c, v-d
                        # Add u-c, u-d (or u-b, u-c), v-a, v-b
                        
                        # Let's define the swaps clearly.
                        # Original: u-a, u-b, v-c, v-d, u-v
                        # Swap 1: u-c, u-d, v-a, v-b, u-v
                        # Swap 2: u-d, u-c, v-b, v-a, u-v (Same as Swap 1? No.)
                        
                        # Let's trace the leaves.
                        # Leaves attached to u: a, b.
                        # Leaves attached to v: c, d.
                        # NNI moves change the pairing.
                        # Current: (a,b) and (c,d) are the "groups" hanging from u and v.
                        # New 1: (a,c) and (b,d).
                        # New 2: (a,d) and (b,c).
                        
                        # Implementation:
                        # Remove edges: (u,a), (u,b), (v,c), (v,d)
                        # Add edges:
                        if swap_type == 1:
                            # u connects to c, d. v connects to a, b.
                            new_adj[u].remove(a)
                            new_adj[a].remove(u)
                            new_adj[u].remove(b)
                            new_adj[b].remove(u)
                            new_adj[v].remove(c)
                            new_adj[c].remove(v)
                            new_adj[v].remove(d)
                            new_adj[d].remove(v)
                            
                            new_adj[u].append(c)
                            new_adj[c].append(u)
                            new_adj[u].append(d)
                            new_adj[d].append(u)
                            new_adj[v].append(a)
                            new_adj[a].append(v)
                            new_adj[v].append(b)
                            new_adj[b].append(v)
                            
                            # Update edge lengths
                            # We keep the length of u-v.
                            # We keep the lengths of the hanging edges?
                            # The edge (u,a) becomes (v,a)? No, (u,a) is removed, (v,a) is added.
                            # We should keep the branch length associated with the leaf/subtree.
                            # So len(v,a) = len(u,a), len(v,b) = len(u,b), len(u,c) = len(v,c), len(u,d) = len(v,d).
                            
                            # Helper to get/set length
                            def get_len(n1, n2):
                                e = (n1, n2) if n1 < n2 else (n2, n1)
                                return new_edge_lengths.get(e, 0.0)
                            
                            def set_len(n1, n2, l):
                                e = (n1, n2) if n1 < n2 else (n2, n1)
                                new_edge_lengths[e] = l
                            
                            la = get_len(u, a)
                            lb = get_len(u, b)
                            lc = get_len(v, c)
                            ld = get_len(v, d)
                            
                            # Remove old
                            del new_edge_lengths[(u,a) if u<a else (a,u)]
                            del new_edge_lengths[(u,b) if u<b else (b,u)]
                            del new_edge_lengths[(v,c) if v<c else (c,v)]
                            del new_edge_lengths[(v,d) if v<d else (d,v)]
                            
                            # Add new
                            set_len(v, a, la)
                            set_len(v, b, lb)
                            set_len(u, c, lc)
                            set_len(u, d, ld)
                            
                        else:
                            # u connects to d, c. v connects to b, a.
                            new_adj[u].remove(a)
                            new_adj[a].remove(u)
                            new_adj[u].remove(b)
                            new_adj[b].remove(u)
                            new_adj[v].remove(c)
                            new_adj[c].remove(v)
                            new_adj[v].remove(d)
                            new_adj[d].remove(v)
                            
                            new_adj[u].append(d)
                            new_adj[d].append(u)
                            new_adj[u].append(c)
                            new_adj[c].append(u)
                            new_adj[v].append(b)
                            new_adj[b].append(v)
                            new_adj[v].append(a)
                            new_adj[a].append(v)
                            
                            def get_len(n1, n2):
                                e = (n1, n2) if n1 < n2 else (n2, n1)
                                return new_edge_lengths.get(e, 0.0)
                            
                            def set_len(n1, n2, l):
                                e = (n1, n2) if n1 < n2 else (n2, n1)
                                new_edge_lengths[e] = l
                            
                            la = get_len(u, a)
                            lb = get_len(u, b)
                            lc = get_len(v, c)
                            ld = get_len(v, d)
                            
                            del new_edge_lengths[(u,a) if u<a else (a,u)]
                            del new_edge_lengths[(u,b) if u<b else (b,u)]
                            del new_edge_lengths[(v,c) if v<c else (c,v)]
                            del new_edge_lengths[(v,d) if v<d else (d,v)]
                            
                            set_len(v, b, lb)
                            set_len(v, a, la)
                            set_len(u, c, lc)
                            set_len(u, d, ld)
                        
                        # Calculate likelihood
                        ll = _jc_log_likelihood(new_adj, node_labels, new_edge_lengths, sequences, labels)
                        
                        if ll > best_ll + 1e-10:
                            best_ll = ll
                            best_adj = new_adj
                            best_node_labels = node_labels
                            best_edge_lengths = new_edge_lengths
                            improved = True
        
        if best_adj is not None:
            adj = best_adj
            current_ll = best_ll
            node_labels = best_node_labels
            edge_lengths = best_edge_lengths
            
    # Convert back to Newick
    return _adj_to_newick(adj, node_labels, edge_lengths)

def _adj_to_newick(adj, node_labels, edge_lengths):
    n_nodes = len(node_labels)
    if n_nodes == 0:
        return ""
        
    # Find a leaf to root
    root = None
    for i in range(n_nodes):
        if node_labels[i] is not None:
            root = i
            break
    if root is None:
        root = 0
        
    visited = set()
    
    def gen(node, parent):
        if node in visited:
            return ""
        visited.add(node)
        
        if node_labels[node] is not None:
            # Leaf
            # Get branch length to parent
            if parent != -1:
                edge = (node, parent) if node < parent else (parent, node)
                bl = edge_lengths.get(edge, 0.0)
                return f"{node_labels[node]}:{bl}"
            else:
                return node_labels[node]
        else:
            children = []
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                child_str = gen(neighbor, node)
                if child_str:
                    children.append(child_str)
            
            if len(children) == 0:
                return ""
            elif len(children) == 1:
                # Unrooted tree, so this shouldn't happen for internal nodes in a binary tree rooted at a leaf?
                # Actually, if we root at a leaf, internal nodes have 2 children.
                # But if the tree is not binary or we have a "cherry" structure?
                # In a binary unrooted tree, internal nodes have degree 3.
                # Rooted at a leaf, the root leaf has 1 child.
                # That child (internal) has 2 children (one is the rest of the tree, one is... wait).
                # Let's trace:
                # Leaf R - Internal I1 - Internal I2 ...
                # I1 is connected to R, I2, and Leaf L1.
                # When rooting at R, I1's children are I2 and L1.
                # So len(children) should be 2.
                # If len(children) == 1, it might be a non-binary node or error.
                # Let's just return it.
                return "(" + children[0] + ")"
            else:
                return "(" + ",".join(children) + ")"
                
    s = gen(root, -1)
    return s + ";"

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        return "(" + ",".join(labels) + ");"
        
    dist = _jc_distance_matrix(sequences, labels)
    
    nj_newick = _nj_tree(labels, dist)
    
    if not _is_valid_newick(nj_newick, labels):
        return "(" + ",".join(labels) + ");"
        
    newick = nj_newick
    
    if not _time_exceeded():
        try:
            refined_newick = _nni_refine(nj_newick, sequences, labels)
            if _is_valid_newick(refined_newick, labels):
                newick = refined_newick
        except Exception:
            pass
            
    if not _is_valid_newick(newick, labels):
        return "(" + ",".join(labels) + ");"
        
    return newick
