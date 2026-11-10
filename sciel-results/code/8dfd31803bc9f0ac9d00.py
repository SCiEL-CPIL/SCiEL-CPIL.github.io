import numpy as np
import math
import time
from collections import deque

def _jc_dist(a, b):
    diffs = sum(1 for x, y in zip(a, b) if x != y)
    p = diffs / len(a)
    if p >= 0.75:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def _nj(dist, labels):
    nodes = list(labels)
    trees = {name: name for name in labels}
    while len(nodes) > 3:
        n = len(nodes)
        r = {a: sum(dist[(a, b)] for b in nodes if b != a) for a in nodes}
        best_crit = float('inf')
        best_pair = None
        for i in range(n):
            for j in range(i+1, n):
                a, b = nodes[i], nodes[j]
                crit = (n - 2) * dist[(a, b)] - r[a] - r[b]
                if crit < best_crit:
                    best_crit = crit
                    best_pair = (a, b)
        a, b = best_pair
        joined = f"({trees[a]},{trees[b]})"
        new = f"n{len(nodes)}_{a}_{b}"
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            dist[(new, x)] = dist[(x, new)] = 0.5 * (dist[(a, x)] + dist[(b, x)] - dist[(a, b)])
        nodes = rest + [new]
        trees[new] = joined
    return "(" + ",".join(trees[x] for x in nodes) + ");"

def _parse_newick(newick):
    """Parse Newick to get adjacency list and branch lengths."""
    neighbors = {}
    branch_lengths = {}
    leaves = set()
    pos = 0
    
    def parse():
        nonlocal pos
        if newick[pos] == '(':
            pos += 1
            children = []
            while True:
                child = parse()
                children.append(child)
                if newick[pos] == ',':
                    pos += 1
                elif newick[pos] == ')':
                    pos += 1
                    break
            # Skip branch length
            bl = 0.0
            if pos < len(newick) and newick[pos] == ':':
                pos += 1
                bl_str = ''
                while pos < len(newick) and newick[pos] not in '(),;':
                    bl_str += newick[pos]
                    pos += 1
                try:
                    bl = float(bl_str)
                except:
                    bl = 0.0
            node = f"_int_{len(neighbors)}"
            neighbors[node] = set(children)
            for c in children:
                neighbors.setdefault(c, set()).add(node)
                branch_lengths[(node, c)] = bl
                branch_lengths[(c, node)] = bl
            return node
        else:
            name = ''
            while pos < len(newick) and newick[pos] not in '(),;':
                name += newick[pos]
                pos += 1
            # Skip branch length
            bl = 0.0
            if pos < len(newick) and newick[pos] == ':':
                pos += 1
                bl_str = ''
                while pos < len(newick) and newick[pos] not in '(),;':
                    bl_str += newick[pos]
                    pos += 1
                try:
                    bl = float(bl_str)
                except:
                    bl = 0.0
            neighbors.setdefault(name, set())
            leaves.add(name)
            return name
    
    newick = newick.rstrip(';')
    root = parse()
    return neighbors, branch_lengths, leaves, root

def _to_newick(neighbors, branch_lengths, leaves):
    """Convert adjacency list to Newick string."""
    internal_nodes = [n for n in neighbors if n not in leaves]
    if not internal_nodes:
        return ""
    root = internal_nodes[0]
    
    parent = {}
    children = {}
    visited = set()
    
    def build(node, par):
        parent[node] = par
        children[node] = []
        for nb in neighbors[node]:
            if nb != par and nb not in visited:
                visited.add(nb)
                children[node].append(nb)
                build(nb, node)
    
    visited.add(root)
    build(root, None)
    
    def to_string(node):
        if node in leaves:
            return node
        if len(children[node]) == 0:
            return f"({node})"
        child_strs = [to_string(c) for c in children[node]]
        return f"({','.join(child_strs)})"
    
    return to_string(root) + ";"

def _ml_likelihood(neighbors, branch_lengths, leaves, sequences):
    """Compute ML log-likelihood for the tree."""
    # Root the tree
    internal_nodes = [n for n in neighbors if n not in leaves]
    if not internal_nodes:
        return 0.0
    root = internal_nodes[0]
    
    # Build parent/children structure
    parent = {}
    children = {}
    visited = set()
    
    def build(node, par):
        parent[node] = par
        children[node] = []
        for nb in neighbors[node]:
            if nb != par and nb not in visited:
                visited.add(nb)
                children[node].append(nb)
                build(nb, node)
    
    visited.add(root)
    build(root, None)
    
    # Get sequences as numpy arrays
    seq_list = [sequences[label] for label in leaves]
    L = len(seq_list[0])
    # Map bases to indices
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    data = np.zeros((L, len(leaves)), dtype=int)
    for i, label in enumerate(leaves):
        for j, base in enumerate(sequences[label]):
            data[j, i] = base_map[base]
    
    # Post-order traversal to compute likelihood
    # We need to process nodes in post-order
    post_order = []
    def get_post_order(node):
        for child in children[node]:
            get_post_order(child)
        post_order.append(node)
    
    get_post_order(root)
    
    # Initialize vectors for leaves
    # vectors[node] is (L, 4)
    vectors = {}
    for node in post_order:
        if node in leaves:
            # One-hot encoding
            v = np.zeros((L, 4))
            # Find index of this leaf in data
            leaf_idx = list(leaves).index(node)
            v[np.arange(L), data[:, leaf_idx]] = 1.0
            vectors[node] = v
        else:
            # Sum over children
            v = np.zeros((L, 4))
            for child in children[node]:
                bl = branch_lengths.get((node, child), 0.0)
                if bl < 0: bl = 0.0
                # Compute transition matrix
                # P = 1/4 * [[1+3e, 1-e, 1-e, 1-e], ...]
                e = math.exp(-4.0 * bl / 3.0)
                # v_child is (L, 4)
                # v_child @ P
                # P is symmetric
                # v_new[i] = 1/4 * ( (1+3e)*v_child[i] + (1-e)*sum(v_child) )
                sum_v = np.sum(vectors[child], axis=1)
                v_child = vectors[child]
                v += (1.0/4.0) * ( (1.0 + 3.0*e) * v_child + (1.0 - e) * (sum_v[:, np.newaxis] - v_child) )
            vectors[node] = v
    
    # Likelihood at root
    # Assuming uniform base frequencies
    ll = np.sum(np.log(np.sum(vectors[root], axis=1) / 4.0))
    return ll

def _optimize_branch_lengths(neighbors, branch_lengths, leaves, sequences):
    """Optimize branch lengths using grid search."""
    # Get all edges
    edges = []
    seen = set()
    for u in neighbors:
        for v in neighbors[u]:
            if (u, v) not in seen:
                edges.append((u, v))
                seen.add((u, v))
                seen.add((v, u))
    
    # Grid search for each edge
    grid = np.linspace(0.0, 1.0, 21)
    for u, v in edges:
        best_ll = -float('inf')
        best_bl = 0.0
        for bl in grid:
            branch_lengths[(u, v)] = bl
            branch_lengths[(v, u)] = bl
            ll = _ml_likelihood(neighbors, branch_lengths, leaves, sequences)
            if ll > best_ll:
                best_ll = ll
                best_bl = bl
        branch_lengths[(u, v)] = best_bl
        branch_lengths[(v, u)] = best_bl
    return branch_lengths

def _nni_ml_improve(neighbors, branch_lengths, leaves, sequences, max_time=10.0):
    """NNI refinement with ML scoring."""
    start_time = time.time()
    best_ll = _ml_likelihood(neighbors, branch_lengths, leaves, sequences)
    improved = True
    while improved and (time.time() - start_time) < max_time:
        improved = False
        # Get internal edges
        internal_edges = []
        seen = set()
        for u in neighbors:
            for v in neighbors[u]:
                if (u, v) not in seen and (v, u) not in seen:
                    if u not in leaves and v not in leaves:
                        internal_edges.append((u, v))
                        seen.add((u, v))
                        seen.add((v, u))
        
        for u, v in internal_edges:
            u_nbs = list(neighbors[u])
            v_nbs = list(neighbors[v])
            a, b = [x for x in u_nbs if x != v]
            c, d = [x for x in v_nbs if x != u]
            
            # Move 1: Swap b and c
            # Original: u-a, u-b, u-v, v-c, v-d
            # New: u-a, u-c, u-v, v-b, v-d
            new_neighbors = {k: set(v) for k, v in neighbors.items()}
            new_bl = dict(branch_lengths)
            
            # Update neighbors
            new_neighbors[u].remove(b)
            new_neighbors[u].add(c)
            new_neighbors[b].remove(u)
            new_neighbors[b].add(v)
            new_neighbors[v].remove(c)
            new_neighbors[v].add(b)
            new_neighbors[c].remove(v)
            new_neighbors[c].add(u)
            
            # Update branch lengths (keep old values for now, optimize later)
            # The edges are (u,c), (v,b), (u,v)
            # We will optimize these 3 edges
            
            # Optimize the 3 affected edges
            edges_to_opt = [(u, c), (v, b), (u, v)]
            grid = np.linspace(0.0, 1.0, 11) # Coarser grid for speed
            for e_u, e_v in edges_to_opt:
                best_ll_e = -float('inf')
                best_bl_e = 0.0
                for bl in grid:
                    new_bl[(e_u, e_v)] = bl
                    new_bl[(e_v, e_u)] = bl
                    ll = _ml_likelihood(new_neighbors, new_bl, leaves, sequences)
                    if ll > best_ll_e:
                        best_ll_e = ll
                        best_bl_e = bl
                new_bl[(e_u, e_v)] = best_bl_e
                new_bl[(e_v, e_u)] = best_bl_e
            
            ll = _ml_likelihood(new_neighbors, new_bl, leaves, sequences)
            if ll > best_ll + 1e-6:
                neighbors = new_neighbors
                branch_lengths = new_bl
                best_ll = ll
                improved = True
                break # Restart sweep
                
    return _to_newick(neighbors, branch_lengths, leaves)

def reconstruct(sequences):
    start_time = time.time()
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    
    # Compute distance matrix
    dist = {}
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            d = _jc_dist(sequences[labels[i]], sequences[labels[j]])
            dist[(labels[i], labels[j])] = d
            dist[(labels[j], labels[i])] = d
    
    # NJ
    newick = _nj(dist, labels)
    
    # Parse
    neighbors, branch_lengths, leaves, root = _parse_newick(newick)
    
    elapsed = time.time() - start_time
    if elapsed > 200:
        return newick
    
    # Optimize branch lengths
    try:
        branch_lengths = _optimize_branch_lengths(neighbors, branch_lengths, leaves, sequences)
    except:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > 240:
        return _to_newick(neighbors, branch_lengths, leaves)
    
    # NNI refinement
    try:
        newick = _nni_ml_improve(neighbors, branch_lengths, leaves, sequences, max_time=240 - elapsed)
    except:
        pass
    
    return newick
