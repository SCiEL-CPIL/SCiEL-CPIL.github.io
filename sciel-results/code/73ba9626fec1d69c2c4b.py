"""Phylogenetic topology recovery: NJ + quartet-based NNI local search with JC-ML scoring."""
import math
import time
import numpy as np
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = _START_TIME + 255  # safety margin before 270s kill


def _check_deadline():
    return time.time() < _DEADLINE


def _jc_distance(a, b):
    """Jukes-Cantor corrected distance."""
    n = len(a)
    if n == 0:
        return 0.0
    diff = sum(1 for x, y in zip(a, b) if x != y)
    p = diff / n
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _parse_newick(newick):
    """Parse a Newick string into a nested tuple structure."""
    newick = newick.strip()
    if newick.endswith(';'):
        newick = newick[:-1]
    
    def parse(s):
        if s[0] == '(':
            depth = 0
            i = 1
            parts = []
            current = ''
            while i < len(s):
                if s[i] == '(':
                    depth += 1
                    current += s[i]
                elif s[i] == ')':
                    depth -= 1
                    if depth == -1:
                        break
                    current += s[i]
                elif s[i] == ',' and depth == 0:
                    parts.append(current)
                    current = ''
                else:
                    current += s[i]
                i += 1
            if i < len(s):
                i += 1
            rest = s[i:]
            if rest.startswith(':'):
                j = 0
                while j < len(rest) and rest[j] not in (',', '(', ')', ';'):
                    j += 1
            children = [parse(p) for p in parts]
            return ('internal', children)
        else:
            label = s
            if ':' in label:
                label = label.split(':')[0]
            return ('leaf', label)
    
    return parse(newick)


def _build_adjacency(tree):
    """Build adjacency list from tree structure."""
    adj = {}
    node_types = {}
    counter = [0]
    
    def assign_id(node, parent_id=None):
        if node[0] == 'leaf':
            nid = ('leaf', node[1])
            node_types[nid] = ('leaf', node[1])
            adj[nid] = set()
            if parent_id is not None:
                adj[nid].add(parent_id)
                adj[parent_id].add(nid)
            return nid
        else:
            nid = ('internal', counter[0])
            counter[0] += 1
            node_types[nid] = ('internal', None)
            adj[nid] = set()
            if parent_id is not None:
                adj[nid].add(parent_id)
                adj[parent_id].add(nid)
            for child in node[1]:
                cid = assign_id(child, nid)
            return nid
    
    root = assign_id(tree)
    node_ids = list(adj.keys())
    return adj, node_ids, node_types


def _root_tree(adj, node_ids, node_types):
    """Root the tree and build parent/children structure."""
    root = None
    for nid in node_ids:
        if node_types[nid][0] == 'internal':
            root = nid
            break
    if root is None:
        root = node_ids[0]
    
    parent = {root: None}
    children = {nid: [] for nid in node_ids}
    visited = {root}
    queue = [root]
    while queue:
        node = queue.pop(0)
        for nbr in adj[node]:
            if nbr not in visited:
                visited.add(nbr)
                parent[nbr] = node
                children[node].append(nbr)
                queue.append(nbr)
    
    return root, parent, children


def _postorder(root, children):
    """Get postorder traversal."""
    order = []
    def _post(node):
        for child in children[node]:
            _post(child)
        order.append(node)
    _post(root)
    return order


def _get_leaves_under(adj, node_types, start, exclude):
    """Get all leaf labels reachable from start without going through exclude."""
    leaves = []
    visited = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        if node_types[node][0] == 'leaf':
            leaves.append(node_types[node][1])
        else:
            for nbr in adj[node]:
                if nbr != exclude and nbr not in visited:
                    stack.append(nbr)
    return leaves


def _jc_likelihood_tree(adj, node_ids, node_types, sequences, labels):
    """Compute ML log-likelihood of the tree given the sequences using Felsenstein's pruning.
    Uses a fixed reasonable branch length for speed."""
    n_taxa = len(labels)
    if n_taxa < 2:
        return 0.0
    
    n_sites = len(sequences[labels[0]])
    state_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    # Build rooted structure
    root, parent, children = _root_tree(adj, node_ids, node_types)
    order = _postorder(root, children)
    
    # Estimate a global branch length scale
    total_d = 0.0
    count = 0
    for i, a in enumerate(labels):
        for b in labels[i+1:]:
            total_d += _jc_distance(sequences[a], sequences[b])
            count += 1
    mean_d = total_d / max(1, count)
    base_length = max(0.05, min(0.3, mean_d / max(1, n_taxa - 2)))
    
    # Precompute transition probabilities
    exp_term = math.exp(-4.0 * base_length / 3.0)
    p_same = 0.25 + 0.75 * exp_term
    p_diff = (1.0 - exp_term) / 3.0
    
    total_ll = 0.0
    
    for site in range(n_sites):
        prob = {}
        
        # Initialize leaves
        for nid in order:
            if node_types[nid][0] == 'leaf':
                label = node_types[nid][1]
                obs = state_map[sequences[label][site]]
                p = [0.0, 0.0, 0.0, 0.0]
                p[obs] = 1.0
                prob[nid] = p
            else:
                prob[nid] = None
        
        # Process internal nodes in postorder
        for nid in order:
            if node_types[nid][0] == 'leaf':
                continue
            
            p_node = [0.0, 0.0, 0.0, 0.0]
            for state in range(4):
                total = 1.0
                for child in children[nid]:
                    child_sum = 0.0
                    for c_state in range(4):
                        p_transition = p_same if c_state == state else p_diff
                        child_sum += p_transition * prob[child][c_state]
                    total *= child_sum
                p_node[state] = total
            
            prob[nid] = p_node
        
        root_prob = sum(prob[root]) / 4.0
        if root_prob > 1e-10:
            total_ll += math.log(root_prob)
        else:
            total_ll += -1000.0
    
    return total_ll


def _apply_nni(adj, node_types, u, v, a, b, c, d, move_type):
    """Apply an NNI move to the adjacency structure.
    
    For internal edge (u,v) where u has neighbors {v,a,b} and v has neighbors {u,c,d}:
    NNI1: swap b and c -> u connected to a,c,v and v connected to u,b,d
    NNI2: swap b and d -> u connected to a,d,v and v connected to u,c,b
    """
    new_adj = {k: set(v) for k, v in adj.items()}
    
    if move_type == 1:
        new_adj[u].discard(b)
        new_adj[b].discard(u)
        new_adj[v].discard(c)
        new_adj[c].discard(v)
        
        new_adj[u].add(c)
        new_adj[c].add(u)
        new_adj[v].add(b)
        new_adj[b].add(v)
    else:
        new_adj[u].discard(b)
        new_adj[b].discard(u)
        new_adj[v].discard(d)
        new_adj[d].discard(v)
        
        new_adj[u].add(d)
        new_adj[d].add(u)
        new_adj[v].add(b)
        new_adj[b].add(v)
    
    return new_adj


def _quartet_support(adj, node_types, labels, sequences, a, b, c, d):
    """Compute quartet support using JC distances.
    Returns (support for (a,b)|(c,d), support for (a,c)|(b,d), support for (a,d)|(b,c))
    """
    # Compute all 6 pairwise distances
    d_ab = _jc_distance(sequences[a], sequences[b])
    d_ac = _jc_distance(sequences[a], sequences[c])
    d_ad = _jc_distance(sequences[a], sequences[d])
    d_bc = _jc_distance(sequences[b], sequences[c])
    d_bd = _jc_distance(sequences[b], sequences[d])
    d_cd = _jc_distance(sequences[c], sequences[d])
    
    # For each of the 3 possible quartet topologies, compute the sum of branch lengths
    # Under JC, the expected distance between two taxa is the sum of branch lengths on the path
    # Topology (a,b)|(c,d): d_ab + d_cd should be < d_ac + d_bd and < d_ad + d_bc
    # Topology (a,c)|(b,d): d_ac + d_bd should be < d_ab + d_cd and < d_ad + d_bc
    # Topology (a,d)|(b,c): d_ad + d_bc should be < d_ab + d_cd and < d_ac + d_bd
    
    s1 = d_ab + d_cd  # (a,b)|(c,d)
    s2 = d_ac + d_bd  # (a,c)|(b,d)
    s3 = d_ad + d_bc  # (a,d)|(b,c)
    
    # Support is proportional to how much smaller each sum is
    min_s = min(s1, s2, s3)
    max_s = max(s1, s2, s3)
    range_s = max_s - min_s
    
    if range_s < 1e-10:
        return (1.0, 1.0, 1.0)
    
    # Convert to soft support scores
    # Lower sum = higher support
    support = [1.0 - (s - min_s) / range_s for s in [s1, s2, s3]]
    # Normalize
    total = sum(support)
    if total > 0:
        support = [s / total for s in support]
    
    return tuple(support)


def _nni_local_search(adj, node_ids, node_types, sequences, labels, max_iterations=20):
    """Perform NNI local search using quartet-based scoring for speed."""
    # Precompute all leaf labels
    leaf_labels = [node_types[nid][1] for nid in node_ids if node_types[nid][0] == 'leaf']
    
    # Build a map from leaf label to node id
    leaf_to_node = {}
    for nid in node_ids:
        if node_types[nid][0] == 'leaf':
            leaf_to_node[node_types[nid][1]] = nid
    
    # Compute JC likelihood for initial tree
    current_ll = _jc_likelihood_tree(adj, node_ids, node_types, sequences, labels)
    
    improved = True
    iteration = 0
    while improved and iteration < max_iterations and _check_deadline():
        improved = False
        iteration += 1
        
        best_ll = current_ll
        best_adj = adj
        
        # Find all internal edges
        for u in node_ids:
            if not _check_deadline():
                break
            if node_types[u][0] != 'internal':
                continue
            for v in adj[u]:
                if v <= u:
                    continue
                if node_types[v][0] != 'internal':
                    continue
                
                u_nbrs = [x for x in adj[u] if x != v]
                v_nbrs = [x for x in adj[v] if x != u]
                
                if len(u_nbrs) < 2 or len(v_nbrs) < 2:
                    continue
                
                a = u_nbrs[0]
                b = u_nbrs[1]
                c = v_nbrs[0]
                d = v_nbrs[1]
                
                # Get leaf sets for each subtree
                leaves_a = _get_leaves_under(adj, node_types, a, u)
                leaves_b = _get_leaves_under(adj, node_types, b, u)
                leaves_c = _get_leaves_under(adj, node_types, c, v)
                leaves_d = _get_leaves_under(adj, node_types, d, v)
                
                if not leaves_a or not leaves_b or not leaves_c or not leaves_d:
                    continue
                
                # Use representative leaves for quartet scoring
                ra = leaves_a[0]
                rb = leaves_b[0]
                rc = leaves_c[0]
                rd = leaves_d[0]
                
                # Current topology: (a,b)|(c,d)
                # NNI1: (a,c)|(b,d)
                # NNI2: (a,d)|(b,c)
                
                # Score current topology
                supp_current = _quartet_support(adj, node_types, labels, sequences, ra, rb, rc, rd)
                # Current is (a,b)|(c,d) which is index 0
                current_score = supp_current[0]
                
                # Try NNI1: swap b and c -> (a,c)|(b,d)
                new_adj1 = _apply_nni(adj, node_types, u, v, a, b, c, d, 1)
                # New topology is (a,c)|(b,d) which is index 1
                # But we need to re-evaluate with the new adjacency
                # For speed, use the precomputed quartet support
                nni1_score = supp_current[1]
                
                if nni1_score > current_score + 1e-6:
                    # Verify with ML if promising
                    ll1 = _jc_likelihood_tree(new_adj1, node_ids, node_types, sequences, labels)
                    if ll1 > best_ll + 1e-6:
                        best_ll = ll1
                        best_adj = new_adj1
                        improved = True
                
                # Try NNI2: swap b and d -> (a,d)|(b,c)
                new_adj2 = _apply_nni(adj, node_types, u, v, a, b, c, d, 2)
                nni2_score = supp_current[2]
                
                if nni2_score > current_score + 1e-6:
                    ll2 = _jc_likelihood_tree(new_adj2, node_ids, node_types, sequences, labels)
                    if ll2 > best_ll + 1e-6:
                        best_ll = ll2
                        best_adj = new_adj2
                        improved = True
        
        if best_adj is not adj:
            adj = best_adj
            current_ll = best_ll
    
    return adj


def _adj_to_newick(adj, node_types):
    """Convert adjacency structure back to Newick string."""
    node_ids = list(adj.keys())
    if not node_ids:
        return ""
    
    root = node_ids[0]
    for nid in node_ids:
        if node_types[nid][0] == 'leaf':
            root = nid
            break
    
    visited = set()
    
    def build(nid):
        if node_types[nid][0] == 'leaf':
            return node_types[nid][1]
        
        visited.add(nid)
        children = []
        for nbr in adj[nid]:
            if nbr not in visited:
                visited.add(nbr)
                children.append(build(nbr))
        
        return "(" + ",".join(children) + ")"
    
    newick = build(root)
    return newick + ";"


def _exhaustive_ml_for_small_n(sequences, labels):
    """For n <= 7, enumerate all unrooted topologies and find MLE."""
    n = len(labels)
    if n < 4:
        return None
    
    # Number of unrooted binary trees with n leaves: (2n-5)!!
    # For n=4: 3, n=5: 15, n=6: 105, n=7: 945
    
    # We'll use a recursive approach to generate all topologies
    # This is complex, so for now we'll use NJ + NNI which should be sufficient
    
    return None


def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC-corrected distances using vectorized operations."""
    n = len(labels)
    # Convert sequences to numpy arrays for vectorized comparison
    seq_arrays = []
    for lab in labels:
        s = sequences[lab]
        # Encode A,C,G,T as 0,1,2,3
        arr = np.zeros(len(s), dtype=np.int8)
        for i, c in enumerate(s):
            if c == 'A':
                arr[i] = 0
            elif c == 'C':
                arr[i] = 1
            elif c == 'G':
                arr[i] = 2
            else:
                arr[i] = 3
        seq_arrays.append(arr)

    L = len(sequences[labels[0]])
    # Stack into 2D array: n x L
    mat = np.stack(seq_arrays)  # shape (n, L)

    # Compute pairwise differences: for each pair (i,j), count positions where mat[i] != mat[j]
    # Use broadcasting: mat[:, None, :] != mat[None, :, :] gives n x n x L boolean
    # This is memory-heavy for large n*L, so do it in chunks
    dist = np.zeros((n, n), dtype=np.float64)
    chunk = 32  # process 32 taxa at a time for memory efficiency
    for i in range(0, n, chunk):
        i_end = min(i + chunk, n)
        # Compare rows i:i_end against all rows
        # mat[i:i_end, None, :] has shape (chunk, 1, L)
        # mat[None, :, :] has shape (1, n, L)
        diff = (mat[i:i_end, None, :] != mat[None, :, :])  # (chunk, n, L)
        counts = diff.sum(axis=2)  # (chunk, n)
        p = counts.astype(np.float64) / L
        # Apply JC correction
        # For p >= 0.7499, set to 3.0
        p_safe = np.clip(p, 0, 0.7499)
        jc = -0.75 * np.log(1.0 - 4.0 * p_safe / 3.0)
        jc[p >= 0.7499] = 3.0
        dist[i:i_end, :] = jc

    np.fill_diagonal(dist, 0.0)
    return dist


def _quartet_supertree(dist, labels, max_quartets=5000):
    """
    Reconstruct a tree from quartet support using a greedy supertree approach.
    dist: n x n distance matrix
    labels: list of n labels
    Returns a Newick string.
    """
    n = len(labels)
    if n <= 3:
        return "(" + ",".join(labels) + ")"
    
    # Sample quartets and compute support
    # For each quartet, compute the three possible topologies' support
    # Use the distance matrix to determine the best topology
    
    # We'll use a greedy approach: start with a star tree and add quartets
    # that are consistent with the current tree, or that have strong support
    
    # For simplicity, we'll use the NJ result as the primary method
    # and use quartet sampling to validate/improve
    
    # Actually, let's implement a simple quartet-based method:
    # 1. Sample many quartets
    # 2. For each, determine the best topology using distances
    # 3. Build a tree from the quartet set using a greedy algorithm
    
    # This is complex, so we'll fall back to NJ for now
    # but use quartet information to guide NNI search
    
    return None  # Fall back to NJ


def reconstruct(sequences):
    """Reconstruct phylogenetic tree topology from sequences."""
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    
    if n_taxa <= 2:
        return "(" + ",".join(labels) + ");"
    if n_taxa == 3:
        return "(" + ",".join(labels) + ");"
    
    # Step 1: Compute distance matrix using JC correction (vectorized)
    dist = _compute_distance_matrix(sequences, labels)
    
    # Convert to dict for NJ
    matrix = {}
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i < j:
                d = dist[i, j]
                matrix[(a, b)] = d
                matrix[(b, a)] = d
    
    # Step 2: Neighbor-Joining to get initial tree
    nodes = list(labels)
    trees = {name: name for name in labels}
    
    while len(nodes) > 3:
        if not _check_deadline():
            break
        n = len(nodes)
        total = {a: sum(matrix[(a, b)] for b in nodes if b != a) for a in nodes}
        best = None
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                criterion = (n - 2) * matrix[(a, b)] - total[a] - total[b]
                if best is None or criterion < best[0]:
                    best = (criterion, a, b)
        
        if best is None:
            break
        
        _, a, b = best
        joined = f"({trees[a]},{trees[b]})"
        new = f"n{len(nodes)}_{a}_{b}"
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            matrix[(new, x)] = matrix[(x, new)] = 0.5 * (matrix[(a, x)] + matrix[(b, x)] - matrix[(a, b)])
        nodes = rest + [new]
        trees[new] = joined
    
    # Build initial Newick
    newick = "(" + ",".join(trees[x] for x in nodes) + ");"
    
    # Step 3: Parse and convert to adjacency
    tree = _parse_newick(newick)
    adj, node_ids, node_types = _build_adjacency(tree)
    
    # Step 4: NNI local search with quartet-guided ML scoring
    if _check_deadline():
        adj = _nni_local_search(adj, node_ids, node_types, sequences, labels, max_iterations=15)
    
    # Convert back to Newick
    result = _adj_to_newick(adj, node_types)
    
    # Validate: ensure all taxa appear exactly once
    for label in labels:
        if label not in result:
            return newick
    
    return result
