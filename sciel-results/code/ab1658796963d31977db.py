import time
import numpy as np
import random

_DEADLINE = 240.0  # seconds wall-clock; evaluator kills at 270s

def _encode_sequences(sequences, labels):
    """Encode DNA sequences to integer array. A=0, C=1, G=2, T=3."""
    n = len(labels)
    seq_len = len(sequences[labels[0]])
    enc = np.zeros((n, seq_len), dtype=np.int8)
    # Mapping: A=65, C=67, G=71, T=84
    mapping = np.zeros(256, dtype=np.int8)
    mapping[65] = 0  # A
    mapping[67] = 1  # C
    mapping[71] = 2  # G
    mapping[84] = 3  # T
    
    for i, lab in enumerate(labels):
        s = sequences[lab]
        arr = np.frombuffer(s.encode('ascii'), dtype=np.int8)
        enc[i] = mapping[arr]
    return enc

def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC-corrected distance matrix using vectorized numpy."""
    n = len(labels)
    seq_len = len(sequences[labels[0]])
    enc = _encode_sequences(sequences, labels)
    
    D = np.zeros((n, n), dtype=np.float64)
    # Vectorized pairwise distance calculation
    # We can compute the number of mismatches for all pairs at once
    # enc is (n, L). We need (n, n) matrix of mismatches.
    # This is memory intensive for large n, but n<=32, so 32x32xL is fine.
    
    # For each pair (i, j), count where enc[i] != enc[j]
    # Using broadcasting: enc[:, np.newaxis, :] != enc[np.newaxis, :, :]
    # This creates a (n, n, L) boolean array.
    # For n=32, L=2000, size is 32*32*2000 = 2M bools = 2MB. Fine.
    
    diff = (enc[:, np.newaxis, :] != enc[np.newaxis, :, :])
    # Sum over the sequence axis (axis 2)
    num_diffs = np.sum(diff, axis=2)
    
    # Convert to proportions
    P = num_diffs / seq_len
    
    # Apply JC correction: d = -3/4 * ln(1 - 4/3 * p)
    # Handle p >= 0.75
    valid = P < 0.75
    D[valid] = -0.75 * np.log(1.0 - (4.0/3.0) * P[valid])
    D[~valid] = 3.0 # Large distance
    
    # Set diagonal to 0
    np.fill_diagonal(D, 0.0)
    
    return D

def _bionj(D, labels, deadline):
    """
    BIONJ (Neighbor Joining with variance correction) algorithm.
    Returns a Newick string.
    """
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    # D is the distance matrix
    # node_names: index -> name
    node_names = {i: labels[i] for i in range(n)}
    remaining = list(range(n))
    
    # We maintain a sub-matrix of distances for the remaining nodes
    # To avoid copying the full matrix, we can use the original D and just track remaining indices
    # But BIONJ requires updating distances for the new node.
    # We will maintain a dictionary of distances for the current remaining nodes.
    # dist[i][j] for i, j in remaining.
    
    # Initialize dist dict
    dist = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[(i, j)] = D[i, j]
    
    counter = n
    tree_edges = [] # List of (node1, node2) for internal connections? No, we build Newick directly.
    
    # We will build the tree by joining pairs and creating new nodes.
    # The final Newick is built by keeping track of the structure.
    # Since we are joining, we can just store the Newick string for the new node.
    
    while len(remaining) > 3:
        if time.time() > deadline - 10:
            # Fallback: star tree
            return "(" + ",".join(node_names[i] for i in remaining) + ")"
        
        m = len(remaining)
        
        # Compute r values: sum of distances from each node to all others in 'remaining'
        r = {i: 0.0 for i in remaining}
        for i in remaining:
            for j in remaining:
                if i != j:
                    r[i] += dist[(i, j)]
        
        # Find the pair (i, j) that minimizes the BIONJ Q criterion
        min_q = float('inf')
        min_i, min_j = remaining[0], remaining[1]
        
        # Precompute sum_dik_djk for each pair?
        # Q_ij = (m-2) d_ij - r_i - r_j - (2/(m-2)) * sum_{k != i,j} d_ik d_jk
        
        for idx_i in range(m):
            i = remaining[idx_i]
            for idx_j in range(idx_i + 1, m):
                j = remaining[idx_j]
                d_ij = dist[(i, j)]
                
                # Calculate sum_{k != i,j} d_ik d_jk
                sum_prod = 0.0
                for k in remaining:
                    if k == i or k == j:
                        continue
                    sum_prod += dist[(i, k)] * dist[(j, k)]
                
                q = (m - 2) * d_ij - r[i] - r[j] - (2.0 / (m - 2)) * sum_prod
                
                if q < min_q:
                    min_q = q
                    min_i = i
                    min_j = j
        
        # Join min_i and min_j
        new_node = counter
        counter += 1
        new_name = f"({node_names[min_i]},{node_names[min_j]})"
        node_names[new_node] = new_name
        
        # Compute distances from new node to all other remaining nodes
        # NJ formula: d(new, k) = 0.5 * (d(i, k) + d(j, k) - d(i, j))
        d_ij = dist[(min_i, min_j)]
        
        for k in remaining:
            if k == min_i or k == min_j:
                continue
            d_new_k = 0.5 * (dist[(min_i, k)] + dist[(min_j, k)] - d_ij)
            dist[(new_node, k)] = d_new_k
            dist[(k, new_node)] = d_new_k
        
        # Remove min_i and min_j from remaining, add new_node
        remaining = [x for x in remaining if x != min_i and x != min_j] + [new_node]
        
        # Clean up dist dict for removed nodes to save memory/time?
        # Not strictly necessary for n<=32, but good practice.
        # for key in list(dist.keys()):
        #     if min_i in key or min_j in key:
        #         del dist[key]
        
    # Return the final 3-taxon tree
    return f"({node_names[remaining[0]]},{node_names[remaining[1]]},{node_names[remaining[2]]})"

def _parse_newick(newick):
    """
    Parse a Newick string into an unrooted tree.
    Returns:
        adj: dict node_id -> set of neighbor node_ids
        leaf_to_node: dict label -> node_id
        node_to_label: dict node_id -> label (for leaves)
    """
    newick = newick.strip().rstrip(';')
    
    adj = {}
    leaf_to_node = {}
    node_to_label = {}
    node_counter = [0]
    
    def next_id():
        nid = node_counter[0]
        node_counter[0] += 1
        adj[nid] = set()
        return nid
    
    def parse(s, pos):
        if pos >= len(s):
            raise ValueError("Unexpected end")
        
        if s[pos] == '(':
            pos += 1
            children = []
            while True:
                child_id, pos = parse(s, pos)
                children.append(child_id)
                if pos < len(s) and s[pos] == ',':
                    pos += 1
                elif pos < len(s) and s[pos] == ')':
                    pos += 1
                    break
                else:
                    raise ValueError(f"Expected , or ) at pos {pos}")
            
            parent_id = next_id()
            for child_id in children:
                adj[parent_id].add(child_id)
                adj[child_id].add(parent_id)
            
            if pos < len(s) and s[pos] == ':':
                pos += 1
                while pos < len(s) and s[pos] not in ',)':
                    pos += 1
            
            return parent_id, pos
        else:
            name_end = pos
            while name_end < len(s) and s[name_end] not in '():,':
                name_end += 1
            name = s[pos:name_end]
            leaf_id = next_id()
            leaf_to_node[name] = leaf_id
            node_to_label[leaf_id] = name
            pos = name_end
            
            if pos < len(s) and s[pos] == ':':
                pos += 1
                while pos < len(s) and s[pos] not in ',)':
                    pos += 1
            
            return leaf_id, pos
    
    parse(newick, 0)
    
    return adj, leaf_to_node, node_to_label

def _fitch_parsimony_score(adj, leaf_to_node, node_to_label, enc):
    """
    Compute the Fitch parsimony score for a given unrooted tree.
    enc: (n_taxa, n_sites) array of int8
    """
    n_nodes = len(adj)
    # Root the tree arbitrarily at node 0 for Fitch algorithm
    # Fitch is unrooted, but we can root it to perform the bottom-up pass.
    
    # Create a rooted tree structure
    parent = {}
    children = {}
    visited = {0}
    queue = [0]
    parent[0] = -1
    children[0] = []
    
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                parent[v] = u
                children[u].append(v)
                if v not in children:
                    children[v] = []
                queue.append(v)
    
    # Post-order traversal
    post_order = []
    def dfs(u):
        for c in children.get(u, []):
            dfs(c)
        post_order.append(u)
    dfs(0)
    
    # Fitch bottom-up pass
    # state[u] is a set of possible states at node u
    # cost[u] is the parsimony cost for the subtree rooted at u
    
    n_sites = enc.shape[1]
    # We can process all sites in parallel using bitmasks or sets.
    # Since n_sites can be up to 2000, and n_nodes up to 32,
    # we can use a list of sets for each node.
    
    # Initialize leaves
    state = {}
    cost = {u: 0 for u in range(n_nodes)}
    
    # Map leaf node id to taxon index
    leaf_to_taxon = {}
    for label, nid in leaf_to_node.items():
        leaf_to_taxon[nid] = int(label[1:])
    
    for u in post_order:
        if u in node_to_label:
            # Leaf
            taxon_idx = leaf_to_taxon[u]
            # State is a list of sets, one per site
            # To be efficient, we can use a list of 4-bit integers or just sets.
            # For simplicity and speed in Python, we'll use a list of sets.
            # But 2000 sites * 32 nodes * set ops might be slow.
            # Let's use a list of bitmasks (int).
            # 0: A, 1: C, 2: G, 3: T
            # Bitmask: 1<<state
            states = []
            seq = enc[taxon_idx]
            for site in range(n_sites):
                states.append(1 << seq[site])
            state[u] = states
        else:
            # Internal node
            # Start with empty set (all bits 0)
            # Intersection of children's states
            # If intersection is empty, union of children's states + cost += 1
            
            # Initialize with all bits set (15) for intersection?
            # No, intersection starts as "any state" which is 15 (1111).
            # But we want the intersection of the sets of states of the children.
            
            # Let's use a list of bitmasks.
            # For each site, compute intersection of children's bitmasks.
            
            child_states = []
            for c in children.get(u, []):
                child_states.append(state[c])
            
            if not child_states:
                # Should not happen for internal node in a tree with >1 leaf
                state[u] = [15] * n_sites
                continue
            
            # Intersection
            inter = [15] * n_sites # 1111
            for cs in child_states:
                for site in range(n_sites):
                    inter[site] &= cs[site]
            
            # Check for empty intersection
            # If inter[site] == 0, then we need to add 1 to cost and take union
            union = [0] * n_sites
            for cs in child_states:
                for site in range(n_sites):
                    union[site] |= cs[site]
            
            final_states = [0] * n_sites
            node_cost = 0
            for site in range(n_sites):
                if inter[site] != 0:
                    final_states[site] = inter[site]
                else:
                    final_states[site] = union[site]
                    node_cost += 1
            
            state[u] = final_states
            cost[u] += node_cost
            
    return cost[0]

def _spr_search_parsimony(newick, sequences, labels, deadline):
    """
    Improve tree topology using SPR moves with Fitch Parsimony criterion.
    """
    start_time = time.time()
    remaining_time = deadline - start_time - 5 # Safety margin
    
    if remaining_time < 10:
        return newick
    
    adj, leaf_to_node, node_to_label = _parse_newick(newick)
    enc = _encode_sequences(sequences, labels)
    
    current_score = _fitch_parsimony_score(adj, leaf_to_node, node_to_label, enc)
    best_score = current_score
    best_adj = {k: set(v) for k, v in adj.items()}
    
    n_nodes = len(adj)
    edges = []
    for u in adj:
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    
    max_rounds = 5
    for round_num in range(max_rounds):
        if time.time() - start_time > remaining_time:
            break
        
        improved = False
        for u, v in edges:
            if time.time() - start_time > remaining_time:
                break
            
            # Prune v from u
            # Assume v is the child (we need to root the tree to determine parent/child)
            # For unrooted SPR, we can pick any edge, remove it, and regraft.
            
            # Remove edge (u, v)
            new_adj = {k: set(val) for k, val in adj.items()}
            new_adj[u].discard(v)
            new_adj[v].discard(u)
            
            # The subtree rooted at v (if we root at u) is the component containing v.
            # Let's find the component containing v in new_adj.
            comp_v = set()
            queue = [v]
            comp_v.add(v)
            while queue:
                node = queue.pop(0)
                for nei in new_adj[node]:
                    if nei not in comp_v:
                        comp_v.add(nei)
                        queue.append(nei)
            
            # Reattach v to any other edge (w, x) not in comp_v
            # To avoid duplicates and invalid moves, we iterate over all edges in the original tree
            # that are not (u, v) and not in comp_v.
            
            for w, x in edges:
                if (w == u and x == v) or (w == v and x == u):
                    continue
                if w in comp_v or x in comp_v:
                    continue
                
                # Try regrafting v to w
                # Remove edge (w, x)
                trial_adj = {k: set(val) for k, val in new_adj.items()}
                trial_adj[w].discard(x)
                trial_adj[x].discard(w)
                
                # Connect v to w
                trial_adj[v].add(w)
                trial_adj[w].add(v)
                
                # Check validity: should be a tree
                # n_edges = n_nodes - 1
                # Also check connectivity (implicitly done if we just moved a subtree)
                
                try:
                    new_score = _fitch_parsimony_score(trial_adj, leaf_to_node, node_to_label, enc)
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = trial_adj
                        improved = True
                except Exception:
                    pass
            
        if not improved:
            break
        
        # Update adj and edges for next round
        adj = {k: set(v) for k, v in best_adj.items()}
        edges = []
        for u in adj:
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
    
    # Convert best_adj back to Newick
    # We need a function to convert adjacency to Newick
    def adj_to_newick(adj, leaf_to_node, node_to_label):
        n_nodes = len(adj)
        if n_nodes == 1:
            return list(leaf_to_node.values())[0]
        
        # Root at 0
        parent = {0: -1}
        children = {0: []}
        visited = {0}
        queue = [0]
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    parent[v] = u
                    children[u].append(v)
                    if v not in children:
                        children[v] = []
                    queue.append(v)
        
        def build_newick(u):
            if u in node_to_label:
                return node_to_label[u]
            if not children[u]:
                return node_to_label.get(u, "?")
            
            child_strs = []
            for c in children[u]:
                child_strs.append(build_newick(c))
            
            return "(" + ",".join(child_strs) + ")"
        
        return build_newick(0)
    
    return adj_to_newick(best_adj, leaf_to_node, node_to_label)

def reconstruct(sequences):
    """
    Reconstruct unrooted phylogenetic tree topology.
    Returns Newick string over labels t0..t(n-1).
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n <= 3:
        if n == 1:
            return labels[0]
        elif n == 2:
            return f"({labels[0]},{labels[1]})"
        else:
            return f"({labels[0]},{labels[1]},{labels[2]})"

    # Compute distance matrix
    D = _compute_distance_matrix(sequences, labels)
    
    # Use BIONJ for topology reconstruction
    result = _bionj(D, labels, deadline)
    
    # Try to improve with SPR using Parsimony if time allows
    remaining = deadline - time.time()
    if remaining > 20:
        improved = _spr_search_parsimony(result, sequences, labels, deadline)
        if improved is not None:
            result = improved
    
    return result + ";"
