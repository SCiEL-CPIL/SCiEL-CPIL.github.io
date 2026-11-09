"""Rewritten program using NumPy-accelerated Neighbor-Joining with NNI optimization."""
import math
import time
import numpy as np

DEADLINE_SECONDS = 270.0
SAFETY_MARGIN = 10.0

def _jc_distances(sequences, labels):
    """Compute pairwise Jukes-Cantor distances using NumPy."""
    n = len(labels)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    
    L = len(sequences[labels[0]])
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    # Convert all sequences to numpy arrays at once for speed
    seq_array = np.zeros((n, L), dtype=np.int8)
    for i, lab in enumerate(labels):
        seq = sequences[lab]
        for j, char in enumerate(seq):
            seq_array[i, j] = char_map.get(char, 0)
            
    # Compute Hamming distances vectorized
    dist_matrix = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        for j in range(i + 1, n):
            diffs = np.sum(seq_array[i] != seq_array[j])
            p = diffs / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    return dist_matrix

def _neighbor_joining(dist_matrix, labels):
    """Standard Neighbor-Joining returning Newick string."""
    n = len(labels)
    if n <= 2:
        return f"({','.join(labels)});"
    
    trees = {label: label for label in labels}
    dist_dict = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_dict[(labels[i], labels[j])] = dist_matrix[i, j]
                
    active = list(labels)
    next_id = 0
    
    while len(active) > 3:
        current_n = len(active)
        
        r = {node: 0.0 for node in active}
        for a in active:
            for b in active:
                if a != b:
                    r[a] += dist_dict[(a, b)]
                    
        best_criterion = float('inf')
        best_a = None
        best_b = None
        best_d = 0
        
        for i, a in enumerate(active):
            for b in active[i + 1:]:
                d_ab = dist_dict[(a, b)]
                criterion = (current_n - 2) * d_ab - r[a] - r[b]
                
                if criterion < best_criterion - 1e-9:
                    best_criterion = criterion
                    best_a = a
                    best_b = b
                    best_d = d_ab
                elif abs(criterion - best_criterion) < 1e-9:
                    if d_ab < best_d:
                        best_criterion = criterion
                        best_a = a
                        best_b = b
                        best_d = d_ab
        
        a, b = best_a, best_b
        d_ab = dist_dict[(a, b)]
        
        new_label = f"n{next_id}"
        next_id += 1
        
        trees[new_label] = f"({trees[a]},{trees[b]})"
        
        rest = [x for x in active if x not in (a, b)]
        
        for x in rest:
            new_d = 0.5 * (dist_dict[(a, x)] + dist_dict[(b, x)] - d_ab)
            dist_dict[(new_label, x)] = new_d
            dist_dict[(x, new_label)] = new_d
            
        active = rest + [new_label]
        
    return "(" + ",".join(trees[x] for x in active) + ");"

def _build_tree_from_newick(newick, leaf_indices):
    """
    Parse Newick into a binary unrooted tree represented as adjacency list.
    Returns:
        adj: dict mapping node_id -> list of neighbor node_ids
        node_type: dict mapping node_id -> 'leaf' or 'internal'
        leaf_map: dict mapping node_id -> leaf index (for leaves)
    """
    # First, we need to parse the Newick to get the structure
    # Then assign node IDs and build adjacency
    
    # Simple recursive parser
    pos = [0]
    s = newick.strip()
    if s.endswith(';'):
        s = s[:-1]
    
    def parse_node():
        """Returns (node_id, children_list) where children_list is list of node_ids"""
        if pos[0] >= len(s):
            return -1, []
        
        if s[pos[0]] == '(':
            pos[0] += 1  # skip '('
            children = []
            while pos[0] < len(s) and s[pos[0]] != ')':
                child_id, _ = parse_node()
                if child_id != -1:
                    children.append(child_id)
                if pos[0] < len(s) and s[pos[0]] == ',':
                    pos[0] += 1  # skip ','
            if pos[0] < len(s) and s[pos[0]] == ')':
                pos[0] += 1  # skip ')'
            
            # This is an internal node
            # Assign a new ID
            node_id = len(_node_counter)
            _node_counter.append('internal')
            return node_id, children
        else:
            # Leaf node
            start = pos[0]
            while pos[0] < len(s) and s[pos[0]] not in ',)':
                pos[0] += 1
            name = s[start:pos[0]]
            node_id = len(_node_counter)
            _node_counter.append('leaf')
            _leaf_names.append(name)
            return node_id, []
    
    _node_counter = []
    _leaf_names = []
    root_id, _ = parse_node()
    
    # Now build the tree from the parsed structure
    # We need to do a DFS to assign edges
    adj = {}
    node_type = {}
    leaf_map = {}
    
    # Re-parse to build structure properly
    pos[0] = 0
    s = newick.strip()
    if s.endswith(';'):
        s = s[:-1]
    
    # We'll use a different approach: assign IDs during parsing and record edges
    node_ids = []  # list of (node_id, name_or_None, children_ids)
    
    def parse_v2():
        if pos[0] >= len(s):
            return -1
        
        if s[pos[0]] == '(':
            pos[0] += 1
            children = []
            while pos[0] < len(s) and s[pos[0]] != ')':
                child_id = parse_v2()
                if child_id != -1:
                    children.append(child_id)
                if pos[0] < len(s) and s[pos[0]] == ',':
                    pos[0] += 1
            if pos[0] < len(s) and s[pos[0]] == ')':
                pos[0] += 1
            
            node_id = len(node_ids)
            node_ids.append((node_id, None, children))
            return node_id
        else:
            start = pos[0]
            while pos[0] < len(s) and s[pos[0]] not in ',)':
                pos[0] += 1
            name = s[start:pos[0]]
            node_id = len(node_ids)
            node_ids.append((node_id, name, []))
            return node_id
    
    parse_v2()
    
    # Build adjacency
    for node_id, name, children in node_ids:
        if name is not None:
            node_type[node_id] = 'leaf'
            leaf_map[node_id] = name
        else:
            node_type[node_id] = 'internal'
        
        if children:
            for child in children:
                if node_id not in adj:
                    adj[node_id] = []
                if child not in adj:
                    adj[child] = []
                adj[node_id].append(child)
                adj[child].append(node_id)
    
    # Ensure all nodes are in adj
    for node_id in range(len(node_ids)):
        if node_id not in adj:
            adj[node_id] = []
    
    return adj, node_type, leaf_map

def _fitch_parsimony_score(adj, node_type, leaf_map, sequences, labels):
    """
    Compute Fitch parsimony score for a tree.
    adj: adjacency dict
    node_type: dict node_id -> 'leaf' or 'internal'
    leaf_map: dict node_id -> leaf name
    sequences: dict leaf_name -> sequence string
    labels: list of leaf names
    """
    n_leaves = len(labels)
    L = len(sequences[labels[0]])
    
    # Build leaf index map
    leaf_index_map = {name: i for i, name in enumerate(labels)}
    
    # For each site, compute Fitch score
    total_score = 0
    
    # Root the tree at an arbitrary leaf for post-order traversal
    # Find a leaf to use as root
    root_leaf = None
    for node_id, ntype in node_type.items():
        if ntype == 'leaf':
            root_leaf = node_id
            break
    
    if root_leaf is None:
        return 0
    
    # Do a DFS from root_leaf to get post-order
    visited = set()
    post_order = []
    
    def dfs(node, parent):
        visited.add(node)
        for neighbor in adj[node]:
            if neighbor != parent and neighbor not in visited:
                dfs(neighbor, node)
        post_order.append(node)
    
    dfs(root_leaf, -1)
    
    # Process in post-order for each site
    # For efficiency, process all sites together using bitsets or just loop
    # Since L can be up to 2000 and nodes up to 63, we loop over sites
    
    for site in range(L):
        # For each node, compute the set of possible states
        # Fitch: leaf gets {observed_state}
        # Internal: intersection of children's sets if non-empty, else union
        
        node_states = {}
        
        for node in post_order:
            if node_type[node] == 'leaf':
                leaf_name = leaf_map[node]
                char = sequences[leaf_name][site]
                # Map char to index
                char_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}[char]
                node_states[node] = (1 << char_idx)  # bitmask
            else:
                # Internal node: combine children
                # Get children (neighbors except parent in the rooted tree)
                # We need to know which neighbors are children in the rooted tree
                # This is tricky with just adjacency. Let's use a different approach.
                pass
        
        # Actually, the above approach doesn't track parent-child relationships properly.
        # Let me redo this with explicit parent tracking.
        
        # Re-do: build parent-child relationships from the rooted tree
        parent_map = {root_leaf: -1}
        children_map = {root_leaf: []}
        
        def dfs2(node, parent):
            for neighbor in adj[node]:
                if neighbor != parent:
                    parent_map[neighbor] = node
                    children_map[node].append(neighbor)
                    dfs2(neighbor, node)
        
        dfs2(root_leaf, -1)
        
        # Now process in post-order
        node_states = {}
        
        for node in post_order:
            if node_type[node] == 'leaf':
                leaf_name = leaf_map[node]
                char = sequences[leaf_name][site]
                char_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}[char]
                node_states[node] = (1 << char_idx)
            else:
                children = children_map.get(node, [])
                if not children:
                    # This shouldn't happen for internal nodes in a proper tree
                    node_states[node] = 0
                    continue
                
                # Combine children's state sets
                combined = 0
                first = True
                for child in children:
                    child_states = node_states[child]
                    if first:
                        combined = child_states
                        first = False
                    else:
                        combined &= child_states
                
                if combined == 0:
                    # Union
                    union = 0
                    for child in children:
                        union |= node_states[child]
                    node_states[node] = union
                    total_score += 1  # One extra step at this site
                else:
                    node_states[node] = combined
    
    return total_score

def _nni_optimize(newick, sequences, labels, time_budget):
    """
    Apply NNI moves to improve the tree topology using Fitch parsimony.
    Returns improved Newick string.
    """
    start_time = time.time()
    
    adj, node_type, leaf_map = _build_tree_from_newick(newick, labels)
    
    # Build leaf index map
    leaf_index_map = {name: i for i, name in enumerate(labels)}
    
    # Compute initial score
    current_score = _fitch_parsimony_score(adj, node_type, leaf_map, sequences, labels)
    
    # Find all internal edges
    # An internal edge connects two internal nodes
    internal_nodes = [n for n, t in node_type.items() if t == 'internal']
    internal_edges = []
    for node in internal_nodes:
        for neighbor in adj[node]:
            if node_type[neighbor] == 'internal' and node < neighbor:
                internal_edges.append((node, neighbor))
    
    improved = True
    while improved:
        improved = False
        
        if time.time() - start_time > time_budget:
            break
        
        for edge_idx, (u, v) in enumerate(internal_edges):
            if time.time() - start_time > time_budget:
                break
            
            # Get the four neighbors
            # u's neighbors except v
            u_neighbors = [n for n in adj[u] if n != v]
            # v's neighbors except u
            v_neighbors = [n for n in adj[v] if n != u]
            
            if len(u_neighbors) != 2 or len(v_neighbors) != 2:
                continue
            
            a, b = u_neighbors
            c, d = v_neighbors
            
            # Original: (a, u, v, b) connected to (c, v, u, d)
            # Actually: a-u-v-b is wrong. The structure is:
            # u connected to a, b, v
            # v connected to c, d, u
            # So the four "branches" are a, b from u and c, d from v
            
            # NNI move 1: swap b and c
            # New: u connected to a, b, c (wait, no)
            # Let me think again.
            # Original: a - u - v - b  (no, that's not right either)
            # u has neighbors: a, b, v
            # v has neighbors: c, d, u
            # So the tree locally looks like:
            #   a
            #   |
            #   u --- v
            #   |     |
            #   b     c
            #        |
            #        d
            # Wait, that's not right. u has 3 neighbors: a, b, v
            # v has 3 neighbors: c, d, u
            # So:
            #   a
            #   |
            #   u --- v
            #   |     |
            #   b     c
            #        |
            #        d
            # Hmm, that gives u degree 3 (a, b, v) and v degree 3 (c, d, u). Correct.
            
            # NNI alternatives:
            # 1. Swap b and c:
            #   a
            #   |
            #   u --- v
            #   |     |
            #   c     b
            #        |
            #        d
            # Wait, that changes which nodes connect to which.
            # Original edges: (a,u), (b,u), (u,v), (c,v), (d,v)
            # NNI 1: remove (b,u) and (c,v), add (b,v) and (c,u)
            # New edges: (a,u), (c,u), (u,v), (b,v), (d,v)
            # This gives u: {a, c, v}, v: {b, d, u}
            
            # 2. Swap b and d:
            # Remove (b,u) and (d,v), add (b,v) and (d,u)
            # New edges: (a,u), (d,u), (u,v), (b,v), (c,v)
            # This gives u: {a, d, v}, v: {b, c, u}
            
            # Try both NNI moves
            for move_type in [1, 2]:
                if time.time() - start_time > time_budget:
                    break
                
                # Save original edges
                orig_edges = [(a, u), (b, u), (u, v), (c, v), (d, v)]
                
                # Apply NNI
                if move_type == 1:
                    # Remove (b,u) and (c,v), add (b,v) and (c,u)
                    adj[u].remove(b)
                    adj[b].remove(u)
                    adj[v].remove(c)
                    adj[c].remove(v)
                    adj[u].append(c)
                    adj[c].append(u)
                    adj[v].append(b)
                    adj[b].append(v)
                else:
                    # Remove (b,u) and (d,v), add (b,v) and (d,u)
                    adj[u].remove(b)
                    adj[b].remove(u)
                    adj[v].remove(d)
                    adj[d].remove(v)
                    adj[u].append(d)
                    adj[d].append(u)
                    adj[v].append(b)
                    adj[b].append(v)
                
                # Compute new score
                new_score = _fitch_parsimony_score(adj, node_type, leaf_map, sequences, labels)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    # Revert
                    if move_type == 1:
                        adj[u].remove(c)
                        adj[c].remove(u)
                        adj[v].remove(b)
                        adj[b].remove(v)
                        adj[u].append(b)
                        adj[b].append(u)
                        adj[v].append(c)
                        adj[c].append(v)
                    else:
                        adj[u].remove(d)
                        adj[d].remove(u)
                        adj[v].remove(b)
                        adj[b].remove(v)
                        adj[u].append(b)
                        adj[b].append(u)
                        adj[v].append(d)
                        adj[d].append(v)
        
        # Update internal_edges if tree changed (topology changed, so edges may have changed)
        if improved:
            internal_nodes = [n for n, t in node_type.items() if t == 'internal']
            internal_edges = []
            for node in internal_nodes:
                for neighbor in adj[node]:
                    if node_type[neighbor] == 'internal' and node < neighbor:
                        internal_edges.append((node, neighbor))
    
    # Convert back to Newick
    return _tree_to_newick(adj, node_type, leaf_map, root_leaf)

def _tree_to_newick(adj, node_type, leaf_map, root):
    """Convert adjacency list back to Newick string."""
    # Find a leaf to root at
    leaf_root = None
    for node, ntype in node_type.items():
        if ntype == 'leaf':
            leaf_root = node
            break
    
    if leaf_root is None:
        return "();"
    
    # Build parent-child relationships
    parent_map = {leaf_root: -1}
    children_map = {leaf_root: []}
    
    def dfs(node, parent):
        for neighbor in adj[node]:
            if neighbor != parent:
                parent_map[neighbor] = node
                children_map[node].append(neighbor)
                dfs(neighbor, node)
    
    dfs(leaf_root, -1)
    
    # Build Newick recursively
    def build_newick(node):
        children = children_map.get(node, [])
        if node_type[node] == 'leaf':
            return leaf_map[node]
        else:
            if len(children) == 0:
                return ""
            child_strings = [build_newick(c) for c in children]
            return "(" + ",".join(child_strings) + ")"
    
    result = build_newick(leaf_root)
    return result + ";"

def reconstruct(sequences):
    start_time = time.time()
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    
    n = len(labels)
    if n <= 2:
        return f"({','.join(labels)});"
        
    # 1. Compute JC distances
    dist_matrix = _jc_distances(sequences, labels)
    
    # 2. Neighbor Joining
    newick = _neighbor_joining(dist_matrix, labels)
    
    # 3. NNI Optimization (if time permits)
    elapsed = time.time() - start_time
    remaining = DEADLINE_SECONDS - SAFETY_MARGIN - elapsed
    
    if remaining > 5.0 and n >= 4:
        try:
            improved_newick = _nni_optimize(newick, sequences, labels, remaining)
            if improved_newick:
                newick = improved_newick
        except Exception:
            # If NNI fails, return the NJ tree
            pass
    
    return newick
