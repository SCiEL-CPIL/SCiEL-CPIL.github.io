"""Phylogenetic topology recovery: BionJ initialization + SSE-based NNI local search."""
import math
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 255  # safety margin before 270s kill


def _check_deadline():
    return time.time() < _DEADLINE


def _jc_distance(a, b):
    """Jukes-Cantor corrected distance."""
    n = len(a)
    if n == 0:
        return 0.0
    diff = 0
    for i in range(n):
        if a[i] != b[i]:
            diff += 1
    p = diff / n
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _bionj_distance_matrix(sequences, labels):
    """Compute BionJ-style distance matrix with bias correction."""
    n = len(labels)
    dist = np.zeros((n, n))
    raw = np.zeros((n, n))
    
    # Compute raw pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
            raw[i, j] = d
            raw[j, i] = d
    
    # Compute mean distances for bias correction
    means = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(n):
            if i != j:
                s += raw[i, j]
        means[i] = s / max(1, n - 1)
    
    # BionJ bias correction
    for i in range(n):
        for j in range(i + 1, n):
            if i == j:
                continue
            # Simple bias correction: subtract means
            corrected = raw[i, j] - 0.5 * (means[i] + means[j])
            dist[i, j] = max(0.0, corrected)
            dist[j, i] = dist[i, j]
    
    return dist


def _neighbor_joining(dist, labels):
    """Neighbor-Joining algorithm. Returns Newick string."""
    n = len(labels)
    if n <= 2:
        return "(" + ",".join(labels) + ");"
    if n == 3:
        return "(" + ",".join(labels) + ");"
    
    nodes = list(range(n))
    tree_str = {i: labels[i] for i in range(n)}
    d = dist.copy()
    
    while len(nodes) > 3:
        if not _check_deadline():
            break
        n_curr = len(nodes)
        
        # Compute row sums
        r = np.zeros(n_curr)
        for i in range(n_curr):
            for j in range(n_curr):
                if i != j:
                    r[i] += d[i, j]
        
        # Find best pair to join
        best_val = float('inf')
        best_i, best_j = -1, -1
        for i in range(n_curr):
            for j in range(i + 1, n_curr):
                val = (n_curr - 2) * d[i, j] - r[i] - r[j]
                if val < best_val:
                    best_val = val
                    best_i, best_j = i, j
        
        if best_i == -1:
            break
        
        # Join nodes best_i and best_j
        new_node = n_curr
        new_str = f"({tree_str[best_i]},{tree_str[best_j]})"
        
        # Create new distance vector
        new_d = np.zeros((n_curr + 1, n_curr + 1))
        
        # Map old indices to new
        remaining = [x for x in range(n_curr) if x != best_i and x != best_j]
        
        for i in range(len(remaining)):
            for j in range(len(remaining)):
                if i == j:
                    continue
                new_d[i, j] = d[remaining[i], remaining[j]]
        
        # Distances from new node to remaining nodes
        for i in range(len(remaining)):
            x = remaining[i]
            new_d[i, len(remaining)] = 0.5 * (d[best_i, x] + d[best_j, x] - d[best_i, best_j])
            new_d[len(remaining), i] = new_d[i, len(remaining)]
        
        d = new_d
        tree_str[new_node] = new_str
        nodes = remaining + [new_node]
    
    # Build final Newick from remaining 3 nodes
    if len(nodes) == 3:
        newick = "(" + ",".join(tree_str[x] for x in nodes) + ");"
    elif len(nodes) == 2:
        newick = "(" + ",".join(tree_str[x] for x in nodes) + ");"
    else:
        newick = "(" + ",".join(tree_str[x] for x in nodes) + ");"
    
    return newick


def _parse_newick(newick):
    """Parse Newick string into adjacency structure."""
    newick = newick.strip().rstrip(';')
    
    adj = {}
    node_types = {}
    counter = [0]
    
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
            nid = ('internal', counter[0])
            counter[0] += 1
            node_types[nid] = ('internal', None)
            adj[nid] = set()
            for child in children:
                cid = child[0] if isinstance(child, tuple) else None
                if isinstance(child, tuple) and child[0] == 'leaf':
                    cid = ('leaf', child[1])
                    if cid not in adj:
                        node_types[cid] = ('leaf', child[1])
                        adj[cid] = set()
                    adj[cid].add(nid)
                    adj[nid].add(cid)
                else:
                    # Already processed
                    if child in adj:
                        adj[child].add(nid)
                        adj[nid].add(child)
            return nid
        else:
            label = s
            if ':' in label:
                label = label.split(':')[0]
            nid = ('leaf', label)
            if nid not in adj:
                node_types[nid] = ('leaf', label)
                adj[nid] = set()
            return nid
    
    root = parse(newick)
    return adj, node_types


def _root_tree(adj, node_types):
    """Root the tree at an internal node."""
    root = None
    for nid in adj:
        if node_types[nid][0] == 'internal':
            root = nid
            break
    if root is None:
        root = list(adj.keys())[0]
    
    parent = {root: None}
    children = {nid: [] for nid in adj}
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


def _get_leaves(adj, node_types):
    """Get all leaf nodes."""
    return [nid for nid in adj if node_types[nid][0] == 'leaf']


def _compute_sse(adj, node_types, dist, labels):
    """Compute Sum of Squared Errors between tree distances and observed distances.
    
    For each pair of leaves, compute the path distance in the tree (using unit edge lengths)
    and compare to the observed distance.
    """
    n = len(labels)
    if n < 2:
        return 0.0
    
    # Build label to node id mapping
    label_to_node = {}
    for nid in adj:
        if node_types[nid][0] == 'leaf':
            label_to_node[node_types[nid][1]] = nid
    
    # Compute all-pairs shortest paths on the tree using unit edge lengths
    # Since it's a tree, we can use BFS from each leaf
    leaf_ids = list(label_to_node.values())
    
    sse = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            li = label_to_node[labels[i]]
            lj = label_to_node[labels[j]]
            
            # BFS to find path length (number of edges)
            from collections import deque
            dist_map = {li: 0}
            queue = deque([li])
            while queue:
                node = queue.popleft()
                if node == lj:
                    break
                for nbr in adj[node]:
                    if nbr not in dist_map:
                        dist_map[nbr] = dist_map[node] + 1
                        queue.append(nbr)
            
            tree_dist = dist_map.get(lj, 0)
            obs_dist = dist[i, j]
            sse += (tree_dist - obs_dist) ** 2
    
    return sse


def _apply_nni(adj, node_types, u, v, a, b, c, d, move_type):
    """Apply an NNI move to the adjacency structure.
    
    For internal edge (u,v) where u has neighbors {v,a,b} and v has neighbors {u,c,d}:
    NNI1: swap b and c -> u:{v,a,c}, v:{u,b,d}
    NNI2: swap b and d -> u:{v,a,d}, v:{u,b,c}
    """
    new_adj = {k: set(v) for k, v in adj.items()}
    
    if move_type == 1:
        # Swap b and c
        new_adj[u].discard(b)
        new_adj[b].discard(u)
        new_adj[v].discard(c)
        new_adj[c].discard(v)
        
        new_adj[u].add(c)
        new_adj[c].add(u)
        new_adj[v].add(b)
        new_adj[b].add(v)
    else:
        # Swap b and d
        new_adj[u].discard(b)
        new_adj[b].discard(u)
        new_adj[v].discard(d)
        new_adj[d].discard(v)
        
        new_adj[u].add(d)
        new_adj[d].add(u)
        new_adj[v].add(b)
        new_adj[b].add(v)
    
    return new_adj


def _nni_local_search(adj, node_types, dist, labels, max_iterations=30):
    """Perform NNI local search using SSE scoring."""
    current_sse = _compute_sse(adj, node_types, dist, labels)
    
    improved = True
    iteration = 0
    while improved and iteration < max_iterations and _check_deadline():
        improved = False
        iteration += 1
        
        best_sse = current_sse
        best_adj = None
        
        # Find all internal edges
        internal_nodes = [nid for nid in adj if node_types[nid][0] == 'internal']
        
        for u in internal_nodes:
            for v in adj[u]:
                if v <= u:
                    continue
                if node_types[v][0] != 'internal':
                    continue
                
                u_nbrs = [x for x in adj[u] if x != v]
                v_nbrs = [x for x in adj[v] if x != u]
                
                if len(u_nbrs) < 2 or len(v_nbrs) < 2:
                    continue
                
                a, b = u_nbrs[0], u_nbrs[1]
                c, d = v_nbrs[0], v_nbrs[1]
                
                # Try NNI1
                new_adj1 = _apply_nni(adj, node_types, u, v, a, b, c, d, 1)
                sse1 = _compute_sse(new_adj1, node_types, dist, labels)
                if sse1 < best_sse - 1e-6:
                    best_sse = sse1
                    best_adj = new_adj1
                
                # Try NNI2
                new_adj2 = _apply_nni(adj, node_types, u, v, a, b, c, d, 2)
                sse2 = _compute_sse(new_adj2, node_types, dist, labels)
                if sse2 < best_sse - 1e-6:
                    best_sse = sse2
                    best_adj = new_adj2
        
        if best_adj is not None:
            adj = best_adj
            current_sse = best_sse
            improved = True
    
    return adj


def _adj_to_newick(adj, node_types):
    """Convert adjacency structure back to Newick string."""
    if not adj:
        return ""
    
    # Find a leaf to start from
    root = None
    for nid in adj:
        if node_types[nid][0] == 'leaf':
            root = nid
            break
    if root is None:
        root = list(adj.keys())[0]
    
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


def reconstruct(sequences):
    """Reconstruct phylogenetic tree topology from sequences."""
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    
    if n_taxa <= 2:
        return "(" + ",".join(labels) + ");"
    if n_taxa == 3:
        return "(" + ",".join(labels) + ");"
    
    # Step 1: Compute BionJ-style distance matrix
    dist = _bionj_distance_matrix(sequences, labels)
    
    # Step 2: Neighbor-Joining to get initial tree
    newick = _neighbor_joining(dist, labels)
    
    # Step 3: Parse and convert to adjacency
    adj, node_types = _parse_newick(newick)
    
    # Step 4: NNI local search with SSE scoring
    if _check_deadline():
        adj = _nni_local_search(adj, node_types, dist, labels, max_iterations=20)
    
    # Convert back to Newick
    result = _adj_to_newick(adj, node_types)
    
    # Validate: ensure all taxa appear exactly once
    for label in labels:
        if label not in result:
            return newick
    
    return result
