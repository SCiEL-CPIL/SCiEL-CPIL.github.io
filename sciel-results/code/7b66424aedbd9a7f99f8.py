import time
import numpy as np

# EVOLVE-BLOCK-START

def _jc_distance_matrix(A):
    """
    Compute Jukes-Cantor distance matrix.
    A: numpy array of shape (n, L) with values 0-3.
    Returns: n x n numpy array.
    """
    n, L = A.shape
    if n < 2:
        return np.zeros((n, n))
    
    # Vectorized Hamming distance
    # A is (n, L). 
    # diff is (n, n, L) boolean.
    # This might be memory heavy for n=32, L=2000. 32*32*2000 = 2M bools. 2MB. Fine.
    diff = A[:, None, :] != A[None, :, :]
    hamming = np.sum(diff, axis=2)
    p = hamming / float(L)
    
    arg = 1.0 - 4.0 * p / 3.0
    arg = np.clip(arg, 1e-12, 1.0)
    d = -0.75 * np.log(arg)
    
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)
    return d

def _run_nj(dist, labels):
    """
    Neighbor Joining.
    Returns: (children_dict, parent_dict, root_id, leaf_ids)
    children: node -> (child1, child2)
    parent: node -> parent
    """
    n = len(labels)
    if n == 1:
        return {}, {0: None}, 0, [0]
    if n == 2:
        return {2: (0, 1)}, {0: 2, 1: 2, 2: None}, 2, [0, 1]
    
    nodes = list(range(n))
    children = {}
    parent = {}
    leaf_ids = list(range(n))
    
    # D is a dict of distances
    D = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                D[(i, j)] = dist[i, j]
    
    new_id = n
    # Active nodes
    active = [True] * (2 * n) # Max nodes is 2n-1
    
    while len(nodes) > 2:
        m = len(nodes)
        # Calculate r_i = sum(D[i, j] for j in nodes, j != i)
        r = {x: 0.0 for x in nodes}
        for x in nodes:
            for y in nodes:
                if x != y:
                    r[x] += D[(x, y)]
        
        best = float('inf')
        a, b = None, None
        for i_idx, x in enumerate(nodes):
            for y in nodes[i_idx + 1:]:
                crit = (m - 2) * D[(x, y)] - r[x] - r[y]
                if crit < best:
                    best = crit
                    a, b = x, y
        
        node = new_id
        new_id += 1
        
        # Add to structure
        children[node] = (a, b)
        parent[a] = node
        parent[b] = node
        
        rest = [x for x in nodes if x != a and x != b]
        d_ab = D[(a, b)]
        for x in rest:
            d_ax = D[(a, x)]
            d_bx = D[(b, x)]
            d_new = 0.5 * (d_ax + d_bx - d_ab)
            D[(node, x)] = d_new
            D[(x, node)] = d_new
        
        nodes.remove(a)
        nodes.remove(b)
        nodes.append(node)
        
        while len(active) <= new_id:
            active.append(False)
        active[node] = True
        active[a] = False
        active[b] = False

    a, b = nodes
    root = new_id
    children[root] = (a, b)
    parent[a] = root
    parent[b] = root
    parent[root] = None
    
    return children, parent, root, leaf_ids

def _fitch_cost(children, root, leaf_states):
    """
    Calculate total Fitch parsimony cost.
    Returns: (cost, node_states)
    node_states: dict node -> array of bitmasks (L,)
    """
    n_leaves = leaf_states.shape[0]
    L = leaf_states.shape[1]
    
    # Post-order traversal
    post_order = []
    stack = [root]
    visited = set()
    while stack:
        node = stack[-1]
        if node < n_leaves:
            post_order.append(node)
            stack.pop()
            visited.add(node)
        else:
            if node in visited:
                stack.pop()
                post_order.append(node)
                continue
            visited.add(node)
            c1, c2 = children[node]
            if c1 not in visited:
                stack.append(c1)
            if c2 not in visited:
                stack.append(c2)
    
    # Process in post-order
    node_states = {}
    total_cost = 0
    
    for node in post_order:
        if node < n_leaves:
            node_states[node] = leaf_states[node]
        else:
            c1, c2 = children[node]
            s1 = node_states[c1]
            s2 = node_states[c2]
            
            inter = s1 & s2
            union = s1 | s2
            
            # Cost is 1 where inter is 0
            cost_mask = (inter == 0)
            total_cost += np.sum(cost_mask)
            
            # Next state is inter if non-zero, else union
            node_states[node] = np.where(inter != 0, inter, union)
            
    return total_cost, node_states

def _build_newick(children, root, labels):
    """
    Generate Newick string from tree structure.
    """
    n_leaves = len(labels)
    
    def build(node):
        if node < n_leaves:
            return labels[node]
        c1, c2 = children[node]
        return f"({build(c1)},{build(c2)})"
    
    return build(root) + ";"

def reconstruct(sequences):
    start_time = time.time()
    deadline = 260.0 # 10s margin from 270s limit
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n <= 2:
        return "(" + ",".join(labels) + ");"
    
    L = len(sequences[labels[0]])
    
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    bit_map = {0: 1, 1: 2, 2: 4, 3: 8}
    
    A = np.zeros((n, L), dtype=np.uint8)
    leaf_states = np.zeros((n, L), dtype=np.uint8)
    
    for i, label in enumerate(labels):
        seq = sequences[label]
        for j, c in enumerate(seq):
            val = char_map[c]
            A[i, j] = val
            leaf_states[i, j] = bit_map[val]
    
    # 1. NJ Tree
    D = _jc_distance_matrix(A)
    children, parent, root, leaf_ids = _run_nj(D, labels)
    
    # 2. NNI Refinement
    current_cost, _ = _fitch_cost(children, root, leaf_states)
    
    # Identify internal edges
    # An internal edge is (u, v) where u is parent of v, u is not root, v is not leaf.
    # Actually, any edge between two internal nodes.
    # We iterate over internal nodes v. If parent[v] is internal, it's an internal edge.
    
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        
        # Get list of internal edges to consider
        edges_to_try = []
        for v in children.keys():
            if v == root:
                continue
            p = parent[v]
            if p is None:
                continue
            # Check if p is internal
            if p in children:
                # v is internal (since it's in children.keys())
                # p is internal
                edges_to_try.append((p, v))
        
        np.random.shuffle(edges_to_try)
        
        for p, v in edges_to_try:
            if time.time() > deadline - 10.0:
                break
            
            # Current topology: p is parent of v.
            # p's other child is w.
            # v's children are a, b.
            w = children[p][0] if children[p][1] == v else children[p][1]
            a, b = children[v]
            
            # We try 2 NNI moves.
            # Move 1: ((w, a), b)
            # Structure:
            # new_node1(w, a)
            # new_node2(new_node1, b)
            # p's child v is replaced by new_node2?
            # No, the edge (p, v) is removed.
            # The 4 branches are T_p (rest), T_w, T_a, T_b.
            # Current: (T_p, (T_w, (T_a, T_b)))? No.
            # Unrooted: The edge (p, v) separates {T_a, T_b} from {T_p, T_w}.
            # So the quartet topology is ((T_a, T_b), (T_p, T_w)).
            
            # Alternative 1: ((T_a, T_w), (T_b, T_p))
            # Alternative 2: ((T_a, T_p), (T_w, T_b))
            
            # To implement Alt 1:
            # We create new_node_x(T_a, T_w) and new_node_y(T_b, T_p).
            # Then we connect new_node_x and new_node_y.
            # This requires re-parenting.
            
            # Let's construct the new children/parent dicts for the trial tree.
            # This is O(n) per trial.
            
            # Trial 1: ((w, a), b) is NOT the standard NNI.
            # Standard NNI on edge (p, v) with children w, v and v's children a, b:
            # 1. Current: ((a, b), w) attached to p.
            # 2. Swap 1: ((a, w), b) attached to p.
            # 3. Swap 2: ((b, w), a) attached to p.
            
            # Let's implement Swap 1: ((a, w), b)
            # New structure:
            # new_node1(a, w)
            # new_node2(new_node1, b)
            # p's child v is replaced by new_node2.
            # So children[p] = (w, new_node2) ? No, w is now child of new_node1.
            # children[p] was (w, v). Now it should be (new_node2, other_child_of_p)?
            # Wait, p has children w and v.
            # If we move w to be sibling of a, then p's children become new_node2 and ...?
            # No, the edge (p, v) is the one being swapped.
            # The 4 subtrees are T_rest (via p), T_w, T_a, T_b.
            # Current: T_rest - p - (T_w, v - (T_a, T_b)).
            # This implies T_w and T_a/T_b are on the same side of the edge (p, rest)?
            # No, p is on the rest side.
            # So T_rest and T_w are on one side?
            # Let's trace:
            # Rest -- p -- w
            #          \
            #           v -- a
            #            \
            #             b
            # Edge (p, v).
            # Side 1: {a, b}
            # Side 2: {w, Rest}
            # Quartet: ((a, b), (w, Rest))
            
            # Alt 1: ((a, w), (b, Rest))
            # Alt 2: ((a, Rest), (w, b))
            
            # To implement Alt 1:
            # We need to connect a and w, and b and Rest.
            # This creates 2 new internal nodes.
            # node_x(a, w)
            # node_y(b, p)  <- p is the root of the Rest side?
            # Then connect node_x and node_y.
            
            # This changes the parent of a, w, b, p.
            
            # Let's write a helper to apply an NNI move and return new children/parent.
            def apply_nni(children, parent, root, n, p, v, mode):
                # p is parent of v.
                # w is other child of p.
                # a, b are children of v.
                w = children[p][0] if children[p][1] == v else children[p][1]
                a, b = children[v]
                
                new_children = children.copy()
                new_parent = parent.copy()
                
                # We need new node IDs.
                # Let's find the max ID.
                max_id = max(list(new_children.keys()) + [root])
                new_id1 = max_id + 1
                new_id2 = max_id + 2
                
                if mode == 1:
                    # ((a, w), (b, Rest))
                    # node1(a, w)
                    # node2(b, p)
                    # Connect node1 and node2.
                    # Who is the parent of whom?
                    # The edge (p, v) is removed.
                    # The new edge is (node1, node2).
                    # One of them must be the "upper" node.
                    # Let's say node2 is the parent of node1?
                    # Then node2's parent is p's parent?
                    # No, p is part of the "Rest" side.
                    # This is tricky.
                    
                    # Let's use a simpler representation:
                    # Just modify the children dict directly.
                    
                    # Current:
                    # p -> (w, v)
                    # v -> (a, b)
                    
                    # Alt 1:
                    # new_p -> (node2, other) ??
                    # No, the topology is unrooted.
                    # We can just pick an arbitrary rooting for the new local structure.
                    # Let's root at p.
                    # Current: p -> (w, v -> (a, b))
                    # Alt 1: p -> (node2 -> (b, node1 -> (a, w)))
                    # Wait, this changes the depth.
                    
                    # Let's just create the 3 possible quartet topologies for the 4 subtrees
                    # and evaluate their cost using the precomputed states.
                    # This is the "incremental" approach.
                    
                    pass # We will not use this helper, we will use the full recalc approach.
            
            # Full Recalculation Approach
            # Construct new children/parent for Alt 1
            # Alt 1: ((a, w), (b, Rest))
            # Let's assume the new edge connects node1(a,w) and node2(b,p).
            # We need to determine the parent-child relationship.
            # Let's say node2 is the parent of node1.
            # Then node2's parent is p's parent?
            # No, p is a child of node2?
            # If node2(b, p), then p is a child of node2.
            # But p was the parent of v.
            # This is getting messy.
            
            # Let's stick to the simplest correct method:
            # 1. Create a copy of children and parent.
            # 2. Remove edge (p, v).
            # 3. Add new edges.
            
            # Let's implement a function that takes the current tree and an NNI move spec
            # and returns the new tree.
            
            def make_nni_tree(children, parent, root, n, p, v, alt):
                # p is parent of v.
                w = children[p][0] if children[p][1] == v else children[p][1]
                a, b = children[v]
                
                new_children = children.copy()
                new_parent = parent.copy()
                
                max_id = max(list(new_children.keys()) + [root])
                new_id1 = max_id + 1
                new_id2 = max_id + 2
                
                # We will always root the new local structure such that
                # the node closer to the original root is the parent.
                # This is not strictly necessary for unrooted trees, but helps for consistency.
                
                if alt == 1:
                    # ((a, w), (b, Rest))
                    # node1(a, w)
                    # node2(b, p)
                    # Connect node1 and node2.
                    # Let's say node2 is the parent of node1.
                    # Then node2's parent is p's parent?
                    # No, p is a child of node2.
                    # So p's parent becomes node2.
                    # And node2's parent is p's old parent?
                    # Wait, p was the parent of v.
                    # If p is now a child of node2, then the edge (p, v) is gone.
                    # The edge (p, old_parent) is replaced by (node2, old_parent).
                    
                    # Let's trace:
                    # Old: old_parent -> p -> (w, v -> (a, b))
                    # New: old_parent -> node2 -> (p -> (b, ???), node1 -> (a, w))
                    # This doesn't seem right.
                    
                    # Let's use a different strategy:
                    # Just swap the children of p and v.
                    # Current: p(w, v), v(a, b)
                    # Alt 1: p(w, new_v), new_v(a, new_w), new_w(b, ???)
                    # No.
                    
                    # Let's use the 4-subtree method properly.
                    # The 4 subtrees are T_a, T_b, T_w, T_rest.
                    # T_rest is rooted at p.
                    # T_a is rooted at a.
                    # T_b is rooted at b.
                    # T_w is rooted at w.
                    
                    # In Alt 1: ((a, w), (b, Rest))
                    # We create node1(a, w).
                    # We create node2(b, p).
                    # Then we connect node1 and node2.
                    # Let's say node2 is the parent of node1.
                    # Then node2's parent is p's parent?
                    # No, p is a child of node2.
                    # So p's parent is node2.
                    # And node2's parent is p's old parent.
                    
                    # Let's implement this.
                    p_parent = new_parent[p]
                    
                    # Create node1
                    new_children[new_id1] = (a, w)
                    new_parent[a] = new_id1
                    new_parent[w] = new_id1
                    
                    # Create node2
                    new_children[new_id2] = (b, p)
                    new_parent[b] = new_id2
                    new_parent[p] = new_id2
                    
                    # Connect node1 and node2
                    # Let's say node2 is parent of node1.
                    new_children[new_id2] = (b, new_id1) # Wait, node2 already has b.
                    # No, node2 has children b and p?
                    # If node2 is parent of node1, then node2's children are b and node1?
                    # And p is... where?
                    # If p is a child of node2, then node2 has children b, p, node1? No, binary tree.
                    
                    # This manual re-wiring is error-prone.
                    
                    # Let's go back to Full Recalculation with a simpler tree modification.
                    # We will just create 3 new internal nodes and re-parent.
                    
                    pass # This is too complex to debug.
            
            # Let's use a different strategy:
            # Since n is small, we can just enumerate all possible NNI moves
            # and for each, construct the new tree by modifying the children dict
            # in a safe way.
            
            # We will define the 3 topologies explicitly.
            # Topology 0: Current
            # Topology 1: ((a, w), (b, Rest))
            # Topology 2: ((a, Rest), (w, b))
            
            # To evaluate Topology 1:
            # We need the states of T_a, T_b, T_w, T_Rest.
            # We have node_states for all nodes.
            # S_a = node_states[a]
            # S_b = node_states[b]
            # S_w = node_states[w]
            # S_Rest = node_states[p] # Is this correct?
            # node_states[p] is the state at p, which is derived from w and v.
            # So S_Rest is NOT node_states[p].
            # S_Rest is the state of the subtree containing the root, excluding the edge (p,v).
            # This is not directly available.
            
            # Therefore, the incremental method is hard without rooting the tree at the edge.
            
            # Given the complexity, I will stick to the **Full Recalculation** method
            # but I will implement the tree modification correctly.
            
            # Let's write a helper `nni_move(children, parent, root, n, p, v, alt)`
            # that returns the new children, parent, root.
            
            def nni_move(children, parent, root, n, p, v, alt):
                w = children[p][0] if children[p][1] == v else children[p][1]
                a, b = children[v]
                
                new_children = children.copy()
                new_parent = parent.copy()
                
                max_id = max(list(new_children.keys()) + [root])
                new_id1 = max_id + 1
                new_id2 = max_id + 2
                
                # We will always create a new root for the local structure if needed,
                # or re-parent existing nodes.
                
                if alt == 1:
                    # ((a, w), (b, Rest))
                    # node1(a, w)
                    # node2(b, p)
                    # Connect node1 and node2.
                    # Let's say node2 is the parent of node1.
                    # Then node2's parent is p's parent.
                    
                    p_parent = new_parent[p]
                    
                    # node1
                    new_children[new_id1] = (a, w)
                    new_parent[a] = new_id1
                    new_parent[w] = new_id1
                    
                    # node2
                    new_children[new_id2] = (b, new_id1)
                    new_parent[b] = new_id2
                    new_parent[new_id1] = new_id2
                    
                    # p is now a child of node2?
                    # No, p was the parent of v.
                    # In the new topology, p is in the "Rest" side.
                    # The edge (p, v) is removed.
                    # The new edge is (node2, p)?
                    # If node2 is parent of node1, and node1 is (a,w), and node2 is (b, node1),
                    # where is p?
                    # p must be a child of node2?
                    # If node2 has children b and node1, it can't have p.
                    
                    # Let's try:
                    # node2(b, p)
                    # node1(a, w)
                    # Connect node2 and node1.
                    # Let's say node2 is parent of node1.
                    # Then node2's children are b and node1.
                    # And p is... a child of node2? No.
                    # p is the root of the Rest subtree.
                    # So p is a child of node2?
                    # If node2 has children b and p, then node1 is where?
                    
                    # This is confusing. Let's use a library or a known algorithm.
                    # Or, since n is small, let's just use a simple heuristic:
                    # Try all 3 quartet topologies for the 4 subtrees
                    # and pick the one with the minimum Fitch cost.
                    # To do this, we need to be able to calculate the cost of a specific
                    # quartet topology given the states of the 4 subtrees.
                    
                    # We can calculate the state of the 4 subtrees by
                    # temporarily detaching the edge (p, v).
                    # State of T_a: node_states[a]
                    # State of T_b: node_states[b]
                    # State of T_w: node_states[w]
                    # State of T_Rest: This is the hard part.
                    
                    # Let's just use the Full Recalculation with a correctly implemented
                    # tree modification.
                    
                    # I will implement a generic tree modifier.
                    
                    pass
            
            # Given the time, I will skip the NNI refinement and rely on NJ + a simple
            # Greedy Split Selection that is correct.
            
            # Actually, the seed program's Greedy Split Selection was flawed.
            # Let's implement a correct Greedy Split Selection.
            
            pass # Skip NNI, use Greedy Splits
            
    # --- Greedy Split Selection ---
    
    def _get_splits_from_tree(children, root, n):
        splits = []
        def dfs(node):
            if node < n:
                return {node}
            c1, c2 = children[node]
            s1 = dfs(c1)
            s2 = dfs(c2)
            s = s1.union(s2)
            if 0 < len(s) < n:
                if min(s) < min(set(range(n)) - s):
                    splits.append(frozenset(s))
                else:
                    splits.append(frozenset(set(range(n)) - s))
            return s
        dfs(root)
        return splits
        
    nj_splits = _get_splits_from_tree(children, root, n)
    
    # Generate candidates: NJ splits + random
    candidates = list(nj_splits)
    rng = np.random.RandomState(42)
    for _ in range(100):
        if time.time() > deadline - 20:
            break
        perm = rng.permutation(n)
        k = rng.randint(2, n-1)
        s = frozenset(perm[:k])
        if min(s) > min(set(range(n)) - s):
            s = frozenset(set(range(n)) - s)
        candidates.append(s)
        
    candidates = list(set(candidates))
    
    # Score candidates with Vectorized Fitch Cost
    taxon_masks = np.zeros((L, 4), dtype=np.uint64)
    for j in range(L):
        for c in range(4):
            mask = 0
            for i in range(n):
                if A[i, j] == c:
                    mask |= (1 << i)
            taxon_masks[j, c] = mask
            
    full_mask = (1 << n) - 1
    
    def calc_split_cost(split):
        M = 0
        for i in split:
            M |= (1 << i)
        M_inv = full_mask ^ M
        
        pS = (taxon_masks & M) != 0
        pSc = (taxon_masks & M_inv) != 0
        overlap = pS & pSc
        has_overlap = overlap.any(axis=1)
        cost = (~has_overlap).sum()
        return int(cost)
        
    split_costs = {}
    for s in candidates:
        if time.time() > deadline - 10:
            break
        split_costs[s] = calc_split_cost(s)
        
    sorted_splits = sorted(split_costs.items(), key=lambda x: x[1])
    selected = []
    target = n - 3
    all_set = set(range(n))
    
    for s, cost in sorted_splits:
        if len(selected) >= target:
            break
        compatible = True
        for s2 in selected:
            i1 = s & s2
            i2 = s & (all_set - s2)
            i3 = (all_set - s) & s2
            i4 = (all_set - s) & (all_set - s2)
            if i1 and i2 and i3 and i4:
                compatible = False
                break
        if compatible:
            selected.append(s)
            
    if not selected:
        return _build_newick(children, root, labels)
        
    def build_tree(subset):
        if len(subset) == 1:
            return labels[list(subset)[0]]
        
        for s in selected:
            A_part = s & subset
            if 0 < len(A_part) < len(subset):
                B_part = subset - A_part
                return f"({build_tree(A_part)},{build_tree(B_part)})"
                
        return "(" + ",".join(labels[i] for i in sorted(subset)) + ")"
        
    final_tree = build_tree(all_set) + ";"
    return final_tree

# EVOLVE-BLOCK-END
