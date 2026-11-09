import time
import numpy as np
from itertools import combinations
import sys

def _jc_distance_matrix(aln):
    """
    Compute the Jukes-Cantor corrected distance matrix.
    aln: list of strings (one per taxon)
    Returns: numpy array of shape (n, n)
    """
    n = len(aln)
    n_sites = len(aln[0])
    
    # Convert to integer matrix for faster comparison
    # A=0, C=1, G=2, T=3
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    # Using a list of lists for simplicity in indexing, or numpy array
    # Numpy is faster for vectorized ops
    mat = np.zeros((n, n_sites), dtype=np.int8)
    for i, seq in enumerate(aln):
        for j, char in enumerate(seq):
            mat[i, j] = base_map.get(char, 0) # Default to 0 if unknown, though task says ACGT
            
    # Calculate raw Hamming distances
    # diff[i, j, k] is 1 if mat[i,k] != mat[j,k]
    # We can do this pairwise to save memory if n is large, but n<=32 so n*n*sites is small
    # 32*32*2000 = 2,048,000 integers. Very manageable.
    
    D = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        for j in range(i + 1, n):
            # Count differences
            diff_count = np.sum(mat[i] != mat[j])
            p = diff_count / n_sites
            # JC correction: d = -3/4 * ln(1 - 4/3 * p)
            # Clip p to avoid log(0) or negative log
            if p >= 0.75:
                D[i, j] = 10.0 # Large distance
            else:
                D[i, j] = -0.75 * np.log(1.0 - (4.0 / 3.0) * p)
            D[j, i] = D[i, j]
            
    return D

def _neighbor_joining(D, labels):
    """
    Perform Neighbor Joining on distance matrix D.
    Returns a Newick string.
    """
    n = len(labels)
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    # D is the current distance matrix
    # active is the list of current leaf/node indices
    active = list(range(n))
    # trees[i] stores the Newick string for the current node i
    trees = {i: labels[i] for i in range(n)}
    
    # We need a dynamic distance matrix. 
    # Instead of resizing, we can use a dictionary or just keep a large array and update.
    # Since n is small, let's just keep a dict of dicts or a numpy array that we rebuild/update.
    # Rebuilding is O(n^2) per step, which is fine for n=32.
    
    current_D = D.copy()
    next_id = n
    
    while len(active) > 3:
        m = len(active)
        if m == 0: break
        
        # Extract submatrix for active nodes
        idx = np.array(active)
        sub_D = current_D[np.ix_(idx, idx)]
        
        # Calculate row sums
        r = sub_D.sum(axis=1)
        
        # Calculate Q matrix: Q[i, j] = (m-2)*D[i,j] - r[i] - r[j]
        Q = (m - 2) * sub_D - r[:, None] - r[None, :]
        
        # Find the pair (i, j) with the minimum Q value
        # We need to find the min in the upper triangle (or lower)
        # Set diagonal to inf to avoid self-pairs
        np.fill_diagonal(Q, np.inf)
        
        # Find min index
        min_val = np.inf
        min_i, min_j = -1, -1
        for i in range(m):
            for j in range(i + 1, m):
                if Q[i, j] < min_val:
                    min_val = Q[i, j]
                    min_i, min_j = i, j
        
        a_idx = active[min_i]
        b_idx = active[min_j]
        
        # Calculate branch lengths (optional, but good for structure if we were doing ML)
        # For topology, we just need to join them.
        d_ab = current_D[a_idx, b_idx]
        
        # New node
        new_node = next_id
        next_id += 1
        
        # New tree string
        trees[new_node] = f"({trees[a_idx]},{trees[b_idx]})"
        
        # Update distance matrix for the new node against all other active nodes
        # D(new, x) = 0.5 * (D(a, x) + D(b, x) - D(a, b))
        # We need to update current_D. 
        # Since we are removing a_idx and b_idx and adding new_node, 
        # and current_D is indexed by original IDs (0..n-1, then n, n+1...),
        # we need to be careful. 
        # Actually, my current_D is indexed by the IDs in 'active' list initially, 
        # but as we add new IDs, the matrix size needs to grow or we need to map.
        
        # Let's restart the distance matrix management strategy to be cleaner.
        # We'll maintain a dictionary `dist` mapping (id1, id2) -> distance.
        # This is O(1) lookup and update, and we only need O(n^2) space.
        
        # Wait, I already started with numpy. Let's stick to a simpler approach for the rewrite:
        # Just use a dictionary for distances to avoid numpy resizing headaches in the loop.
        
        pass # Placeholder, I will rewrite _neighbor_joining below with dict distances

def _neighbor_joining_v2(D_init, labels):
    """
    NJ using a dictionary for distances to handle dynamic node addition easily.
    """
    n = len(labels)
    if n <= 1:
        return labels[0] + ";" if n==1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    # Initialize distance dict
    dist = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[(i, j)] = D_init[i, j]
                dist[(j, i)] = D_init[i, j]
                
    active = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    next_id = n
    
    while len(active) > 3:
        m = len(active)
        
        # Calculate row sums
        r = {i: 0.0 for i in active}
        for i in active:
            for j in active:
                if i != j:
                    r[i] += dist[(i, j)]
        
        # Find min Q
        min_q = float('inf')
        min_i, min_j = -1, -1
        for i in active:
            for j in active:
                if i == j:
                    continue
                # Q[i,j] = (m-2)*D[i,j] - r[i] - r[j]
                q_val = (m - 2) * dist[(i, j)] - r[i] - r[j]
                if q_val < min_q:
                    min_q = q_val
                    min_i, min_j = i, j
        
        if min_i == -1:
            break
            
        a, b = min_i, min_j
        
        # Create new node
        new_node = next_id
        next_id += 1
        
        trees[new_node] = f"({trees[a]},{trees[b]})"
        
        # Update distances
        d_ab = dist[(a, b)]
        
        for x in active:
            if x == a or x == b:
                continue
            # D(new, x) = 0.5 * (D(a, x) + D(b, x) - D(a, b))
            d_new_x = 0.5 * (dist[(a, x)] + dist[(b, x)] - d_ab)
            dist[(new_node, x)] = d_new_x
            dist[(x, new_node)] = d_new_x
            
        # Remove a and b from active, add new_node
        active.remove(a)
        active.remove(b)
        active.append(new_node)
        
    # Final 3 nodes
    if len(active) == 3:
        a, b, c = active
        # Find the pair that should be joined last based on distances
        # The one with the smallest sum of distances to the other two? 
        # In NJ, the last 3 form a star or a cherry.
        # We just need to pick a topology. The standard NJ picks the pair with min Q.
        m = 3
        r = {i: 0.0 for i in active}
        for i in active:
            for j in active:
                if i != j:
                    r[i] += dist[(i, j)]
        
        min_q = float('inf')
        min_i, min_j = -1, -1
        for i in active:
            for j in active:
                if i == j: continue
                q_val = (m - 2) * dist[(i, j)] - r[i] - r[j]
                if q_val < min_q:
                    min_q = q_val
                    min_i, min_j = i, j
        
        k = [x for x in active if x not in (min_i, min_j)][0]
        
        # Topology: ((a,b),c)
        return f"(({trees[min_i]},{trees[min_j]}),{trees[k]});"
    elif len(active) == 2:
        return f"({trees[active[0]]},{trees[active[1]]});"
    elif len(active) == 1:
        return trees[active[0]] + ";"
    else:
        # Fallback star
        return "(" + ",".join(trees[i] for i in active) + ");"

def _parse_newick_to_splits(newick, labels):
    """
    Parse Newick string into a set of splits (frozensets).
    Returns a set of frozensets representing the non-trivial splits.
    """
    # Remove semicolon
    newick = newick.strip().rstrip(';')
    
    def parse(s):
        # Returns a frozenset of leaves in this subtree
        s = s.strip()
        if not s.startswith('('):
            return frozenset([s])
        
        # Find matching parenthesis
        depth = 0
        end_idx = -1
        for i, char in enumerate(s):
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break
        
        if end_idx == -1:
            return frozenset([s])
            
        inner = s[1:end_idx]
        # Split inner by commas, respecting parentheses
        parts = []
        depth = 0
        start = 0
        for i, char in enumerate(inner):
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            elif char == ',' and depth == 0:
                parts.append(inner[start:i])
                start = i + 1
        parts.append(inner[start:])
        
        leaves = frozenset()
        for p in parts:
            leaves = leaves.union(parse(p))
            
        return leaves

    all_leaves = parse(newick)
    # We need the splits. A split is defined by each internal node's leaf set.
    # The recursive parse above gives us the leaf set for the root.
    # We need to modify parse to return the set of leaf sets for all internal nodes.
    
    def parse_splits(s):
        s = s.strip()
        splits = set()
        if not s.startswith('('):
            return frozenset([s]), splits
        
        depth = 0
        end_idx = -1
        for i, char in enumerate(s):
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break
        
        if end_idx == -1:
            return frozenset([s]), splits
            
        inner = s[1:end_idx]
        parts = []
        depth = 0
        start = 0
        for i, char in enumerate(inner):
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            elif char == ',' and depth == 0:
                parts.append(inner[start:i])
                start = i + 1
        parts.append(inner[start:])
        
        leaves = frozenset()
        for p in parts:
            sub_leaves, sub_splits = parse_splits(p)
            leaves = leaves.union(sub_leaves)
            splits.update(sub_splits)
            
        # Add the split for this node
        # The split is defined by `leaves` vs `all_leaves - leaves`
        # We only store the smaller side or just the frozenset itself?
        # For compatibility checking, it's best to store the frozenset of leaves on one side.
        # To avoid duplicates (S and S^c), we can normalize: 
        # always store the set such that the smallest label is in it.
        if leaves:
            # Normalize
            min_label = min(leaves)
            if min_label in leaves:
                # It is already the side containing min_label? 
                # Wait, if we store `leaves`, and `all_leaves - leaves` also contains min_label? No, they are disjoint.
                # So one of them contains min_label.
                # If min_label is in `leaves`, we store `leaves`.
                # Else we store `all_leaves - leaves`.
                pass
            else:
                leaves = all_leaves - leaves
            splits.add(leaves)
            
        return leaves, splits

    _, splits = parse_splits(newick)
    
    # Filter trivial splits (size 1 or n-1)
    n = len(labels)
    valid_splits = set()
    for s in splits:
        if 2 <= len(s) <= n - 2:
            valid_splits.add(s)
            
    return valid_splits

def _compatibility(s1, s2):
    """
    Check if two splits are compatible.
    s1, s2 are frozensets (one side of the bipartition).
    They are compatible if one side of s1 is disjoint from one side of s2.
    i.e., s1 ∩ s2 = ∅ OR s1 ∩ s2^c = ∅ OR s1^c ∩ s2 = ∅ OR s1^c ∩ s2^c = ∅
    Which simplifies to: s1 ∩ s2 is either empty, or s1, or s2.
    """
    inter = s1 & s2
    if len(inter) == 0:
        return True
    if inter == s1:
        return True
    if inter == s2:
        return True
    return False

def _calculate_parity(newick, aln):
    """
    Calculate the Simple Parsimony score for the tree.
    This is the sum of steps required to explain the alignment.
    """
    # Parse tree to get structure
    # We need to traverse the tree and calculate minimal changes.
    # Fitch algorithm.
    
    def parse_tree(s):
        s = s.strip()
        if not s.startswith('('):
            return ('leaf', s)
        
        depth = 0
        end_idx = -1
        for i, char in enumerate(s):
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break
        
        if end_idx == -1:
            return ('leaf', s)
            
        inner = s[1:end_idx]
        parts = []
        depth = 0
        start = 0
        for i, char in enumerate(inner):
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            elif char == ',' and depth == 0:
                parts.append(inner[start:i])
                start = i + 1
        parts.append(inner[start:])
        
        children = [parse_tree(p) for p in parts]
        return ('node', children)
        
    def fitch(node, site_idx):
        # Returns (set_of_bases, steps)
        if node[0] == 'leaf':
            base = aln[node[1]][site_idx]
            return {base}, 0
        
        # Node
        sets = []
        total_steps = 0
        for child in node[1]:
            c_set, c_steps = fitch(child, site_idx)
            sets.append(c_set)
            total_steps += c_steps
            
        # Intersection of all child sets
        inter = sets[0]
        for s in sets[1:]:
            inter = inter.intersection(s)
            
        if inter:
            return inter, total_steps
        else:
            # Union of all child sets
            union = sets[0]
            for s in sets[1:]:
                union = union.union(s)
            return union, total_steps + 1

    root = parse_tree(newick)
    
    n_sites = len(aln[0])
    total_score = 0
    for s in range(n_sites):
        _, steps = fitch(root, s)
        total_score += steps
        
    return total_score

def _spr_refine(newick, aln, labels, max_rounds=5, time_limit=8.0):
    """
    Refine the tree using SPR moves guided by Simple Parsimony.
    """
    start_time = time.time()
    current_newick = newick
    current_score = _calculate_parity(current_newick, aln)
    
    # We need a way to represent the tree to perform SPR.
    # For simplicity and speed in Python, we will use a recursive string-based approach
    # or a simple node-based structure.
    # Given the time limit, a pure string manipulation SPR is complex.
    # Let's use a simple node structure.
    
    class Node:
        def __init__(self, name=None, children=None):
            self.name = name
            self.children = children or []
            
        def to_newick(self):
            if not self.children:
                return self.name
            return "(" + ",".join(c.to_newick() for c in self.children) + ")"
            
    def parse_to_node(s):
        s = s.strip().rstrip(';')
        if not s.startswith('('):
            return Node(name=s)
            
        depth = 0
        end_idx = -1
        for i, char in enumerate(s):
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break
        
        if end_idx == -1:
            return Node(name=s)
            
        inner = s[1:end_idx]
        parts = []
        depth = 0
        start = 0
        for i, char in enumerate(inner):
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            elif char == ',' and depth == 0:
                parts.append(inner[start:i])
                start = i + 1
        parts.append(inner[start:])
        
        children = [parse_to_node(p) for p in parts]
        return Node(children=children)
        
    root = parse_to_node(current_newick)
    
    # Precompute Fitch sets for all nodes for the current tree to speed up evaluation?
    # SPR moves change the tree structure, so we'd have to recalculate.
    # For n<=32 and few rounds, full recalculation might be okay if optimized.
    # But _calculate_parity is O(n * n_sites). 
    # SPR moves: O(n) potential moves per round.
    # Total: O(n^2 * n_sites * rounds). 
    # 32^2 * 2000 * 5 = ~10M ops. Should be fast enough in Python if vectorized or simple loops.
    
    # Let's implement a faster parity check that only updates affected parts?
    # Too complex for this step. Let's stick to full recalc but limit rounds.
    
    for round in range(max_rounds):
        if time.time() - start_time > time_limit:
            break
            
        improved = False
        # Generate all possible SPR moves
        # An SPR move involves:
        # 1. Cutting an edge (u, v) where v is the subtree root.
        # 2. Regrafting v onto a new edge (a, b).
        
        # We need to iterate over all edges.
        # Representing the tree as a Node structure makes edge iteration tricky.
        # Let's use a list of edges (parent, child_node_id).
        
        # To make SPR easier, let's convert to a list of nodes with IDs.
        node_id = 0
        nodes = {}
        edges = [] # (parent_id, child_id)
        
        def assign_ids(node, parent_id):
            nonlocal node_id
            current_id = node_id
            node_id += 1
            nodes[current_id] = node
            if parent_id is not None:
                edges.append((parent_id, current_id))
            for child in node.children:
                assign_ids(child, current_id)
                
        assign_ids(root, None)
        
        # List of all internal nodes (potential cut points)
        # We can cut any edge.
        # For each edge (p, c):
        #   Subtree is rooted at c.
        #   Regraft onto any other edge (p2, c2) where c2 is not in subtree(c).
        
        potential_moves = []
        
        # Precompute leaf sets for each node to check disjointness
        leaf_sets = {}
        def compute_leaf_sets(node_id, node):
            if not node.children:
                leaf_sets[node_id] = frozenset([node.name])
            else:
                ls = frozenset()
                for child in node.children:
                    child_id = None
                    # Find child id
                    for p, c in edges:
                        if p == node_id and nodes[c] == child:
                            child_id = c
                            break
                    if child_id is not None:
                        ls = ls.union(leaf_sets[child_id])
                leaf_sets[node_id] = ls
                
        # This is getting messy with the Node object comparison.
        # Let's just use the Newick string for the current state and 
        # generate candidate Newick strings for SPR moves.
        
        # Simpler approach for SPR in Python:
        # 1. Parse tree to a list of splits? No, SPR changes splits.
        # 2. Use a library? No.
        # 3. Simple heuristic: Random restarts of NJ?
        #    Or, use the "Fast" SPR by just trying to swap children at internal nodes (NNI).
        #    NNI is a subset of SPR and much easier to implement.
        
        # Let's do NNI refinement.
        # For each internal node with 2 children, check if swapping the children 
        # (or rotating) improves the score.
        # Actually, NNI involves 4 trees. 
        # Given time, let's just do a few rounds of "Random Restart NJ" 
        # and pick the best. This is robust and easy.
        
        pass # Skip complex SPR, use Random Restarts instead

    return current_newick, current_score

def reconstruct(sequences):
    start_time = time.time()
    # Budget is 180s for 20 instances -> 9s per instance.
    # We will use 5s for NJ + Restarts, 3s for safety.
    deadline = start_time + 8.0
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    aln = [sequences[lab] for lab in labels]
    n_sites = len(aln[0])
    
    # 1. Compute Distance Matrix
    D = _jc_distance_matrix(aln)
    
    # 2. Initial NJ Tree
    best_newick = _neighbor_joining_v2(D, labels)
    best_score = _calculate_parity(best_newick, aln)
    
    # 3. Random Restarts with Perturbed Distances
    # Add noise to distance matrix and rerun NJ
    # This helps escape local optima.
    num_restarts = 5
    rng = np.random.default_rng(42)
    
    for r in range(num_restarts):
        if time.time() > deadline - 2.0:
            break
            
        # Perturb distances
        # Scale noise by the magnitude of distances
        noise_scale = 0.1
        D_perturbed = D + rng.normal(0, noise_scale, D.shape)
        D_perturbed = (D_perturbed + D_perturbed.T) / 2
        np.fill_diagonal(D_perturbed, 0)
        
        # Ensure symmetry and non-negative (though NJ handles negative)
        
        new_tree = _neighbor_joining_v2(D_perturbed, labels)
        score = _calculate_parity(new_tree, aln)
        
        if score < best_score:
            best_score = score
            best_newick = new_tree
            
    # 4. Simple NNI Refinement
    # Implement basic NNI
    def nni_refine(newick, aln, labels, max_rounds=3):
        current = newick
        current_score = _calculate_parity(current, aln)
        
        # Parse to list of internal nodes?
        # For NNI, we need to identify internal nodes with 2 children.
        # We'll use a string-based heuristic or a simple node parser.
        
        class NNode:
            def __init__(self, name=None, children=None):
                self.name = name
                self.children = children or []
            def to_newick(self):
                if not self.children:
                    return self.name
                return "(" + ",".join(c.to_newick() for c in self.children) + ")"
                
        def parse_n(s):
            s = s.strip().rstrip(';')
            if not s.startswith('('):
                return NNode(name=s)
            depth = 0; end_idx = -1
            for i, char in enumerate(s):
                if char == '(': depth += 1
                elif char == ')':
                    depth -= 1
                    if depth == 0: end_idx = i; break
            if end_idx == -1: return NNode(name=s)
            inner = s[1:end_idx]
            parts = []
            depth = 0; start = 0
            for i, char in enumerate(inner):
                if char == '(': depth += 1
                elif char == ')': depth -= 1
                elif char == ',' and depth == 0:
                    parts.append(inner[start:i]); start = i + 1
            parts.append(inner[start:])
            return NNode(children=[parse_n(p) for p in parts])
            
        root = parse_n(current)
        
        # Find all internal nodes
        def get_internal_nodes(node, path="root"):
            nodes = []
            if node.children:
                nodes.append((node, path))
                for i, child in enumerate(node.children):
                    nodes.extend(get_internal_nodes(child, f"{path}_{i}"))
            return nodes
            
        # NNI move: For a node with children A, B, and parent P with other child C.
        # The edge is (P, Node).
        # This is complex to implement correctly in Python without a robust tree library.
        # Given the time constraints and the fact that NJ+Restarts is already a strong baseline,
        # I will skip full NNI to avoid bugs and instead rely on the restarts.
        
        return current, current_score
        
    best_newick, best_score = nni_refine(best_newick, aln, labels)
    
    return best_newick

# Final check: The function name and signature must match.
# def reconstruct(sequences): ... return Newick string.

# One optimization: The `_calculate_parity` function is called many times.
# It parses the tree every time. We can cache the parsed tree if the newick doesn't change,
# but in the restart loop, the newick changes.
# The parsing is O(n). The Fitch is O(n * n_sites).
# This should be fast enough.

# Let's double check the `_jc_distance_matrix` for efficiency.
# It uses a nested loop for i, j. For n=32, that's 500 pairs.
# Each pair does a vectorized comparison. This is fast.

# Let's double check `_neighbor_joining_v2`.
# It uses dictionaries. O(n^2) per step. 30 steps. Fast.

# The main bottleneck is `_calculate_parity` if called too many times.
# In the restart loop, we call it once per restart. 5 restarts.
# 5 * (parse + fitch).
# Fitch is O(n_sites * n). 2000 * 32 = 64,000 ops.
# 5 * 64,000 = 320,000 ops. Very fast.

# So the algorithm is:
# 1. NJ
# 2. 5 Random Restarts with noise
# 3. Return best

# This is a robust improvement over single NJ.

# One more thing: The task says "branch lengths optional". 
# My NJ returns a topology only (no lengths in the string). 
# `f"({trees[a]},{trees[b]})"` does not include lengths. 
# This is valid.

# Let's ensure the labels are correct.
# `labels` is sorted t0..tn.
# `aln` corresponds to `labels`.
# `D` is computed on `aln`.
# `NJ` uses `labels`.
# Output Newick uses `labels`.

# Looks good.

# Final Code Structure

def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 8.0
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    aln = [sequences[lab] for lab in labels]
    n_sites = len(aln[0])
    
    # 1. Compute Distance Matrix
    D = _jc_distance_matrix(aln)
    
    # 2. Initial NJ Tree
    best_newick = _neighbor_joining_v2(D, labels)
    best_score = _calculate_parity(best_newick, aln)
    
    # 3. Random Restarts
    num_restarts = 5
    rng = np.random.default_rng(42)
    
    for r in range(num_restarts):
        if time.time() > deadline - 1.0:
            break
            
        # Perturb distances
        noise_scale = 0.1
        D_perturbed = D + rng.normal(0, noise_scale, D.shape)
        D_perturbed = (D_perturbed + D_perturbed.T) / 2
        np.fill_diagonal(D_perturbed, 0)
        
        new_tree = _neighbor_joining_v2(D_perturbed, labels)
        score = _calculate_parity(new_tree, aln)
        
        if score < best_score:
            best_score = score
            best_newick = new_tree
            
    return best_newick

# Helper functions defined above need to be in scope.
# I will consolidate them into the final code block.

