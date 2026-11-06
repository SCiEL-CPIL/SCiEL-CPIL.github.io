# EVOLVE-BLOCK-START
import time
import math
import numpy as np
import sys

_START = None
_DEADLINE_FRACTION = 0.85 # Stop at 85% of 270s to be safe

def _now():
    return time.time()

def _init_time():
    global _START
    if _START is None:
        _START = time.time()

def _time_left():
    _init_time()
    return max(0.0, 270.0 * _DEADLINE_FRACTION - (time.time() - _START))

def _seqs_to_array(sequences):
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    seqs = [sequences[l] for l in labels]
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    # Using uint8 for compactness
    arr = np.array([[mapping[c] for c in s] for s in seqs], dtype=np.uint8)
    return labels, arr

def _jc_matrix(arr):
    n, L = arr.shape
    # Calculate pairwise differences efficiently
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
        return labels[0], set()
    if n == 2:
        return f"({labels[0]},{labels[1]})", set()
    
    all_leaves = set(labels)
    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    leaf_sets = {i: {labels[i]} for i in range(n)}
    
    # Copy dist to dict for easy access/modification
    d = {}
    for i in range(n):
        d[i] = {}
        for j in range(n):
            d[i][j] = dist[i][j]
    
    splits = set()
    
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
        trees[new_id] = f"({trees[a]},{trees[b]})"
        leaf_sets[new_id] = leaf_sets[a] | leaf_sets[b]
        
        # Record split
        side1 = frozenset(leaf_sets[new_id])
        side2 = frozenset(all_leaves - leaf_sets[new_id])
        # Canonical form: lexicographically smaller frozenset
        if tuple(sorted(side1)) < tuple(sorted(side2)):
            split = side1
        else:
            split = side2
        if 1 < len(split) < n - 1:
            splits.add(split)
            
        d[new_id] = {}
        for x in nodes:
            if x == a or x == b:
                continue
            d[new_id][x] = 0.5 * (d[a][x] + d[b][x] - d[a][b])
            d[x][new_id] = d[new_id][x]
            
        nodes.remove(a)
        nodes.remove(b)
        nodes.append(new_id)
        
    final_tree = "(" + ",".join(trees[x] for x in nodes) + ")"
    return final_tree, splits

# --- Maximum Parsimony Implementation ---

class TreeNode:
    __slots__ = ['name', 'parent', 'children', 'idx']
    def __init__(self, name=None, idx=None):
        self.name = name
        self.parent = None
        self.children = []
        self.idx = idx # Index in the data array if leaf, None if internal

def parse_newick_to_tree(newick, labels):
    """
    Parse a Newick string into a TreeNode structure.
    Returns the root node.
    """
    s = newick.strip()
    if s.endswith(';'):
        s = s[:-1]
    
    # We need to map names to indices
    name_to_idx = {l: i for i, l in enumerate(labels)}
    
    root = TreeNode()
    stack = [root]
    i = 0
    while i < len(s):
        c = s[i]
        if c == '(':
            new_node = TreeNode()
            stack[-1].children.append(new_node)
            new_node.parent = stack[-1]
            stack.append(new_node)
        elif c == ',':
            pass
        elif c == ')':
            stack.pop()
        elif c == ':':
            while i < len(s) and s[i] not in ',)':
                i += 1
            i -= 1
        else:
            start = i
            while i < len(s) and s[i] not in '(),':
                i += 1
            name = s[start:i]
            if name in name_to_idx:
                stack[-1].name = name
                stack[-1].idx = name_to_idx[name]
            else:
                stack[-1].name = name
                stack[-1].idx = None
        i += 1
    
    # Unroot the tree if necessary (promote single child)
    while len(root.children) == 1:
        root = root.children[0]
        root.parent = None
    return root

def get_postorder(root):
    order = []
    stack = [root]
    while stack:
        node = stack.pop()
        order.append(node)
        for child in node.children:
            stack.append(child)
    return order

def fitch_score_vectorized(root, data):
    """
    Calculate Fitch parsimony score for all sites simultaneously.
    data: np array of shape (n_taxa, L) with values 0-3.
    """
    order = get_postorder(root)
    # states[node] is an array of shape (L,) containing bitmask states (uint8)
    states = {}
    
    total_score = 0
    
    # Process in reverse post-order (leaves to root)
    for node in reversed(order):
        if not node.children:
            # Leaf
            idx = node.idx
            # Convert 0-3 to bitmask: 1<<val
            # data[idx] is array of 0-3
            # We need bitmasks.
            # 0->1, 1->2, 2->4, 3->8
            # We can precompute a lookup table?
            # Or just use np.where?
            # Simpler: create a lookup array
            # LUT[0]=1, LUT[1]=2, LUT[2]=4, LUT[3]=8
            # states[node] = LUT[data[idx]]
            # Since data is uint8, we can index a precomputed array.
            # But we don't have a global LUT.
            # Let's create it once outside? Or just use a vectorized shift?
            # 1 << data[idx] works in numpy? Yes, if data is integer.
            states[node] = 1 << data[idx]
        else:
            # Internal
            # Get states of children
            # Note: children are already processed because we are in reversed post-order
            s1 = states[node.children[0]]
            if len(node.children) == 1:
                states[node] = s1
                continue
            
            s2 = states[node.children[1]]
            # If more than 2 children (shouldn't happen in binary tree, but safe)
            for c in node.children[2:]:
                sc = states[c]
                # Union of s2 and sc, then intersect with s1?
                # Fitch is associative.
                # Let's just assume binary for now, as NJ produces binary trees.
                # If polytomy, we can iteratively combine.
                s2 = s2 | sc # This is not strictly Fitch for >2 children in one step without intermediate scores, 
                             # but for unrooted binary trees converted to rooted, internal nodes have 2 children 
                             # except root which has 2 children (if we unroot properly) or 3?
                             # Standard unrooted binary tree rooted at an edge: root has 2 children.
                             # All other internal nodes have 2 children.
                             # Leaves have 0.
                             # So binary assumption holds.
            
            inter = s1 & s2
            non_empty = inter > 0
            # Score: count of sites where intersection is empty
            total_score += np.sum(~non_empty)
            
            # New state: intersection if non-empty, else union
            states[node] = np.where(non_empty, inter, s1 | s2)
            
    return total_score

def tree_to_newick(root):
    """
    Convert TreeNode structure to Newick string.
    """
    if not root.children:
        return root.name
    
    parts = []
    for child in root.children:
        parts.append(tree_to_newick(child))
    
    return "(" + ",".join(parts) + ")"

def spr_refine(root, data, time_limit=10.0):
    """
    Refine tree using SPR moves and Fitch score.
    """
    current_score = fitch_score_vectorized(root, data)
    best_score = current_score
    best_root = root
    
    start_time = time.time()
    
    # Precompute postorder for score calculation? 
    # Fitch score recalculates postorder every time.
    # We can cache the structure if we modify it in place?
    # SPR modifies structure.
    
    # Helper to get all edges (parent, child) where child is not root
    def get_edges(node):
        edges = []
        stack = [node]
        while stack:
            n = stack.pop()
            if n.parent is not None:
                edges.append((n.parent, n))
            for c in n.children:
                stack.append(c)
        return edges

    # Helper to perform SPR and return new root
    # This modifies the tree in place.
    # We need to be able to revert if the move is bad.
    # Easier: Clone the tree? Too slow.
    # Alternative: Perform move, calc score, if bad, revert.
    
    def perform_spr(parent, child, new_parent):
        """
        Prune 'child' from 'parent'.
        Regraft 'child' onto 'new_parent'.
        Returns new_root.
        """
        # 1. Detach child from parent
        parent.children.remove(child)
        child.parent = None
        
        # 2. Regraft
        # new_parent is a node.
        # We insert 'child' as a child of 'new_parent'.
        # But if new_parent is a leaf, we need to create a new internal node?
        # No, in an unrooted tree, if we regraft onto a leaf, that leaf becomes internal.
        # However, our representation is rooted.
        # If new_parent is a leaf (no children), adding a child makes it internal.
        # This is valid.
        
        # If new_parent is internal, adding a child makes it 3-way (if it had 2) or 4-way.
        # This is valid for parsimony.
        
        # BUT: We must ensure we don't create a cycle.
        # 'child' must not be an ancestor of 'new_parent'.
        # And 'new_parent' must not be 'parent' (trivial) or 'child'.
        
        # Check if new_parent is in the subtree of child?
        # If so, cycle.
        # We can check by traversing up from new_parent to root.
        # If we hit child, it's a cycle.
        
        # Let's assume the caller filters valid moves.
        
        # Add child to new_parent
        new_parent.children.append(child)
        child.parent = new_parent
        
        # Update root if parent was root?
        # If parent was root, and we removed child, parent might have 1 child left.
        # If parent has 0 children, it becomes a leaf?
        # If parent was root, and now has 1 child, we should promote that child to root?
        # Or keep parent as root with 1 child?
        # For Fitch, root with 1 child is fine.
        # But for SPR validity, we usually keep the tree binary?
        # No, parsimony allows polytomies.
        
        # Determine new root
        # If parent was root, root is still parent (unless parent became empty, which doesn't happen)
        # If new_parent was root, root is still new_parent.
        # Root doesn't change unless we detach the root? We never detach root.
        # Wait, if parent was root, and we remove child, root still exists.
        
        return root # Root pointer doesn't change in this simple representation?
        # Actually, if we prune from root, root remains root.
        # If we regraft onto root, root remains root.
        # So root is stable.
    
    # To allow reverting, we need to save the state.
    # State: 
    # 1. parent.children list (remove child)
    # 2. child.parent (set to None)
    # 3. new_parent.children list (add child)
    # 4. child.parent (set to new_parent)
    
    # We can just try the move, calculate score, and if bad, undo.
    
    def undo_spr(parent, child, new_parent):
        # Undo reattach
        new_parent.children.remove(child)
        child.parent = None
        # Undo detach
        parent.children.append(child)
        child.parent = parent
    
    # Main loop
    while _time_left() > time_limit:
        edges = get_edges(root)
        if not edges:
            break
            
        improved = False
        # Shuffle edges to avoid bias?
        # np.random.shuffle(edges)
        
        for parent, child in edges:
            if _time_left() < time_limit:
                break
                
            # Find valid new_parent candidates
            # All nodes except:
            # 1. child itself
            # 2. parent (trivial)
            # 3. Ancestors of child (would create cycle)
            # 4. Descendants of child (would create cycle? No, regrafting onto descendant is invalid)
            
            # Get all nodes
            all_nodes = get_postorder(root)
            
            # Find ancestors of child
            ancestors = set()
            curr = child.parent
            while curr:
                ancestors.add(curr)
                curr = curr.parent
            
            # Find descendants of child
            descendants = set()
            stack = [child]
            while stack:
                n = stack.pop()
                descendants.add(n)
                for c in n.children:
                    stack.append(c)
            
            valid_parents = []
            for n in all_nodes:
                if n == child or n == parent:
                    continue
                if n in ancestors or n in descendants:
                    continue
                valid_parents.append(n)
            
            if not valid_parents:
                continue
                
            # Try each valid parent
            for new_parent in valid_parents:
                # Perform move
                perform_spr(parent, child, new_parent)
                
                # Calculate score
                new_score = fitch_score_vectorized(root, data)
                
                if new_score < current_score:
                    # Accept move
                    current_score = new_score
                    improved = True
                    # Update best if needed?
                    # We are doing hill climbing, so current becomes best.
                    break # Restart loop with new topology
                else:
                    # Undo move
                    undo_spr(parent, child, new_parent)
                    
        if not improved:
            break
            
    return root, current_score

def reconstruct(sequences):
    global _START
    _START = time.time()
    
    labels, arr = _seqs_to_array(sequences)
    n = len(labels)
    
    d_full = _jc_matrix(arr)
    tree_full, splits_full = _nj(d_full, labels)
    
    candidates = [(tree_full, splits_full)]
    
    L = arr.shape[1]
    num_bootstraps = 0
    max_bootstraps = 100 # Reduced to allow time for SPR
    
    # Time budget check
    # NJ is fast. Let's do 50 bootstraps.
    while num_bootstraps < max_bootstraps and _time_left() > 20.0:
        indices = np.random.randint(0, L, L)
        arr_boot = arr[:, indices]
        d_boot = _jc_matrix(arr_boot)
        tree_boot, splits_boot = _nj(d_boot, labels)
        candidates.append((tree_boot, splits_boot))
        num_bootstraps += 1
        
    # Select best candidate
    best_tree = tree_full
    if len(candidates) > 1:
        best_idx = 0
        best_score = float('inf')
        n_cands = len(candidates)
        
        for i in range(n_cands):
            score = 0
            splits_i = candidates[i][1]
            for j in range(n_cands):
                if i != j:
                    splits_j = candidates[j][1]
                    score += len(splits_i.symmetric_difference(splits_j))
            score /= (n_cands - 1)
            if score < best_score:
                best_score = score
                best_idx = i
                
        best_tree = candidates[best_idx][0]
        
    # Refine with SPR
    root = parse_newick_to_tree(best_tree, labels)
    refined_root, _ = spr_refine(root, arr, time_limit=5.0)
    final_newick = tree_to_newick(refined_root)
    
    return final_newick

# EVOLVE-BLOCK-END
