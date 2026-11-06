"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''Phylogenetic reconstruction: NJ + quartet voting + NNI refinement under Jukes-Cantor.'''
import math
import time
import numpy as np
from itertools import combinations


def _jc_distance(a, b):
    differences = sum(1 for x, y in zip(a, b) if x != y)
    p = differences / len(a)
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _seqs_to_array(sequences, labels):
    n = len(labels)
    L = len(sequences[labels[0]])
    code = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    arr = np.zeros((n, L), dtype=np.int8)
    for i, lab in enumerate(labels):
        seq = sequences[lab]
        for j, c in enumerate(seq):
            arr[i, j] = code[c]
    return arr


def _jc_matrix(sequences, labels):
    n = len(labels)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
            D[i, j] = d
            D[j, i] = d
    return D


def _nj_reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n <= 3:
        return "(" + ",".join(labels) + ");"
    
    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    D = _jc_matrix(sequences, labels)
    
    while len(nodes) > 3:
        m = len(nodes)
        r = np.zeros(m)
        for i, a in enumerate(nodes):
            for j, b in enumerate(nodes):
                if a != b:
                    r[i] += D[a, b]
        
        best_criterion = float('inf')
        best_i, best_j = -1, -1
        for i in range(m):
            for j in range(i + 1, m):
                criterion = (m - 2) * D[nodes[i], nodes[j]] - r[i] - r[j]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_i, best_j = i, j
        
        a, b = nodes[best_i], nodes[best_j]
        new_node = max(nodes) + 1
        trees[new_node] = f"({trees[a]},{trees[b]})"
        
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            D[new_node, x] = D[x, new_node] = 0.5 * (D[a, x] + D[b, x] - D[a, b])
        
        nodes = rest + [new_node]
    
    return "(" + ",".join(trees[x] for x in nodes) + ");"


def _parse_tree(s):
    """Parse Newick into nested tuple structure. Returns ('node', left, right) or ('leaf', label, None)."""
    s = s.strip().rstrip(';')
    pos = [0]
    
    def parse_node():
        if pos[0] >= len(s):
            return None
        if s[pos[0]] == '(':
            pos[0] += 1
            left = parse_node()
            if pos[0] < len(s) and s[pos[0]] == ',':
                pos[0] += 1
                right = parse_node()
            else:
                right = None
            if pos[0] < len(s) and s[pos[0]] == ')':
                pos[0] += 1
            if pos[0] < len(s) and s[pos[0]] == ':':
                pos[0] += 1
                while pos[0] < len(s) and s[pos[0]] not in '(),;':
                    pos[0] += 1
            return ('node', left, right)
        else:
            start = pos[0]
            while pos[0] < len(s) and s[pos[0]] not in '(),;':
                pos[0] += 1
            label = s[start:pos[0]]
            if pos[0] < len(s) and s[pos[0]] == ':':
                pos[0] += 1
                while pos[0] < len(s) and s[pos[0]] not in '(),;':
                    pos[0] += 1
            return ('leaf', label, None)
    
    return parse_node()


def _to_newick(node):
    if node is None:
        return ''
    if node[0] == 'leaf':
        return node[1]
    else:
        return f"({_to_newick(node[1])},{_to_newick(node[2])})"


def _get_leaves(node, leaves=None):
    if leaves is None:
        leaves = []
    if node is None:
        return leaves
    if node[0] == 'leaf':
        leaves.append(node[1])
    else:
        _get_leaves(node[1], leaves)
        _get_leaves(node[2], leaves)
    return leaves


def _quartet_vote(D, labels):
    """
    For every 4-taxon subset, determine the best quartet topology using distance sum.
    Returns a dict mapping frozenset of 4 labels to the best split (pair of pairs).
    """
    n = len(labels)
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    quartets = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    a, b, c, d = labels[i], labels[j], labels[k], labels[l]
                    ia, ib, ic, id_ = label_to_idx[a], label_to_idx[b], label_to_idx[c], label_to_idx[d]
                    
                    # Three possible topologies:
                    # 0: ((a,b)|(c,d)) -> score = D[ia,ib] + D[ic,id_]
                    # 1: ((a,c)|(b,d)) -> score = D[ia,ic] + D[ib,id_]
                    # 2: ((a,d)|(b,c)) -> score = D[ia,id_] + D[ib,ic]
                    
                    s0 = D[ia, ib] + D[ic, id_]
                    s1 = D[ia, ic] + D[ib, id_]
                    s2 = D[ia, id_] + D[ib, ic]
                    
                    scores = [s0, s1, s2]
                    best = min(range(3), key=lambda x: scores[x])
                    
                    # Store the split as a frozenset of two frozensets
                    if best == 0:
                        split = frozenset(frozenset({a, b}), frozenset({c, d}))
                    elif best == 1:
                        split = frozenset(frozenset({a, c}), frozenset({b, d}))
                    else:
                        split = frozenset(frozenset({a, d}), frozenset({b, c}))
                    
                    quartets[frozenset({a, b, c, d})] = split
    
    return quartets


def _build_tree_from_quartets(quartets, labels):
    """
    Build a tree that agrees with as many quartets as possible.
    Use a greedy approach: start with NJ, then fix conflicts via NNI.
    """
    # First get NJ tree as starting point
    # Then we'll refine it in _nni_refine using quartet scores
    return None  # We'll just use NJ as starting point


def _quartet_score_for_tree(tree, D, label_to_idx, quartets):
    """
    Compute how many quartets the tree agrees with.
    For each internal edge, it defines a split. For each 4-taxon set,
    check if the tree's induced topology matches the quartet vote.
    Returns (num_agree, num_total).
    """
    # Extract all splits from the tree
    # A tree with n taxa has n-3 internal edges, each defining a split
    
    def extract_splits(node, splits=None):
        if splits is None:
            splits = []
        if node is None or node[0] == 'leaf':
            return splits
        left_leaves = frozenset(_get_leaves(node[1]))
        right_leaves = frozenset(_get_leaves(node[2]))
        if len(left_leaves) >= 2 and len(right_leaves) >= 2:
            splits.append((left_leaves, right_leaves))
        extract_splits(node[1], splits)
        extract_splits(node[2], splits)
        return splits
    
    splits = extract_splits(tree)
    
    # For each quartet, check if the tree agrees
    agree = 0
    total = len(quartets)
    
    for qset, voted_split in quartets.items():
        # voted_split is frozenset of two frozensets
        # Check if the tree induces this split
        # A tree agrees with a quartet split if there's an edge that separates
        # the two pairs correctly
        
        # For quartet {a,b,c,d} with voted split {X, Y} where X and Y are pairs:
        # The tree agrees if there exists a split (S, complement) such that
        # X ⊆ S and Y ⊆ complement, or X ⊆ complement and Y ⊆ S
        
        found = False
        for (left, right) in splits:
            # Check if the split separates the two pairs
            for pair1, pair2 in [(voted_split[0], voted_split[1]), (voted_split[1], voted_split[0])]:
                if pair1 <= left and pair2 <= right:
                    found = True
                    break
                if pair1 <= right and pair2 <= left:
                    found = True
                    break
            if found:
                break
        
        if found:
            agree += 1
    
    return agree, total


def _nni_refine_quartet(tree, D, labels, quartets, deadline):
    """
    Refine tree using NNI moves to maximize quartet agreement.
    """
    n = len(labels)
    if n < 4:
        return tree
    
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    
    # Get current quartet agreement
    current_agree, current_total = _quartet_score_for_tree(tree, D, label_to_idx, quartets)
    
    improved = True
    max_rounds = 10
    
    for _ in range(max_rounds):
        if time.time() > deadline - 1.0:
            break
        if not improved:
            break
        improved = False
        
        # Try NNI moves at each internal edge
        # For each internal edge (connecting two internal nodes), try the 2 alternative topologies
        
        def try_nni_moves(node, parent=None):
            """Recursively try NNI moves. Returns (new_node, improved)."""
            if node is None or node[0] == 'leaf':
                return node, False
            
            if node[1] is None or node[2] is None:
                return node, False
            
            # First, recurse into children
            new_left, imp1 = try_nni_moves(node[1])
            new_right, imp2 = try_nni_moves(node[2])
            node = ('node', new_left, new_right)
            
            if imp1 or imp2:
                return node, True
            
            # Now try NNI at this node
            # Current: node = (A, B) where A and B are subtrees
            # NNI alternatives:
            # If A = (A1, A2) and B is a subtree, we can try:
            #   ((A1, B), A2) or ((A2, B), A1)
            # If B = (B1, B2) and A is a subtree, we can try:
            #   ((B1, A), B2) or ((B2, A), B1)
            
            best_node = node
            best_agree = current_agree
            
            # Try NNI if left child is internal
            if node[1][0] == 'node':
                A1 = node[1][1]
                A2 = node[1][2]
                B = node[2]
                
                # Alternative 1: ((A1, B), A2)
                cand1 = ('node', ('node', A1, B), A2)
                agree1, _ = _quartet_score_for_tree(cand1, D, label_to_idx, quartets)
                if agree1 > best_agree:
                    best_agree = agree1
                    best_node = cand1
                
                # Alternative 2: ((A2, B), A1)
                cand2 = ('node', ('node', A2, B), A1)
                agree2, _ = _quartet_score_for_tree(cand2, D, label_to_idx, quartets)
                if agree2 > best_agree:
                    best_agree = agree2
                    best_node = cand2
            
            # Try NNI if right child is internal
            if node[2][0] == 'node':
                B1 = node[2][1]
                B2 = node[2][2]
                A = node[1]
                
                # Alternative 3: ((B1, A), B2)
                cand3 = ('node', ('node', B1, A), B2)
                agree3, _ = _quartet_score_for_tree(cand3, D, label_to_idx, quartets)
                if agree3 > best_agree:
                    best_agree = agree3
                    best_node = cand3
                
                # Alternative 4: ((B2, A), B1)
                cand4 = ('node', ('node', B2, A), B1)
                agree4, _ = _quartet_score_for_tree(cand4, D, label_to_idx, quartets)
                if agree4 > best_agree:
                    best_agree = agree4
                    best_node = cand4
            
            if best_agree > current_agree:
                return best_node, True
            
            return node, False
        
        new_tree, improved = try_nni_moves(tree)
        if improved:
            tree = new_tree
            current_agree, _ = _quartet_score_for_tree(tree, D, label_to_idx, quartets)
    
    return tree


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n <= 3:
        return "(" + ",".join(labels) + ");"
    
    start = time.time()
    deadline = start + 8.5  # Safe margin under 180s/20 = 9s per instance
    
    try:
        # Step 1: Compute JC distance matrix
        D = _jc_matrix(sequences, labels)
        
        # Step 2: Get quartet votes
        quartets = _quartet_vote(D, labels)
        
        # Step 3: Get NJ tree as starting point
        tree_newick = _nj_reconstruct(sequences)
        tree = _parse_tree(tree_newick)
        
        # Step 4: NNI refinement using quartet agreement
        if time.time() < deadline - 1.0:
            tree = _nni_refine_quartet(tree, D, labels, quartets, deadline)
        
        result = _to_newick(tree) + ";"
        if result and result.strip():
            return result
    except Exception:
        pass
    
    # Fallback to NJ
    try:
        result = _nj_reconstruct(sequences)
        if result and result.strip():
            return result
    except Exception:
        pass
    
    # Last resort: star tree
    return "(" + ",".join(labels) + ");"
# EVOLVE-BLOCK-END
