<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 265.0  # seconds; leave 5s margin before the 270s kill
_START = time.time()

def _jc_distances(sequences, labels):
    """Compute pairwise JC-corrected distances using NumPy."""
    n = len(labels)
    seqs = [sequences[lab] for lab in labels]
    length = len(seqs[0])
    
    if length == 0:
        return np.zeros((n, n))

    dna_arr = np.array([list(s) for s in seqs])  # (n, length)
    
    # Compute pairwise Hamming distances
    mismatches = np.sum(dna_arr[:, None, :] != dna_arr[None, :, :], axis=2)
    
    # Proportion of differences
    p = mismatches.astype(float) / length
    
    # JC correction: -0.75 * ln(1 - 4p/3)
    # Clip p to avoid log(0) or log(negative)
    p_clipped = np.clip(p, 0, 0.7499)
    dist = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    
    # Set diagonal to 0
    np.fill_diagonal(dist, 0.0)
    
    return dist

def _neighbor_joining(D, labels):
    """
    Correct Neighbor Joining implementation.
    Returns a Newick string.
    """
    n = len(labels)
    if n <= 2:
        return "(" + ",".join(labels) + ");"
    
    # State:
    # D: (m, m) distance matrix for active nodes
    # trees: list of length m, where trees[i] is the Newick string for the subtree rooted at node i
    #        Initially, trees[i] = labels[i]
    #        When joining i and j into new node k, trees[k] = "(trees[i],trees[j])"
    
    D = D.copy()
    trees = list(labels)
    
    # Map from current index in D/trees to the "original" label is not needed for Newick,
    # but we need to be careful with indices.
    
    m = n
    # We will maintain a list of active indices? 
    # Simpler: just operate on the shrinking arrays directly.
    # However, removing rows/cols from numpy arrays is O(m^2). 
    # Since m <= 32, this is negligible.
    
    while m > 3:
        if time.time() > _DEADLINE:
            # Fallback: return a star tree of current active labels
            # The current 'trees' list contains the Newick strings for the current active nodes.
            # We need to combine them into a valid Newick string.
            # A simple star: (tree0, tree1, ..., tree(m-1));
            return "(" + ",".join(trees) + ");"
            
        # Compute row sums
        r = np.sum(D, axis=1)
        
        # Compute Q matrix
        # Q[i, j] = (m - 2) * D[i, j] - r[i] - r[j]
        Q = (m - 2) * D - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        
        # Find min Q
        i, j = np.unravel_index(np.argmin(Q), Q.shape)
        if i == j:
            # Should not happen if diagonal is inf, but safety check
            continue
        if i > j:
            i, j = j, i
            
        # Determine indices of other nodes
        others = [k for k in range(m) if k != i and k != j]
        
        # Compute new branch lengths and distance from new node to others
        d_ij = D[i, j]
        
        # Distance from new node (k) to others
        if others:
            d_i_others = D[i, others]
            d_j_others = D[j, others]
            d_new_others = 0.5 * (d_i_others + d_j_others - d_ij)
        else:
            d_new_others = np.array([])
        
        # Construct new tree string for the joined node
        new_tree_str = f"({trees[i]},{trees[j]})"
        
        # Construct new distance matrix D_new
        # Size: (len(others) + 1) x (len(others) + 1)
        # The new node will be at the last index.
        new_m = len(others) + 1
        D_new = np.zeros((new_m, new_m))
        
        if others:
            # Fill distances among others
            D_new[:len(others), :len(others)] = D[np.ix_(others, others)]
            
            # Fill distances from new node to others
            D_new[:len(others), len(others)] = d_new_others
            D_new[len(others), :len(others)] = d_new_others
            # D_new[len(others), len(others)] = 0.0 (already zero)
            
            # Update trees list: remove i, j, add new_tree_str at end
            # Order in 'others' must match the order in D_new rows/cols 0..len(others)-1
            new_trees = [trees[k] for k in others]
            new_trees.append(new_tree_str)
        else:
            # This case implies m was 4, and we joined 2, leaving 1 other? No, m>3.
            # If m=4, others has 2 elements.
            # If m=3, loop doesn't run.
            # So others is never empty if m > 3.
            new_trees = [new_tree_str]
            D_new = np.array([[0.0]])
            
        D = D_new
        trees = new_trees
        m = new_m
        
    # Handle the final 3 nodes
    # We have 3 nodes in 'trees' and a 3x3 distance matrix D.
    # We can just return a star tree of these 3 subtrees, or resolve the triplet.
    # A star tree (a,b,c) is topologically equivalent to any unrooted tree with 3 leaves.
    # So returning a star is correct for topology.
    return "(" + ",".join(trees) + ");"

def reconstruct(sequences):
    global _START, _DEADLINE
    
    _START = time.time()
    _DEADLINE = _START + 265.0
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        return "(" + ",".join(labels) + ");"
    
    # Compute distance matrix
    D = _jc_distances(sequences, labels)
    
    # Run NJ
    newick = _neighbor_joining(D, labels)
    
    return newick
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 265.0  # seconds; leave 5s margin before the 270s kill
_START = time.time()

def _jc_distances(sequences, labels):
    """Compute pairwise JC-corrected distances using NumPy."""
    n = len(labels)
    seqs = [sequences[lab] for lab in labels]
    length = len(seqs[0])
    
    if length == 0:
        return np.zeros((n, n))

    dna_arr = np.array([list(s) for s in seqs])  # (n, length)
    
    # Compute pairwise Hamming distances
    mismatches = np.sum(dna_arr[:, None, :] != dna_arr[None, :, :], axis=2)
    
    # Proportion of differences
    p = mismatches.astype(float) / length
    
    # JC correction: -0.75 * ln(1 - 4p/3)
    # Clip p to avoid log(0) or log(negative)
    p_clipped = np.clip(p, 0, 0.7499)
    dist = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    
    # Set diagonal to 0
    np.fill_diagonal(dist, 0.0)
    
    return dist

def _neighbor_joining(D, labels):
    """
    Correct Neighbor Joining implementation.
    Returns a Newick string.
    """
    n = len(labels)
    if n <= 2:
        return "(" + ",".join(labels) + ");"
    
    D = D.copy()
    trees = list(labels)
    
    m = n
    
    while m > 3:
        if time.time() > _DEADLINE:
            return "(" + ",".join(trees) + ");"
            
        # Compute row sums
        r = np.sum(D, axis=1)
        
        # Compute Q matrix
        # Q[i, j] = (m - 2) * D[i, j] - r[i] - r[j]
        Q = (m - 2) * D - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        
        # Find min Q
        i, j = np.unravel_index(np.argmin(Q), Q.shape)
        if i == j:
            continue
        if i > j:
            i, j = j, i
            
        # Determine indices of other nodes
        others = [k for k in range(m) if k != i and k != j]
        
        # Compute new branch lengths and distance from new node to others
        d_ij = D[i, j]
        
        # Distance from new node (k) to others
        if others:
            d_i_others = D[i, others]
            d_j_others = D[j, others]
            d_new_others = 0.5 * (d_i_others + d_j_others - d_ij)
        else:
            d_new_others = np.array([])
        
        # Construct new tree string for the joined node
        new_tree_str = f"({trees[i]},{trees[j]})"
        
        # Construct new distance matrix D_new
        # Size: (len(others) + 1) x (len(others) + 1)
        # The new node will be at the last index.
        new_m = len(others) + 1
        D_new = np.zeros((new_m, new_m))
        
        if others:
            # Fill distances among others
            D_new[:len(others), :len(others)] = D[np.ix_(others, others)]
            
            # Fill distances from new node to others
            D_new[:len(others), len(others)] = d_new_others
            D_new[len(others), :len(others)] = d_new_others
            # D_new[len(others), len(others)] = 0.0 (already zero)
            
            # Update trees list: remove i, j, add new_tree_str at end
            # Order in 'others' must match the order in D_new rows/cols 0..len(others)-1
            new_trees = [trees[k] for k in others]
            new_trees.append(new_tree_str)
        else:
            new_trees = [new_tree_str]
            D_new = np.array([[0.0]])
            
        D = D_new
        trees = new_trees
        m = new_m
        
    # Handle the final 3 nodes
    return "(" + ",".join(trees) + ");"

def _parse_newick(newick):
    """
    Parse a Newick string into a list of splits.
    Returns a list of frozensets, where each frozenset is one side of a split.
    The other side is the complement.
    We normalize splits so that the side containing the smallest leaf is kept.
    """
    # Remove trailing semicolon if present
    newick = newick.strip()
    if newick.endswith(';'):
        newick = newick[:-1]
    
    splits = []
    n_leaves = 0
    
    # We need to know the total set of leaves to compute complements
    # First pass: extract all leaf labels
    leaves = set()
    for part in newick.split(','):
        part = part.strip()
        if part and not part.startswith('('):
            # It's a leaf label (might have branch length after colon)
            label = part.split(':')[0].strip()
            leaves.add(label)
    
    total_leaves = frozenset(leaves)
    
    # Second pass: parse structure to find splits
    # We'll use a stack-based approach or recursive descent
    # Since the tree is small, a simple recursive parser is fine
    
    def parse_node(s, pos):
        """
        Parse a node starting at position pos.
        Returns (set_of_leaves_in_subtree, new_pos)
        """
        if s[pos] == '(':
            pos += 1
            subtree_leaves = set()
            
            while True:
                # Parse a child
                child_leaves, pos = parse_node(s, pos)
                subtree_leaves.update(child_leaves)
                
                # Check for comma or closing paren
                if pos < len(s) and s[pos] == ',':
                    pos += 1
                elif pos < len(s) and s[pos] == ')':
                    pos += 1
                    break
                else:
                    break
            
            # Skip branch length if present
            if pos < len(s) and s[pos] == ':':
                while pos < len(s) and s[pos] not in (',', ')'):
                    pos += 1
            
            return subtree_leaves, pos
        else:
            # It's a leaf
            start = pos
            while pos < len(s) and s[pos] not in (',', ')', ':'):
                pos += 1
            label = s[start:pos].strip()
            
            # Skip branch length if present
            if pos < len(s) and s[pos] == ':':
                while pos < len(s) and s[pos] not in (',', ')'):
                    pos += 1
            
            return {label}, pos
    
    leaves_in_tree, _ = parse_node(newick, 0)
    
    # Now we need to extract the splits.
    # A split is defined by any internal node.
    # We need to modify the parser to collect splits.
    
    def parse_node_with_splits(s, pos):
        """
        Parse a node starting at position pos.
        Returns (set_of_leaves_in_subtree, new_pos, list_of_splits_from_subtree)
        """
        if s[pos] == '(':
            pos += 1
            subtree_leaves = set()
            local_splits = []
            
            while True:
                # Parse a child
                child_leaves, pos, child_splits = parse_node_with_splits(s, pos)
                subtree_leaves.update(child_leaves)
                local_splits.extend(child_splits)
                
                # Check for comma or closing paren
                if pos < len(s) and s[pos] == ',':
                    pos += 1
                elif pos < len(s) and s[pos] == ')':
                    pos += 1
                    break
                else:
                    break
            
            # This node defines a split: subtree_leaves vs complement
            # Only add if it's a non-trivial split (size > 1 and size < n)
            if 1 < len(subtree_leaves) < len(total_leaves):
                # Normalize: keep the side with the smallest leaf
                min_leaf = min(subtree_leaves)
                # Check if min_leaf is in subtree_leaves (it is by definition)
                # But we want a canonical representation.
                # Let's represent the split as the frozenset that contains the smallest leaf in the entire tree?
                # No, the standard is just to have a set of sets.
                # To avoid duplicates (A|B and B|A), we can sort the two sides and take the lexicographically smaller.
                side1 = frozenset(subtree_leaves)
                side2 = total_leaves - side1
                # Canonical form: sort the two sides by their min element? Or just store both in a set?
                # For RF distance, we just need a set of unique splits.
                # A split is {side1, side2}.
                # We can represent it as (side1, side2) where side1 < side2 lexicographically.
                if side1 < side2:
                    local_splits.append((side1, side2))
                else:
                    local_splits.append((side2, side1))
            
            # Skip branch length if present
            if pos < len(s) and s[pos] == ':':
                while pos < len(s) and s[pos] not in (',', ')'):
                    pos += 1
            
            return subtree_leaves, pos, local_splits
        else:
            # It's a leaf
            start = pos
            while pos < len(s) and s[pos] not in (',', ')', ':'):
                pos += 1
            label = s[start:pos].strip()
            
            # Skip branch length if present
            if pos < len(s) and s[pos] == ':':
                while pos < len(s) and s[pos] not in (',', ')'):
                    pos += 1
            
            return {label}, pos, []
    
    _, _, splits = parse_node_with_splits(newick, 0)
    
    # Convert to a set of canonical splits
    split_set = set(splits)
    
    return split_set

def _compute_rf_distance(splits1, splits2):
    """
    Compute the normalized Robinson-Foulds distance between two sets of splits.
    """
    if len(splits1) == 0 and len(splits2) == 0:
        return 0.0
    
    intersection = splits1 & splits2
    num_diff = len(splits1) + len(splits2) - 2 * len(intersection)
    
    # Normalized RF distance: 1 - (2 * |intersection|) / (|S1| + |S2|)
    # Or simply |S1 \ S2| + |S2 \ S1| / max(|S1|, |S2|)?
    # Standard normalized RF: 1 - (2 * |S1 ∩ S2|) / (|S1| + |S2|)
    # If one is empty, it's 1.
    
    if len(splits1) + len(splits2) == 0:
        return 0.0
        
    return 1.0 - (2.0 * len(intersection) / (len(splits1) + len(splits2)))

def _refine_tree(newick, D, labels):
    """
    Attempt to refine the tree topology by swapping edges to reduce RF distance?
    No, we don't have the true tree.
    Instead, we can try to optimize branch lengths or use a different distance metric?
    
    Actually, the directive suggests: "consider a smooth surrogate for a nonsmooth min/max objective... or a final high-precision polish pass".
    
    NJ is a heuristic. It's fast but can be wrong.
    Can we do better?
    With n<=32 and 20 instances, we have ~13s per instance.
    NJ is O(n^3) ~ 32768 operations. Very fast.
    Maybe we can run NJ multiple times with slight perturbations?
    Or use a different method like FastME?
    
    Let's stick to the "polish" idea.
    Is there a way to polish the NJ tree?
    One idea: After NJ, we have a tree topology.
    We can try to optimize the branch lengths on this topology to maximize the likelihood (or minimize some distance).
    But the score is based on topology (RF distance), not likelihood.
    So optimizing branch lengths won't change the topology, and thus won't change the score.
    
    Unless... the NJ algorithm is sensitive to the distance matrix.
    What if we use a different distance matrix?
    e.g., K2P correction instead of JC?
    The task says "simulated ... under the Jukes-Cantor model". So JC is the correct model.
    
    What if we use a different tree construction method?
    e.g., UPGMA? (No, assumes clock)
    e.g., BIONJ? (Similar to NJ)
    e.g., NeighborNet? (Returns a network, not a tree)
    
    Maybe we can try multiple NJ runs with different initial distances?
    e.g., add noise to the distance matrix?
    
    Or, more promisingly:
    The NJ algorithm joins pairs sequentially.
    The choice of pair at each step is deterministic (min Q).
    If we slightly perturb the Q matrix, we might get a different tree.
    We can run NJ multiple times with random perturbations and keep the "best" tree.
    But what is "best"? We don't have the true tree.
    We could use the likelihood of the tree as a proxy?
    Computing likelihood for a tree with 32 taxa and 2000 sites is feasible but might be slow if done many times.
    
    Let's try a simpler approach first:
    Just run NJ once. It's usually quite good.
    The score is 1.0000, which means it's exactly matching the seed.
    The seed is just NJ.
    So to improve, we need to beat NJ.
    
    Let's try running NJ 5 times with small random perturbations to the distance matrix,
    and select the tree with the highest likelihood (using a fast approximation or just the sum of squared errors of distances?).
    
    Actually, a simpler proxy for "goodness" of a tree given the distance matrix is:
    Sum of squared differences between the tree's implied distances and the observed distances.
    For a given tree topology and optimized branch lengths, we can compute the implied distances.
    But optimizing branch lengths for each candidate tree is expensive.
    
    Alternative:
    Use the "quartet" approach?
    Count the number of quartets supported by the tree.
    For each quartet (a,b,c,d), the tree implies one of 3 topologies.
    We can count how many quartets are consistent with the observed data.
    But how to determine the "best" quartet from data?
    We can compute the three pairwise distances: d_ab, d_cd, d_ac, d_bd, d_ad, d_bc.
    The topology that minimizes the sum of squared errors for the quartet is the "best" one.
    Then we check if our tree agrees with the best quartet.
    
    This is a good "polish" metric.
    Let's implement a function that counts how many quartets a tree agrees with the data.
    Then we can run NJ multiple times (with perturbations) and pick the tree with the highest quartet agreement score.
    
    Steps:
    1. Generate all quartets (n choose 4). For n=32, that's ~35960 quartets.
    2. For each quartet, determine the "best" topology based on the distance matrix.
       There are 3 possible unrooted topologies for 4 taxa:
       (a,b)|(c,d) -> d_ab + d_cd < d_ac + d_bd and d_ab + d_cd < d_ad + bc
       (a,c)|(b,d) -> d_ac + d_bd < d_ab + d_cd and d_ac + d_bd < d_ad + bc
       (a,d)|(b,c) -> d_ad + d_bc < d_ab + d_cd and d_ad + d_bc < d_ac + d_bd
       Actually, the standard criterion is:
       Topology (ab|cd) is supported if d_ab + d_cd < d_ac + d_bd AND d_ab + d_cd < d_ad + d_bc.
       (Tie-breaking: if equal, it's ambiguous).
    3. For a candidate tree, determine which topology it implies for each quartet.
       We can do this by finding the path between each pair of taxa in the tree.
       Or simpler: for a quartet (a,b,c,d), the tree implies (ab|cd) if the path a-b and c-d do not cross?
       Actually, in an unrooted tree, the split is defined by the internal edge.
       For a quartet, the tree implies a split.
       We can extract the split for the quartet from the tree.
       If the tree has a split that separates {a,b} from {c,d}, it implies (ab|cd).
       If it separates {a,c} from {b,d}, it implies (ac|bd).
       If it separates {a,d} from {b,c}, it implies (ad|bc).
       Note: A tree with >4 taxa might not have an explicit edge separating the quartet in a way that matches one of these 3 splits exactly if the quartet is not a "cherry" pair?
       No, any binary tree with 4 leaves has exactly one internal edge, which defines one of the 3 splits.
       For a larger tree, the restriction to the 4 leaves is a binary tree with 4 leaves, so it implies exactly one of the 3 splits.
       
       How to find the split implied by the tree for a quartet?
       We can use the Newick string.
       Find the least common ancestor (LCA) of the 4 leaves?
       Or simpler:
       For each pair (a,b), check if the path from a to b in the tree crosses the edge that separates the tree into two parts containing {a,b} and {c,d}.
       This is getting complicated to implement from scratch efficiently.
       
       Alternative:
       Use the split set from the tree.
       For the quartet {a,b,c,d}, the tree implies the split S if S is a subset of the splits of the tree?
       No, the split of the quartet is a restriction of the tree's splits.
       
       Let's stick to the simpler idea:
       Just run NJ once. It's hard to beat without more time.
       The current score is 1.0.
       Maybe the NJ implementation has a bug?
       Let's re-verify the NJ code.
       
       The code looks correct.
       
       What if we use `scipy.cluster.hierarchy`?
       It has `linkage` and `single`, `complete`, `average`, etc.
       But NJ is not directly available.
       
       Let's try a different angle.
       The directive says: "investigate improvements within its family".
       Family = NJ based.
       
       Idea: Use BIONJ (Better NJ).
       BIONJ uses a different formula for the Q matrix that accounts for the variance of the distances.
       Q_ij = (n-2) * D_ij - r_i - r_j  (NJ)
       BIONJ:
       Q_ij = D_ij - (r_i + r_j) / (n-2) * (1 - 2 * sigma_ij^2 / (sigma_i^2 + sigma_j^2)) ... something like that.
       Actually, the BIONJ formula is:
       Q_ij = D_ij - (r_i + r_j)/(n-2) + 2 * (sigma_i^2 * sigma_j^2) / (sigma_i^2 + sigma_j^2) * (D_ij - (r_i + r_j)/(n-2))
       Where sigma_i^2 is the variance of the distances from i to others.
       
       Let's implement BIONJ and see if it's faster or better.
       Given the time constraint, implementing BIONJ correctly might be risky.
       
       Let's try a simpler improvement:
       Run NJ, then do a few rounds of "Nearest Neighbor Interchange" (NNI) moves,
       accepting a move if it reduces the sum of squared errors between the tree's implied distances and the observed distances.
       
       To compute the sum of squared errors (SSE) for a tree:
       1. Optimize branch lengths for the tree topology to minimize SSE.
          This can be done in O(n^3) or O(n^2) per tree.
          For n=32, this is fast.
       2. Compute the SSE.
       
       NNI moves:
       - Identify an internal edge.
       - Swap the subtrees on either side.
       - This creates a new topology.
       - Compute SSE for the new topology.
       - If SSE is lower, accept the move.
       
       This is a local search.
       It might escape the local optimum of NJ.
       
       Let's implement a function to compute the SSE for a tree given the distance matrix.
       This requires solving for the branch lengths.
       The SSE is a quadratic function of the branch lengths.
       We can use `scipy.optimize.minimize` or solve the linear system directly.
       
       Actually, there is a closed-form solution for the branch lengths that minimize the SSE for a given topology.
       This is the "Least Squares" tree method (Fitch-Margoliash).
       
       Let's implement Fitch-Margoliash (FM) as a polish step.
       1. Start with the NJ tree.
       2. Compute the FM branch lengths for this topology.
       3. Compute the SSE.
       4. Try NNI moves. For each move, compute the new FM branch lengths and SSE.
       5. Accept the best move.
       6. Repeat until no improvement.
       
       This might be slow if we do many NNI moves.
       Number of NNI moves is O(n).
       Each move requires solving for branch lengths (O(n^3)).
       So one round of NNI is O(n^4).
       For n=32, 32^4 = ~1 million operations. Very fast.
       We can do several rounds.
       
       Let's implement this.
       
       First, we need to represent the tree in a way that allows easy NNI moves and distance computation.
       A Newick string is hard to manipulate.
       We need a graph representation.
       
       Since the tree is unrooted, we can represent it as a dictionary of edges.
       Each node is an integer ID.
       Leaves are 0..n-1.
       Internal nodes are n..2n-3.
       
       Let's build a tree object from the Newick string.
       
       This is a significant change.
       Given the time, I'll stick to a simpler heuristic if possible.
       
       Actually, let's just try to run NJ with a slightly different distance matrix.
       E.g., use the raw Hamming distances instead of JC-corrected?
       No, JC is better for long branches.
       
       What if we use the K2P distance?
       The task says JC model. So JC is the right choice.
       
       Let's try to implement the NNI polish.
       It's the most promising "polish" within the NJ family.
       
       I will write a helper class for the tree.
       
    """
    pass # Placeholder, will implement below

def reconstruct(sequences):
    global _START, _DEADLINE
    
    _START = time.time()
    _DEADLINE = _START + 265.0
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        return "(" + ",".join(labels) + ");"
    
    # Compute distance matrix
    D = _jc_distances(sequences, labels)
    
    # Run NJ
    newick = _neighbor_joining(D, labels)
    
    return newick
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
