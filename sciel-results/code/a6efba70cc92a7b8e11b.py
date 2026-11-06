<<<<<<< SEARCH
def _exhaustive_quartet_search(dist_matrix, labels):
    """
    Exhaustive quartet-based topology search for small n (<= 7).
    Enumerates all unrooted binary tree topologies and scores them.
    """
    n = len(labels)
    if n <= 3:
        if n == 1:
            return labels[0] + ";"
        elif n == 2:
            return "(" + labels[0] + "," + labels[1] + ");"
        else:
            return "(" + labels[0] + "," + labels[1] + "," + labels[2] + ");"
    
    # For n=4, there are 3 possible topologies
    # For n=5, there are 15 possible topologies
    # For n=6, there are 105 possible topologies
    # For n=7, there are 945 possible topologies
    
    # We'll use a recursive approach to generate all binary unrooted trees
    # and score each one using the quartet consistency criterion.
    
    # Actually, for small n, we can use a different approach:
    # Generate all possible sets of splits that form a valid tree topology.
    
    # For n taxa, a valid unrooted binary tree has exactly 2n-3 non-trivial splits.
    # Each split partitions the leaves into two sets.
    
    # Let's use a different strategy: recursively build trees by joining subtrees.
    
    # Represent a tree as a frozenset of frozensets (the splits).
    
    def generate_all_splits(n_leaves, leaf_set):
        """Generate all possible valid sets of splits for n_leaves."""
        if n_leaves <= 1:
            return [set()]
        
        # For n_leaves > 1, we need to find all ways to partition
        # This is complex. Let's use a different approach.
        
        pass
    
    # Alternative: Use the fact that for small n, we can enumerate all 
    # possible binary tree topologies using a recursive construction.
    
    # A binary unrooted tree on n leaves can be built by taking two smaller
    # binary trees and joining them at a new internal node.
    
    # Let's represent trees as nested tuples of leaf indices.
    
    def all_binary_trees(leaves):
        """Generate all binary unrooted tree topologies for the given leaves."""
        n = len(leaves)
        if n == 1:
            yield frozenset({leaves[0]})
        elif n == 2:
            yield frozenset({leaves[0], leaves[1]})
        else:
            # Choose a subset S of leaves of size k where 1 <= k <= n-1
            # and n-k >= 1. To avoid duplicates, we require k <= n-k.
            # Also, to ensure binary trees, we need both sides to be valid.
            
            # Actually, for unrooted binary trees, we need to be careful.
            # Let's use a different representation.
            
            # A binary unrooted tree can be represented by its set of splits.
            # But generating all valid sets of splits is complex.
            
            # Simpler: recursively choose how to split the leaves into two groups,
            # generate all trees for each group, and join them.
            
            # To avoid duplicates, we fix one leaf (say, leaves[0]) to be in the 
            # first group, and vary the size of the first group from 1 to n-1.
            
            # For each valid partition, generate all trees for each side and combine.
            
            # This is still complex. Let's use a library or a simpler heuristic.
            
            # Given the complexity, let's use a different approach:
            # Use the quartet consistency criterion directly.
            
            # For each quartet, determine the best split.
            # Then, check if the collection of best splits is consistent 
            # (i.e., forms a valid tree topology).
            
            # If consistent, reconstruct the tree from the splits.
            
            pass
    
    # Given the complexity of enumerating all tree topologies correctly,
    # let's use a simpler but effective approach for n <= 7:
    # Use the quartet consistency criterion to determine the best splits,
    # then reconstruct the tree.
    
    # Step 1: For each quartet, determine the best split.
    quartet_splits = {}
    for quartet in combinations(range(n), 4):
        best_split = _quartet_score(dist_matrix, quartet)
        # Store the split as a frozenset of the smaller side
        if len(best_split) > n // 2:
            best_split = frozenset(set(range(n)) - set(best_split))
        quartet_splits[quartet] = best_split
    
    # Step 2: Reconstruct the tree from the splits.
    # This is non-trivial. Let's use a different approach.
    
    # For very small n (n <= 5), we can enumerate all topologies.
    # For n = 4: 3 topologies
    # For n = 5: 15 topologies
    # For n = 6: 105 topologies
    # For n = 7: 945 topologies
    
    # Let's implement a scoring function for a given tree topology.
    
    def score_tree(tree_splits):
        """
        Score a tree topology given by its set of non-trivial splits.
        Use the quartet consistency criterion: for each quartet,
        check if the tree's restriction to that quartet matches the
        best quartet split.
        """
        # Count how many quartets are consistent
        consistent = 0
        total_quartets = 0
        
        for quartet in combinations(range(n), 4):
            a, b, c, d = quartet
            total_quartets += 1
            
            # Determine which split the tree implies for this quartet
            # For each of the 3 possible splits, check if it's in the tree
            
            # Split {a,b} | {c,d}
            split_ab = frozenset({a, b})
            split_ac = frozenset({a, c})
            split_ad = frozenset({a, d})
            
            # Check if any of these splits (or their complements) are in the tree
            # A split is in the tree if it matches one of the tree's splits
            # or its complement matches one of the tree's splits.
            
            def split_in_tree(split):
                if split in tree_splits:
                    return True
                complement = frozenset(set(range(n)) - set(split))
                if complement in tree_splits:
                    return True
                return False
            
            if split_in_tree(split_ab):
                implied_split = frozenset({a, b})
            elif split_in_tree(split_ac):
                implied_split = frozenset({a, c})
            elif split_in_tree(split_ad):
                implied_split = frozenset({a, d})
            else:
                # This shouldn't happen for a valid tree
                continue
            
            # Compare with the best quartet split
            best_split = quartet_splits[quartet]
            if len(best_split) > n // 2:
                best_split = frozenset(set(range(n)) - set(best_split))
            
            # Check if implied_split matches best_split (or its complement)
            if implied_split == best_split or implied_split == frozenset(set(range(n)) - set(best_split)):
                consistent += 1
        
        return consistent / total_quartets if total_quartets > 0 else 0
    
    # Now, we need to generate all possible tree topologies and score them.
    # This is feasible for n <= 7.
    
    # Let's use a recursive approach to generate all binary tree topologies.
    
    def generate_tree_splits(leaves, all_splits_set):
        """
        Recursively generate all possible sets of splits for binary trees
        on the given leaves.
        """
        n_leaves = len(leaves)
        if n_leaves <= 2:
            yield set()
        else:
            # Choose a subset S of leaves to form one side of the top-level split.
            # To avoid duplicates, we fix leaves[0] to be in S, and vary |S| from 1 to n_leaves-1.
            
            # For each valid partition, generate all splits for each side and combine.
            
            # This is still complex. Let's use a different strategy.
            
            # Given the time constraints, let's use a heuristic:
            # Use the quartet consistency to guide the tree construction.
            
            pass
    
    # Given the complexity, let's use a simpler approach:
    # For n <= 5, enumerate all topologies explicitly.
    # For n = 6, 7, use NJ as a fallback.
    
    if n <= 5:
        # Enumerate all topologies
        best_score = -1
        best_tree = None
        
        # Generate all possible binary tree topologies
        # For n=4: 3 topologies
        # For n=5: 15 topologies
        
        # We'll represent a tree by its set of non-trivial splits.
        
        def all_topologies(n_leaves):
            """Generate all binary unrooted tree topologies for n_leaves."""
            if n_leaves <= 2:
                yield set()
            else:
                # Choose the first split
                # The first split partitions the leaves into two groups of sizes k and n-k
                # where 1 <= k <= n-1. To avoid duplicates, we require k <= n-k.
                
                # For each valid k, choose which leaves go in the first group.
                # To avoid duplicates, we fix leaf 0 to be in the first group.
                
                for k in range(1, n_leaves // 2 + 1):
                    # Choose k-1 other leaves to join leaf 0 in the first group
                    other_leaves = list(range(1, n_leaves))
                    for subset in combinations(other_leaves, k - 1):
                        group1 = frozenset([0] + list(subset))
                        group2 = frozenset(set(range(n_leaves)) - set(group1))
                        
                        # Generate all topologies for group1
                        for splits1 in all_topologies(len(group1)):
                            # Generate all topologies for group2
                            for splits2 in all_topologies(len(group2)):
                                # Combine: the top-level split is group1
                                # Plus all splits from group1 and group2
                                all_splits = set(splits1) | set(splits2)
                                all_splits.add(group1)
                                yield all_splits
        
        # Score each topology
        for tree_splits in all_topologies(n):
            # Convert to frozensets for comparison
            tree_splits_frozen = set(frozenset(s) if len(s) <= n // 2 else frozenset(set(range(n)) - set(s)) for s in tree_splits)
            
            # Remove trivial splits (size 1 or n-1)
            tree_splits_frozen = {s for s in tree_splits_frozen if 1 < len(s) < n - 1}
            
            score = score_tree(tree_splits_frozen)
            
            if score > best_score:
                best_score = score
                best_tree = tree_splits_frozen
        
        # Reconstruct Newick from the best tree's splits
        newick = _splits_to_newick(best_tree, labels)
        return newick
    else:
        # For n > 5, fall back to NJ
        return _neighbor_joining(dist_matrix, labels)
=======
def _exhaustive_quartet_search(dist_matrix, labels):
    """
    Exhaustive quartet-based topology search for small n (<= 7).
    Enumerates all unrooted binary tree topologies and scores them using SSE.
    """
    n = len(labels)
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"
    if n == 3:
        return "(" + labels[0] + "," + labels[1] + "," + labels[2] + ");"
    
    # Generate all binary unrooted tree topologies
    def all_topologies(n_leaves):
        """Generate all binary unrooted tree topologies as sets of splits."""
        if n_leaves <= 2:
            yield set()
        else:
            for k in range(1, n_leaves // 2 + 1):
                other_leaves = list(range(1, n_leaves))
                for subset in combinations(other_leaves, k - 1):
                    group1 = frozenset([0] + list(subset))
                    group2 = frozenset(set(range(n_leaves)) - set(group1))
                    
                    for splits1 in all_topologies(len(group1)):
                        for splits2 in all_topologies(len(group2)):
                            all_splits = set(splits1) | set(splits2)
                            all_splits.add(group1)
                            yield all_splits
    
    # Score each topology using SSE
    best_sse = float('inf')
    best_splits = None
    
    for tree_splits in all_topologies(n):
        # Normalize splits to use smaller side
        normalized = set()
        for s in tree_splits:
            if len(s) <= n // 2:
                normalized.add(frozenset(s))
            else:
                normalized.add(frozenset(set(range(n)) - set(s)))
        # Remove trivial splits
        normalized = {s for s in normalized if 1 < len(s) < n - 1}
        
        # Build tree tuple from splits for SSE computation
        tree_tuple = _splits_to_tree_tuple(normalized, n)
        if tree_tuple is None:
            continue
        
        sse = _compute_sse(tree_tuple, dist_matrix)
        if sse < best_sse:
            best_sse = sse
            best_splits = normalized
    
    # Reconstruct Newick from the best tree's splits
    newick = _splits_to_newick(best_splits, labels)
    return newick
>>>>>>> REPLACE

