import time

# EVOLVE-BLOCK-START
def construct():
    """
    Construct 101 nonempty lists, row r containing distinct denominators in 1..10
    whose modular inverses sum to r modulo 101. Minimize the largest row length.
    """
    P = 101
    deadline = time.time() + 50  # safety margin before 60s (or 90s) limit
    
    # Precompute inverses
    inv = {x: pow(x, -1, P) for x in range(1, 11)}
    
    # We need to assign to each residue r in 0..100 a non-empty subset of {1..10}
    # such that sum of inverses ≡ r (mod 101).
    # Minimize the maximum subset size across all 101 residues.
    
    # Strategy: Try to find an assignment where max size is as small as possible.
    # The minimum possible max size is 1 (if we can cover all 101 residues with singletons),
    # but there are only 10 singletons, so max size must be at least 11 (since 101 residues
    # and each subset of size k covers at most C(10,k) residues... wait, no, each subset
    # maps to exactly one residue).
    
    # Actually, each subset maps to exactly one residue. So we have 2^10 - 1 = 1023 subsets,
    # each mapping to one of 101 residues. We need to pick one subset for each residue.
    # We want to minimize the maximum size of the chosen subsets.
    
    # Let's check: can we cover all 101 residues using only subsets of size <= 1? No, only 10 subsets of size 1.
    # Size <= 2? C(10,1) + C(10,2) = 10 + 45 = 55 subsets. Not enough for 101.
    # Size <= 3? 10 + 45 + 120 = 175 subsets. Enough in count, but do they cover all residues?
    
    # Let's compute all subset sums and check coverage by size.
    
    # Precompute all subsets grouped by size
    subsets_by_size = {s: [] for s in range(1, 11)}
    
    for mask in range(1, 1 << 10):
        subset = []
        s = 0
        for x in range(1, 11):
            if mask >> (x - 1) & 1:
                subset.append(x)
                s += inv[x]
        r = s % P
        sz = len(subset)
        subsets_by_size[sz].append((r, subset))
    
    # For each max_size k, check if we can assign subsets of size <= k to cover all 101 residues
    # This is a bipartite matching problem: residues on left, subsets on right.
    # But we want the minimum k such that a perfect matching exists.
    
    # For a given k, the available subsets are those with size <= k.
    # We need to check if there's a system of distinct representatives.
    
    # Let's try k = 3 first, then k = 4, etc.
    
    from collections import defaultdict
    
    def can_cover(max_size):
        """Check if subsets of size <= max_size can cover all 101 residues."""
        # Build adjacency: for each residue, which subsets (by index) can cover it
        # Use Hopcroft-Karp or simple augmenting path since 101 nodes is small
        
        # Collect all subsets with size <= max_size
        all_subsets = []
        for sz in range(1, max_size + 1):
            for (r, subset) in subsets_by_size[sz]:
                all_subsets.append((r, subset))
        
        # For each residue, list of subset indices
        residue_to_subsets = defaultdict(list)
        for idx, (r, subset) in enumerate(all_subsets):
            residue_to_subsets[r].append(idx)
        
        # Check if every residue has at least one subset
        for r in range(P):
            if not residue_to_subsets[r]:
                return None
        
        # Bipartite matching: left = residues (0..100), right = subsets (0..len-1)
        # We need a matching that covers all left nodes.
        
        n_left = P
        n_right = len(all_subsets)
        
        match_left = [-1] * n_left  # match_left[r] = subset index assigned to r
        match_right = [-1] * n_right  # match_right[idx] = residue assigned to idx
        
        def try_augment(r, visited):
            for idx in residue_to_subsets[r]:
                if visited[idx]:
                    continue
                visited[idx] = True
                if match_right[idx] == -1 or try_augment(match_right[idx], visited):
                    match_right[idx] = r
                    match_left[r] = idx
                    return True
            return False
        
        for r in range(n_left):
            visited = [False] * n_right
            if not try_augment(r, visited):
                return None
        
        # Build result
        result = [None] * P
        for r in range(P):
            idx = match_left[r]
            result[r] = all_subsets[idx][1]
        return result
    
    # Try increasing max_size
    for max_size in range(1, 11):
        if time.time() > deadline:
            break
        result = can_cover(max_size)
        if result is not None:
            return result
    
    # Fallback: use the original approach (minimize individual sizes, not max)
    best = [None] * P
    for mask in range(1, 1 << 10):
        subset = []
        s = 0
        for x in range(1, 11):
            if mask >> (x - 1) & 1:
                subset.append(x)
                s += inv[x]
        r = s % P
        sz = len(subset)
        if best[r] is None or sz < best[r][0]:
            best[r] = (sz, subset)
    
    return [best[r][1] for r in range(P)]
# EVOLVE-BLOCK-END
