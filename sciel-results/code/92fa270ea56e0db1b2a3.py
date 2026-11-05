import time
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    """
    Construct 101 nonempty lists, row r containing distinct denominators in 1..10
    whose modular inverses sum to r modulo 101. Minimize the largest row length.
    """
    P = 101
    deadline = time.time() + 50  # safety margin before 90s limit
    
    # Precompute inverses
    inv = {x: pow(x, -1, P) for x in range(1, 11)}
    
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
    
    def hopcroft_karp(residue_to_subsets, n_left, n_right):
        """
        Hopcroft-Karp algorithm for maximum bipartite matching.
        residue_to_subsets: dict mapping left node -> list of right nodes
        Returns match_left array or None if matching doesn't cover all left nodes.
        """
        match_left = [-1] * n_left
        match_right = [-1] * n_right
        dist = [0] * n_left
        
        def bfs():
            from collections import deque
            queue = deque()
            for u in range(n_left):
                if match_left[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = -1
            found = False
            while queue:
                u = queue.popleft()
                for v in residue_to_subsets[u]:
                    mu = match_right[v]
                    if mu == -1:
                        found = True
                    elif dist[mu] == -1:
                        dist[mu] = dist[u] + 1
                        queue.append(mu)
            return found
        
        def dfs(u):
            for v in residue_to_subsets[u]:
                mu = match_right[v]
                if mu == -1 or (dist[mu] == dist[u] + 1 and dfs(mu)):
                    match_left[u] = v
                    match_right[v] = u
                    return True
            dist[u] = -1
            return False
        
        matching = 0
        while bfs():
            for u in range(n_left):
                if match_left[u] == -1:
                    if dfs(u):
                        matching += 1
        
        if matching < n_left:
            return None
        
        return match_left
    
    def can_cover(max_size):
        """Check if subsets of size <= max_size can cover all 101 residues."""
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
        
        n_left = P
        n_right = len(all_subsets)
        
        # Convert to list of lists for Hopcroft-Karp
        adj = [residue_to_subsets[r] for r in range(P)]
        
        match_left = hopcroft_karp(adj, n_left, n_right)
        if match_left is None:
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
