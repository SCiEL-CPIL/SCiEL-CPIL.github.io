"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def construct():
    # Deadline guard
    deadline = time.time() + 59.0
    
    # Precompute modular inverses for denominators 1 to 10 modulo 101
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, 101)
        
    # Precompute all non-empty subsets of {1, ..., 10}
    # For each mask, compute: sum mod 101, size, and list of denominators
    all_subsets = []  # list of (residue, size, terms_list, mask)
    for mask in range(1, 1 << 10):
        current_sum = 0
        size = 0
        terms = []
        for i in range(1, 11):
            if mask & (1 << (i - 1)):
                current_sum += inv[i]
                size += 1
                terms.append(i)
        r = current_sum % 101
        all_subsets.append((r, size, terms, mask))
    
    # Group subsets by residue
    subsets_by_residue = defaultdict(list)
    for r, size, terms, mask in all_subsets:
        subsets_by_residue[r].append((size, mask))
    
    # We want to assign to each residue r a subset that sums to r,
    # such that no two residues get the same subset,
    # minimizing the maximum size used.
    
    # Binary search on K: minimum maximum row length
    # For a given K, check if there exists an assignment where each residue
    # gets a distinct subset of size <= K
    
    def hopcroft_karp(max_size):
        """Check if perfect matching exists in bipartite graph where
        left side = 101 residues, right side = subsets of size <= max_size.
        Edge exists if subset sums to the residue.
        Returns the matching (dict: residue -> mask) or None if no perfect matching."""
        
        # Build adjacency: for each residue, list of subset masks of size <= max_size
        adj = defaultdict(list)
        for r in range(101):
            for size, mask in subsets_by_residue[r]:
                if size <= max_size:
                    adj[r].append(mask)
        
        # Hopcroft-Karp
        # matchR[mask] = residue matched to this mask, or -1
        # matchL[r] = mask matched to this residue, or -1
        matchL = {}
        matchR = {}
        
        def bfs():
            """Find shortest augmenting paths. Returns dist for unmatched left nodes."""
            dist = {}
            queue = deque()
            
            for r in range(101):
                if r not in matchL:
                    dist[r] = 0
                    queue.append(r)
                else:
                    dist[r] = float('inf')
            
            found = False
            while queue:
                r = queue.popleft()
                for mask in adj[r]:
                    next_r = matchR.get(mask, -1)
                    if next_r == -1:
                        found = True
                    elif next_r not in dist:
                        dist[next_r] = dist[r] + 1
                        queue.append(next_r)
            
            return dist, found
        
        def dfs(r, dist):
            """Try to find augmenting path from residue r."""
            for mask in adj[r]:
                next_r = matchR.get(mask, -1)
                if next_r == -1:
                    # Augmenting path found
                    matchL[r] = mask
                    matchR[mask] = r
                    return True
                elif next_r not in dist or dist[next_r] != dist[r] + 1:
                    continue
                if dfs(next_r, dist):
                    matchL[r] = mask
                    matchR[mask] = r
                    return True
            dist[r] = float('inf')
            return False
        
        while True:
            if time.time() > deadline:
                return None
            dist, found = bfs()
            if not found:
                break
            for r in range(101):
                if r not in matchL:
                    dfs(r, dist)
        
        # Check if we have a perfect matching
        if len(matchL) == 101:
            return matchL
        return None
    
    # Binary search on K
    lo, hi = 1, 10
    best_K = 10
    best_matching = None
    
    while lo <= hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        matching = hopcroft_karp(mid)
        if matching is not None:
            best_K = mid
            best_matching = matching
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_matching is None:
        # Fallback: just pick minimum size for each residue (may have duplicates)
        result = []
        for r in range(101):
            if subsets_by_residue[r]:
                # Find min size
                min_size = min(s for s, m in subsets_by_residue[r])
                for s, m in subsets_by_residue[r]:
                    if s == min_size:
                        terms = []
                        for i in range(1, 11):
                            if m & (1 << (i - 1)):
                                terms.append(i)
                        result.append(terms)
                        break
            else:
                result.append([])
        return result
    
    # Construct result from best_matching
    # mask_to_terms cache
    mask_to_terms = {}
    for r, size, terms, mask in all_subsets:
        mask_to_terms[mask] = terms
    
    result = []
    for r in range(101):
        mask = best_matching[r]
        result.append(mask_to_terms[mask])
    
    return result
# EVOLVE-BLOCK-END
