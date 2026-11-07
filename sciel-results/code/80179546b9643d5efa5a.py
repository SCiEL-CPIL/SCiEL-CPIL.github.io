"""Rewritten program using bitmask-based exact search for cochromatic partition."""
# EVOLVE-BLOCK-START
import itertools
from functools import lru_cache

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    full_mask = (1 << n) - 1
    
    # Precompute all maximal cliques and maximal independent sets
    # A subset S is a clique if for all u,v in S, u~v
    # A subset S is an independent set if for all u,v in S, u!~v
    
    def is_clique(mask):
        # Check if vertices in mask form a clique
        # For each vertex u in mask, all other vertices in mask must be adjacent to u
        m = mask
        while m:
            lsb = m & (-m)
            u = lsb.bit_length() - 1
            m -= lsb
            # All other vertices in mask must be neighbors of u
            others = mask ^ lsb
            if others & ~adj[u]:
                return False
        return True
    
    def is_independent(mask):
        # Check if vertices in mask form an independent set
        m = mask
        while m:
            lsb = m & (-m)
            u = lsb.bit_length() - 1
            m -= lsb
            others = mask ^ lsb
            if others & adj[u]:
                return False
        return True
    
    # Enumerate all subsets and check which are cliques or independent sets
    # For n=14, there are 2^14 = 16384 subsets, which is manageable
    clique_masks = set()
    indep_masks = set()
    
    for mask in range(1, full_mask + 1):
        if is_clique(mask):
            clique_masks.add(mask)
        if is_independent(mask):
            indep_masks.add(mask)
    
    # We want to partition the full set into minimum number of parts,
    # where each part is either a clique or an independent set.
    # Use DP: dp[mask] = minimum number of parts to cover mask
    # dp[0] = 0
    # dp[mask] = min over all valid submasks S of mask (where S is clique or indep) of 1 + dp[mask ^ S]
    
    # To make this efficient, for each mask, we only need to consider maximal
    # cliques/independent sets that are subsets of mask. But actually, we can
    # just iterate over all valid submasks.
    
    # Better approach: for each mask, find all valid submasks S (clique or indep)
    # and take min(1 + dp[mask ^ S])
    
    # Precompute list of valid masks (cliques and independent sets, non-empty)
    valid_masks = list(clique_masks | indep_masks)
    
    # Sort valid masks by size descending to potentially find good solutions faster
    # But for DP, order doesn't matter much
    
    # DP with memoization
    import sys
    sys.setrecursionlimit(10000)
    
    @lru_cache(maxsize=None)
    def dp(mask):
        if mask == 0:
            return 0
        best = float('inf')
        # Try all valid submasks of mask
        # Iterate over submasks of mask
        sub = mask
        while sub:
            if sub in clique_masks or sub in indep_masks:
                rest = mask ^ sub
                val = 1 + dp(rest)
                if val < best:
                    best = val
            sub = (sub - 1) & mask
        return best
    
    # The answer is dp(full_mask)
    min_parts = dp(full_mask)
    
    # Now we need to reconstruct the actual partition
    # We'll use a similar DP but track the choices
    
    @lru_cache(maxsize=None)
    def reconstruct(mask):
        if mask == 0:
            return []
        best_val = float('inf')
        best_sub = None
        sub = mask
        while sub:
            if sub in clique_masks or sub in indep_masks:
                rest = mask ^ sub
                val = 1 + dp(rest)
                if val < best_val:
                    best_val = val
                    best_sub = sub
            sub = (sub - 1) & mask
        
        if best_sub is None:
            # Should not happen, but fallback
            return [list(range(n))]
        
        rest = mask ^ best_sub
        sub_vertices = [i for i in range(n) if (best_sub >> i) & 1]
        remaining_parts = reconstruct(rest)
        return [sub_vertices] + remaining_parts
    
    partition = reconstruct(full_mask)
    
    # Convert to list of lists, ensuring each part is a list of ints
    result = []
    for part in partition:
        result.append(list(part))
    
    return result

# EVOLVE-BLOCK-END
