# EVOLVE-BLOCK-START
import numpy as np
from functools import lru_cache


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a maximum square-difference-free set modulo 205 using a bitset-based
    branch-and-bound Maximum Independent Set solver with symmetry breaking
    and improved greedy lower bounds.
    """
    m = 205
    k = 2

    # Compute forbidden differences: nonzero squares mod 205
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, k, m))
    forbidden.discard(0)

    # Build adjacency bitsets
    adj = [0] * m
    for i in range(m):
        for f in forbidden:
            j1 = (i + f) % m
            j2 = (i - f) % m
            adj[i] |= (1 << j1)
            adj[i] |= (1 << j2)
    for i in range(m):
        adj[i] &= ~(1 << i)

    full_mask = (1 << m) - 1

    def greedy_color_bound(candidates_bitset):
        """
        Returns an upper bound on the size of the Maximum Independent Set in G[candidates].
        Computes greedy coloring of the complement graph \bar{G}[candidates].
        """
        if candidates_bitset == 0:
            return 0
        
        remaining = candidates_bitset
        color_count = 0
        
        while remaining != 0:
            # Find a maximal clique in G induced by 'remaining' (color class in \bar{G})
            clique = 0
            temp = remaining
            while temp != 0:
                lb = temp & (-temp)
                v = lb.bit_length() - 1
                
                if (adj[v] & clique) == clique:
                    clique |= lb
                    temp &= adj[v]
                
                temp &= ~(1 << v)
            
            color_count += 1
            remaining &= ~clique
        
        return color_count

    def greedy_mis_lower_bound(candidates_bitset):
        """
        Greedy heuristic to find a large independent set.
        Picks vertices with minimum degree in the current candidate set.
        """
        s = 0
        cand = candidates_bitset
        while cand != 0:
            best_v = -1
            min_deg = m + 1
            temp = cand
            while temp != 0:
                lb = temp & (-temp)
                v = lb.bit_length() - 1
                deg = bin(adj[v] & cand).count('1')
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                temp ^= lb
            
            if best_v == -1:
                break
            
            s |= (1 << best_v)
            cand &= ~(1 << best_v) & ~adj[best_v]
        
        return s

    best_size = 0
    best_set = 0

    def branch_and_bound(candidates_bitset, current_set, current_size):
        nonlocal best_size, best_set
        
        if candidates_bitset == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set
            return
        
        # Upper bound
        ub = current_size + greedy_color_bound(candidates_bitset)
        if ub <= best_size:
            return
        
        # Pick a vertex to branch on
        # Heuristic: pick vertex with highest degree in candidate subgraph
        # This often leads to stronger pruning
        temp = candidates_bitset
        best_v = -1
        best_deg = -1
        while temp != 0:
            lb = temp & (-temp)
            v = lb.bit_length() - 1
            deg = bin(adj[v] & candidates_bitset).count('1')
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp ^= lb
        
        if best_v == -1:
            return
        
        # Branch 1: Include best_v
        # Remove best_v and all its neighbors from candidates
        new_candidates = candidates_bitset & ~(1 << best_v) & ~adj[best_v]
        new_set = current_set | (1 << best_v)
        branch_and_bound(new_candidates, new_set, current_size + 1)
        
        # Branch 2: Exclude best_v
        new_candidates = candidates_bitset & ~(1 << best_v)
        branch_and_bound(new_candidates, current_set, current_size)

    # Symmetry Breaking:
    # The graph is vertex-transitive.
    # We can split the search into two cases:
    # 1. Vertex 0 is in the independent set.
    # 2. Vertex 0 is not in the independent set.
    
    # Case 1: 0 in S
    # If 0 is in S, no neighbor of 0 can be in S.
    # Candidates: All vertices except 0 and its neighbors.
    cand1 = full_mask & ~(1 << 0) & ~adj[0]
    set1 = 1 << 0
    # Update best with greedy on this subproblem
    g_set = greedy_mis_lower_bound(cand1)
    g_size = bin(g_set).count('1')
    if g_size + 1 > best_size:
        best_size = g_size + 1
        best_set = set1 | g_set
        
    branch_and_bound(cand1, set1, 1)

    # Case 2: 0 not in S
    # Candidates: All vertices except 0.
    cand2 = full_mask & ~(1 << 0)
    set2 = 0
    # Update best with greedy on this subproblem
    g_set2 = greedy_mis_lower_bound(cand2)
    g_size2 = bin(g_set2).count('1')
    if g_size2 > best_size:
        best_size = g_size2
        best_set = g_set2
        
    branch_and_bound(cand2, set2, 0)

    # Extract the best set
    result = []
    for v in range(m):
        if (best_set >> v) & 1:
            result.append(v)

    return np.array(result, dtype=np.int64)


# EVOLVE-BLOCK-END
