"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
from collections import defaultdict

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

_DEADLINE = _time.monotonic() + 78.0

def _time_left():
    return _time.monotonic() < _DEADLINE

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list and set
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # 1. Find a large clique greedily
    # We try to build a clique by adding vertices one by one.
    # Start with a random vertex, then try to add others that are connected to all current members.
    # To maximize size, we can try starting from different vertices or using a simple heuristic.
    # Given n=24, we can afford a bit of search.
    
    best_clique = []
    best_paths = []
    
    # Try all vertices as the starting point for the greedy clique construction
    for start in range(n):
        if not _time_left():
            break
            
        clique = [start]
        clique_set = {start}
        
        # Candidates to add: neighbors of all vertices in the clique
        # We sort candidates by degree to prioritize "hub" vertices which might allow more Steiner points
        candidates = sorted(adj[start], key=lambda x: -len(adj[x]))
        
        for c in candidates:
            if not _time_left():
                break
            if c in clique_set:
                continue
            # Check if c is connected to all in clique
            is_connected = all(c in adj[v] for v in clique)
            if is_connected:
                clique.append(c)
                clique_set.add(c)
        
        # Now we have a clique 'clique'. Let's try to form the subdivision.
        # We need to assign a unique Steiner vertex to each pair in the clique.
        # A pair (u, v) needs a Steiner vertex s such that s is adjacent to u and v, and s is not in the clique.
        
        # Let's collect all pairs
        pairs = []
        for i in range(len(clique)):
            for j in range(i+1, len(clique)):
                pairs.append((clique[i], clique[j]))
                
        # For each pair, find potential Steiner vertices
        # Potential Steiner vertices are those not in the clique and adjacent to both endpoints.
        
        # We can model this as a bipartite matching problem:
        # Left side: Pairs
        # Right side: Candidate Steiner vertices (V \ clique)
        # Edge if Steiner is adjacent to both endpoints of the pair.
        
        # However, we want to maximize the number of branches.
        # If we can't match all pairs for this clique, maybe a smaller clique works?
        # Or maybe we can just take the maximum matching and remove the unmatched pairs?
        # No, the requirement is "one simple path for each unordered pair of branches".
        # So if we have k branches, we need k*(k-1)/2 disjoint Steiner vertices.
        
        # Let's try to find a perfect matching for the pairs.
        # If perfect matching exists, this clique is valid.
        # If not, we can try to remove some branches from the clique to reduce the number of pairs,
        # or try a different clique.
        
        # To be efficient, let's first check if a perfect matching is possible.
        # We can use Hopcroft-Karp or a simple DFS based matching since n is small.
        
        # Let's identify candidate steiners
        steiners = [v for v in range(n) if v not in clique_set]
        
        # Build adjacency for matching
        # pair_index -> list of steiner_indices
        pair_steiner_adj = defaultdict(list)
        for i, (u, v) in enumerate(pairs):
            for j, s in enumerate(steiners):
                if s in adj[u] and s in adj[v]:
                    pair_steiner_adj[i].append(j)
                    
        # Simple DFS based bipartite matching (Kuhn's algorithm)
        # Since n is small (max 24 vertices, max 276 pairs, max 24 steiners), 
        # wait, the number of steiners is at most 24.
        # The number of pairs is k*(k-1)/2.
        # For a perfect matching, we need len(pairs) <= len(steiners).
        # So k*(k-1)/2 <= 24 => k is at most 7 (7*6/2 = 21), 8*7/2=28 > 24.
        # So the maximum possible number of branches is 7!
        # The current program found 4 (score 1.33, seed likely 3).
        # If we can find a clique of size 7 with a valid matching, we get score 7/3 = 2.33.
        
        if len(pairs) > len(steiners):
            continue # Cannot have perfect matching
            
        # Try to find perfect matching
        # match[steiner_idx] = pair_idx
        match = [-1] * len(steiners)
        
        def try_kuhn(v, used):
            for u in pair_steiner_adj[v]:
                if used[u]:
                    continue
                used[u] = True
                if match[u] == -1 or try_kuhn(match[u], used):
                    match[u] = v
                    return True
            return False
        
        # We need to match all pairs.
        # Note: Kuhn's algorithm is usually run for each vertex on the left.
        # Here left is pairs, right is steiners.
        # We want to match all pairs.
        
        # To improve success rate, we can sort pairs by the number of available steiners (most constrained first).
        # But Kuhn's is greedy. Let's just run it.
        
        # Actually, to ensure we find a maximum matching, we should run DFS for each pair.
        # Since we need a PERFECT matching (all pairs matched), we can just check if the size of max matching == len(pairs).
        
        # Let's run Kuhn's for each pair.
        # To make it faster, we can order pairs by degree (ascending).
        pair_order = sorted(range(len(pairs)), key=lambda i: len(pair_steiner_adj[i]))
        
        # Reset match for this attempt
        match = [-1] * len(steiners)
        
        for p_idx in pair_order:
            used = [False] * len(steiners)
            if not try_kuhn(p_idx, used):
                break # Failed to match this pair, so no perfect matching with this greedy order?
                # Actually, Kuhn's algorithm finds a maximum matching if we iterate all vertices.
                # If we break early, we might miss a matching if the order was bad.
                # However, for small graphs, a single pass is often enough or we can retry.
                # Given the small size, let's just check if all are matched after one pass.
                # If not, we can try a different order or just skip this clique.
                pass
        
        # Check if all pairs are matched
        matched_pairs = set(match)
        if len(matched_pairs) == len(pairs):
            # We found a valid subdivision for this clique!
            if len(clique) > len(best_clique):
                best_clique = clique
                # Construct paths
                paths = []
                # match[steiner_idx] = pair_idx
                # We need pair_idx -> steiner_idx
                pair_to_steiner = {}
                for s_idx, p_idx in enumerate(match):
                    if p_idx != -1:
                        pair_to_steiner[p_idx] = s_idx
                        
                for p_idx, (u, v) in enumerate(pairs):
                    s_idx = pair_to_steiner[p_idx]
                    s = steiners[s_idx]
                    paths.append([u, s, v])
                    
                best_paths = paths
                
                # Since max k is 7, and we are iterating start vertices, 
                # if we find k=7, we can stop? 
                # Actually, we want the MAXIMUM.
                # Let's keep searching to see if we can find another 7 or confirm 7 is max.
                # But 7 is the theoretical max given 24 vertices.
                # So if we find 7, we can break.
                if len(best_clique) == 7:
                    break

    # Fallback if no clique > 2 found (should not happen with this graph)
    if len(best_clique) < 2:
        # Find any edge
        if edges:
            u, v = edges[0]
            best_clique = [u, v]
            best_paths = [[u, v]]
        else:
            best_clique = [0, 1] # Invalid if no edge, but graph has edges
            best_paths = [[0, 1]]
            
    return {
        'branches': [int(x) for x in best_clique],
        'paths': [[int(x) for x in p] for p in best_paths]
    }

# EVOLVE-BLOCK-END
