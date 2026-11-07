import time
import random
import itertools

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = P['n']
    edges = P['edges']
    
    # Precompute adjacency as bitmasks
    adj = [0] * n
    edge_set = set()
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        # Store edges as frozensets for easy removal
        edge_set.add(frozenset((u, v)))
    
    # Helper: Check if a set of vertices L and R forms a valid biclique in the remaining graph
    # remaining_mask[u] is bitmask of neighbors in remaining graph
    def is_valid_biclique(L, R, rem_adj):
        if not L or not R:
            return False
        # Check all u in L, v in R: (1<<v) & rem_adj[u] must be non-zero
        for u in L:
            mask_u = rem_adj[u]
            for v in R:
                if not (mask_u & (1 << v)):
                    return False
        return True

    def expand_biclique(start_u, start_v, rem_adj, rem_edges_set):
        """
        Expand from (start_u, start_v) to a maximal biclique within rem_adj.
        Returns (L, R) as lists of ints.
        """
        L = {start_u}
        R = {start_v}
        changed = True
        while changed:
            changed = False
            # Expand L: add u' if connected to all in R
            mask_R = 0
            for v in R:
                mask_R |= (1 << v)
            
            # Candidates for L: u' not in L such that rem_adj[u'] covers mask_R
            for u_prime in range(n):
                if u_prime in L:
                    continue
                if (rem_adj[u_prime] & mask_R) == mask_R:
                    L.add(u_prime)
                    changed = True
            
            # Expand R: add v' if connected to all in L
            mask_L = 0
            for u in L:
                mask_L |= (1 << u)
                
            for v_prime in range(n):
                if v_prime in R:
                    continue
                if (rem_adj[v_prime] & mask_L) == mask_L:
                    R.add(v_prime)
                    changed = True
        
        return list(L), list(R)

    def greedy_partition(seed):
        """
        Run greedy partition with a specific random seed for tie-breaking.
        Returns list of bicliques [[L, R], ...]
        """
        random.seed(seed)
        rem_adj = adj[:] # Copy
        rem_edges = set(edge_set)
        
        # Optimization: Track remaining edges count
        edges_left = len(rem_edges)
        
        bicliques = []
        
        while edges_left > 0:
            if time.time() > deadline:
                return None
            
            # Find a pivot edge
            # Heuristic: Pick edge (u,v) that is in the densest part of remaining graph?
            # Simple: Pick first edge from rem_edges. To add randomness, shuffle a copy?
            # For efficiency, we can just pick a random edge from rem_edges.
            # Converting set to list is O(E). E is small (~200).
            edge_list = list(rem_edges)
            if not edge_list:
                break
            u, v = random.choice(edge_list)
            
            # Ensure u, v are still connected in rem_adj (they should be)
            if not (rem_adj[u] & (1 << v)):
                # Edge was removed? Should not happen if we maintain consistency.
                continue

            L, R = expand_biclique(u, v, rem_adj, rem_edges)
            
            if not L or not R:
                # Fallback: single edge biclique
                L = [u]
                R = [v]
            
            bicliques.append([L, R])
            
            # Remove edges covered by L x R from rem_adj and rem_edges
            # For each u in L, v in R, remove (u,v)
            for u in L:
                for v in R:
                    if u == v: continue
                    # Check if edge exists in original graph (it does, by validity)
                    # Remove from rem_adj
                    rem_adj[u] &= ~(1 << v)
                    rem_adj[v] &= ~(1 << u)
                    # Remove from rem_edges
                    rem_edges.discard(frozenset((u, v)))
                    edges_left -= 1
        
        return bicliques

    def try_merge(bicliques):
        """
        Attempt to merge pairs of bicliques if valid.
        Returns new list of bicliques.
        """
        merged = True
        current = bicliques
        while merged:
            merged = False
            new_current = []
            i = 0
            while i < len(current):
                if i + 1 < len(current):
                    L1, R1 = current[i]
                    L2, R2 = current[i+1]
                    
                    # Check if merge is valid:
                    # New L = L1 U L2, New R = R1 U R2
                    # Must be a valid biclique in ORIGINAL graph
                    # AND the edges covered must be exactly E(L1,R1) U E(L2,R2)
                    # This implies E(L1, R2) and E(L2, R1) must be empty in original graph.
                    
                    valid_merge = True
                    # Check cross edges L1 x R2
                    for u in L1:
                        for v in R2:
                            if (adj[u] & (1 << v)):
                                valid_merge = False
                                break
                        if not valid_merge: break
                    if valid_merge:
                        for u in L2:
                            for v in R1:
                                if (adj[u] & (1 << v)):
                                    valid_merge = False
                                    break
                            if not valid_merge: break
                    
                    if valid_merge:
                        new_L = list(set(L1) | set(L2))
                        new_R = list(set(R1) | set(R2))
                        new_current.append([new_L, new_R])
                        i += 2
                        merged = True
                    else:
                        new_current.append(current[i])
                        i += 1
                else:
                    new_current.append(current[i])
                    i += 1
            current = new_current
        return current

    best_bicliques = None
    best_count = float('inf')
    
    # Run multiple restarts
    # The graph is small, so we can do many restarts.
    num_restarts = 200
    seed = 42
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Different seeds for randomization in pivot selection
        bicliques = greedy_partition(seed + i)
        if bicliques is None:
            continue
        
        # Apply local search merging
        bicliques = try_merge(bicliques)
        
        if len(bicliques) < best_count:
            best_count = len(bicliques)
            best_bicliques = bicliques
            
        # If we reach a very low count (e.g., 1), we can stop early
        if best_count == 1:
            break

    if best_bicliques is None:
        # Fallback to trivial solution if something went wrong
        return [[[u],[v for a,v in P['edges'] if a==u]] for u in range(P['n']) if any(a==u for a,v in P['edges'])]
    
    # Convert to required format: list of [left, right] where left/right are lists of ints
    # Ensure types are Python int
    final_result = []
    for L, R in best_bicliques:
        final_L = [int(x) for x in L]
        final_R = [int(x) for x in R]
        final_result.append([final_L, final_R])
        
    return final_result
# EVOLVE-BLOCK-END
