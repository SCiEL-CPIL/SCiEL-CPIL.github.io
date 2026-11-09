import time
import random
import numpy as np

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    n = P['n']
    edges = P['edges']
    
    # Build initial adjacency and edge set
    init_adj = [set() for _ in range(n)]
    init_edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        init_edge_set.add((u, v))
        init_adj[u].add(v)
        init_adj[v].add(u)

    def expand_biclique(L, R, adj):
        """Expand L and R to a maximal biclique in the given adjacency."""
        L = set(L)
        R = set(R)
        if not L or not R:
            return L, R
        changed = True
        while changed:
            changed = False
            new_L = set()
            for x in L:
                if R.issubset(adj[x]):
                    new_L.add(x)
            if len(new_L) != len(L):
                L = new_L
                changed = True
            new_R = set()
            for y in R:
                if L.issubset(adj[y]):
                    new_R.add(y)
            if len(new_R) != len(R):
                R = new_R
                changed = True
        return L, R

    def get_max_biclique(u, v, adj):
        # Orientation 1: u in L, v in R
        L1 = set(adj[v])
        R1 = set(adj[u])
        if u not in L1 or v not in R1:
            size1 = 0
            L1, R1 = set(), set()
        else:
            L1, R1 = expand_biclique(L1, R1, adj)
            size1 = len(L1) * len(R1)

        # Orientation 2: u in R, v in L
        L2 = set(adj[u])
        R2 = set(adj[v])
        if v not in L2 or u not in R2:
            size2 = 0
            L2, R2 = set(), set()
        else:
            L2, R2 = expand_biclique(L2, R2, adj)
            size2 = len(L2) * len(R2)

        if size1 >= size2:
            return size1, list(L1), list(R1)
        else:
            return size2, list(L2), list(R2)

    def greedy_partition(adj, remaining, seed_rng=None):
        """Greedy set-cover: at each step pick the biclique covering most uncovered edges."""
        if seed_rng is not None:
            rng = seed_rng
        else:
            rng = random.Random()
        
        bicliques = []
        remaining = set(remaining)
        adj = [set(s) for s in adj]
        
        while remaining:
            best_size = -1
            best_L = None
            best_R = None
            best_edges = None
            
            edges_list = list(remaining)
            rng.shuffle(edges_list)
            
            for u, v in edges_list:
                size, L, R = get_max_biclique(u, v, adj)
                if size > best_size:
                    best_size = size
                    best_L = L
                    best_R = R
                    # Compute which edges this biclique covers
                    covered = set()
                    for x in L:
                        for y in R:
                            e = (min(x, y), max(x, y))
                            if e in remaining:
                                covered.add(e)
                    best_edges = covered
            
            if best_L is None or best_size <= 0:
                # Fallback: single edge
                u, v = next(iter(remaining))
                bicliques.append([[u], [v]])
                remaining.remove((min(u,v), max(u,v)))
                continue
                
            bicliques.append([list(best_L), list(best_R)])
            
            # Remove covered edges
            for e in best_edges:
                remaining.discard(e)
                u, v = e
                adj[u].discard(v)
                adj[v].discard(u)
                
        return bicliques

    def try_merge(bicliques, adj):
        """Try to merge pairs of bicliques if their union is still a valid biclique."""
        changed = True
        while changed:
            changed = False
            new_bicliques = []
            i = 0
            while i < len(bicliques):
                if i + 1 < len(bicliques):
                    L1, R1 = bicliques[i]
                    L2, R2 = bicliques[i+1]
                    # Try merging: (L1∪L2, R1∪R2) must be a biclique
                    merged_L = set(L1) | set(L2)
                    merged_R = set(R1) | set(R2)
                    # Check all cross edges exist
                    valid = True
                    for x in merged_L:
                        for y in merged_R:
                            if x != y and y not in adj[x]:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        new_bicliques.append([list(merged_L), list(merged_R)])
                        i += 2
                        changed = True
                    else:
                        new_bicliques.append(bicliques[i])
                        i += 1
                else:
                    new_bicliques.append(bicliques[i])
                    i += 1
            bicliques = new_bicliques
        return bicliques

    def try_cross_merge(bicliques, adj):
        """Try to merge bicliques with different pairings: (L1,R2) and (L2,R1) or (L1∪L2, R1) etc."""
        changed = True
        while changed:
            changed = False
            new_bicliques = []
            i = 0
            while i < len(bicliques):
                if i + 1 < len(bicliques):
                    L1, R1 = bicliques[i]
                    L2, R2 = bicliques[i+1]
                    # Try cross-merge 1: (L1, R1∪R2) and (L2, R1∪R2) -> merge to (L1∪L2, R1∪R2)
                    # Actually try: can we form (L1∪L2, R1) and (L1∪L2, R2)? No, that's not a merge.
                    # Try: (L1, R1∪R2) is valid? If so, and (L2, R1∪R2) is valid?
                    # Better: try to see if L1∪L2 with R1 is a biclique, or L1 with R1∪R2, etc.
                    
                    # Attempt 1: merged = (L1∪L2, R1∪R2) - already handled by try_merge
                    # Attempt 2: Can we reassign? Try (L1, R1∪R2) as one biclique if valid
                    # This is getting complex. Let's try a simpler heuristic:
                    # Check if (L1∪L2, R1) is a valid biclique
                    merged_L = set(L1) | set(L2)
                    merged_R1 = set(R1)
                    valid1 = all(y in adj[x] for x in merged_L for y in merged_R1 if x != y)
                    if valid1 and len(merged_L) > len(L1) and len(merged_L) > len(L2):
                        # We can potentially combine, but we need to handle R2 separately
                        # This is complex, so let's skip for now and rely on try_merge
                        pass
                    
                    new_bicliques.append(bicliques[i])
                    i += 1
                else:
                    new_bicliques.append(bicliques[i])
                    i += 1
            bicliques = new_bicliques
        return bicliques

    def validate(bicliques, edge_set):
        """Check that bicliques form a valid partition."""
        covered = set()
        for L, R in bicliques:
            if not L or not R:
                return False
            for x in L:
                for y in R:
                    if x == y:
                        return False
                    e = (min(x, y), max(x, y))
                    if e not in edge_set:
                        return False
                    if e in covered:
                        return False
                    covered.add(e)
        return covered == edge_set

    def local_search(bicliques, adj, edge_set):
        """Try to improve by moving vertices between bicliques or splitting."""
        # For now, just try merging again with shuffled order
        import random as _random
        _random.shuffle(bicliques)
        return try_merge(bicliques, adj)

    # Run multiple restarts with different random seeds
    best_sol = None
    best_count = float('inf')
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    num_restarts = 300
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        rng = random.Random(restart * 7919 + 13)
        sol = greedy_partition(init_adj, init_edge_set, seed_rng=rng)
        
        # Try to merge
        sol = try_merge(sol, init_adj)
        
        # Additional local search: shuffle and re-merge
        if time.time() < deadline - 2:
            import random as _random
            _random.Random(restart).shuffle(sol)
            sol = try_merge(sol, init_adj)
        
        if validate(sol, init_edge_set):
            if len(sol) < best_count:
                best_count = len(sol)
                best_sol = sol
                # Early exit if we found a very good solution
                if best_count <= 4:
                    break
    
    if best_sol is None:
        # Fallback: star decomposition
        result = []
        for u in range(n):
            neighbors = list(init_adj[u])
            if neighbors:
                result.append([[u], neighbors])
        return result
    
    # Convert to proper Python types
    final_result = []
    for L, R in best_sol:
        final_result.append([[int(x) for x in L], [int(x) for x in R]])
    
    return final_result
