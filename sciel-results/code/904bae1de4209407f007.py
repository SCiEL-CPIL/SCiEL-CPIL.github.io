"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    n = P['n']
    edges = P['edges']
    m = len(edges)
    
    # Build adjacency sets
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Edge index: (u,v) -> index (with u < v)
    edge_idx = {}
    for i, (u, v) in enumerate(edges):
        a, b = (u, v) if u < v else (v, u)
        edge_idx[(a, b)] = i
    
    # Find all maximal bicliques using a simple exhaustive approach
    # For a graph with 30 vertices and ~150 edges, we can enumerate
    # all pairs of vertices as potential "core" of a biclique and expand.
    
    # A biclique is defined by (L, R) where every (l,r) in LxR is an edge.
    # We want maximal bicliques: cannot add any vertex to L or R while maintaining the property.
    
    # Strategy: for each subset of vertices as one side, find the other side.
    # But that's exponential. Instead, use the fact that a biclique's left side
    # must be a clique in the "co-graph" restricted appropriately... 
    # 
    # Simpler approach for n=30: enumerate all possible left sides as subsets of size 1, 2, 3
    # (bicliques with left side >= 4 are rare in random graphs and we can still get good coverage)
    # Actually, let's try a smarter approach.
    
    # For each edge (u,v), we can build the maximal biclique containing it by:
    # 1. Start with L={u}, R={v}
    # 2. Expand L by adding any vertex w that is adjacent to all of R
    # 3. Expand R by adding any vertex w that is adjacent to all of L
    # 4. Repeat until no more can be added
    # This gives the maximal biclique for that starting edge.
    
    # But different edges may give the same maximal biclique, so we deduplicate.
    
    # Actually, a cleaner approach: enumerate all maximal bicliques by
    # considering each vertex as being in L, and for each subset of neighbors as R candidates.
    
    # Given n=30, let's use a different strategy:
    # For each pair (u,v) as an initial edge, greedily expand to get a biclique.
    # Collect all distinct bicliques found.
    
    bicliques = set()
    
    for i in range(n):
        for j in range(n):
            if i == j or (i, j) not in edge_idx and (j, i) not in edge_idx:
                continue
            # Check if edge exists
            a, b = (i, j) if i < j else (j, i)
            if (a, b) not in edge_idx:
                continue
            
            # Build maximal biclique starting with L={i}, R={j}
            L = {i}
            R = {j}
            changed = True
            while changed:
                changed = False
                # Try to add vertices to L
                for w in range(n):
                    if w not in L and w not in R:
                        # w can join L if adjacent to all in R
                        if all(w in adj[r] for r in R):
                            L.add(w)
                            changed = True
                # Try to add vertices to R
                for w in range(n):
                    if w not in L and w not in R:
                        # w can join R if adjacent to all in L
                        if all(w in adj[l] for l in L):
                            R.add(w)
                            changed = True
            
            # Store the biclique as a frozenset pair
            biclique = (frozenset(L), frozenset(R))
            bicliques.add(biclique)
            # Also try the swap
            biclique2 = (frozenset(R), frozenset(L))
            bicliques.add(biclique2)
    
    # Now we have a collection of maximal bicliques.
    # Convert to list of (L_list, R_list, edge_set)
    biclique_list = []
    for L, R in bicliques:
        L_list = list(L)
        R_list = list(R)
        if len(L_list) == 0 or len(R_list) == 0:
            continue
        # Compute the set of edges covered
        e_set = set()
        for u in L_list:
            for v in R_list:
                a, b = (u, v) if u < v else (v, u)
                if (a, b) in edge_idx:
                    e_set.add(edge_idx[(a, b)])
        if len(e_set) == 0:
            continue
        biclique_list.append((L_list, R_list, e_set))
    
    # Now solve minimum set cover: cover all m edges with minimum bicliques
    # Use branch and bound
    
    m = len(edges)
    full_mask = (1 << m) - 1
    
    # Convert edge sets to bitmasks
    biclique_masks = []
    for L_list, R_list, e_set in biclique_list:
        mask = 0
        for e in e_set:
            mask |= (1 << e)
        biclique_masks.append((L_list, R_list, mask))
    
    # Sort by number of edges covered (descending) for better pruning
    biclique_masks.sort(key=lambda x: bin(x[2]).count('1'), reverse=True)
    
    # Remove dominated bicliques: if biclique A's mask is subset of biclique B's mask, A is dominated
    # Actually for set cover, we want to keep all but we can prune dominated ones
    # A set is dominated if there's another set that covers all its elements plus more
    # But with bitmasks of up to 150 bits, this is expensive. Let's skip for now.
    
    best = [m + 1]  # Best number of bicliques found
    best_solution = []
    
    # Precompute for greedy lower bound
    # For branch and bound, use a simple greedy lower bound
    
    def greedy_lb(remaining_mask):
        """Lower bound on number of sets needed to cover remaining_mask."""
        if remaining_mask == 0:
            return 0
        # Greedy: pick set covering most uncovered edges
        covered = 0
        lb = 0
        rem = remaining_mask
        while rem != 0:
            best_cover = 0
            for _, _, mask in biclique_masks:
                c = bin(rem & mask).count('1')
                if c > best_cover:
                    best_cover = c
            if best_cover == 0:
                return float('inf')
            lb += 1
            # Find which set achieved best_cover and remove its coverage
            # Actually for LB we just need to estimate, so use a simpler approach
            # Count total uncovered edges
            total = bin(rem).count('1')
            if best_cover == 0:
                return float('inf')
            rem = rem  # This doesn't work properly for a real LB
            break
        # Better LB: use the fact that max coverage per set is max_cover
        total_uncovered = bin(remaining_mask).count('1')
        max_cover = 0
        for _, _, mask in biclique_masks:
            c = bin(mask).count('1')
            if c > max_cover:
                max_cover = c
        if max_cover == 0:
            return float('inf')
        return (total_uncovered + max_cover - 1) // max_cover
    
    # Better lower bound: for each edge, if it's only covered by k sets, we need at least 1
    # Use LP relaxation style: for each edge, sum of x_j >= 1 where x_j is 1 if set j chosen
    # Dual: for each set, y_i >= 0, sum y_i over edges in set <= 1, maximize sum y_i
    # This is too complex. Use simple cardinality LB.
    
    def simple_lb(remaining_mask):
        """Lower bound: at least ceil(unc_edges / max_edges_per_set)"""
        if remaining_mask == 0:
            return 0
        unc = bin(remaining_mask).count('1')
        max_cov = 0
        for _, _, mask in biclique_masks:
            c = bin(mask & remaining_mask).count('1')
            if c > max_cov:
                max_cov = c
        if max_cov == 0:
            return float('inf')
        return (unc + max_cov - 1) // max_cov
    
    import sys
    sys.setrecursionlimit(10000)
    
    def bbsolve(idx, remaining_mask, count, current_solution):
        if remaining_mask == 0:
            if count < best[0]:
                best[0] = count
                best_solution[:] = list(current_solution)
            return
        
        if count >= best[0]:
            return
        
        lb = simple_lb(remaining_mask)
        if count + lb >= best[0]:
            return
        
        if idx >= len(biclique_masks):
            return
        
        # Option 1: skip this biclique
        bbsolve(idx + 1, remaining_mask, count, current_solution)
        
        if best[0] <= count:
            return
        
        # Option 2: include this biclique
        L_list, R_list, mask = biclique_masks[idx]
        new_remaining = remaining_mask & ~mask
        if new_remaining != remaining_mask:  # Only include if it covers something
            current_solution.append((L_list, R_list))
            bbsolve(idx + 1, new_remaining, count + 1, current_solution)
            current_solution.pop()
    
    # Initialize with a greedy solution to get a good upper bound
    def greedy_solve():
        rem = full_mask
        sol = []
        while rem != 0:
            best_idx = -1
            best_cov = 0
            for i, (L, R, mask) in enumerate(biclique_masks):
                c = bin(rem & mask).count('1')
                if c > best_cov:
                    best_cov = c
                    best_idx = i
            if best_idx < 0 or best_cov == 0:
                # Fallback: use individual edges
                for i, (u, v) in enumerate(edges):
                    if (1 << i) & rem:
                        sol.append(([u], [v]))
                        rem &= ~(1 << i)
                break
            L, R, mask = biclique_masks[best_idx]
            sol.append((L, R))
            rem &= ~mask
        return sol
    
    greedy_sol = greedy_solve()
    best[0] = len(greedy_sol)
    best_solution[:] = list(greedy_sol)
    
    # Now run branch and bound
    bbsolve(0, full_mask, 0, [])
    
    # Convert best_solution to the required format
    result = []
    for L_list, R_list in best_solution:
        result.append([list(L_list), list(R_list)])
    
    return result
# EVOLVE-BLOCK-END
