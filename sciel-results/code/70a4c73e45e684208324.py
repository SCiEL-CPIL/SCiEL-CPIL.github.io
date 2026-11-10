import numpy as np
import time
import sys
from itertools import combinations

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 600.0  # 600s limit, keep margin
    
    # 1. Generate Candidate Pool
    # Norm 2 vectors: 2 * C(11, 2) = 110 vectors (e.g., (1,1,0...), (1,-1,0...))
    # Norm 4 vectors: 4 * C(11, 2) = 440 vectors (e.g., (2,0...), (1,1,1,0...)) 
    # Wait, norm squared 4 can be:
    # - One coordinate is 2: 11 * 2 = 22 vectors
    # - Three coordinates are 1: C(11,3) * 2^3 = 165 * 8 = 1320 vectors
    # Total norm 4 candidates = 22 + 1320 = 1342
    # Total norm 2 candidates = 110
    # Total pool size ~ 1452. This is manageable for MIS.
    
    candidates = []
    # Add Norm 2 vectors
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = s1
                    v[j] = s2
                    candidates.append(v)
                    
    # Add Norm 4 vectors
    # Type 1: (2, 0, ..., 0)
    for i in range(d):
        for s in [1, -1]:
            v = np.zeros(d, dtype=np.int64)
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: (1, 1, 1, 0, ...)
    for i, j, k in combinations(range(d), 3):
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                for s3 in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = s1
                    v[j] = s2
                    v[k] = s3
                    candidates.append(v)
                    
    n = len(candidates)
    # Sort by norm descending? Or just keep as is. 
    # For MIS, ordering by degree is often better, but let's try a simple heuristic first.
    
    # 2. Build Conflict Graph
    # Two vectors u, v conflict if ||u-v||^2 < max_norm
    # max_norm will be 4 (from norm 4 vectors). 
    # If we include norm 4 vectors, the max squared norm in the set is 4.
    # Condition: ||u-v||^2 >= 4.
    # If u and v are both norm 4, ||u-v||^2 = ||u||^2 + ||v||^2 - 2 u.v = 8 - 2 u.v.
    # We need 8 - 2 u.v >= 4 => 2 u.v <= 4 => u.v <= 2.
    # If u is norm 2, v is norm 4: ||u-v||^2 = 2 + 4 - 2 u.v = 6 - 2 u.v >= 4 => 2 u.v <= 2 => u.v <= 1.
    # If u, v are both norm 2: ||u-v||^2 = 2 + 2 - 2 u.v = 4 - 2 u.v >= 4 => u.v <= 0.
    
    # Let's compute the conflict matrix.
    # Conflict[i][j] is True if they CANNOT be together.
    # We want an Independent Set, so we want a set where Conflict[i][j] is False for all i!=j.
    
    conflict = np.zeros((n, n), dtype=bool)
    
    # Precompute norms and dot products for speed
    # V is n x d
    V = np.array(candidates, dtype=np.int64)
    
    # Squared norms
    norms = np.sum(V * V, axis=1)
    max_norm = np.max(norms) # Should be 4
    
    # Dot products
    # This is n x n x d -> n x n. For n~1500, this is 1500*1500*11 ~ 25M ops, fast in numpy.
    dots = V @ V.T
    
    # Distance squared
    dist_sq = norms[:, None] + norms[None, :] - 2 * dots
    
    # Conflict if dist_sq < max_norm
    # Note: dist_sq[i][i] = 0, which is < 4, so self-conflict is True. We ignore self in MIS.
    conflict = dist_sq < max_norm
    
    # 3. Solve Maximum Independent Set
    # We use a branch and bound algorithm.
    # To make it efficient, we order vertices by degree (descending) in the conflict graph.
    # High degree vertices are more constrained, so deciding them early prunes the tree.
    
    degrees = np.sum(conflict, axis=1)
    order = np.argsort(degrees)[::-1]
    
    # Reorder conflict matrix
    # This is a bit expensive for large n, but n is small enough.
    # Alternatively, we can just use the order in the search.
    
    # Let's implement a simple recursive MIS with bitsets for speed.
    # Python bitsets are just integers.
    
    # Convert conflict to bitsets
    # conflict_bits[i] is an integer where bit j is 1 if i and j conflict.
    conflict_bits = [0] * n
    for i in range(n):
        bits = 0
        for j in range(n):
            if conflict[i][j]:
                bits |= (1 << j)
        conflict_bits[i] = bits
    
    best_set = []
    best_size = 0
    current_set = []
    
    # Global variables for recursive function
    # We need to pass the available candidates.
    # To speed up, we can use a greedy coloring bound.
    
    def greedy_coloring(candidates_list):
        """
        Greedily color the subgraph induced by candidates_list.
        Returns the number of colors used, which is an upper bound on the clique size (or independent set size in complement).
        Here we are looking for Independent Set in G, which is Clique in G_bar.
        But we are working with G (conflict graph). Independent Set in G = set of non-conflicting nodes.
        So we want to find a set of nodes such that no two are connected in G.
        This is equivalent to finding a Clique in the complement graph G_bar.
        
        Alternatively, we can just bound the size of the independent set.
        The greedy coloring bound for MIS is:
        Color the vertices of the subgraph. The number of colors is an upper bound on the size of the maximum independent set?
        No, the number of colors is an upper bound on the size of the maximum CLIQUE in the subgraph.
        Since MIS in G = Clique in G_bar, we should apply coloring to G_bar?
        
        Let's stick to the standard MIS branch and bound.
        We maintain a set of candidates P.
        We pick a vertex v from P.
        Branch 1: Include v. Then we remove v and all its neighbors from P.
        Branch 2: Exclude v. Then we remove v from P.
        
        To prune, we need an upper bound on the size of the MIS in P.
        A simple bound is |P|.
        A better bound is the greedy coloring of the subgraph induced by P.
        Wait, if we color G[P], the number of colors k means we can partition P into k independent sets.
        So the size of the MIS in G[P] is at most the sum of the sizes of the largest independent sets? No.
        
        Actually, the standard bound for MIS using coloring is:
        If we color the graph G[P] with c colors, then the maximum independent set in G[P] is at most c?
        No, that's for Clauses in SAT or something else.
        
        Let's use a simpler bound: The size of the maximum independent set is at most the number of vertices.
        To get a tighter bound, we can use the fact that if we pick a vertex v, the MIS in P is at most 1 + MIS(P \ N[v]) or MIS(P \ {v}).
        
        Let's try a simpler heuristic first: Greedy MIS.
        Then use local search (swap) to improve it.
        """
        pass

    # Given the complexity of implementing a fast, robust MIS solver in Python from scratch within time limits,
    # and knowing that n ~ 1500, a pure Python recursive B&B might be slow.
    # However, the conflict graph is likely sparse or has structure.
    
    # Let's try a Greedy approach first with a better ordering, then Local Search.
    # 1. Greedy: Pick vertex with lowest degree in current candidate set? Or highest?
    #    For MIS, picking a vertex with LOW degree is often better because it conflicts with fewer others.
    
    # Let's try:
    # 1. Start with empty set.
    # 2. Iterate through vertices sorted by degree ascending.
    # 3. If vertex doesn't conflict with any in current set, add it.
    
    # This is the seed program's strategy essentially, but with a larger pool.
    # The seed program got 110.
    # Inspiration 2 got 209.
    # Inspiration 3 got 110.
    
    # Let's implement a simple Local Search (Swap) to improve the greedy solution.
    
    # Greedy Initial Solution
    # Sort by degree ascending
    deg_sorted = np.argsort(degrees)
    
    selected = []
    selected_set = set()
    
    for i in deg_sorted:
        # Check if i conflicts with any selected
        ok = True
        for j in selected:
            if conflict[i][j]:
                ok = False
                break
        if ok:
            selected.append(i)
            selected_set.add(i)
            
    best_size = len(selected)
    best_selected = list(selected)
    
    # Local Search: Try to swap out one selected vertex for one non-selected vertex
    # Or add a vertex if possible (shouldn't be possible if greedy was maximal)
    # Or swap: remove v1, add v2 if v2 doesn't conflict with (selected - {v1})
    
    # To make this fast, we can limit the number of iterations.
    
    # Precompute non-neighbors?
    # For each i, non_neighbors[i] is the set of j such that not conflict[i][j] and i != j.
    
    # Let's do a few rounds of swaps.
    
    time_limit = 500.0 # Leave 100s for finalization
    
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        # For each selected vertex, try to replace it with a non-selected vertex
        # This is O(k * (n-k) * k) which is expensive.
        # k ~ 200, n ~ 1500. 200 * 1300 * 200 = 52M ops. Might be too slow in Python.
        
        # Alternative: Randomized restarts of greedy with different orderings.
        # Or, just use the greedy result.
        
        # Let's try a simpler improvement: 
        # Check if we can add any vertex (should be none).
        # Check if we can swap.
        
        # To keep it simple and fast, let's just do a few random restarts of the greedy algorithm
        # with random tie-breaking or different orderings.
        
        # Actually, let's just return the greedy result for now. 
        # The pool expansion from 110 to 1500 candidates should significantly increase the count.
        # Norm 2 alone gave 110.
        # Norm 4 adds many more.
        
        # Let's estimate: 
        # If we pack norm 4 vectors, they are "further apart" relative to their norm?
        # Max norm is 4. Min dist >= 4.
        # For norm 2 vectors, min dist >= 4 means u.v <= 0.
        # For norm 4 vectors, min dist >= 4 means u.v <= 2.
        # This is a looser constraint for norm 4 vectors.
        # So we can pack more norm 4 vectors.
        
        # Let's just return the greedy solution.
        break
        
    # Convert selected indices to vectors
    if best_selected:
        result = V[best_selected]
    else:
        # Fallback to seed solution if something went wrong
        result = np.array(candidates[:110])
        
    # Ensure int64 and rounding
    result = np.around(result).astype(np.int64)
    
    return result

if __name__ == "__main__":
    res = kissing_number11()
    print(res.shape)
    print(res[:5])
