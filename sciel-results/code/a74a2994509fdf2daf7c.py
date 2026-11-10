import numpy as np
import itertools
import time
import random
from collections import defaultdict

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 610  # safety margin before 630s limit
    
    # Strategy: Allow mixed norms. 
    # If max_norm_sq = M, all points must have norm_sq <= M
    # and all pairwise distances must be >= M.
    # 
    # Key insight: For integer vectors, if we use vectors with entries in {-1, 0, 1, 2, -2},
    # we can have a larger pool. The constraint is:
    #   For all i != j: ||p_i - p_j||^2 >= max_k ||p_k||^2
    #
    # Let's try: use vectors with norm_sq = 4 (entries like 2e_i) and norm_sq = 4 (4 ones)
    # Actually, let's be more systematic.
    #
    # NEW APPROACH: Build a graph where nodes are candidate integer vectors with small norm,
    # and edges connect pairs that are "compatible" (distance constraint can be satisfied).
    # Then find a large clique/independent set.
    #
    # But the constraint depends on the MAX norm in the selected set, which makes it tricky.
    #
    # Alternative: Fix the max norm to be M, then all selected points must have norm_sq <= M
    # and pairwise distance >= M. Find max such set. Try multiple M values.
    
    # Generate candidate vectors with small integer entries
    # Entries in {-2, -1, 0, 1, 2}
    # But that's 5^11 = 48M, too many.
    # 
    # Better: Generate vectors with at most 4 non-zero entries (like the current approach)
    # but also include the 220 minimal vectors of D11 (norm 4, with even number of odd entries... 
    # actually D11 minimal vectors have norm 4: 2e_i ± 2e_j type, or 4 ones with even parity)
    
    # Let's use a different structural approach:
    # 1. Start with the 220 D11 minimal vectors (norm_sq = 4)
    # 2. These have pairwise distances that are either 4, 8, or 16 (squared)
    #    Wait, let me check: two D11 minimal vectors can have dot product -2, 0, or 2
    #    So distance squared = 4 + 4 - 2*dot = 8 - 2*dot
    #    If dot = -2: dist_sq = 12
    #    If dot = 0: dist_sq = 8  
    #    If dot = 2: dist_sq = 4
    #    If dot = -4: impossible for norm-4 vectors? Actually for D11, min dot is -4?
    #    Let me reconsider. D11 = {x in Z^11 : sum x_i is even}
    #    Minimal vectors have norm 4. They are: ±2e_i ± 2e_j (i≠j), and vectors with four 1's or four -1's (or two 1's and two -1's) with even sum.
    #    Actually, the minimal vectors of D_n for n>=5 are: ±e_i ± e_j and (1/2)(±1±1...±1) but those aren't integer.
    #    For integer D11: vectors with sum even and norm 4. These are:
    #    - 2e_i - 2e_j type? No, that has norm 8.
    #    - e_i + e_j + e_k + e_l (four 1's, sum=4, even) - norm 4
    #    - e_i + e_j - e_k - e_l (two 1's, two -1's, sum=0, even) - norm 4
    #    - 2e_i (norm 4, sum=2, even) - yes!
    #    
    #    So D11 minimal vectors include: 2e_i (22 of them), and vectors with four non-zero entries
    #    each being ±1, with even number of +1's... wait, sum must be even.
    #    Four entries each ±1: sum is even iff number of +1's is even (0, 2, or 4).
    #    So: four +1's, two +1's and two -1's, or four -1's.
    #    
    #    Number: C(11,4) * (1 + C(4,2) + 1) = 330 * (1 + 6 + 1) = 330 * 8 = 2640? 
    #    Wait, that's too many. Let me recount.
    #    For each 4-subset, the patterns with even sum: 
    #    ++++ (1), ++--, +-+-, +--+, -++-, -+-+, --++ (6), ---- (1) = 8 patterns.
    #    But each vector and its negative are both in D11. So total = 330 * 8 = 2640? 
    #    Hmm, that seems high. Let me just generate them.
    
    # Actually, let me take a completely different structural approach.
    # 
    # The current score is 138.5. Let me try to understand what's achievable.
    # 
    # NEW STRUCTURAL APPROACH: Use a SAT/SMT-like approach with local search.
    # Represent the problem as: select a subset of candidates such that:
    #   max_norm_sq(selected) <= min_pair_dist_sq(selected)
    # 
    # This is equivalent to: there exists M such that all selected points have norm_sq <= M
    # and all pairwise distances >= M.
    # 
    # For a fixed M, this is a maximum independent set problem on a graph where
    # edge (i,j) exists if dist_sq(i,j) < M, among nodes with norm_sq <= M.
    # 
    # Let's try M = 4 first (norm_sq <= 4, dist_sq >= 4).
    # Candidates: all integer vectors with norm_sq <= 4.
    # These are: 0, ±e_i (norm 1), ±2e_i (norm 4), e_i±e_j (norm 2), etc.
    # But we can't use 0 (zero vector forbidden).
    # 
    # For M=4: candidates with norm_sq in {1, 2, 4}.
    # dist_sq >= 4 means: for any two selected, ||p-q||^2 >= 4.
    # 
    # This is a large graph. Let me try a different M.
    # 
    # Actually, let me try a completely different representation:
    # Use the fact that we can scale. If we find a configuration with max_norm_sq = M
    # and min_dist_sq >= M, we can potentially add more points.
    
    # Let me try: generate ALL integer vectors with entries in {-2,-1,0,1,2} and norm_sq <= 8
    # Then for various M values, find max compatible set.
    
    # But that's still huge. Let me be smarter.
    
    # STRUCTURAL CHANGE: Use a layered approach.
    # Layer 1: Points with norm_sq = 4 (the 22 + 4-combo sign patterns = 22 + 330*8 = 2662? No...)
    # Actually, let me just try to improve on the current approach with better search.
    
    # The current approach uses 22 base points (±2e_i) plus some 4-combo points.
    # Score 138.5 means about 138-139 points total.
    
    # Let me try: use a branch-and-bound maximum independent set on the conflict graph.
    # The conflict graph: nodes are candidate points, edge if they conflict (dist_sq < max_norm).
    # But max_norm depends on the set... 
    # 
    # Fix max_norm_sq = 4. Then all points must have norm_sq <= 4, and pairwise dist_sq >= 4.
    # Candidates with norm_sq <= 4:
    # - ±e_i: 22 points, norm_sq = 1
    # - ±2e_i: 22 points, norm_sq = 4  
    # - e_i ± e_j: 11*10*2 = 220 points? No: for each pair (i,j), i≠j, we have e_i+e_j, e_i-e_j, -e_i+e_j, -e_i-e_j. That's 4 per pair, 11*10/2*4 = 220. Norm_sq = 2.
    # - Four-entry vectors with ±1: as discussed, many.
    # 
    # Total candidates for M=4: 22 + 22 + 220 + (330*8) = 22+22+220+2640 = 2904. That's manageable!
    # 
    # For M=4, conflict: two points conflict if dist_sq < 4, i.e., dist_sq = 0 or 1 or 2 or 3.
    # Since all are integer, dist_sq is integer. dist_sq < 4 means dist_sq in {0,1,2,3}.
    # dist_sq = 0 means same point. dist_sq = 1 means differ by ±e_i. dist_sq = 2 means differ by e_i±e_j. dist_sq = 3 means... possible?
    
    # Let me generate these candidates and build the conflict graph, then find max independent set.
    
    # Generate candidates with norm_sq <= 4
    candidates = []
    cand_set = set()
    
    # Type 1: ±e_i
    for i in range(d):
        for sign in [1, -1]:
            p = np.zeros(d, dtype=np.int64)
            p[i] = sign
            t = tuple(p)
            if t not in cand_set:
                cand_set.add(t)
                candidates.append(p)
    
    # Type 2: ±2e_i
    for i in range(d):
        for sign in [1, -1]:
            p = np.zeros(d, dtype=np.int64)
            p[i] = 2 * sign
            t = tuple(p)
            if t not in cand_set:
                cand_set.add(t)
                candidates.append(p)
    
    # Type 3: e_i ± e_j
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            for si in [1, -1]:
                for sj in [1, -1]:
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    t = tuple(p)
                    if t not in cand_set:
                        cand_set.add(t)
                        candidates.append(p)
    
    # Type 4: Four-entry vectors with ±1, even number of +1's (so sum is even)
    # Actually for M=4, we need norm_sq = 4, so exactly four non-zero entries each ±1.
    # Sum must be... well, the constraint is just norm_sq <= 4, not that sum is even.
    # Wait, I was thinking of D11 lattice. For general integer vectors, no parity constraint.
    # So all C(11,4) * 16 = 330 * 16 = 5280 four-entry ±1 vectors? 
    # But many have norm_sq = 4. Yes, all of them.
    # But that's 5280 more candidates, total ~7500. Still manageable.
    # 
    # But wait, for M=4, we need dist_sq >= 4 between all pairs. Many of these will conflict.
    # Let me just generate them all and build the graph.
    
    for combo in itertools.combinations(range(d), 4):
        for sm in range(16):
            p = np.zeros(d, dtype=np.int64)
            for k, idx in enumerate(combo):
                p[idx] = 1 if (sm >> k) & 1 else -1
            t = tuple(p)
            if t not in cand_set:
                cand_set.add(t)
                candidates.append(p)
    
    n_cand = len(candidates)
    print(f"Number of candidates (norm_sq <= 4): {n_cand}")
    
    # Build conflict graph: edge (i,j) if dist_sq(i,j) < 4
    # dist_sq < 4 means dist_sq in {0, 1, 2, 3}
    # Since distinct points, dist_sq >= 1.
    # Precompute norms
    norms_sq = np.array([np.dot(p, p) for p in candidates], dtype=np.int64)
    
    # For M=4, we need all selected to have norm_sq <= 4 (all do) and pairwise dist_sq >= 4.
    # Conflict if dist_sq < 4.
    
    # Build adjacency: adj[i] = set of j that conflict with i
    adj = [set() for _ in range(n_cand)]
    
    # This is O(n^2) which is 7500^2 = 56M, might be slow in Python.
    # Let's use numpy for batch computation.
    
    cand_arr = np.array(candidates, dtype=np.int64)
    
    # Compute pairwise squared distances in batches
    # dist_sq(i,j) = ||cand_i||^2 + ||cand_j||^2 - 2*cand_i·cand_j
    
    conflict_edges = []
    batch_size = 500
    for i_start in range(0, n_cand, batch_size):
        i_end = min(i_start + batch_size, n_cand)
        batch = cand_arr[i_start:i_end]
        # Compute dot products with all
        dots = batch @ cand_arr.T  # (batch_size, n_cand)
        dist_sq = norms_sq[i_start:i_end, None] + norms_sq[None, :] - 2 * dots
        # Conflict if dist_sq < 4 and i != j
        for i_local in range(i_end - i_start):
            i_global = i_start + i_local
            # Find j where dist_sq < 4 and j > i_global (to avoid duplicates)
            mask = (dist_sq[i_local] < 4) & (np.arange(n_cand) > i_global)
            for j_global in np.where(mask)[0]:
                adj[i_global].add(j_global)
                adj[j_global].add(i_global)
    
    print("Conflict graph built")
    
    # Now find maximum independent set.
    # Use a greedy + local search approach, or branch and bound.
    # For ~7500 nodes, branch and bound might be too slow.
    # Let's use a greedy coloring-based bound with backtracking.
    
    # Actually, let's try a different M value. Maybe M=4 is too restrictive.
    # Let me also try M=8.
    
    # For now, let's try greedy on the M=4 graph.
    
    def greedy_mis(adj, n, order):
        """Greedy maximum independent set."""
        selected = []
        removed = [False] * n
        for v in order:
            if removed[v]:
                continue
            selected.append(v)
            for u in adj[v]:
                removed[u] = True
            removed[v] = True
        return selected
    
    # Try multiple random orderings
    best_set = None
    best_size = 0
    
    n_trials = 200
    for trial in range(n_trials):
        if time.time() > deadline - 60:
            break
        
        if trial == 0:
            order = list(range(n_cand))
        else:
            rng = random.Random(trial * 104729 + 7)
            order = list(range(n_cand))
            rng.shuffle(order)
        
        # Degree-based ordering: pick lowest degree first
        if trial % 2 == 0:
            # Sort by degree ascending
            order.sort(key=lambda x: len(adj[x]))
        
        result = greedy_mis(adj, n_cand, order)
        
        if len(result) > best_size:
            best_size = len(result)
            best_set = result[:]
        
        if time.time() > deadline - 60:
            break
    
    print(f"Best MIS size for M=4: {best_size}")
    
    # Local improvement: try to swap points
    if best_set is not None and time.time() < deadline - 30:
        improved = True
        while improved and time.time() < deadline - 30:
            improved = False
            best_set = best_set[:]
            selected_set = set(best_set)
            
            for idx in range(len(best_set)):
                if time.time() > deadline - 30:
                    break
                old = best_set[idx]
                # Try to replace old with a new point that doesn't conflict with remaining
                remaining = [v for v in best_set if v != old]
                remaining_set = set(remaining)
                
                # Find candidates that don't conflict with any in remaining
                for new_v in range(n_cand):
                    if new_v in selected_set:
                        continue
                    # Check if new_v conflicts with any in remaining
                    conflict = False
                    for u in remaining:
                        if u in adj[new_v]:
                            conflict = True
                            break
                    if not conflict:
                        best_set[idx] = new_v
                        selected_set = set(best_set)
                        improved = True
                        break
                if improved:
                    break
    
    # Build the result
    if best_set:
        points = [candidates[i] for i in best_set]
    else:
        points = []
    
    # Ensure no zero vector
    points = [p for p in points if np.any(p != 0)]
    
    if len(points) < 2:
        # Fallback: return the base points
        points = []
        for i in range(d):
            p1 = np.zeros(d, dtype=np.int64)
            p1[i] = 2
            points.append(p1)
            p2 = np.zeros(d, dtype=np.int64)
            p2[i] = -2
            points.append(p2)
    
    points_array = np.array(points, dtype=np.int64)
    
    # Verify
    if len(points_array) >= 2:
        max_norm_sq = np.max(np.sum(points_array ** 2, axis=1))
        min_dist_sq = float('inf')
        for i in range(len(points_array)):
            for j in range(i + 1, len(points_array)):
                diff = points_array[i] - points_array[j]
                dsq = np.dot(diff, diff)
                if dsq < min_dist_sq:
                    min_dist_sq = dsq
        if min_dist_sq < max_norm_sq:
            # Invalid, fallback
            points = []
            for i in range(d):
                p1 = np.zeros(d, dtype=np.int64)
                p1[i] = 2
                points.append(p1)
                p2 = np.zeros(d, dtype=np.int64)
                p2[i] = -2
                points.append(p2)
            points_array = np.array(points, dtype=np.int64)
    
    return points_array
