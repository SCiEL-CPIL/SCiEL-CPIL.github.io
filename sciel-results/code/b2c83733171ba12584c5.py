import time
import math

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    # Distance squared on triangular lattice: (a,b) represents a + b*omega, omega=exp(i*pi/3)
    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da*da + db*db + da*db

    def count_edges(points):
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if dist_sq(points[i], points[j]) == 1:
                    count += 1
        return count

    # Hex distance from origin
    def hex_dist(p):
        a, b = p
        return (abs(a) + abs(b) + abs(a+b)) // 2

    # Generate candidate pool: Radius 5 hexagon (91 points)
    # This provides a larger search space to ensure we don't miss the optimal configuration
    # if the greedy start is slightly off, though radius 3 is usually sufficient for n=32.
    max_radius = 5
    all_points = []
    for a in range(-max_radius, max_radius + 1):
        for b in range(-max_radius, max_radius + 1):
            if hex_dist([a, b]) <= max_radius:
                all_points.append([a, b])
    
    n_candidates = len(all_points)
    
    # Map point to index for fast lookup
    point_to_idx = {tuple(p): i for i, p in enumerate(all_points)}
    
    # Precompute neighbors (distance squared == 1) for each candidate
    neighbors = [set() for _ in range(n_candidates)]
    for i in range(n_candidates):
        for j in range(i+1, n_candidates):
            if dist_sq(all_points[i], all_points[j]) == 1:
                neighbors[i].add(j)
                neighbors[j].add(i)

    # Initial configuration: Greedy center-out
    def angle(p):
        a, b = p
        x = a + b/2.0
        y = b * math.sqrt(3) / 2.0
        return math.atan2(y, x)
        
    candidates_sorted = sorted(range(n_candidates), key=lambda i: (hex_dist(all_points[i]), angle(all_points[i])))
    
    # Select first 32 points
    initial_selected = set(candidates_sorted[:32])
    current_selected = list(initial_selected)
    current_edges = count_edges([all_points[i] for i in current_selected])
    
    # Local Search with Incremental Delta Calculation
    # deg[i] = number of neighbors of i in current_selected
    deg = [0] * n_candidates
    selected_set = set(current_selected)
    
    # Initialize deg
    for i in current_selected:
        for n in neighbors[i]:
            if n in selected_set:
                deg[i] += 1
                # Note: This double counts? No, deg[i] is count of neighbors in set.
                # Wait, the loop above is wrong.
                # deg[i] should be sum(1 for n in neighbors[i] if n in selected_set)
                # The loop `for i in current_selected: for n in neighbors[i]: if n in selected_set: deg[i] += 1`
                # This correctly calculates deg[i] for i in selected_set.
                # What about i not in selected_set?
                # We need deg for ALL nodes to calculate delta for swaps.
                # So we must calculate deg for all nodes.
                
    # Recalculate deg for all nodes correctly
    deg = [0] * n_candidates
    for i in range(n_candidates):
        count = 0
        for n in neighbors[i]:
            if n in selected_set:
                count += 1
        deg[i] = count

    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        # Iterate over all selected points and all unselected points
        # To ensure we don't miss improvements, we iterate through all pairs.
        # We apply the first improving swap found (First-Improvement Hill Climbing).
        
        for i in current_selected:
            if time.time() > deadline - 5.0:
                break
            
            # Optional: Randomize order of i or j to escape local minima?
            # For now, deterministic order is fine if we restart or use larger pool.
            
            for j in range(n_candidates):
                if j in selected_set:
                    continue
                
                # Calculate delta
                # delta = (new_edges) - (old_edges)
                # new_edges = old_edges - deg[i] + deg[j] - (1 if i~j else 0)
                # Wait, let's derive carefully.
                # E(S) = sum_{k in S} deg_S(k) / 2
                # S' = S \ {i} U {j}
                # E(S') = E(S) - (edges incident to i in S) + (edges incident to j in S')
                # Edges incident to i in S: deg[i] (since i in S)
                # Edges incident to j in S': 
                #   j is connected to neighbors in S' = (S \ {i}) U {j}
                #   Neighbors of j in S \ {i}: deg[j] - (1 if i in N(j) else 0)
                #   Neighbors of j in {j}: 0
                #   So edges incident to j in S' = deg[j] - (1 if i~j else 0)
                # 
                # Delta = (deg[j] - is_adj) - deg[i]
                
                is_adj = 1 if j in neighbors[i] else 0
                delta = (deg[j] - is_adj) - deg[i]
                
                if delta > 0:
                    # Perform swap
                    selected_set.remove(i)
                    selected_set.add(j)
                    
                    # Update deg incrementally
                    # 1. Update deg for neighbors of i
                    #    For k in N(i):
                    #    if k in S' (i.e., k in S and k != i), deg[k] -= 1
                    #    if k not in S', deg[k] unchanged
                    #    Note: i is removed from S.
                    #    So for k in N(i):
                    #       if k in selected_set: deg[k] -= 1
                    #       else: deg[k] unchanged
                    #    (Since i is no longer in S, it doesn't contribute to deg[k] if k is not in S? 
                    #     No, deg[k] is |N(k) cap S|. If k not in S, deg[k] is still |N(k) cap S|.
                    #     i was in S. If i in N(k), then deg[k] included i.
                    #     Now i not in S. So if i in N(k), deg[k] decreases by 1.
                    #     This is true regardless of whether k is in S or not?
                    #     Yes, deg[k] is defined for all k.
                    #     So: for k in N(i): if i in N(k) (always true), deg[k] -= 1.
                    #     Wait, is i in N(k)? Yes, graph is undirected.
                    #     So for all k in N(i), deg[k] -= 1.
                    
                    for k in neighbors[i]:
                        deg[k] -= 1
                        
                    # 2. Update deg for neighbors of j
                    #    For k in N(j):
                    #    if k in S' (i.e., k in S and k != i? No, k in S' means k in S \ {i} U {j})
                    #    j is added to S.
                    #    So for k in N(j), if k in S', deg[k] += 1.
                    #    S' = S \ {i} U {j}.
                    #    So k in S' iff (k in S and k != i) or (k == j).
                    #    Since j in N(j) is false, k != j.
                    #    So k in S' iff k in S and k != i.
                    #    So for k in N(j):
                    #       if k in S and k != i: deg[k] += 1
                    #       else: deg[k] unchanged
                    #    Note: i is in S (before removal).
                    #    So if k == i, we skip increment.
                    #    But we already decremented deg[i] in step 1?
                    #    Let's check deg[i].
                    #    Step 1: for k in N(i), deg[k] -= 1.
                    #    If i in N(j), then j in N(i). So i is in N(j) is false? No.
                    #    If i and j are adjacent, i in N(j) and j in N(i).
                    #    Step 1: k=i is in N(j)? No, we iterate N(i).
                    #    If j in N(i), then j is in N(i). So deg[j] -= 1.
                    #    Step 2: iterate N(j). i is in N(j).
                    #    Condition: if i in S and i != i? False. So no increment for i.
                    #    So deg[i] is decremented by 1 (from step 1) and not incremented.
                    #    Is this correct?
                    #    deg[i] in S: |N(i) cap S|.
                    #    deg[i] in S': |N(i) cap S'|.
                    #    S' = S \ {i} U {j}.
                    #    N(i) cap S = {k in N(i) | k in S}.
                    #    N(i) cap S' = {k in N(i) | k in S \ {i} U {j}}.
                    #    Since i not in N(i), N(i) cap (S \ {i}) = N(i) cap S.
                    #    If j in N(i), then j is in N(i) cap S'.
                    #    So deg[i]_new = deg[i]_old + (1 if j in N(i) else 0).
                    #    
                    #    Let's re-evaluate the incremental updates for deg[i] and deg[j].
                    #    
                    #    Current logic:
                    #    1. for k in N(i): deg[k] -= 1
                    #       This includes deg[i]? No, i not in N(i).
                    #       This includes deg[j] if j in N(i).
                    #    2. for k in N(j): if k in S and k != i: deg[k] += 1
                    #       This includes deg[j]? No, j not in N(j).
                    #       This includes deg[i] if i in N(j) and i in S and i != i? False.
                    #    
                    #    So deg[i] is NOT updated by these loops?
                    #    But deg[i] must change!
                    #    deg[i]_old = |N(i) cap S|.
                    #    deg[i]_new = |N(i) cap S'|.
                    #    Change: - (1 if i in N(i) [False]) + (1 if j in N(i) [True if adj])
                    #    So deg[i] should increase by 1 if adj, else 0.
                    #    
                    #    deg[j]_old = |N(j) cap S|.
                    #    deg[j]_new = |N(j) cap S'|.
                    #    Change: - (1 if i in N(j) [True if adj]) + (1 if j in N(j) [False])
                    #    So deg[j] should decrease by 1 if adj, else 0.
                    #    
                    #    So we need explicit updates for deg[i] and deg[j].
                    
                    # Update deg[i]
                    if is_adj:
                        deg[i] += 1
                    # deg[i] was decremented? No, i not in N(i).
                    # But wait, did we decrement deg[i] in step 1?
                    # Step 1 iterates N(i). i is not in N(i). So deg[i] unchanged so far.
                    # Correct.
                    
                    # Update deg[j]
                    if is_adj:
                        deg[j] -= 1
                    # Did we increment deg[j] in step 2?
                    # Step 2 iterates N(j). j is not in N(j). So deg[j] unchanged so far.
                    # But did we decrement deg[j] in step 1?
                    # Step 1 iterates N(i). If is_adj, j in N(i). So deg[j] was decremented by 1 in step 1.
                    # So if is_adj:
                    #   Step 1: deg[j] -= 1
                    #   Step 2: deg[j] unchanged (j not in N(j))
                    #   Explicit: deg[j] -= 1
                    #   Total change: -2?
                    #   Expected change: -1.
                    #   So there is a double count.
                    
                    # Let's trace carefully.
                    # deg[k] = |N(k) cap S|.
                    # S -> S' = S \ {i} U {j}.
                    
                    # For k in N(i) \ {j}:
                    #   k in S? If yes, k in S' (since k != i, j).
                    #   deg[k]_old included i (since i in S, i in N(k)).
                    #   deg[k]_new does not include i.
                    #   So deg[k] -= 1.
                    #   If k not in S, deg[k] unchanged.
                    
                    # For k in N(j) \ {i}:
                    #   k in S? If yes, k in S'.
                    #   deg[k]_old did not include j (j not in S).
                    #   deg[k]_new includes j.
                    #   So deg[k] += 1.
                    #   If k not in S, deg[k] unchanged.
                    
                    # For k = i:
                    #   deg[i]_old = |N(i) cap S|.
                    #   deg[i]_new = |N(i) cap S'|.
                    #   S' = S \ {i} U {j}.
                    #   N(i) cap S' = (N(i) cap S) \ {i} U (N(i) cap {j}).
                    #   i not in N(i). So (N(i) cap S) \ {i} = N(i) cap S.
                    #   N(i) cap {j} = {j} if j in N(i) else empty.
                    #   So deg[i]_new = deg[i]_old + (1 if is_adj else 0).
                    
                    # For k = j:
                    #   deg[j]_old = |N(j) cap S|.
                    #   deg[j]_new = |N(j) cap S'|.
                    #   N(j) cap S' = (N(j) cap S) \ {i} U (N(j) cap {j}).
                    #   j not in N(j). So N(j) cap {j} = empty.
                    #   N(j) cap S' = (N(j) cap S) \ {i}.
                    #   So deg[j]_new = deg[j]_old - (1 if is_adj else 0).

                    # So the updates are:
                    # 1. for k in N(i):
                    #      if k == i: continue (impossible)
                    #      if k == j: 
                    #          # Handled by deg[j] update?
                    #          # Let's separate j.
                    #      else:
                    #          if k in S: deg[k] -= 1
                    # 2. for k in N(j):
                    #      if k == j: continue (impossible)
                    #      if k == i:
                    #          # Handled by deg[i] update?
                    #      else:
                    #          if k in S: deg[k] += 1
                    # 3. deg[i] += (1 if is_adj else 0)
                    # 4. deg[j] -= (1 if is_adj else 0)
                    
                    # Let's implement this.
                    
                    # Undo previous incorrect incremental attempts in thought process.
                    # We will do it cleanly.
                    
                    # 1. Neighbors of i (excluding j)
                    for k in neighbors[i]:
                        if k == j:
                            continue
                        if k in selected_set:
                            deg[k] -= 1
                            
                    # 2. Neighbors of j (excluding i)
                    for k in neighbors[j]:
                        if k == i:
                            continue
                        if k in selected_set:
                            deg[k] += 1
                            
                    # 3. Update deg[i]
                    if is_adj:
                        deg[i] += 1
                        
                    # 4. Update deg[j]
                    if is_adj:
                        deg[j] -= 1
                        # Note: deg[j] was decremented in step 1 if is_adj?
                        # No, we skipped k=j in step 1.
                        # So deg[j] is only updated in step 4.
                        # Correct.
                        
                    # Update current_edges
                    current_edges += delta
                    improved = True
                    break
            if improved:
                break
        
    # Final result
    result = [all_points[i] for i in current_selected]
    
    # Verify
    assert len(result) == 32
    assert len(set(tuple(p) for p in result)) == 32
    for p in result:
        assert abs(p[0]) <= 64 and abs(p[1]) <= 64
        
    return result
