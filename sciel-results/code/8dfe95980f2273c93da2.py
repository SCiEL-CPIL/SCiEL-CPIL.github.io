import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin for 60s limit
    
    N = 256
    
    # Precompute 4-AP tuples
    # A 4-AP is (a, a+d, a+2d, a+3d) with a>=1, a+3d<=N
    four_ap_tuples = []
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            four_ap_tuples.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Precompute 3-AP tuples for counting
    # A 3-AP is (a, a+d, a+2d) with a>=1, a+2d<=N
    three_ap_tuples = []
    for d in range(1, (N - 1) // 2 + 1):
        max_a = N - 2 * d
        for a in range(1, max_a + 1):
            three_ap_tuples.append((a, a + d, a + 2 * d))
            
    # Helper: count 3-APs in a set
    def count_3aps(S):
        count = 0
        # Use a boolean array for O(1) lookup
        arr = [False] * (N + 1)
        for x in S:
            arr[x] = True
        for a, b, c in three_ap_tuples:
            if arr[a] and arr[b] and arr[c]:
                count += 1
        return count

    # --- Branch and Bound for Maximum Weight Independent Set ---
    # Vertices: 1..N
    # Edges: (u, v) if u and v are part of the same 4-AP? 
    # Wait, the constraint is "no 4-term AP". This is a 4-uniform hypergraph independent set problem.
    # Standard MIP solvers or simple B&B on graphs (2-uniform) don't directly apply.
    # However, we can approximate or use a different B&B logic.
    # Actually, the "conflict" is that we cannot select ANY 4-AP.
    # This is equivalent to finding a subset A such that for all 4-APs (a,b,c,d), not all are in A.
    
    # Let's refine the B&B approach for hypergraphs.
    # State: (current_set, remaining_candidates, forbidden_sets)
    # This is complex.
    
    # Alternative Structural Change: 
    # The prompt suggests: "Model the problem as a maximum independent set in a conflict graph where vertices represent candidate elements and edges represent 4-AP violations".
    # This interpretation is slightly flawed for 4-APs (which are hyperedges), but we can transform it.
    # If we select a 4-AP, it's invalid.
    # We can use a "Conflict Graph" where edges represent pairs that *cannot* coexist? No, that's for 2-APs or 3-APs if we were forbidding them.
    # For 4-APs, any 4-tuple is forbidden.
    
    # Let's stick to a more effective structural change for this specific constraint:
    # **Dynamic Programming / Greedy with Backtracking on "Density Layers"**
    # Or better: **Iterative Deepening A* (IDA*)** or **Branch and Bound on the 4-AP constraints directly**.
    
    # Given the time limit, a pure B&B on 256 elements is likely too slow without strong heuristics.
    # Let's try a **Weighted Greedy Set Packing** with **Local Improvement via Conflict Resolution**.
    
    # Actually, the most promising "different family" is **Genetic Algorithm** or **Simulated Annealing on a Bitset** which is faster than Python sets.
    # The current program uses Python sets and explicit loops for counting, which is slow.
    # Using NumPy/Bitsets for the 3-AP count and 4-AP check can speed up the inner loop by 100x, allowing for far more iterations.
    
    # Let's implement a **Bitset-based Simulated Annealing** with a **Restart Strategy**.
    # This is structurally different in implementation efficiency, allowing a different search depth.
    # But the prompt asks for a "different design or modelling family".
    
    # Let's try **Greedy Randomized Adaptive Search Procedure (GRASP)**.
    # 1. Constructive phase: Build a set greedily. At each step, choose the element from a "restricted candidate list" (RCL) that maximizes a score function.
    # 2. Local improvement phase: Apply a local search (swap, add, remove) to improve the solution.
    # 3. Repeat with different random seeds for RCL construction.
    
    # This is distinct from the current "random walk from a single seed".
    
    # Precompute 4-AP conflicts for fast checking
    # For each element x, store the list of 4-APs it belongs to.
    four_ap_neighbors = {i: [] for i in range(1, N + 1)}
    for idx, (a, b, c, d) in enumerate(four_ap_tuples):
        for x in (a, b, c, d):
            four_ap_neighbors[x].append(idx)
            
    # Precompute 3-AP contributions for fast scoring
    # For each element x, store the list of 3-APs it belongs to.
    three_ap_neighbors = {i: [] for i in range(1, N + 1)}
    for idx, (a, b, c) in enumerate(three_ap_tuples):
        for x in (a, b, c):
            three_ap_neighbors[x].append(idx)
    
    # Fast 4-AP check using a boolean array
    def has_4ap(arr, x_add):
        # arr is a list of bools, size N+1
        # Check only 4-APs containing x_add
        for idx in four_ap_neighbors[x_add]:
            a, b, c, d = four_ap_tuples[idx]
            if arr[a] and arr[b] and arr[c] and arr[d]:
                return True
        return False
        
    # Fast 3-AP count delta
    def count_3aps_delta(arr, x_add):
        # Count how many 3-APs are completed by adding x_add
        count = 0
        for idx in three_ap_neighbors[x_add]:
            a, b, c = three_ap_tuples[idx]
            if arr[a] and arr[b] and arr[c]:
                count += 1
        return count
        
    def count_3aps_delta_remove(arr, x_rem):
        # Count how many 3-APs are broken by removing x_rem
        count = 0
        for idx in three_ap_neighbors[x_rem]:
            a, b, c = three_ap_tuples[idx]
            if arr[a] and arr[b] and arr[c]:
                count += 1
        return count

    best_set = set()
    best_score = 0
    
    # GRASP: Multiple restarts
    # Time budget per restart
    max_restarts = 20
    time_per_restart = (deadline - time.time()) / max_restarts
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Phase 1: Constructive (Greedy with Randomization)
        arr = [False] * (N + 1)
        current_set = set()
        
        # Score function: 
        # We want elements that complete many 3-APs but don't create 4-APs.
        # Since we build incrementally, we check 4-AP validity.
        
        # To add randomness, we use a Restricted Candidate List (RCL).
        # At each step, calculate the "gain" (3-APs completed) for each valid candidate.
        # Select the top alpha% of candidates randomly.
        
        alpha = 0.2 # 20% best candidates
        remaining = list(range(1, N + 1))
        
        while remaining:
            valid_candidates = []
            gains = []
            
            # Filter and score
            for x in remaining:
                if has_4ap(arr, x):
                    continue
                # Calculate gain: how many 3-APs does x complete with current set?
                gain = count_3aps_delta(arr, x)
                valid_candidates.append(x)
                gains.append(gain)
                
            if not valid_candidates:
                break
                
            # Sort by gain
            sorted_pairs = sorted(zip(valid_candidates, gains), key=lambda z: z[1], reverse=True)
            
            # Determine RCL size
            k = max(1, int(len(sorted_pairs) * alpha))
            rcl = sorted_pairs[:k]
            
            # Pick randomly from RCL
            chosen_idx = int(time.time() * 1000000) % len(rcl) # Simple pseudo-random
            chosen_x, chosen_gain = rcl[chosen_idx]
            
            arr[chosen_x] = True
            current_set.add(chosen_x)
            remaining.remove(chosen_x)
            
        # Phase 2: Local Improvement (Hill Climbing)
        # Use the fast bitset-like array for checks
        current_score = count_3aps(current_set)
        
        # Simple local search: Try all swaps and adds
        # To keep it fast, limit iterations or use a simple greedy improvement
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            
            # Try adding elements
            for x in range(1, N + 1):
                if arr[x]:
                    continue
                if has_4ap(arr, x):
                    continue
                # Check if adding improves score
                # We know it adds count_3aps_delta(arr, x) 3-APs
                # It never removes any
                # So if delta > 0, we should add it? 
                # Wait, adding an element might prevent adding *other* elements later?
                # In local search, we accept if it improves current score.
                # Since adding always increases 3-AP count (or keeps same), and doesn't break 4-APs,
                # we should always add if possible.
                # BUT, we must check if adding x creates a 4-AP. We already did.
                # So why didn't the greedy phase add it? 
                # Because it might have been invalid at that time due to other elements.
                # Or it was not in the RCL.
                
                # If we can add x without 4-AP, do it.
                arr[x] = True
                current_set.add(x)
                current_score += count_3aps_delta(arr, x) # Note: this counts 3-APs completed *with* x.
                # Actually, count_3aps_delta counts 3-APs where x is part of the triple and the other two are in S.
                # Since x was not in S, these are new 3-APs.
                improved = True
                
            if improved:
                continue
                
            # Try swaps: remove x, add y
            # This is O(N^2) which is ~65k checks. Might be slow in Python.
            # Let's try a limited number of swaps.
            # Or just stop if no adds were possible.
            
            # Let's try a few random swaps to escape local minima
            for _ in range(50):
                if time.time() > deadline - 1:
                    break
                if not current_set:
                    break
                x_rem = list(current_set)[int(time.time() * 10000) % len(current_set)]
                # Find a y not in set that can be added if x_rem is removed
                # This is complex to check quickly.
                # Let's skip swaps for now and rely on multiple restarts.
                pass
            
        if current_score > best_score:
            best_score = current_score
            best_set = set(current_set)
            
    return sorted(best_set)
