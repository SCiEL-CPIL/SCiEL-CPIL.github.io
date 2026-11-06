import time
import random
import math
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLOR = 120

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

CYCLES = []
for a in range(N):
    for b in range(a+1, N):
        for c in range(b+1, N):
            for d in range(c+1, N):
                CYCLES.append((edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a)))
                CYCLES.append((edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a)))
                CYCLES.append((edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a)))

NUM_CYCLES = len(CYCLES)

edge_to_cycles = [[] for _ in range(NUM_EDGES)]
for c_idx, cycle in enumerate(CYCLES):
    for e in cycle:
        edge_to_cycles[e].append(c_idx)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Initial valid coloring: 1 color
    current_colors = [0] * NUM_EDGES
    
    # Precompute cycle edge sets for faster distinct counting if needed, 
    # but we maintain counts dynamically.
    counts = np.zeros((NUM_CYCLES, MAX_COLOR), dtype=np.int8)
    distinct_counts = np.zeros(NUM_CYCLES, dtype=np.int8)
    
    # Initialize counts for color 0
    for c_idx, cycle in enumerate(CYCLES):
        distinct_counts[c_idx] = 1
        for e in cycle:
            counts[c_idx, 0] += 1
            
    global_color_counts = [0] * MAX_COLOR
    global_color_counts[0] = NUM_EDGES
    
    current_score = 1
    best_colors = current_colors[:]
    best_score = 1
    
    # Precompute edge degrees (number of cycles per edge)
    edge_degrees = [len(edge_to_cycles[e]) for e in range(NUM_EDGES)]
    
    # Greedy color introduction
    # We want to introduce new colors one by one.
    # For each new color, we find an edge to assign it to such that the coloring remains valid.
    # To maximize the chance of success, we prioritize edges that are "less constrained" 
    # in terms of potential future conflicts, or simply pick the best available edge.
    
    # A better heuristic: 
    # 1. Start with all edges color 0.
    # 2. Try to add color 1. Find an edge e such that changing it to 1 doesn't create a 4-distinct cycle.
    #    Since all are 0, changing any edge to 1 makes cycles containing it have {0, 1}. Distinct count becomes 2. Valid.
    #    Which edge? Any edge works for the first new color. 
    #    But to allow more colors later, we should pick an edge that minimizes the number of cycles 
    #    that now have 2 colors? No, all edges have similar degree.
    #    Actually, let's look at the structure.
    
    # Let's try a different structural approach:
    # Partition edges into groups based on some property, and assign distinct colors to groups?
    # No, that's what the current random search is struggling to find efficiently.
    
    # Let's try a "Simulated Annealing with Restart" but with a smarter move generator.
    # Instead of random edge + random color, try:
    # 1. Pick a random edge.
    # 2. Try to assign it a color that is currently used by few other edges in its cycles?
    # Or better:
    # 3. Pick a random edge currently with color C.
    # 4. Try to change it to a NEW color (not currently used in any of its cycles) if possible?
    #    If a cycle has colors {A, B, C}, adding D makes it invalid.
    #    So we can only add a new color K to edge e if NO cycle containing e already has 3 distinct colors 
    #    that are not K? No, if a cycle has {A, B, C} and we change one edge to K, it becomes {A, B, K} (if the edge was A) 
    #    or {A, B, C, K} (if the edge was not A,B,C? Impossible, edge is part of cycle).
    #    If edge e has color A, and cycle has {A, B, C}, changing e to K gives {K, B, C}. Distinct count is 3. Valid!
    #    Wait, if cycle has {A, B, C} (3 distinct), and we change an edge with color A to K, the new set is {K, B, C}. 
    #    If K is not B or C, distinct count is 3. Still valid.
    #    If K is B, distinct count is 2. Valid.
    #    So we can always change an edge to a new color K if the cycle doesn't have 3 distinct colors 
    #    OTHER than the color of the edge?
    #    Let's re-verify:
    #    Cycle edges: e1(A), e2(B), e3(C), e4(A). Distinct: {A, B, C}.
    #    Change e1 to K. New colors: K, B, C, A. Distinct: {K, B, C, A}.
    #    If K is new, distinct count is 4. INVALID.
    #    So my previous deduction was wrong.
    #    If a cycle has 3 distinct colors, changing any edge in it to a 4th distinct color makes it invalid.
    #    Therefore, to introduce a new color K, we must pick an edge e such that 
    #    NO cycle containing e currently has 3 distinct colors.
    #    If a cycle has 2 distinct colors {A, B}, and we change an edge with color A to K, 
    #    the new colors are {K, B, ...}. If the other edges are A or B, the new set is {K, B} or {K, B, A? No, A is gone from that edge}.
    #    Example: e1(A), e2(B), e3(A), e4(B). Change e1 to K. New: K, B, A, B. Distinct {K, B, A}. Count 3. Valid.
    #    So, we can introduce a new color K on edge e if for every cycle c containing e:
    #    - If distinct_counts[c] <= 2: Always safe?
    #      If distinct is 1 {A}, change e(A) to K -> {K}. Distinct 1. Safe.
    #      If distinct is 2 {A, B}, and e is A. Change to K. Others are A or B. New set {K, A?, B}. 
    #      If there is another A, set is {K, A, B}. Distinct 3. Safe.
    #      If no other A, set is {K, B}. Distinct 2. Safe.
    #    - If distinct_counts[c] == 3: Unsafe, as shown above.
    #    - If distinct_counts[c] == 4: Already invalid (shouldn't happen).
    
    # So the rule is: An edge e can be assigned a new color K (not currently in the cycle) 
    # ONLY IF all cycles containing e have distinct_counts <= 2.
    # Wait, what if K is already in the cycle?
    # If K is in the cycle, we are not introducing a "new" color to the global palette necessarily, 
    # but we are changing the local configuration.
    # However, our goal is to maximize GLOBAL distinct colors.
    # So we should focus on introducing colors that are NEW to the global palette.
    
    # Strategy:
    # 1. Maintain list of unused colors.
    # 2. Iterate. Find an edge e such that all cycles containing e have distinct_counts <= 2.
    # 3. If such an edge exists, assign it the next unused color.
    # 4. Update counts.
    # 5. Repeat until no such edge exists or time runs out.
    
    # This is a greedy constructive heuristic.
    
    unused_colors = list(range(1, MAX_COLOR))
    
    # To speed up finding valid edges, we can maintain a set of "candidate" edges.
    # An edge is a candidate if all its cycles have distinct_counts <= 2.
    # Initially, all edges are candidates (distinct_counts are 1).
    
    candidate_edges = set(range(NUM_EDGES))
    
    while time.time() < deadline and unused_colors:
        if not candidate_edges:
            break
            
        # Pick an edge from candidates. 
        # Heuristic: Pick the one with highest degree? Or random?
        # High degree might be more "central" and constrain others more? 
        # Or low degree might be safer?
        # Let's try random for now, or maybe lowest degree to preserve flexibility of high-degree edges?
        # Actually, if we use a high-degree edge, we "spend" its flexibility. 
        # If we use a low-degree edge, we preserve high-degree edges for later when constraints are tighter?
        # Let's try picking the edge with the LOWEST degree among candidates.
        
        # Optimization: Instead of iterating all candidates to find min degree, 
        # we can just pick one randomly or the first one. 
        # Given the small N=16, NUM_EDGES=120, iterating is fast.
        
        best_e = -1
        min_deg = float('inf')
        for e in candidate_edges:
            if edge_degrees[e] < min_deg:
                min_deg = edge_degrees[e]
                best_e = e
                
        if best_e == -1:
            break
            
        new_color = unused_colors.pop(0)
        
        # Assign new_color to best_e
        old_color = current_colors[best_e]
        current_colors[best_e] = new_color
        
        # Update counts
        global_color_counts[old_color] -= 1
        global_color_counts[new_color] += 1
        
        # Update cycle counts and candidate status
        affected_cycles = edge_to_cycles[best_e]
        
        # Remove best_e from candidates? 
        # After assignment, best_e might still be a candidate for future changes?
        # But we are doing constructive addition. We don't plan to change it back.
        # However, other edges' candidacy might change.
        
        # We need to check if any OTHER edge becomes invalid (loses candidacy) 
        # or if best_e itself is no longer "safe" for further changes (though we won't change it).
        # Actually, the condition for an edge e to be assignable a NEW global color is 
        # that all its cycles have distinct_counts <= 2.
        # When we change best_e, the distinct_counts of its cycles might increase from 1 to 2, or 2 to 3.
        # If a cycle's distinct_count becomes 3, then ALL edges in that cycle are NO LONGER candidates 
        # for receiving a NEW global color (because assigning a new color to any edge in a 3-distinct cycle makes it 4-distinct).
        
        # So, for each affected cycle:
        for c_idx in affected_cycles:
            counts[c_idx, old_color] -= 1
            if counts[c_idx, old_color] == 0:
                distinct_counts[c_idx] -= 1
            counts[c_idx, new_color] += 1
            if counts[c_idx, new_color] == 1:
                distinct_counts[c_idx] += 1
                
            if distinct_counts[c_idx] == 3:
                # This cycle is now "saturated" for new global colors.
                # All edges in this cycle must be removed from candidate_edges.
                for e in CYCLES[c_idx]:
                    candidate_edges.discard(e)
                    
        # Also, best_e itself: if any of its cycles is now 3-distinct, it's removed.
        # The loop above handles this if best_e is in the cycle (it is).
        
        # Update score
        if global_color_counts[new_color] == 1:
            # It was already 1? No, we popped it, so it was 0.
            # Wait, global_color_counts[new_color] was 0. Now it is 1.
            pass # Score increase is implicit in the number of used colors.
            
        # Track best
        current_score = sum(1 for c in global_color_counts if c > 0)
        if current_score > best_score:
            best_score = current_score
            best_colors = current_colors[:]
            
    return best_colors
