import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119

def _edge_index(i, j):
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def _build_cycles():
    cycles = []
    for a in range(N - 3):
        for b in range(a + 1, N - 2):
            for c in range(b + 1, N - 1):
                for d in range(c + 1, N):
                    # Cycle 1: a-b-c-d-a
                    cycles.append((_edge_index(a,b), _edge_index(b,c), _edge_index(c,d), _edge_index(a,d)))
                    # Cycle 2: a-b-d-c-a
                    cycles.append((_edge_index(a,b), _edge_index(b,d), _edge_index(c,d), _edge_index(a,c)))
                    # Cycle 3: a-c-b-d-a
                    cycles.append((_edge_index(a,c), _edge_index(b,c), _edge_index(b,d), _edge_index(a,d)))
    return cycles

CYCLES = _build_cycles()
NUM_CYCLES = len(CYCLES)

def _build_edge_to_cycles():
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cyc in enumerate(CYCLES):
        for e in cyc:
            edge_to_cycles[e].append(ci)
    return edge_to_cycles

EDGE_TO_CYCLES = _build_edge_to_cycles()

def _count_violations(colors):
    violations = 0
    for cyc in CYCLES:
        c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            violations += 1
    return violations

def _count_distinct_colors(colors):
    return len(set(colors))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Initial state: "Lowest Set Bit" coloring on GF(2)^4
    # This guarantees 0 violations and 4 colors.
    # Vertices 0..15 are 4-bit vectors.
    # c(u, v) = index of least significant bit where u and v differ.
    init_colors = np.zeros(NUM_EDGES, dtype=np.int32)
    for i in range(N):
        for j in range(i + 1, N):
            diff = i ^ j
            # Find index of lowest set bit
            # diff is non-zero
            lsb_index = 0
            while (diff & 1) == 0:
                diff >>= 1
                lsb_index += 1
            init_colors[_edge_index(i, j)] = lsb_index
            
    colors = init_colors.copy()
    best_colors = colors.copy()
    best_score = _count_distinct_colors(colors) - 1000 * _count_violations(colors)
    current_violations = _count_violations(colors)
    current_distinct = _count_distinct_colors(colors)
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.999
    
    random.seed(42)
    
    # Precompute distinct color set for fast update?
    # Updating distinct count is O(1) if we track counts, but set(colors) is O(E).
    # E=120, so O(120) is fine.
    
    while time.time() < deadline:
        # Pick a random edge
        e = random.randint(0, NUM_EDGES - 1)
        old_color = colors[e]
        
        # Pick a new color
        # Bias towards new colors to increase diversity?
        # Or random?
        # Let's try random first, but if we are valid, bias towards unused colors.
        if current_violations == 0:
            # Try to introduce a new color
            if random.random() < 0.5:
                # Pick a color not in current set
                used = set(colors.tolist())
                unused = [c for c in range(MAX_COLOR + 1) if c not in used]
                if unused:
                    new_color = random.choice(unused)
                else:
                    new_color = random.randint(0, MAX_COLOR)
            else:
                new_color = random.randint(0, MAX_COLOR)
        else:
            new_color = random.randint(0, MAX_COLOR)
            
        if new_color == old_color:
            continue
            
        # Calculate delta violations
        # Only cycles containing e are affected
        delta_violations = 0
        for ci in EDGE_TO_CYCLES[e]:
            cyc = CYCLES[ci]
            # Get current colors
            c0 = colors[cyc[0]]
            c1 = colors[cyc[1]]
            c2 = colors[cyc[2]]
            c3 = colors[cyc[3]]
            
            # Check if current is violation
            # 4 distinct colors
            is_viol_before = (c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3)
            
            # Calculate new colors
            # Replace old_color with new_color in the cycle
            # Note: cyc contains e.
            # We need to identify which position in cyc is e.
            # Instead of searching, we can just check the set of colors.
            # New set of colors in cycle:
            # {c0, c1, c2, c3} with one instance of old_color replaced by new_color.
            
            # Efficient check:
            # If old_color == new_color, no change (handled above).
            # Let S be the set of colors {c0, c1, c2, c3}.
            # If old_color is in S, remove one instance? No, sets don't have multiplicity.
            # We need to check if the *multiset* has 4 distinct elements.
            
            # Let's just construct the new list of 4 colors.
            new_c0 = new_color if cyc[0] == e else c0
            new_c1 = new_color if cyc[1] == e else c1
            new_c2 = new_color if cyc[2] == e else c2
            new_c3 = new_color if cyc[3] == e else c3
            
            is_viol_after = (new_c0 != new_c1 and new_c0 != new_c2 and new_c0 != new_c3 and new_c1 != new_c2 and new_c1 != new_c3 and new_c2 != new_c3)
            
            if is_viol_before and not is_viol_after:
                delta_violations -= 1
            elif not is_viol_before and is_viol_after:
                delta_violations += 1
        
        # Calculate delta distinct colors
        # This is tricky to do incrementally without a counter map.
        # For now, calculate if we accept.
        
        # Objective: Maximize Distinct, Minimize Violations.
        # Score = Distinct - 1000 * Violations.
        # Delta Score = Delta Distinct - 1000 * Delta Violations.
        
        # We don't know Delta Distinct yet.
        # Approximation: 
        # If we change an edge from old to new:
        # - If old_color is unique to this edge, distinct decreases by 1 (if new is already used) or stays same (if new is unused).
        # - If new_color is new, distinct increases by 1.
        
        # Let's just calculate full score if we accept? No, too slow.
        # Let's use a heuristic acceptance:
        # 1. If delta_violations < 0 (fewer violations), accept.
        # 2. If delta_violations == 0, accept if we are likely increasing diversity.
        #    - If new_color not in current set, accept.
        #    - Else, accept with prob 0.5.
        # 3. If delta_violations > 0, accept with prob exp(-1000 * delta_violations / T).
        
        accept = False
        if delta_violations < 0:
            accept = True
        elif delta_violations == 0:
            # Check if new color is new
            if new_color not in set(colors.tolist()):
                accept = True
            else:
                # Neutral move, accept with some probability to explore
                if random.random() < 0.3:
                    accept = True
        else:
            # Penalty for violations is high
            prob = np.exp(-1000.0 * delta_violations / T)
            if random.random() < prob:
                accept = True
                
        if accept:
            colors[e] = new_color
            current_violations += delta_violations
            # Update distinct count (expensive but necessary for tracking)
            # Optimization: Only recalc if we suspect change?
            # For now, recalc every 100 steps or if accepted?
            # Recalcing set(colors) is O(120).
            current_distinct = _count_distinct_colors(colors)
            
            score = current_distinct - 1000 * current_violations
            if score > best_score:
                best_score = score
                best_colors = colors.copy()
                
        T *= cooling
        if T < 0.01:
            T = 0.01

    # Final check: if best_colors has violations, try to repair or fallback
    # Since we tracked best_score which penalizes violations, best_colors should be valid or near-valid.
    # If best_colors has violations, it's invalid for the task.
    # The task requires NO violations.
    # So we must return a valid coloring.
    
    # If best_colors is invalid, we need a valid fallback.
    # The initial state was valid.
    # We can keep a separate "best_valid" tracker.
    
    # Let's re-run the loop logic with a strict "best_valid" tracker.
    
    return best_colors.tolist()

# Refinement: The above code might return an invalid coloring if the best score was achieved with violations (unlikely due to high penalty, but possible if distinct count is huge).
# To be safe, we should track the best VALID coloring separately.

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Initial state: "Lowest Set Bit" coloring
    init_colors = np.zeros(NUM_EDGES, dtype=np.int32)
    for i in range(N):
        for j in range(i + 1, N):
            diff = i ^ j
            lsb_index = 0
            while (diff & 1) == 0:
                diff >>= 1
                lsb_index += 1
            init_colors[_edge_index(i, j)] = lsb_index
            
    colors = init_colors.copy()
    best_valid_colors = colors.copy()
    best_valid_distinct = _count_distinct_colors(colors)
    
    current_violations = _count_violations(colors)
    current_distinct = _count_distinct_colors(colors)
    
    T = 1.0
    cooling = 0.999
    
    random.seed(42)
    
    while time.time() < deadline:
        e = random.randint(0, NUM_EDGES - 1)
        old_color = colors[e]
        
        # Bias towards new colors if valid
        if current_violations == 0:
            if random.random() < 0.7:
                used = set(colors.tolist())
                unused = [c for c in range(MAX_COLOR + 1) if c not in used]
                if unused:
                    new_color = random.choice(unused)
                else:
                    new_color = random.randint(0, MAX_COLOR)
            else:
                new_color = random.randint(0, MAX_COLOR)
        else:
            new_color = random.randint(0, MAX_COLOR)
            
        if new_color == old_color:
            continue
            
        delta_violations = 0
        for ci in EDGE_TO_CYCLES[e]:
            cyc = CYCLES[ci]
            c0 = colors[cyc[0]]
            c1 = colors[cyc[1]]
            c2 = colors[cyc[2]]
            c3 = colors[cyc[3]]
            
            is_viol_before = (c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3)
            
            new_c0 = new_color if cyc[0] == e else c0
            new_c1 = new_color if cyc[1] == e else c1
            new_c2 = new_color if cyc[2] == e else c2
            new_c3 = new_color if cyc[3] == e else c3
            
            is_viol_after = (new_c0 != new_c1 and new_c0 != new_c2 and new_c0 != new_c3 and new_c1 != new_c2 and new_c1 != new_c3 and new_c2 != new_c3)
            
            if is_viol_before and not is_viol_after:
                delta_violations -= 1
            elif not is_viol_before and is_viol_after:
                delta_violations += 1
        
        accept = False
        if delta_violations < 0:
            accept = True
        elif delta_violations == 0:
            # Check if new color is new
            # Optimization: maintain a set of used colors?
            # set(colors.tolist()) is O(E).
            # Let's just accept with prob 0.5 if delta_violations == 0.
            # The diversity gain will be captured in the "best_valid" update.
            if random.random() < 0.5:
                accept = True
        else:
            prob = np.exp(-500.0 * delta_violations / T)
            if random.random() < prob:
                accept = True
                
        if accept:
            colors[e] = new_color
            current_violations += delta_violations
            
            if current_violations == 0:
                current_distinct = _count_distinct_colors(colors)
                if current_distinct > best_valid_distinct:
                    best_valid_distinct = current_distinct
                    best_valid_colors = colors.copy()
            else:
                # If invalid, we don't update best_valid
                pass
                
        T *= cooling
        if T < 0.01:
            T = 0.01

    return best_valid_colors.tolist()
