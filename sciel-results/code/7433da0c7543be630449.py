import time
import random
import numpy as np
import sys

# Precompute K4s
# Vertices 0..15
# Edges (i, j) with i < j
edge_idx = {}
edges_list = []
for i in range(16):
    for j in range(i+1, 16):
        edge_idx[(i, j)] = len(edges_list)
        edges_list.append((i, j))

# List of K4s: each is a list of 6 edge indices
k4s = []
for a in range(16):
    for b in range(a+1, 16):
        for c in range(b+1, 16):
            for d in range(c+1, 16):
                # Vertices a, b, c, d
                # Edges: (a,b), (a,c), (a,d), (b,c), (b,d), (c,d)
                e1 = edge_idx[(a, b)]
                e2 = edge_idx[(a, c)]
                e3 = edge_idx[(a, d)]
                e4 = edge_idx[(b, c)]
                e5 = edge_idx[(b, d)]
                e6 = edge_idx[(c, d)]
                k4s.append([e1, e2, e3, e4, e5, e6])

# Map edge index to list of K4 indices containing it
edge_to_k4 = [[] for _ in range(120)]
for k_idx, k4 in enumerate(k4s):
    for e in k4:
        edge_to_k4[e].append(k_idx)

def get_colours(k4, edges):
    # k4 is list of 6 edge indices
    # edges is array of colours
    return set(edges[e] for e in k4)

def count_violations(edges):
    # Returns list of bools and count
    violations = [False] * len(k4s)
    count = 0
    for k_idx, k4 in enumerate(k4s):
        if len(get_colours(k4, edges)) < 5:
            violations[k_idx] = True
            count += 1
    return violations, count

def solve(K, time_limit, initial_state=None):
    start_time = time.time()
    
    if initial_state is not None:
        edges = np.array(initial_state, dtype=int)
    else:
        # Structured Initialization: Use Z_16 difference
        # Color of (i,j) is (j-i) % 16, mapped to 0..K-1 if K<16?
        # Actually, let's just use (j-i)%16 and then reduce mod K if needed, 
        # or better, use a random permutation of Z_16 mapped to 0..K-1?
        # For K=13, we can't map 16 values to 13 uniquely without collisions.
        # Let's use (j-i) % K directly for initial state if K <= 16.
        # This creates a circulant-like structure.
        edges = np.array([(j - i) % K for i, j in edges_list], dtype=int)
        
    violations, violation_count = count_violations(edges)
    
    if violation_count == 0:
        return edges, True
        
    # Simulated Annealing
    # Objective: Minimize violation_count. If 0, minimize distinct colors used.
    # But the task asks to minimize colors used subject to validity.
    # So we first need a valid coloring.
    # SA will try to reach violation_count = 0.
    
    current_temp = 10.0
    cooling_rate = 0.9999
    
    best_edges = edges.copy()
    best_violations = violation_count
    
    # Track colors used for the "minimize colors" part once valid
    # But first priority is validity.
    
    while time.time() - start_time < time_limit:
        if violation_count == 0:
            # Try to reduce colors? 
            # For now, just return if we found a valid one quickly.
            # The outer loop will try smaller K.
            return edges, True

        # Pick a random violating K4
        # To speed up, pick random K4, check if violating.
        # If most are valid, this is slow. But we start with many violations.
        k_idx = random.randint(0, len(k4s) - 1)
        
        if not violations[k_idx]:
            # If we picked a valid one, maybe try to perturb a random edge?
            # Or just skip.
            # To ensure exploration, let's pick a random edge from a random K4
            # even if it's valid, to allow "bad" moves that might lead to better states.
            # But standard SA usually focuses on the objective function.
            # Let's stick to fixing violations primarily.
            # If we can't find a violation quickly, it means we are close to valid.
            # Let's pick a random edge from the whole graph.
            e_idx = random.randint(0, 119)
        else:
            k4 = k4s[k_idx]
            e_idx = random.choice(k4)
        
        old_colour = edges[e_idx]
        new_colour = random.randint(0, K-1)
        if new_colour == old_colour:
            continue
            
        edges[e_idx] = new_colour
        
        # Calculate delta in violations
        # We need to check all K4s containing e_idx
        delta_viol = 0
        affected_k4s = edge_to_k4[e_idx]
        
        # Save old states to revert if needed? 
        # No, just calculate new count and compare.
        # But we need to update the violations array.
        
        # For simplicity and speed in Python, we'll just recalculate the affected K4s
        # and update the global count.
        
        for aff_k_idx in affected_k4s:
            old_status = violations[aff_k_idx]
            new_status = len(get_colours(k4s[aff_k_idx], edges)) < 5
            if old_status != new_status:
                violations[aff_k_idx] = new_status
                if new_status:
                    delta_viol += 1
                else:
                    delta_viol -= 1
        
        new_violation_count = violation_count + delta_viol
        
        # Acceptance criterion
        if new_violation_count < violation_count:
            violation_count = new_violation_count
            if violation_count < best_violations:
                best_violations = violation_count
                best_edges = edges.copy()
        else:
            # Accept with probability exp(-delta / T)
            # delta is positive (worse)
            if random.random() < np.exp(-delta_viol / current_temp):
                violation_count = new_violation_count
            else:
                # Revert
                edges[e_idx] = old_colour
                # Revert violations
                for aff_k_idx in affected_k4s:
                    # We need to recalculate or just store old status?
                    # Since we only changed one edge, we can recalc if we stored old colors?
                    # Easier: Just recalc the status for affected K4s based on reverted edge?
                    # Actually, we can just recalc the set of colours for these K4s.
                    # But wait, if we reverted, the status is back to what it was.
                    # We didn't store the old status in a list, we just modified violations[].
                    # So we need to restore violations[] for affected K4s.
                    # Let's just recalc them. It's only ~10-20 K4s per edge.
                    pass # We'll handle this by recalculating if we revert? 
                    # Actually, the code above modifies violations[] in place.
                    # If we revert edges, we MUST revert violations[].
                    # Let's refactor: Don't update violations[] until we accept.
                    
    # Refactor for correctness:
    # We should calculate delta without updating the state, then update if accepted.
    
    return best_edges, best_violations == 0

# Corrected Solve Function
def solve_sa(K, time_limit, initial_state=None):
    start_time = time.time()
    
    if initial_state is not None:
        edges = np.array(initial_state, dtype=int)
    else:
        # Structured Initialization: (j-i) % K
        edges = np.array([(j - i) % K for i, j in edges_list], dtype=int)
        
    violations, violation_count = count_violations(edges)
    
    if violation_count == 0:
        return edges, True
        
    current_temp = 50.0
    cooling_rate = 0.99995
    
    best_edges = edges.copy()
    best_violations = violation_count
    
    while time.time() - start_time < time_limit:
        if violation_count == 0:
            return edges, True

        # Pick a random edge
        # Bias towards edges in violating K4s?
        # Let's just pick a random edge for simplicity, SA explores broadly.
        e_idx = random.randint(0, 119)
        
        old_colour = edges[e_idx]
        new_colour = random.randint(0, K-1)
        if new_colour == old_colour:
            continue
            
        edges[e_idx] = new_colour
        
        # Calculate delta
        delta_viol = 0
        affected_k4s = edge_to_k4[e_idx]
        
        # Store old statuses to revert if needed
        old_statuses = []
        for aff_k_idx in affected_k4s:
            old_statuses.append(violations[aff_k_idx])
            new_status = len(get_colours(k4s[aff_k_idx], edges)) < 5
            if old_statuses[-1] != new_status:
                if new_status:
                    delta_viol += 1
                else:
                    delta_viol -= 1
        
        new_violation_count = violation_count + delta_viol
        
        # Acceptance
        if new_violation_count < violation_count or random.random() < np.exp(-delta_viol / current_temp):
            # Accept
            violation_count = new_violation_count
            # Update violations array
            for i, aff_k_idx in enumerate(affected_k4s):
                violations[aff_k_idx] = len(get_colours(k4s[aff_k_idx], edges)) < 5
                
            if violation_count < best_violations:
                best_violations = violation_count
                best_edges = edges.copy()
                
            if violation_count == 0:
                return edges, True
        else:
            # Revert
            edges[e_idx] = old_colour
            # Revert violations
            for i, aff_k_idx in enumerate(affected_k4s):
                violations[aff_k_idx] = old_statuses[i]
        
        current_temp *= cooling_rate
        if current_temp < 0.1:
            current_temp = 0.1
            
    return best_edges, best_violations == 0

def construct():
    t_start = time.time()
    deadline = 140.0 # Safety margin
    
    # Try K=13 first
    if time.time() - t_start < deadline - 20:
        res13, valid13 = solve_sa(13, min(10.0, deadline - (time.time() - t_start) - 20))
        if valid13:
            return res13.tolist()
    
    # Try K=14
    if time.time() - t_start < deadline - 10:
        # Use best from 13 as initial if available, else structured
        init = res13 if 'res13' in locals() else None
        res14, valid14 = solve_sa(14, min(20.0, deadline - (time.time() - t_start) - 10), initial_state=init)
        if valid14:
            return res14.tolist()
        
        # Store best 14
        best_14 = res14
    
    # Try K=15
    if time.time() - t_start < deadline - 2:
        init = best_14 if 'best_14' in locals() else (res13 if 'res13' in locals() else None)
        res15, valid15 = solve_sa(15, min(30.0, deadline - (time.time() - t_start) - 2), initial_state=init)
        if valid15:
            return res15.tolist()

    # Fallback: Return a known valid coloring if we found any valid one in previous steps
    # If we found a valid 13, we returned.
    # If we found a valid 14, we returned.
    # If we found a valid 15, we returned.
    # If none valid, we must return something. The task says "Minimize the number of colours used. A valid colouring certifies..."
    # If we don't find a valid one, we should probably return the one with the fewest violations?
    # But the interface expects 120 integers. If invalid, score is 0.
    # So we should try our best to find a valid one.
    
    # If we still haven't found one, let's do a quick random search for K=15
    # Or just return the last best_edges from K=15?
    # It might be invalid.
    
    # Let's assume the SA found something close.
    # If we have a valid 15, we returned.
    # If not, we are in trouble.
    # Let's try one more time with a longer run for K=15 if time permits.
    
    remaining = deadline - (time.time() - t_start)
    if remaining > 5:
         res15_retry, valid15_retry = solve_sa(15, remaining, initial_state=best_14 if 'best_14' in locals() else None)
         if valid15_retry:
             return res15_retry.tolist()
    
    # If all else fails, return a structured coloring (might be invalid)
    # Or a known valid one if we hardcoded one? No, we don't have one.
    # The seed program's fallback was (i+j)%17. Let's use that.
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

# Ensure global variables are initialized before construct is called
# They are defined at module level in the draft, so they are ready.

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Result length: {len(res)}")
    print(f"First few: {res[:10]}")
