import numpy as np
import time
import random
import sys
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit
    
    d = 7
    n = 1 << d
    num_edges = d * (1 << (d - 1))  # 448
    
    # 1. Precompute Edge-Index Mapping
    edge_to_idx = {}
    idx_to_edge = []
    idx = 0
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_idx[(u, v)] = idx
                idx_to_edge.append((u, v))
                idx += 1
                
    assert idx == num_edges
    
    # 2. Precompute Face Constraints
    # For each face, store the 4 edge indices.
    # For each edge, store the list of faces it belongs to.
    faces = []
    edge_to_faces = [[] for _ in range(num_edges)]
    
    for i in range(d):
        for j in range(i + 1, d):
            for u in range(n):
                if (u & (1 << i)) == 0 and (u & (1 << j)) == 0:
                    v1 = u
                    v2 = u ^ (1 << i)
                    v3 = u ^ (1 << j)
                    v4 = u ^ (1 << i) ^ (1 << j)
                    
                    e1 = (min(v1, v2), max(v1, v2))
                    e2 = (min(v1, v3), max(v1, v3))
                    e3 = (min(v2, v4), max(v2, v4))
                    e4 = (min(v3, v4), max(v3, v4))
                    
                    ie1 = edge_to_idx[e1]
                    ie2 = edge_to_idx[e2]
                    ie3 = edge_to_idx[e3]
                    ie4 = edge_to_idx[e4]
                    
                    face_indices = (ie1, ie2, ie3, ie4)
                    faces.append(face_indices)
                    
                    for ie in face_indices:
                        edge_to_faces[ie].append(len(faces) - 1)
                        
    num_faces = len(faces)
    
    # Helper: Check if adding edge e is valid
    def is_valid_add(selected, e):
        for f_idx in edge_to_faces[e]:
            f_edges = faces[f_idx]
            count = 0
            for oe in f_edges:
                if oe != e and selected[oe]:
                    count += 1
            if count == 3:
                return False
        return True

    # Helper: Greedy fill - add all possible valid edges
    def greedy_fill(selected):
        changed = True
        while changed:
            changed = False
            for e in range(num_edges):
                if not selected[e]:
                    if is_valid_add(selected, e):
                        selected[e] = True
                        changed = True
        return selected

    # New Strategy: Simulated Annealing with a focus on "2-for-1" and "3-for-2" moves
    # The current local search is greedy and gets stuck. SA allows worse moves to escape local optima.
    # We will also use a more aggressive initial construction based on random bit-flip patterns.
    
    def random_initial():
        # Start with all edges, then remove to fix conflicts?
        # Or start with random subset and fill?
        # Let's try: Random subset of ~60% edges, then greedy fill.
        selected = np.zeros(num_edges, dtype=bool)
        for e in range(num_edges):
            if random.random() < 0.6:
                selected[e] = True
        # Fix any immediate violations? 
        # Actually, greedy_fill assumes no violations exist in the 'selected' set.
        # If we start with random, we might have violations.
        # So we must start with a valid set.
        
        # Valid start: Greedy with random order.
        edges = list(range(num_edges))
        random.shuffle(edges)
        for e in edges:
            if is_valid_add(selected, e):
                selected[e] = True
        return selected

    def get_conflict_count(selected):
        # Not needed if we maintain validity, but good for debugging.
        # We maintain validity in SA by only making valid moves.
        return 0

    # SA Move Generation:
    # 1. Single flip: Remove 1 edge, Add 0 (always valid, lowers count). Or Add 1 (if valid).
    # 2. Swap: Remove 1, Add 1.
    # 3. 2-for-1: Remove 1, Add 2.
    # 4. 3-for-2: Remove 2, Add 3.
    
    # To make this efficient, we precompute for each edge the "potential" adds if it were removed.
    # But this is dynamic.
    
    # Let's implement a simple SA loop.
    # State: selected (numpy bool array)
    # Objective: Maximize sum(selected)
    
    # Temperature schedule
    T_start = 1.0
    T_end = 0.01
    alpha = 0.999
    
    def simulated_annealing(initial_selected, max_steps):
        selected = initial_selected.copy()
        current_count = np.sum(selected)
        best_selected = selected.copy()
        best_count = current_count
        
        T = T_start
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Choose a move type
            move_type = random.random()
            
            if move_type < 0.5:
                # Single flip
                # Try to add a random unselected edge
                unselected = np.where(~selected)[0]
                if len(unselected) > 0:
                    e = random.choice(unselected)
                    if is_valid_add(selected, e):
                        selected[e] = True
                        new_count = current_count + 1
                        if new_count > best_count:
                            best_count = new_count
                            best_selected = selected.copy()
                        # Always accept if better
                        current_count = new_count
                    else:
                        # Can't add, try to remove a random selected edge
                        selected_edges = np.where(selected)[0]
                        if len(selected_edges) > 0:
                            e = random.choice(selected_edges)
                            selected[e] = False
                            new_count = current_count - 1
                            # Accept with probability exp(-delta/T)
                            delta = new_count - current_count # -1
                            if random.random() < np.exp(delta / T):
                                current_count = new_count
                            else:
                                selected[e] = True
                else:
                    # All selected? Unlikely.
                    pass
                    
            elif move_type < 0.8:
                # Swap: Remove 1, Add 1
                # This is complex to generate efficiently.
                # Let's stick to 2-for-1 and 1-for-0 for now, as they are more impactful.
                # Actually, let's try 2-for-1.
                pass # Skip for simplicity in first pass, focus on 2-for-1 below
            
            else:
                # 2-for-1 Move: Remove 1 edge, Add 2 edges
                # 1. Pick a random selected edge to remove
                selected_edges = np.where(selected)[0]
                if len(selected_edges) == 0:
                    continue
                e_rem = random.choice(selected_edges)
                selected[e_rem] = False
                count_after_rem = current_count - 1
                
                # 2. Find candidates that are now valid
                candidates = []
                for x in np.where(~selected)[0]:
                    if is_valid_add(selected, x):
                        candidates.append(x)
                
                # 3. Pick 2 from candidates that don't conflict with each other
                # We need to check if adding c1 and c2 together is valid.
                # Since they were both valid individually after removing e_rem,
                # they might conflict with each other in a face where the other 2 edges are selected.
                
                added = 0
                # Try to add 2
                if len(candidates) >= 2:
                    # Shuffle to avoid bias
                    random.shuffle(candidates)
                    # Try pairs? Or just greedy pick 2?
                    # Greedy pick:
                    added_edges = []
                    for c in candidates:
                        if len(added_edges) == 2:
                            break
                        # Check if c is valid given current selected + added_edges
                        # We need a temporary check
                        # is_valid_add checks against 'selected'.
                        # We need to check against 'selected' + 'added_edges'.
                        
                        # Let's just try to add them one by one to 'selected'
                        # and see if we can get 2.
                        if is_valid_add(selected, c):
                            selected[c] = True
                            added_edges.append(c)
                            added += 1
                            # Check if we can add another?
                            # We break out of inner loop if we added 2.
                    
                    # If we added < 2, we might want to revert the ones we added?
                    # No, if we added 1, we are at net 0 change (removed 1, added 1).
                    # If we added 2, we are at net +1.
                    
                new_count = np.sum(selected)
                delta = new_count - current_count
                
                # Acceptance
                if delta > 0:
                    current_count = new_count
                    if new_count > best_count:
                        best_count = new_count
                        best_selected = selected.copy()
                else:
                    # delta <= 0
                    if random.random() < np.exp(delta / T):
                        current_count = new_count
                    else:
                        # Revert
                        # We need to know what we changed.
                        # We removed e_rem.
                        # We added some edges from added_edges.
                        # Revert means: remove added_edges, re-add e_rem.
                        for c in added_edges:
                            selected[c] = False
                        selected[e_rem] = True
                        # current_count stays the same
                    
            T *= alpha
            if T < T_end:
                T = T_end
                
        return best_selected, best_count

    # Run multiple SA instances with different random seeds
    best_overall = 0
    best_overall_selected = None
    
    seed = 0
    while time.time() < deadline - 5:
        random.seed(seed)
        initial = random_initial()
        count = np.sum(initial)
        
        # Run SA for a limited number of steps
        # Estimate steps: 120s / 10 restarts = 12s per restart.
        # Each step is fast (microseconds). Let's allow up to 100,000 steps or time limit.
        sa_selected, sa_count = simulated_annealing(initial, 50000)
        
        if sa_count > best_overall:
            best_overall = sa_count
            best_overall_selected = sa_selected.copy()
        
        seed += 1
        if time.time() > deadline - 10:
            break

    # If SA didn't find anything better than a simple greedy, use greedy
    if best_overall_selected is None or best_overall < 300:
        # Fallback to greedy
        random.seed(0)
        edges = list(range(num_edges))
        random.shuffle(edges)
        selected = np.zeros(num_edges, dtype=bool)
        for e in edges:
            if is_valid_add(selected, e):
                selected[e] = True
        best_overall_selected = selected
        best_overall = np.sum(selected)

    # Final polish: Greedy fill on the best found
    # Note: SA maintains validity, so no fill needed, but maybe we can add more?
    # is_valid_add in SA already tries to add.
    
    # Convert best_overall_selected to list of edges
    result = []
    for i in range(num_edges):
        if best_overall_selected[i]:
            result.append(list(idx_to_edge[i]))
            
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))
