import numpy as np
import random
import time

N = 16
GRID_POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = len(GRID_POINTS)

# Precompute squared distances
coords = np.array(GRID_POINTS, dtype=np.int32)
diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
dist_sq = np.sum(diffs * diffs, axis=-1)

# Precompute distance sets for each point (to check isosceles conditions quickly)
# dist_sets[i] is a set of squared distances from point i to all points
dist_sets = [set(dist_sq[i, :].tolist()) for i in range(NUM_POINTS)]

def is_valid_addition(p, S_indices):
    """
    Check if point p can be added to set S_indices without forming an isosceles triangle.
    S_indices is a list of indices.
    """
    if len(S_indices) <= 1:
        return True
    
    # Get distances from p to all points in S
    # S_indices is a list, so we can index dist_sq with it
    d_pS = dist_sq[p, S_indices]
    
    # Check if all distances from p to S are unique (Vertex p isosceles check)
    if len(set(d_pS)) != len(d_pS):
        return False
        
    # Check if d(p, a) equals d(a, b) for any b in S (Vertex a isosceles check)
    # And d(p, b) equals d(a, b) for any a in S (Vertex b isosceles check)
    # Both are covered by: for each a in S, is d(p, a) in dist_sets[a]?
    # If d(p, a) == d(a, b), then triangle (p, a, b) is isosceles at a.
    # If d(p, b) == d(a, b), then triangle (p, a, b) is isosceles at b.
    # The loop iterates over all a in S.
    # When a is the vertex, it checks if d(p, a) is a distance from a to any other point.
    # dist_sets[a] contains d(a, x) for all x.
    # So if d(p, a) is in dist_sets[a], it means there exists x such that d(p, a) = d(a, x).
    # If x is in S, we have an isosceles triangle.
    # Note: dist_sets[a] includes d(a, a)=0. d(p, a) is never 0.
    # dist_sets[a] includes d(a, p) if p was in S? No, p is not in S yet.
    # But dist_sets is precomputed for ALL points.
    # So d(p, a) in dist_sets[a] implies d(p, a) = d(a, x) for some x in 0..NUM_POINTS-1.
    # If x is in S, it's a conflict.
    # If x is not in S, it's not a conflict (triangle not formed).
    # Wait, my logic above "is d(p, a) in dist_sets[a]" checks against ALL points, not just S.
    # This is too strict!
    # I need to check if d(p, a) = d(a, b) for b IN S.
    
    # Let's revert to the slightly slower but correct O(|S|^2) or O(|S| * log |S|) check
    # Or optimize the check:
    # For each a in S:
    #   d_pa = dist_sq[p, a]
    #   # Check if d_pa is in { dist_sq[a, b] for b in S }
    #   # This is equivalent to checking if d_pa is in the set of distances from a to S.
    
    # Precomputing "dist_to_S" sets dynamically is expensive.
    # Let's stick to the loop but make it fast.
    
    # Optimization:
    # For each a in S_indices:
    #   d_pa = dist_sq[p, a]
    #   # We need to know if any b in S_indices has dist_sq[a, b] == d_pa
    #   # We can iterate b in S_indices and check dist_sq[a, b]
    
    # This is O(|S|^2). With |S| ~ 30, 900 ops.
    # 256 * 900 = 230,400 ops per M(S) calculation.
    # 1000 steps = 230M ops. Might be too slow in Python.
    
    # Let's try a different vectorization:
    # For a candidate p and set S:
    # 1. d_pS = dist_sq[p, S] (unique check)
    # 2. Matrix D_SS = dist_sq[np.ix_(S, S)]
    # 3. We need to check if for any a in S, d_pS[a] is in D_SS[a, :] (excluding self)
    # 4. And for any b in S, d_pS[b] is in D_SS[:, b] (excluding self) -- same as 3 by symmetry of check?
    #    No, 3 checks vertex a. 4 checks vertex b.
    #    Actually, condition "d(p, a) == d(a, b)" is symmetric to "d(p, b) == d(b, a)" if we swap a and b?
    #    No.
    #    Let's just check:
    #    For each i (index in S):
    #       d_pi = d_pS[i]
    #       row_i = D_SS[i, :]
    #       # Check if d_pi is in row_i (excluding index i)
    #       # And check if d_pi is in col_i? No, that's vertex b.
    #       # Vertex b check: d(p, b) == d(a, b).
    #       # This is: for each j (index in S), is d_pS[j] in D_SS[:, j]?
    #    
    #    We can combine:
    #    Create a boolean matrix conflict = np.zeros_like(D_SS, dtype=bool)
    #    # Vertex p check already done.
    #    # Vertex a check: d(p, a) == d(a, b)
    #    # For each a, d_pS[a] should not appear in D_SS[a, :] (except at a)
    #    # Vectorized: 
    #    # We want to know if any element in D_SS[i, :] equals d_pS[i]
    #    # D_SS[i, :] == d_pS[i]
    #    # This creates a boolean row.
    #    
    #    Let's do this:
    #    matches_vertex_a = (D_SS == d_pS[:, np.newaxis]) # Check if D_SS[i, j] == d_pS[i]
    #    # This checks if d(a, b) == d(p, a). Vertex a.
    #    # We need to exclude j==i (where d(a,a)=0, d(p,a)>0, so no match).
    #    
    #    matches_vertex_b = (D_SS == d_pS[np.newaxis, :]) # Check if D_SS[i, j] == d_pS[j]
    #    # This checks if d(a, b) == d(p, b). Vertex b.
    #    
    #    If any True in matches_vertex_a or matches_vertex_b, invalid.
    
    if len(S_indices) == 0:
        return True
        
    S_arr = np.array(S_indices, dtype=np.int32)
    D_SS = dist_sq[np.ix_(S_arr, S_arr)]
    
    # Vertex p: unique distances
    if len(set(d_pS)) != len(d_pS):
        return False
        
    # Vertex a: d(a, b) == d(p, a)
    # D_SS[i, j] == d_pS[i]
    # Note: D_SS[i, i] = 0. d_pS[i] > 0. So diagonal is safe.
    if np.any(D_SS == d_pS[:, np.newaxis]):
        return False
        
    # Vertex b: d(a, b) == d(p, b)
    # D_SS[i, j] == d_pS[j]
    if np.any(D_SS == d_pS[np.newaxis, :]):
        return False
        
    return True

def calculate_maximality_penalty(S_indices):
    """
    Returns the number of points p not in S that CAN be added to S.
    """
    S_set = set(S_indices)
    count = 0
    for p in range(NUM_POINTS):
        if p in S_set:
            continue
        if is_valid_addition(p, S_indices):
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 280
    
    # 1. Initial Greedy Construction
    # Random shuffle
    order = list(range(NUM_POINTS))
    random.shuffle(order)
    
    S_indices = []
    for p in order:
        if is_valid_addition(p, S_indices):
            S_indices.append(p)
            
    # S_indices is now a maximal set (greedy maximal)
    best_size = len(S_indices)
    best_set = S_indices.copy()
    best_max_penalty = calculate_maximality_penalty(S_indices)
    
    # 2. Simulated Annealing
    T = 10.0
    cooling_rate = 0.999
    
    current_S = S_indices.copy()
    current_penalty = best_max_penalty
    current_size = best_size
    current_cost = current_size + 100 * current_penalty
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Generate Neighbor
        # 50% Remove, 50% Add/Swap
        if random.random() < 0.5 and len(current_S) > 1:
            # Remove
            idx_to_remove = random.randint(0, len(current_S) - 1)
            new_S = current_S[:idx_to_remove] + current_S[idx_to_remove+1:]
        else:
            # Try Add
            p = random.randint(0, NUM_POINTS - 1)
            if p in current_S:
                # Try Swap
                if len(current_S) > 0:
                    idx_to_remove = random.randint(0, len(current_S) - 1)
                    temp_S = current_S[:idx_to_remove] + current_S[idx_to_remove+1:]
                    if is_valid_addition(p, temp_S):
                        new_S = temp_S + [p]
                    else:
                        continue # Invalid move
                else:
                    continue
            else:
                if is_valid_addition(p, current_S):
                    new_S = current_S + [p]
                else:
                    continue
        
        if not new_S:
            continue
            
        new_size = len(new_S)
        # Only calculate penalty if size is promising or to track cost
        # Penalty calculation is expensive.
        # Optimization: Only calculate penalty if new_size is close to best or if we want to accept a non-maximal state.
        # In SA, we accept non-maximal states (high penalty) to escape local minima.
        new_penalty = calculate_maximality_penalty(new_S)
        new_cost = new_size + 100 * new_penalty
        
        delta_cost = new_cost - current_cost
        
        accept = False
        if delta_cost < 0:
            accept = True
        else:
            # Accept with probability exp(-delta / T)
            if T > 0:
                prob = np.exp(-delta_cost / T)
                if random.random() < prob:
                    accept = True
        
        if accept:
            current_S = new_S
            current_size = new_size
            current_penalty = new_penalty
            current_cost = new_cost
            
            if current_penalty == 0:
                if current_size < best_size:
                    best_size = current_size
                    best_set = current_S.copy()
                    best_max_penalty = 0
            else:
                # Update best non-maximal? No, we only care about maximal.
                # But we can track best overall cost to guide search?
                # No, just track best maximal.
                pass
            
            if best_size <= 10:
                break
                
        T *= cooling_rate
        
    # 3. Final Check and Repair
    # If best_set is not maximal, make it maximal by adding points greedily
    # This ensures validity, though it might increase size.
    final_set = best_set.copy()
    # Ensure maximal
    changed = True
    while changed and time.time() < deadline - 10:
        changed = False
        # Shuffle candidates to get different maximal extensions
        candidates = [p for p in range(NUM_POINTS) if p not in final_set]
        random.shuffle(candidates)
        for p in candidates:
            if is_valid_addition(p, final_set):
                final_set.append(p)
                changed = True
                break # Restart with new set to ensure proper maximal order?
                # Actually, just add all valid points?
                # Greedy maximal: add any valid point.
                # To minimize size, we should add points that block the most others?
                # For now, just add all valid points to ensure maximality.
        
    # If the repair made it larger than best_size, and best_set was already maximal, 
    # we should keep best_set.
    # But best_set IS maximal (penalty 0).
    # If best_set penalty was not 0, we repaired it.
    
    # Check if final_set is better?
    # If best_set was maximal, final_set == best_set (no changes needed).
    # If best_set was not maximal, final_set is larger.
    
    return [GRID_POINTS[i] for i in final_set]

if __name__ == "__main__":
    res = construct()
    print(f"Found set of size {len(res)}")
