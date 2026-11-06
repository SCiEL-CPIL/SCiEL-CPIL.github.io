import time
import random
import math
import numpy as np

def construct():
    d = 7
    n = 1 << d  # 128 vertices
    deadline = time.time() + 115  # Safety margin

    # 1. Precompute edges
    all_edges = []
    edge_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(all_edges)
                all_edges.append((x, y))
                edge_idx[(x, y)] = idx
    m = len(all_edges)  # 448

    # 2. Precompute faces
    face_edges = []
    edge_faces = [[] for _ in range(m)]
    
    for i in range(d):
        for j in range(i + 1, d):
            for x in range(n):
                if (x >> i) & 1 == 0 and (x >> j) & 1 == 0:
                    a = x
                    b = x ^ (1 << i)
                    c = x ^ (1 << j)
                    d2 = x ^ (1 << i) ^ (1 << j)
                    
                    e1 = edge_idx[(min(a, b), max(a, b))]
                    e2 = edge_idx[(min(b, d2), max(b, d2))]
                    e3 = edge_idx[(min(d2, c), max(d2, c))]
                    e4 = edge_idx[(min(c, a), max(c, a))]
                    
                    fid = len(face_edges)
                    face_edges.append([e1, e2, e3, e4])
                    for ei in (e1, e2, e3, e4):
                        edge_faces[ei].append(fid)
                        
    n_faces = len(face_edges)  # 672

    # Precompute face lists as tuples for fast set operations
    edge_faces_set = [set(f) for f in edge_faces]
    
    # Convert to numpy arrays for vectorized operations where possible
    # But let's stick to lists for now for clarity, and optimize the inner loop.

    # 3. Greedy Hitting Set Initialization
    def greedy_init():
        R = [False] * m
        face_count = [0] * n_faces
        uncovered = [True] * n_faces
        num_uncovered = n_faces
        
        # Track for each edge, how many uncovered faces it hits
        # We can recompute gain efficiently
        # But for the initial greedy, simple O(m * avg_faces) per step is fine since m=448, faces=672
        
        while num_uncovered > 0:
            best_e = -1
            best_gain = -1
            for e in range(m):
                if R[e]:
                    continue
                gain = 0
                for f in edge_faces[e]:
                    if uncovered[f]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_e = e
            
            if best_e == -1:
                break
                
            R[best_e] = True
            for f in edge_faces[best_e]:
                if uncovered[f]:
                    uncovered[f] = False
                    num_uncovered -= 1
                face_count[f] += 1
                
        return R, face_count

    R, face_count = greedy_init()
    current_size = sum(R)
    best_size = current_size
    best_R = R[:]
    
    # Objective: Minimize size(R) subject to all faces covered (uncovered == 0)
    # Soft constraint: cost = size + lambda * uncovered
    # We want to drive uncovered to 0.
    
    LAMBDA = 5.0
    current_uncovered = sum(1 for c in face_count if c == 0)
    
    def calc_cost(size, unc):
        return size + LAMBDA * unc

    current_cost = calc_cost(current_size, current_uncovered)

    # 4. Simulated Annealing
    T = 1.5
    T_min = 0.001
    cooling = 0.9995  # Slower cooling
    
    # Precompute for each edge, the faces it touches.
    # For delta calculation:
    # If we remove e:
    #   For each face f in e:
    #     if face_count[f] == 1: uncovered increases by 1
    # If we add e:
    #   For each face f in e:
    #     if face_count[f] == 0: uncovered decreases by 1
    
    # For swaps (remove e1, add e2):
    #   Need to handle interaction on shared faces.
    
    # We will use a mix of moves:
    # 1. Single flip (add/remove)
    # 2. Swap (remove one, add another)
    # 3. 2-opt (remove two, add two) - maybe later
    
    start_time = time.time()
    
    while time.time() < deadline:
        if T < T_min:
            T = T_min
            
        # Choose move type
        r = random.random()
        
        if r < 0.4:
            # Single flip
            e = random.randint(0, m - 1)
            if R[e]:
                # Remove e
                delta_unc = 0
                for f in edge_faces[e]:
                    if face_count[f] == 1:
                        delta_unc += 1
                new_size = current_size - 1
                new_unc = current_uncovered + delta_unc
                new_cost = calc_cost(new_size, new_unc)
                delta_cost = new_cost - current_cost
                
                # Apply
                R[e] = False
                for f in edge_faces[e]:
                    face_count[f] -= 1
                current_size = new_size
                current_uncovered = new_unc
            else:
                # Add e
                delta_unc = 0
                for f in edge_faces[e]:
                    if face_count[f] == 0:
                        delta_unc -= 1  # Uncovered decreases
                new_size = current_size + 1
                new_unc = current_uncovered + delta_unc
                new_cost = calc_cost(new_size, new_unc)
                delta_cost = new_cost - current_cost
                
                # Apply
                R[e] = True
                for f in edge_faces[e]:
                    face_count[f] += 1
                current_size = new_size
                current_uncovered = new_unc
            
            # Acceptance
            accept = False
            if delta_cost <= 0:
                accept = True
            else:
                if random.random() < math.exp(-delta_cost / T):
                    accept = True
            
            if not accept:
                # Revert
                if R[e]:
                    # We added, so remove
                    R[e] = False
                    for f in edge_faces[e]:
                        face_count[f] -= 1
                    current_size -= 1
                    # Revert uncovered
                    # We need to know how many faces went from 0 to 1.
                    # This is tricky if we don't store it.
                    # Let's just recalculate delta_unc by scanning again?
                    # Or better: store the change.
                    # Actually, for add: delta_unc was negative.
                    # Let's just recompute current_uncovered from face_count?
                    # That's O(n_faces) = 672. Might be too slow for every rejection.
                    # Alternative: store the list of affected faces and their old counts?
                    # Or: accept only if delta_cost <= 0 or with probability.
                    # If rejected, revert.
                    # To revert add:
                    #   For each f in e:
                    #      if face_count[f] was 0 (before add), it is now 1. So uncovered decreases by 1 for each.
                    #      We don't know which were 0.
                    # Let's precompute: when adding, store the number of faces that were 0.
                    # Let's modify the apply step to return delta_unc.
                    
                    # Wait, I already calculated delta_unc.
                    # For add: delta_unc is negative.
                    # Revert: current_uncovered -= delta_unc (which adds back)
                    current_uncovered -= delta_unc
                else:
                    # We removed, so add
                    R[e] = True
                    for f in edge_faces[e]:
                        face_count[f] += 1
                    current_size += 1
                    current_uncovered -= delta_unc
            else:
                # Accepted
                pass # State is already updated

        elif r < 0.8:
            # Swap: remove e1 (in R), add e2 (not in R)
            e1 = random.randint(0, m - 1)
            e2 = random.randint(0, m - 1)
            if e1 == e2:
                continue
            if not R[e1] or R[e2]:
                # Not a valid swap (must remove one in R, add one not in R)
                # We can swap states: R[e1] -> False, R[e2] -> True
                # But only if R[e1] is True and R[e2] is False.
                # If both True or both False, size doesn't change, but we can still try?
                # Let's stick to size-preserving swaps for now to keep size stable.
                if R[e1] == R[e2]:
                    continue
                else:
                    # Flip the one that is in R to False, and the other to True?
                    # No, that changes size.
                    # We want to keep size constant.
                    # So we must pick e1 in R and e2 not in R.
                    # If R[e1] and not R[e2], swap.
                    # If not R[e1] and R[e2], swap (remove e2, add e1).
                    if R[e1] and not R[e2]:
                        rem, add = e1, e2
                    else:
                        rem, add = e2, e1
                    
                    # Calculate delta
                    # Faces involved: union of faces of rem and add
                    # For each face f:
                    #   old_c = face_count[f]
                    #   new_c = old_c - (1 if f in rem else 0) + (1 if f in add else 0)
                    #   if old_c > 0 and new_c == 0: inc_unc += 1
                    #   if old_c == 0 and new_c > 0: dec_unc += 1
                    
                    delta_unc = 0
                    # We need to iterate over all faces in rem or add.
                    # Using sets for efficiency
                    faces_rem = edge_faces_set[rem]
                    faces_add = edge_faces_set[add]
                    
                    # To avoid double counting, we can iterate over the union
                    # But we need to know the change for each face.
                    # Let's just iterate over faces_rem and faces_add, and handle overlaps carefully.
                    # Or simpler: iterate over all faces in the union.
                    
                    # Create a set of unique faces
                    all_affected = faces_rem | faces_add
                    
                    for f in all_affected:
                        old_c = face_count[f]
                        new_c = old_c
                        if f in faces_rem:
                            new_c -= 1
                        if f in faces_add:
                            new_c += 1
                        if old_c == 0 and new_c > 0:
                            delta_unc -= 1  # Uncovered decreases
                        elif old_c > 0 and new_c == 0:
                            delta_unc += 1  # Uncovered increases
                    
                    new_size = current_size  # Size unchanged
                    new_unc = current_uncovered + delta_unc
                    new_cost = calc_cost(new_size, new_unc)
                    delta_cost = new_cost - current_cost
                    
                    # Apply
                    R[rem] = False
                    R[add] = True
                    for f in faces_rem:
                        face_count[f] -= 1
                    for f in faces_add:
                        face_count[f] += 1
                    current_uncovered = new_unc
                    
                    # Acceptance
                    accept = False
                    if delta_cost <= 0:
                        accept = True
                    else:
                        if random.random() < math.exp(-delta_cost / T):
                            accept = True
                    
                    if not accept:
                        # Revert
                        R[rem] = True
                        R[add] = False
                        for f in faces_rem:
                            face_count[f] += 1
                        for f in faces_add:
                            face_count[f] -= 1
                        current_uncovered -= delta_unc

        # Update best
        if current_uncovered == 0 and current_size < best_size:
            best_size = current_size
            best_R = R[:]
            
        T *= cooling
        # Occasionally restart T if stuck?
        # Or use adaptive cooling?

    # 5. Final Greedy Polish
    # Try to remove edges greedily if they don't uncover any face.
    # This is a local search step:
    # For each edge e in best_R (in random order), try to remove it.
    # If removing it doesn't make any face uncovered (i.e., all faces still have count >= 1),
    # then remove it.
    
    # First, ensure best_R is valid.
    fc_best = [0] * n_faces
    for e in range(m):
        if best_R[e]:
            for f in edge_faces[e]:
                fc_best[f] += 1
    
    # If not valid, fix it.
    # But our SA should maintain validity if we start valid and only accept valid moves?
    # No, we use a soft constraint, so we might be invalid.
    # We must ensure the final result is valid.
    
    is_valid = all(c > 0 for c in fc_best)
    
    if not is_valid:
        # Fall back to greedy init which is valid
        R_g, fc_g = greedy_init()
        best_R = R_g
        best_size = sum(R_g)
        # And do the polish on this valid set
        fc_best = fc_g[:]
    else:
        # Use the best_R found by SA
        pass

    # Polish: Remove unnecessary edges
    # We want to minimize |R|.
    # Iterate until no improvement.
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle edges in R
        edges_in_R = [e for e in range(m) if best_R[e]]
        random.shuffle(edges_in_R)
        for e in edges_in_R:
            # Try to remove e
            # Check if any face becomes uncovered
            can_remove = True
            for f in edge_faces[e]:
                if fc_best[f] == 1:
                    can_remove = False
                    break
            if can_remove:
                best_R[e] = False
                best_size -= 1
                for f in edge_faces[e]:
                    fc_best[f] -= 1
                improved = True

    # 6. Construct Output
    out = []
    for e in range(m):
        if not best_R[e]:
            x, y = all_edges[e]
            out.append([x, y])
            
    return out

if __name__ == "__main__":
    t0 = time.time()
    edges = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s, Edges: {len(edges)}")
    
    # Validate
    d = 7
    n = 1 << d
    all_edges_set = set()
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges_set.add((x, y))
                
    selected_set = set()
    for u, v in edges:
        selected_set.add((min(u, v), max(u, v)))
        
    # Check no face has 4 edges
    valid = True
    for i in range(d):
        for j in range(i + 1, d):
            for x in range(n):
                if (x >> i) & 1 == 0 and (x >> j) & 1 == 0:
                    a = x
                    b = x ^ (1 << i)
                    c = x ^ (1 << j)
                    d2 = x ^ (1 << i) ^ (1 << j)
                    
                    e1 = (min(a, b), max(a, b))
                    e2 = (min(b, d2), max(b, d2))
                    e3 = (min(d2, c), max(d2, c))
                    e4 = (min(c, a), max(c, a))
                    
                    cnt = 0
                    if e1 in selected_set: cnt += 1
                    if e2 in selected_set: cnt += 1
                    if e3 in selected_set: cnt += 1
                    if e4 in selected_set: cnt += 1
                    
                    if cnt == 4:
                        print(f"Invalid face found at {x}, dims {i},{j}")
                        valid = False
                        break
            if not valid: break
        if not valid: break
    if valid:
        print("Valid!")
