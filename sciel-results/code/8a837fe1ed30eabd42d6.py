import time
import random
import math
import numpy as np

def construct():
    d = 7
    n = 1 << d  # 128 vertices
    deadline = time.time() + 118  # Safety margin (150s kill, 118s budget)

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
    # A face is defined by two dimensions i, j and 5 fixed bits.
    # There are C(7,2) * 2^5 = 21 * 32 = 672 faces.
    face_edges = []
    edge_faces = [[] for _ in range(m)]
    
    # Iterate over all pairs of dimensions
    for i in range(d):
        for j in range(i + 1, d):
            # Iterate over all combinations of the other d-2 bits
            for x in range(n):
                if (x >> i) & 1 == 0 and (x >> j) & 1 == 0:
                    a = x
                    b = x ^ (1 << i)
                    c = x ^ (1 << j)
                    d2 = x ^ (1 << i) ^ (1 << j)
                    
                    # Get edge indices for the 4 edges of the face
                    e1 = edge_idx[(min(a, b), max(a, b))]
                    e2 = edge_idx[(min(b, d2), max(b, d2))]
                    e3 = edge_idx[(min(d2, c), max(d2, c))]
                    e4 = edge_idx[(min(c, a), max(c, a))]
                    
                    fid = len(face_edges)
                    face_edges.append([e1, e2, e3, e4])
                    for ei in (e1, e2, e3, e4):
                        edge_faces[ei].append(fid)
                        
    n_faces = len(face_edges) # 672

    # 3. Greedy Hitting Set Initialization
    # We want to find a set R of edges such that every face has at least one edge in R.
    # Minimize |R|.
    
    def greedy_init():
        R = [False] * m
        face_count = [0] * n_faces
        uncovered = [True] * n_faces
        num_uncovered = n_faces
        
        # Precompute for each edge, the list of faces it covers
        # edge_faces is already this.
        
        while num_uncovered > 0:
            best_e = -1
            best_gain = -1
            # Find edge that covers most uncovered faces
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
                
            # Add best_e to R
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
    
    # Calculate uncovered count for current state
    def get_uncovered_count(fc):
        cnt = 0
        for c in fc:
            if c == 0:
                cnt += 1
        return cnt

    current_uncovered = get_uncovered_count(face_count)
    
    # Objective function for SA
    # Minimize: size(R) + lambda * uncovered
    # We want uncovered to be 0.
    LAMBDA = 10.0
    
    def get_cost(size, unc):
        return size + LAMBDA * unc

    current_cost = get_cost(current_size, current_uncovered)
    
    # 4. Simulated Annealing with 2-Opt moves
    # Start temperature
    T = 2.0
    T_min = 0.001
    cooling = 0.999
    
    # Precompute random edge list? No.
    
    # For faster SA, we can try swapping edges.
    # Move: Remove e1, Add e2.
    # Or just Add/Remove as before, but with better acceptance.
    
    # Let's try a different approach: Local Search on the complement (selected edges)
    # We want to maximize selected edges.
    # Start with selected = all edges not in R_greedy.
    # Actually, the hitting set R is the "removed" edges.
    # We want to minimize |R|.
    
    # Let's stick to optimizing R.
    
    while time.time() < deadline:
        if T < T_min:
            T = T_min
            break
            
        # Pick a random move
        # Type 0: Flip single edge
        # Type 1: Swap two edges (remove one, add another)
        
        move_type = random.random()
        
        if move_type < 0.5:
            # Single flip
            e = random.randint(0, m - 1)
            old_state = R[e]
            
            # Calculate delta
            if R[e]:
                # Remove e
                # Count faces that will become uncovered (count 1 -> 0)
                inc_unc = 0
                for f in edge_faces[e]:
                    if face_count[f] == 1:
                        inc_unc += 1
                new_size = current_size - 1
                new_uncovered = current_uncovered + inc_unc
                new_cost = get_cost(new_size, new_uncovered)
                delta_cost = new_cost - current_cost
                
                # Apply
                R[e] = False
                for f in edge_faces[e]:
                    face_count[f] -= 1
                current_size = new_size
                current_uncovered = new_uncovered
            else:
                # Add e
                # Count faces that will become covered (count 0 -> 1)
                dec_unc = 0
                for f in edge_faces[e]:
                    if face_count[f] == 0:
                        dec_unc += 1
                new_size = current_size + 1
                new_uncovered = current_uncovered - dec_unc
                new_cost = get_cost(new_size, new_uncovered)
                delta_cost = new_cost - current_cost
                
                # Apply
                R[e] = True
                for f in edge_faces[e]:
                    face_count[f] += 1
                current_size = new_size
                current_uncovered = new_uncovered
            
            # Acceptance
            accept = False
            if delta_cost < 0:
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
                    # Revert uncovered: we added dec_unc faces. So current_uncovered += dec_unc
                    # But we don't have dec_unc stored easily.
                    # Recalculate? No.
                    # Let's store the delta in a variable.
                    # Actually, we can just recompute current_uncovered from face_count?
                    # That's O(n_faces) = 672. Acceptable?
                    # 118s / 672 ops ~ 175k ops/sec. Python is slower.
                    # Let's try to be smarter.
                    # Store the change in uncovered.
                    pass
                else:
                    # We removed, so add
                    R[e] = True
                    for f in edge_faces[e]:
                        face_count[f] += 1
                    current_size += 1
                    pass

            # This revert logic is messy. Let's use a cleaner SA loop.
            # I'll rewrite the loop body to be cleaner.
            
        else:
            # Swap
            e1 = random.randint(0, m - 1)
            e2 = random.randint(0, m - 1)
            if e1 == e2:
                continue
            # Only consider swap if one is in R and one is not?
            # Or just try to swap states.
            # If R[e1] and not R[e2]: remove e1, add e2.
            # If not R[e1] and R[e2]: add e1, remove e2.
            # If both in R or both not: no change in size, might change uncovered.
            
            if R[e1] != R[e2]:
                # Remove the one in R, add the one not in R
                e_rem = e1 if R[e1] else e2
                e_add = e2 if R[e1] else e1
                
                # Calculate delta for remove
                inc_unc_rem = 0
                for f in edge_faces[e_rem]:
                    if face_count[f] == 1:
                        inc_unc_rem += 1
                # Calculate delta for add
                dec_unc_add = 0
                for f in edge_faces[e_add]:
                    if face_count[f] == 0:
                        dec_unc_add += 1
                        
                # Note: if a face is in both e_rem and e_add, the counts interact.
                # face_count[f] changes by -1 (rem) and +1 (add) -> net 0.
                # So if face_count[f] was 1, it becomes 0 then 1. Uncovered stays same?
                # Wait.
                # If f is in e_rem: count 1 -> 0 (uncovered +1)
                # If f is also in e_add: count 0 -> 1 (uncovered -1)
                # Net change 0.
                # My simple counting above (inc_unc_rem and dec_unc_add independently) is WRONG for shared faces.
                
                # Correct calculation:
                # For each face f involved in e_rem or e_add:
                # old_count = face_count[f]
                # new_count = old_count - (1 if f in e_rem else 0) + (1 if f in e_add else 0)
                # if old_count == 0 and new_count > 0: dec_unc += 1
                # if old_count > 0 and new_count == 0: inc_unc += 1
                
                inc_unc = 0
                dec_unc = 0
                affected_faces = set(edge_faces[e_rem] + edge_faces[e_add])
                for f in affected_faces:
                    old_c = face_count[f]
                    new_c = old_c
                    if f in edge_faces[e_rem]: new_c -= 1
                    if f in edge_faces[e_add]: new_c += 1
                    if old_c == 0 and new_c > 0: dec_unc += 1
                    if old_c > 0 and new_c == 0: inc_unc += 1
                
                new_size = current_size # size unchanged
                new_uncovered = current_uncovered - dec_unc + inc_unc
                new_cost = get_cost(new_size, new_uncovered)
                delta_cost = new_cost - current_cost
                
                # Apply
                R[e_rem] = False
                R[e_add] = True
                for f in edge_faces[e_rem]: face_count[f] -= 1
                for f in edge_faces[e_add]: face_count[f] += 1
                current_uncovered = new_uncovered
                # current_size unchanged
                
                # Acceptance
                accept = False
                if delta_cost < 0:
                    accept = True
                else:
                    if random.random() < math.exp(-delta_cost / T):
                        accept = True
                
                if not accept:
                    # Revert
                    R[e_rem] = True
                    R[e_add] = False
                    for f in edge_faces[e_rem]: face_count[f] += 1
                    for f in edge_faces[e_add]: face_count[f] -= 1
                    current_uncovered = current_uncovered + dec_unc - inc_unc

        # Update best
        if current_uncovered == 0 and current_size < best_size:
            best_size = current_size
            best_R = R[:]
            
        T *= cooling

    # 5. Final Greedy Fill (if any uncovered faces remain, try to fix them)
    # If SA didn't reach 0 uncovered, we must ensure validity.
    # The problem requires a valid set.
    # If best_R is not valid (uncovered > 0), we must add edges until valid.
    # But we want to maximize edges, so we want MIN removed edges.
    # If we have a valid set, we return it.
    
    # Check validity of best_R
    fc_best = [0] * n_faces
    for e in range(m):
        if best_R[e]:
            for f in edge_faces[e]:
                fc_best[f] += 1
                
    is_valid = all(c > 0 for c in fc_best)
    
    if not is_valid:
        # Fallback: Use the greedy init result which is guaranteed valid
        R_g, fc_g = greedy_init()
        best_R = R_g
        best_size = sum(R_g)

    # 6. Construct Output
    # Selected edges are those NOT in R
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
