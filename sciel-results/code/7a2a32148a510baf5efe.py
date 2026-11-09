# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using Simulated Annealing with incremental updates.
    
    Strategy:
    1. Precompute field arithmetic tables for F_49.
    2. Initialize N using the Hermitian unital construction.
    3. Precompute line memberships and initialize missing_count and witness_count.
    4. Use Simulated Annealing to minimize |N| subject to validity (witness_count > 0 for all).
       - Moves: Flip a random point (add or remove).
       - Incremental update of missing_count and witness_count.
       - Accept moves based on energy change (invalidity count) and size change.
    5. Keep the best valid solution found.
    """
    deadline = time.time() + 620  # Safety margin before 630s limit
    
    p = 7
    q = p * p  # 49

    # Field F_q = F_p[alpha], alpha^2 = k
    k_field = 3  # 3 is a non-square in F_7 (3^3=27=6=-1, so 3 is non-square)
    
    def to_int(a, b):
        return a + b * p

    def to_pair(x):
        return (x % p, (x // p) % p)

    # Precompute field arithmetic tables
    f_add = np.zeros((q, q), dtype=np.int32)
    f_mul = np.zeros((q, q), dtype=np.int32)
    f_neg = np.zeros(q, dtype=np.int32)

    for x in range(q):
        xa, xb = to_pair(x)
        f_neg[x] = to_int((-xa) % p, (-xb) % p)
        for y in range(q):
            ya, yb = to_pair(y)
            f_add[x, y] = to_int((xa + ya) % p, (xb + yb) % p)
            real = (xa * ya + xb * yb * k_field) % p
            imag = (xa * yb + xb * ya) % p
            f_mul[x, y] = to_int(real, imag)

    # Directions: (1, m) for m in F_49, and (0, 1)
    directions = []
    for m in range(q):
        directions.append((1, m))
    directions.append((0, 1))
    n_dir = len(directions)

    # All points in F_49^2
    # Indexing: point (x, y) -> idx = x * q + y
    point_index = {}
    all_points = []
    for x_idx in range(q):
        for y_idx in range(q):
            idx = x_idx * q + y_idx
            point_index[(x_idx, y_idx)] = idx
            all_points.append((x_idx, y_idx))
    
    idx_to_point = [None] * (q * q)
    for pt, idx in point_index.items():
        idx_to_point[idx] = pt

    # Build initial N (Hermitian Unital)
    def build_initial_in_N():
        in_N = np.ones(q * q, dtype=bool)
        y_special = set()
        for u in range(p):
            for v in range(p):
                y_special.add(to_int((u * u) % p, v))
        
        for x_idx in range(q):
            xa, xb = to_pair(x_idx)
            trace_x = (2 * xa) % p
            for y_idx in range(q):
                ya, yb = to_pair(y_idx)
                norm_y = (ya * ya - yb * yb * k_field) % p
                if (trace_x + norm_y) % p == 0:
                    if y_idx not in y_special:
                        in_N[point_index[(x_idx, y_idx)]] = False
        return in_N

    # Precompute for each point p_idx and direction d_idx, the list of other points on that line
    # line_points[p_idx, d_idx] = list of indices of other points on the line through p_idx in direction d_idx
    # Actually, we need to update missing_count for all x such that p is on the line through x in direction d.
    # This is equivalent to: for each direction d, the lines partition the plane.
    # Let's precompute the lines.
    # lines[d_idx] = list of sets of points.
    
    # For incremental update, we need:
    # For a point p, and a direction d, find the line L passing through p with direction d.
    # Then for every other point x on L, p is a "missing" point for x's witness in direction d if p not in N.
    # So we need to update missing_count[x, d] for all x in L \ {p}.
    
    # Precompute line membership:
    # line_id[p_idx, d_idx] -> id of the line
    # line_members[line_id] -> list of points on the line
    
    # There are n_dir * q lines total? No.
    # For each direction d, there are q lines (parallel classes).
    # Total lines = n_dir * q = 50 * 49 = 2450.
    
    line_members = []
    point_line_id = np.zeros((q * q, n_dir), dtype=np.int32)
    
    line_id_counter = 0
    for d_idx, (vx, vy) in enumerate(directions):
        # Find a representative for each line in this direction
        # We can iterate through all points and assign them to lines.
        # A point (x,y) is on a line with direction (vx,vy) if it can be written as (x0,y0) + t(vx,vy).
        # It's easier to just group them.
        
        # For direction (1, m), lines are y - m*x = c.
        # For direction (0, 1), lines are x = c.
        
        assigned = set()
        for p_idx in range(q * q):
            if p_idx in assigned:
                continue
            
            px, py = idx_to_point[p_idx]
            # Find all points on this line
            members = []
            for t in range(q):
                tx = f_mul[t, vx]
                ty = f_mul[t, vy]
                nx = f_add[px, tx]
                ny = f_add[py, ty]
                n_idx = point_index[(nx, ny)]
                members.append(n_idx)
                assigned.add(n_idx)
            
            # Assign line ID
            for m_idx in members:
                point_line_id[m_idx, d_idx] = line_id_counter
            
            line_members.append(members)
            line_id_counter += 1
            
    # Initialize Simulated Annealing
    in_N = build_initial_in_N()
    
    # missing_count[x, d]: number of points on line through x in direction d (excluding x) that are NOT in N
    missing_count = np.zeros((q * q, n_dir), dtype=np.int32)
    witness_count = np.zeros(q * q, dtype=np.int32)
    
    # Initialize counts
    for d_idx in range(n_dir):
        for line_id in range(line_id_counter):
            members = line_members[line_id]
            # For each x in members, check how many others are not in N
            # This is O(q^2 * n_dir * q) which is too slow if done naively.
            # Better: for each line, count how many points are NOT in N.
            # Let k = number of points in line NOT in N.
            # For each x in line:
            # If x is in N: missing_count[x, d] = k (since x is not counted as missing for itself, but the other k-1 are? No.)
            # Definition: missing_count[x, d] = |{y in L(x,d) \ {x} : y not in N}|
            # If x in N: missing_count[x, d] = (total not in N on line) - (0 if x not in N else 1? No, x is excluded).
            # Let S = {y in L : y not in N}.
            # If x in N: x not in S. missing_count[x, d] = |S|.
            # If x not in N: x in S. missing_count[x, d] = |S| - 1.
            
            not_in_N_on_line = [m for m in members if not in_N[m]]
            k = len(not_in_N_on_line)
            not_in_N_set = set(not_in_N_on_line)
            
            for x in members:
                if in_N[x]:
                    missing_count[x, d_idx] = k
                else:
                    missing_count[x, d_idx] = k - 1
                
                if missing_count[x, d_idx] == 0:
                    witness_count[x] += 1
    
    # Current state
    current_size = np.sum(in_N)
    current_invalid = np.sum(witness_count == 0)
    
    best_in_N = in_N.copy()
    best_size = current_size
    
    # SA Parameters
    T_start = 10.0
    T_end = 0.1
    cooling_rate = 0.9999
    max_iters = 5000000 # Adjust based on speed
    
    # Precompute random indices for speed
    # We will pick a random point and decide to flip it.
    
    t = T_start
    iter_count = 0
    last_time = time.time()
    
    while t > T_end and time.time() < deadline - 10:
        # Pick random point
        p_idx = np.random.randint(0, q * q)
        
        # Decide action: Remove if in N, Add if not in N
        # Bias: We want to minimize size, so prefer removing.
        # But if invalid, we might need to add.
        # Let's just flip.
        
        action_remove = in_N[p_idx]
        
        # Calculate Delta
        delta_size = -1 if action_remove else 1
        
        # Calculate Delta Invalid
        # We need to check which points' witness_count changes.
        # Only points on lines passing through p_idx are affected.
        
        # Identify affected (x, d) pairs
        # For each direction d, p_idx is on a specific line L_d.
        # For each other point x in L_d, missing_count[x, d] changes by +/- 1.
        # If missing_count[x, d] crosses 0, witness_count[x] changes.
        
        new_invalid = 0
        old_invalid = current_invalid
        
        # We need to simulate the change to know new_invalid.
        # Let's store the changes to revert if rejected.
        
        changes = [] # (x, d, old_missing, new_missing, old_witness, new_witness)
        
        for d_idx in range(n_dir):
            lid = point_line_id[p_idx, d_idx]
            members = line_members[lid]
            
            # p_idx is in members.
            # For each x in members \ {p_idx}:
            # If removing p:
            #   If p was in N (it is), then p was NOT missing. Now p IS missing.
            #   So missing_count[x, d] increases by 1.
            # If adding p:
            #   If p was not in N (it is not), then p WAS missing. Now p IS NOT missing.
            #   So missing_count[x, d] decreases by 1.
            
            delta_missing = 1 if action_remove else -1
            
            for x in members:
                if x == p_idx:
                    continue
                
                old_miss = missing_count[x, d_idx]
                new_miss = old_miss + delta_missing
                
                old_wit = witness_count[x]
                new_wit = old_wit
                
                if old_miss == 0 and new_miss > 0:
                    new_wit = old_wit - 1
                elif old_miss > 0 and new_miss == 0:
                    new_wit = old_wit + 1
                
                changes.append((x, d_idx, old_miss, new_miss, old_wit, new_wit))
                
                # Apply tentatively
                missing_count[x, d_idx] = new_miss
                witness_count[x] = new_wit
                
                if new_wit == 0 and old_wit > 0:
                    new_invalid += 1
                elif new_wit > 0 and old_wit == 0:
                    new_invalid -= 1
        
        # Also, p_idx itself might change witness status?
        # No, witness_count[p_idx] depends on lines through p_idx.
        # The lines through p_idx are the same lines.
        # Does missing_count[p_idx, d] change?
        # missing_count[p_idx, d] counts points on L(p_idx, d) \ {p_idx} not in N.
        # Changing N status of p_idx does NOT affect missing_count[p_idx, d].
        # So witness_count[p_idx] does NOT change.
        
        delta_invalid = new_invalid - old_invalid
        
        # Objective: Minimize Size + Lambda * Invalid
        # Let's use Energy E = Invalid * 1000 + Size
        # Delta E = delta_invalid * 1000 + delta_size
        
        delta_E = delta_invalid * 1000 + delta_size
        
        accept = False
        if delta_E < 0:
            accept = True
        else:
            # Boltzmann acceptance
            # P = exp(-delta_E / T)
            # If delta_E is large, P is small.
            if delta_E < 50 * t: # Optimization: avoid underflow
                prob = np.exp(-delta_E / t)
                if np.random.random() < prob:
                    accept = True
        
        if accept:
            # Commit changes
            in_N[p_idx] = not in_N[p_idx]
            current_size += delta_size
            current_invalid = new_invalid
            
            if current_invalid == 0 and current_size < best_size:
                best_size = current_size
                best_in_N = in_N.copy()
        else:
            # Revert changes
            for x, d_idx, old_miss, new_miss, old_wit, new_wit in changes:
                missing_count[x, d_idx] = old_miss
                witness_count[x] = old_wit
        
        iter_count += 1
        if iter_count % 1000 == 0:
            t *= cooling_rate
            # Check time
            if time.time() - last_time > 10:
                last_time = time.time()
                # Optional: Reheat or adjust
                pass

    # Final check: ensure best_in_N is valid
    # Recalculate validity to be safe
    final_missing = np.zeros((q * q, n_dir), dtype=np.int32)
    final_witness = np.zeros(q * q, dtype=np.int32)
    
    for d_idx in range(n_dir):
        for lid in range(line_id_counter):
            members = line_members[lid]
            not_in_N_on_line = [m for m in members if not best_in_N[m]]
            k = len(not_in_N_on_line)
            not_in_N_set = set(not_in_N_on_line)
            
            for x in members:
                if best_in_N[x]:
                    final_missing[x, d_idx] = k
                else:
                    final_missing[x, d_idx] = k - 1
                
                if final_missing[x, d_idx] == 0:
                    final_witness[x] += 1
    
    if np.any(final_witness == 0):
        # Fallback to initial unital if SA failed to maintain validity
        # Or find a valid subset?
        # For now, return the initial unital if invalid
        best_in_N = build_initial_in_N()
        # Re-verify unital is valid (it should be)
        
    points = []
    for i in range(q * q):
        if best_in_N[i]:
            x, y = idx_to_point[i]
            xa, xb = to_pair(x)
            ya, yb = to_pair(y)
            points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
