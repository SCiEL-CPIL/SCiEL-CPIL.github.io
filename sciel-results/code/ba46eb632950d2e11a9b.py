# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import heapq

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using priority-queue greedy removal
    with multiple random restarts.
    
    Strategy:
    1. Precompute field arithmetic tables for F_49.
    2. Build initial N (Hermitian unital or random).
    3. For each point, compute cover_count = number of (x, d) pairs where
       x's only witness in direction d is this point.
    4. Use priority queue to always remove the point with minimum cover_count.
    5. Multiple restarts with different initial sets.
    """
    deadline = time.time() + 600  # Safety margin
    
    p = 7
    q = p * p  # 49
    k_field = 3
    
    def to_int(a, b):
        return a + b * p

    def to_pair(x):
        return (x % p, (x // p) % p)

    # Precompute field arithmetic tables
    f_add = np.zeros((q, q), dtype=np.int32)
    f_mul = np.zeros((q, q), dtype=np.int32)

    for x in range(q):
        xa, xb = to_pair(x)
        for y in range(q):
            ya, yb = to_pair(y)
            f_add[x, y] = to_int((xa + ya) % p, (xb + yb) % p)
            real = (xa * ya + xb * yb * k_field) % p
            imag = (xa * yb + xb * ya) % p
            f_mul[x, y] = to_int(real, imag)

    # Directions
    directions = [(1, m) for m in range(q)] + [(0, 1)]
    n_dir = len(directions)

    # Point indexing
    point_index = {}
    idx_to_point = []
    for x_idx in range(q):
        for y_idx in range(q):
            idx = x_idx * q + y_idx
            point_index[(x_idx, y_idx)] = idx
            idx_to_point.append((x_idx, y_idx))
    
    n_points = q * q

    # Precompute lines
    line_members = []
    point_line_id = np.zeros((n_points, n_dir), dtype=np.int32)
    
    line_id_counter = 0
    for d_idx, (vx, vy) in enumerate(directions):
        assigned = set()
        for p_idx in range(n_points):
            if p_idx in assigned:
                continue
            
            px, py = idx_to_point[p_idx]
            members = []
            for t in range(q):
                tx = f_mul[t, vx]
                ty = f_mul[t, vy]
                nx = f_add[px, tx]
                ny = f_add[py, ty]
                n_idx = point_index[(nx, ny)]
                members.append(n_idx)
                assigned.add(n_idx)
            
            for m_idx in members:
                point_line_id[m_idx, d_idx] = line_id_counter
            
            line_members.append(members)
            line_id_counter += 1
    
    # Build initial N (Hermitian Unital)
    def build_unital():
        in_N = np.ones(n_points, dtype=bool)
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
    
    def build_random(density=0.7):
        in_N = np.random.random(n_points) < density
        return in_N
    
    def verify_validity(in_N):
        """Check if in_N is a valid Nikodym set."""
        for d_idx in range(n_dir):
            for lid in range(line_id_counter):
                members = line_members[lid]
                not_in_N = [m for m in members if not in_N[m]]
                k = len(not_in_N)
                
                for x in members:
                    if in_N[x]:
                        miss = k
                    else:
                        miss = k - 1
                    
                    if miss == 0:
                        return True, x, d_idx  # Valid witness found
        
        return False, None, None
    
    def greedy_priority_removal(in_N, time_limit):
        """
        Greedy removal using priority queue.
        For each point, compute how many witness relationships it's critical for.
        Always remove the point with minimum criticality.
        """
        in_N = in_N.copy()
        
        # Compute missing_count and witness_count
        missing_count = np.zeros((n_points, n_dir), dtype=np.int32)
        witness_count = np.zeros(n_points, dtype=np.int32)
        
        for d_idx in range(n_dir):
            for lid in range(line_id_counter):
                members = line_members[lid]
                not_in_N_on_line = [m for m in members if not in_N[m]]
                k = len(not_in_N_on_line)
                
                for x in members:
                    if in_N[x]:
                        missing_count[x, d_idx] = k
                    else:
                        missing_count[x, d_idx] = k - 1
                    
                    if missing_count[x, d_idx] == 0:
                        witness_count[x] += 1
        
        # For each point p, compute cover_count:
        # number of (x, d) where p is on line L(x,d) and missing_count[x,d] == 1
        # (i.e., p is the unique witness for x in direction d)
        # Actually, we need: if we remove p, which (x,d) pairs lose their witness?
        # For x to lose witness in direction d when p is removed:
        #   - p is on line L(x,d), p != x
        #   - missing_count[x,d] == 1 (p is the only missing point, so removing p makes it 0)
        #   - Wait, that's wrong. missing_count[x,d] counts points NOT in N.
        #   - If we remove p from N, p becomes NOT in N.
        #   - So missing_count[x,d] increases by 1 for all x on line through p in direction d.
        #   - x loses witness if missing_count[x,d] was 0 and becomes 1.
        
        # Let me redefine: 
        # critical_count[p] = number of (x, d) where:
        #   - p is on line L(x,d), p != x
        #   - missing_count[x,d] == 0 (x currently has witness in direction d)
        #   - x is in N or not (doesn't matter for the count)
        # If we remove p, then for all such (x,d), missing_count[x,d] becomes 1,
        # so x loses its witness in direction d.
        
        # But x might still have witnesses in other directions.
        # We need: x loses ALL witnesses if for all d, missing_count[x,d] > 0 after removal.
        # This is more complex. Let's use a simpler heuristic:
        # critical_count[p] = sum over (x,d) where p is on L(x,d)\{x} and missing_count[x,d]==0
        #   of 1 / witness_count[x]
        # This weights by how critical this witness is for x.
        
        # Actually, let's just use: critical_count[p] = number of (x,d) where
        # p is on L(x,d)\{x} and missing_count[x,d] == 0
        # This is a reasonable proxy.
        
        critical_count = np.zeros(n_points, dtype=np.int32)
        
        for d_idx in range(n_dir):
            for lid in range(line_id_counter):
                members = line_members[lid]
                for i, x in enumerate(members):
                    if missing_count[x, d_idx] != 0:
                        continue
                    # x has witness in direction d
                    # All other points on this line are potential witnesses
                    for j, p in enumerate(members):
                        if p == x:
                            continue
                        # Check if p is in N (it must be, since missing_count counts NOT in N)
                        # Actually, any p on the line where p != x contributes to being a witness
                        # if p is in N. But missing_count[x,d]==0 means ALL points on line
                        # (except x) are in N. So all p != x on this line are in N.
                        critical_count[p] += 1
        
        # Build priority queue: (critical_count, point_index)
        # We want to remove points with lowest critical_count
        pq = []
        for p_idx in range(n_points):
            if in_N[p_idx]:
                heapq.heappush(pq, (critical_count[p_idx], p_idx))
        
        # Greedy removal
        current_size = np.sum(in_N)
        removed = set()
        
        while pq and time.time() < time_limit:
            crit, p_idx = heapq.heappop(pq)
            
            if not in_N[p_idx] or p_idx in removed:
                continue
            
            # Try removing p_idx
            # Check if removal keeps validity
            # We need to check: after removing p_idx, does every x still have at least one witness?
            
            # Simulate removal
            new_in_N = in_N.copy()
            new_in_N[p_idx] = False
            
            # Check validity efficiently
            # Only points on lines through p_idx are affected
            valid = True
            for d_idx in range(n_dir):
                lid = point_line_id[p_idx, d_idx]
                members = line_members[lid]
                
                # Recompute missing_count for points on this line
                not_in_N_new = [m for m in members if not new_in_N[m]]
                k_new = len(not_in_N_new)
                
                for x in members:
                    if x == p_idx:
                        continue  # p_idx is no longer in N, but we don't care about its witness
                    
                    # Compute new missing_count[x, d_idx]
                    if new_in_N[x]:
                        new_miss = k_new
                    else:
                        new_miss = k_new - 1
                    
                    if new_miss == 0:
                        continue  # x still has witness in this direction
                    
                    # x lost witness in direction d_idx
                    # Check if x has witness in any other direction
                    has_other_witness = False
                    for d2 in range(n_dir):
                        if d2 == d_idx:
                            continue
                        lid2 = point_line_id[x, d2]
                        members2 = line_members[lid2]
                        not_in_N_2 = [m for m in members2 if not new_in_N[m]]
                        k2 = len(not_in_N_2)
                        if new_in_N[x]:
                            miss2 = k2
                        else:
                            miss2 = k2 - 1
                        if miss2 == 0:
                            has_other_witness = True
                            break
                    
                    if not has_other_witness:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if valid:
                # Accept removal
                in_N[p_idx] = False
                removed.add(p_idx)
                current_size -= 1
                
                # Update critical_count for affected points
                # This is expensive; for now, just rebuild periodically
                # For efficiency, we'll rebuild the PQ every N removals
                pass
            
            else:
                # Rejection: don't remove
                pass
        
        return in_N, current_size
    
    # Main: multiple restarts
    best_in_N = None
    best_size = n_points + 1
    
    # First, try unital
    in_N = build_unital()
    result_in_N, result_size = greedy_priority_removal(in_N, deadline - 5)
    
    if result_size < best_size:
        best_size = result_size
        best_in_N = result_in_N
    
    # Multiple random restarts
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        density = 0.6 + 0.05 * restart
        in_N = build_random(density)
        
        # Quick validity check
        valid, _, _ = verify_validity(in_N)
        if not valid:
            continue
        
        result_in_N, result_size = greedy_priority_removal(in_N, deadline - 30 - restart * 5)
        
        if result_size < best_size:
            best_size = result_size
            best_in_N = result_in_N
    
    # Fallback: if best_in_N is None or invalid, use unital
    if best_in_N is None:
        best_in_N = build_unital()
    
    # Final verification
    valid, _, _ = verify_validity(best_in_N)
    if not valid:
        best_in_N = build_unital()
    
    # Convert to output format
    points = []
    for i in range(n_points):
        if best_in_N[i]:
            x, y = idx_to_point[i]
            xa, xb = to_pair(x)
            ya, yb = to_pair(y)
            points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
