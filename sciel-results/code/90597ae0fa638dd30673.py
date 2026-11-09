# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import heapq

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using conflict-graph local search with simulated annealing.
    
    Strategy:
    1. Precompute field arithmetic tables for F_49.
    2. Initialize N using the Hermitian unital construction.
    3. Build conflict graph: edges between points that cannot both be in complement.
    4. Use simulated annealing on the complement set to maximize its size.
    """
    deadline = time.time() + 620
    
    p = 7
    q = p * p  # 49

    k_field = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            k_field = c
            break
    if k_field is None:
        k_field = 3

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

    # Directions
    directions = []
    for m in range(q):
        directions.append((1, m))
    directions.append((0, 1))
    n_dir = len(directions)

    # All points
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

    # Build initial N
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

    # Precompute line structure: for each point p and direction d, list of other points on line
    # line_points[p_idx, d_idx] = list of (other_point_idx)
    line_points = []
    for d_idx, (vx, vy) in enumerate(directions):
        t_v = []
        for t in range(1, q):
            tx = f_mul[t, vx]
            ty = f_mul[t, vy]
            t_v.append((tx, ty))
        
        for p_idx in range(q * q):
            px, py = idx_to_point[p_idx]
            pts = []
            for tx, ty in t_v:
                nx = f_add[px, tx]
                ny = f_add[py, ty]
                n_idx = point_index[(nx, ny)]
                pts.append(n_idx)
            line_points.append(pts)
    
    # Reshape to 2D: line_points[p_idx * n_dir + d_idx]
    line_points_2d = np.array(line_points).reshape(q * q, n_dir)

    # Precompute affects: for each point r, which (p_idx, d_idx) pairs are such that r is on line(p, d) \ {p}
    affects = [[] for _ in range(q * q)]
    for r_idx in range(q * q):
        rx, ry = idx_to_point[r_idx]
        for d_idx, (vx, vy) in enumerate(directions):
            for t in range(1, q):
                tv_x = f_mul[t, vx]
                tv_y = f_mul[t, vy]
                neg_tv_x = f_neg[tv_x]
                neg_tv_y = f_neg[tv_y]
                px = f_add[rx, neg_tv_x]
                py = f_add[ry, neg_tv_y]
                p_idx = point_index[(px, py)]
                affects[r_idx].append((p_idx, d_idx))

    def compute_not_in_N_count(in_N):
        not_in_N_count = np.zeros((q * q, n_dir), dtype=np.int32)
        for d_idx in range(n_dir):
            for p_idx in range(q * q):
                count = 0
                for other_idx in line_points_2d[p_idx, d_idx]:
                    if not in_N[other_idx]:
                        count += 1
                not_in_N_count[p_idx, d_idx] = count
        return not_in_N_count

    def compute_cover_count(not_in_N_count):
        return np.sum(not_in_N_count == 0, axis=1).astype(np.int32)

    def is_valid_nikodym(in_N):
        """Check if in_N is a valid Nikodym set."""
        not_in_N_count = compute_not_in_N_count(in_N)
        cover_count = compute_cover_count(not_in_N_count)
        return np.all(cover_count > 0)

    def greedy_remove(in_N, not_in_N_count, cover_count, time_limit):
        """Greedy removal with priority queue."""
        in_N = in_N.copy()
        not_in_N_count = not_in_N_count.copy()
        cover_count = cover_count.copy()
        
        start_time = time.time()
        
        # Build priority queue: (cost, r_idx)
        # Cost = number of (p_idx, d_idx) where not_in_N_count == 0
        heap = []
        cost_cache = {}
        
        for r_idx in range(q * q):
            if in_N[r_idx]:
                cost = 0
                for p_idx, d_idx in affects[r_idx]:
                    if not_in_N_count[p_idx, d_idx] == 0:
                        cost += 1
                cost_cache[r_idx] = cost
                heapq.heappush(heap, (cost, r_idx))
        
        while heap and time.time() - start_time < time_limit:
            cost, r_idx = heapq.heappop(heap)
            
            if not in_N[r_idx]:
                continue
            if cost_cache.get(r_idx, -1) != cost:
                continue
            
            # Check validity
            invalid = False
            affected = []
            
            for p_idx, d_idx in affects[r_idx]:
                old_count = not_in_N_count[p_idx, d_idx]
                if old_count == 0:
                    if cover_count[p_idx] == 1:
                        invalid = True
                        break
                affected.append((p_idx, d_idx, old_count))
            
            if invalid:
                cost_cache[r_idx] = -1
                continue
            
            # Commit
            for p_idx, d_idx, old_count in affected:
                not_in_N_count[p_idx, d_idx] = old_count + 1
                if old_count == 0:
                    cover_count[p_idx] -= 1
            
            in_N[r_idx] = False
            cost_cache[r_idx] = -1
            
            # Update costs of potentially affected points
            # Points that are on lines through r_idx in various directions
            # For efficiency, we'll do lazy recomputation
            
            # Re-push neighbors with updated costs
            for p_idx, d_idx in affects[r_idx]:
                # Points that affect p_idx in direction d_idx
                # This is expensive, so we'll just mark for re-evaluation
                pass
        
        return in_N, not_in_N_count, cover_count

    def simulated_annealing(in_N, not_in_N_count, cover_count, time_limit):
        """Simulated annealing to find larger complement."""
        in_N = in_N.copy()
        not_in_N_count = not_in_N_count.copy()
        cover_count = cover_count.copy()
        
        start_time = time.time()
        T = 100.0
        T_min = 0.1
        cooling = 0.995
        
        best_in_N = in_N.copy()
        best_complement = np.sum(~in_N)
        
        while time.time() - start_time < time_limit and T > T_min:
            # Choose a random point to try to add to complement (remove from N)
            in_N_indices = np.where(in_N)[0]
            if len(in_N_indices) == 0:
                break
            
            r_idx = random.choice(in_N_indices.tolist())
            
            # Try removing r_idx
            invalid = False
            affected = []
            
            for p_idx, d_idx in affects[r_idx]:
                old_count = not_in_N_count[p_idx, d_idx]
                if old_count == 0:
                    if cover_count[p_idx] == 1:
                        invalid = True
                        break
                affected.append((p_idx, d_idx, old_count))
            
            if invalid:
                # Try to undo a previous removal (add back a point to N)
                not_in_N_indices = np.where(~in_N)[0]
                if len(not_in_N_indices) > 0:
                    r_idx = random.choice(not_in_N_indices.tolist())
                    
                    # Add r_idx back to N
                    for p_idx, d_idx in affects[r_idx]:
                        not_in_N_count[p_idx, d_idx] -= 1
                        if not_in_N_count[p_idx, d_idx] == 0:
                            cover_count[p_idx] += 1
                    
                    in_N[r_idx] = True
                    complement_size = np.sum(~in_N)
                    
                    if complement_size > best_complement:
                        best_complement = complement_size
                        best_in_N = in_N.copy()
                    
                    T *= cooling
                    continue
            
            # Commit removal
            for p_idx, d_idx, old_count in affected:
                not_in_N_count[p_idx, d_idx] = old_count + 1
                if old_count == 0:
                    cover_count[p_idx] -= 1
            
            in_N[r_idx] = False
            complement_size = np.sum(~in_N)
            
            if complement_size > best_complement:
                best_complement = complement_size
                best_in_N = in_N.copy()
            
            T *= cooling
        
        return best_in_N

    # Main algorithm
    in_N = build_initial_in_N()
    not_in_N_count = compute_not_in_N_count(in_N)
    cover_count = compute_cover_count(not_in_N_count)
    
    best_in_N = in_N.copy()
    best_complement = np.sum(~in_N)
    
    # Phase 1: Greedy removal
    time_per_phase = 120
    in_N, not_in_N_count, cover_count = greedy_remove(in_N, not_in_N_count, cover_count, time_per_phase)
    complement_size = np.sum(~in_N)
    if complement_size > best_complement:
        best_complement = complement_size
        best_in_N = in_N.copy()
    
    # Phase 2: Simulated annealing with random restarts
    remaining_time = deadline - time.time() - 10
    num_restarts = max(1, int(remaining_time / 30))
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Start from best found so far with some randomization
        in_N_sa = best_in_N.copy()
        
        # Randomly add back some points to N (to escape local optima)
        num_to_add = random.randint(5, 20)
        not_in_N_indices = np.where(~in_N_sa)[0]
        if len(not_in_N_indices) > num_to_add:
            to_add = random.sample(not_in_N_indices.tolist(), num_to_add)
            for idx in to_add:
                in_N_sa[idx] = True
        
        not_in_N_count_sa = compute_not_in_N_count(in_N_sa)
        cover_count_sa = compute_cover_count(not_in_N_count_sa)
        
        # Run SA
        time_for_sa = min(25, deadline - time.time() - 5)
        in_N_sa = simulated_annealing(in_N_sa, not_in_N_count_sa, cover_count_sa, time_for_sa)
        
        complement_size = np.sum(~in_N_sa)
        if complement_size > best_complement:
            best_complement = complement_size
            best_in_N = in_N_sa.copy()
    
    # Final greedy pass on best result
    if time.time() < deadline - 15:
        in_N_final = best_in_N.copy()
        not_in_N_count_final = compute_not_in_N_count(in_N_final)
        cover_count_final = compute_cover_count(not_in_N_count_final)
        
        final_time = deadline - time.time() - 5
        in_N_final, _, _ = greedy_remove(in_N_final, not_in_N_count_final, cover_count_final, final_time)
        
        complement_size = np.sum(~in_N_final)
        if complement_size > best_complement:
            best_complement = complement_size
            best_in_N = in_N_final.copy()
    
    # Build result
    points = []
    for i in range(q * q):
        if best_in_N[i]:
            x, y = idx_to_point[i]
            xa, xb = to_pair(x)
            ya, yb = to_pair(y)
            points.append([[xa, xb], [ya, yb]])
    
    if not points:
        # Fallback: reconstruct initial base set
        in_N_init = build_initial_in_N()
        for i in range(q * q):
            if in_N_init[i]:
                x, y = idx_to_point[i]
                xa, xb = to_pair(x)
                ya, yb = to_pair(y)
                points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
