import time
import random
import math
from collections import Counter
import numpy as np

def construct():
    deadline = time.time() + 80.0  # Safety margin
    random.seed(42)
    
    N = 30
    BOUND = 30
    
    def get_initial_configs():
        configs = []
        # 1. 5x6 Grid
        # Centered at 0
        # x in [-2, 2], y in [-2.5, 2.5] -> not integer.
        # Let's use x in [-2, 2] (5 values), y in [-2, 3] (6 values).
        grid_5x6 = [(x, y) for x in range(-2, 3) for y in range(-2, 4)]
        configs.append(grid_5x6)
        
        # 2. 3x10 Grid
        grid_3x10 = [(x, y) for x in range(-1, 2) for y in range(-5, 5)]
        configs.append(grid_3x10)
        
        # 3. 6x5 Grid (Transposed)
        grid_6x5 = [(x, y) for x in range(-3, 3) for y in range(-2, 3)]
        configs.append(grid_6x5)
        
        # 4. Random dense cluster
        pts = []
        seen = set()
        while len(pts) < 30:
            x = random.randint(-10, 10)
            y = random.randint(-10, 10)
            if (x, y) not in seen:
                seen.add((x, y))
                pts.append((x, y))
        configs.append(pts)
        
        return configs

    def compute_areas(points):
        """
        points: list of (x, y)
        Returns Counter of doubled areas
        """
        points_arr = np.array(points, dtype=np.int64)
        # Compute all differences
        # Shape: (N, N, 2)
        # diff[i, j] = points[i] - points[j]
        # We need cross products of (p_j - p_i) and (p_k - p_i)
        
        # Efficient way:
        # For each i, compute vectors to all j > i and k > j?
        # Or just compute all pairwise cross products?
        
        # Let's use the property: Area(i, j, k) = 0.5 * | (p_j - p_i) x (p_k - p_i) |
        # Cross product in 2D: (x1, y1) x (x2, y2) = x1*y2 - y1*x2
        
        # We can vectorize this for all i.
        # For a fixed i, we have vectors v_j = p_j - p_i for all j != i.
        # We need cross(v_j, v_k) for j < k.
        
        areas = []
        for i in range(N):
            # Vectors from i to all other points
            # p_j - p_i
            vecs = points_arr - points_arr[i]
            # Zero out i-th row
            vecs[i] = 0
            
            # We need cross products of pairs (j, k) with j < k, j,k != i
            # This is tricky to vectorize directly for all i in one shot without huge memory.
            # N=30 is small.
            
            # Let's just do the triple loop but with numpy for the inner product?
            # No, pure python triple loop is ~4000 iterations. That's fast.
            # But we want speed.
            
            # Alternative:
            # Precompute all pairwise cross products?
            # C[i, j] = (p_j - p_i) x (p_k - p_i) ... no, depends on k.
            
            # Let's stick to the optimized python loop or simple numpy.
            # 4060 triples.
            # In numpy:
            # We can create arrays of indices.
            pass
        
        # Actually, a simple loop in Python is ~1ms for 4000 triples.
        # Let's use a slightly optimized Python approach first, then switch to NumPy if needed.
        # But the prompt says NumPy is available. Let's use NumPy for the full count.
        
        # Vectorized full count:
        # Create arrays for all triples (i, j, k) with i < j < k
        idx_i, idx_j, idx_k = [], [], []
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    idx_i.append(i)
                    idx_j.append(j)
                    idx_k.append(k)
                    
        idx_i = np.array(idx_i)
        idx_j = np.array(idx_j)
        idx_k = np.array(idx_k)
        
        pi = points_arr[idx_i]
        pj = points_arr[idx_j]
        pk = points_arr[idx_k]
        
        # (pj - pi) x (pk - pi)
        # (xj - xi)(yk - yi) - (yj - yi)(xk - xi)
        area_vals = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
        area_vals = np.abs(area_vals)
        
        # Exclude zero
        area_vals = area_vals[area_vals > 0]
        
        if len(area_vals) == 0:
            return Counter(), 0, 0
            
        counts = Counter(area_vals.tolist())
        max_area, max_count = counts.most_common(1)[0]
        return counts, max_area, max_count

    def delta_count(points, idx, new_pt, target_area):
        """
        Calculate change in count of target_area if points[idx] is replaced by new_pt.
        Only considers triples involving idx.
        """
        points_arr = np.array(points, dtype=np.int64)
        pi = points_arr[idx]
        new_pi = np.array(new_pt, dtype=np.int64)
        
        # Other points
        other_indices = [j for j in range(N) if j != idx]
        p_others = points_arr[other_indices]
        
        # Calculate areas with old point
        # Vectors from pi to p_others
        vecs_old = p_others - pi
        # Cross products of pairs in vecs_old
        # We need cross(vecs_old[j], vecs_old[k]) for all j < k in other_indices
        # There are 29 other points. Pairs = 29*28/2 = 406.
        
        # Vectorized calculation for 406 pairs
        # Create index pairs for the 29 points
        j_idx, k_idx = np.triu_indices(29, k=1)
        
        # Get the vectors
        v_j_old = vecs_old[j_idx]
        v_k_old = vecs_old[k_idx]
        
        # Cross product: x1*y2 - y1*x2
        areas_old = np.abs(v_j_old[:, 0] * v_k_old[:, 1] - v_j_old[:, 1] * v_k_old[:, 0])
        count_old = np.sum(areas_old == target_area)
        
        # Calculate areas with new point
        vecs_new = p_others - new_pi
        v_j_new = vecs_new[j_idx]
        v_k_new = vecs_new[k_idx]
        
        areas_new = np.abs(v_j_new[:, 0] * v_k_new[:, 1] - v_j_new[:, 1] * v_k_new[:, 0])
        count_new = np.sum(areas_new == target_area)
        
        return count_new - count_old

    def solve_diophantine(A, B, C, bound):
        """
        Solve A*x + B*y = C for integer x, y in [-bound, bound].
        Returns list of (x, y).
        """
        if A == 0 and B == 0:
            if C == 0:
                return [(x, y) for x in range(-bound, bound+1) for y in range(-bound, bound+1)]
            else:
                return []
        
        g = math.gcd(A, B)
        if C % g != 0:
            return []
            
        A_prime = A // g
        B_prime = B // g
        C_prime = C // g
        
        # Find one solution
        # Use extended Euclidean algorithm
        def egcd(a, b):
            if b == 0:
                return (a, 1, 0)
            g, x1, y1 = egcd(b, a % b)
            return (g, y1, x1 - (a // b) * y1)
            
        g, x, y = egcd(A_prime, B_prime)
        # A_prime * x + B_prime * y = g = 1
        # So A_prime * (x * C_prime) + B_prime * (y * C_prime) = C_prime
        x0 = x * C_prime
        y0 = y * C_prime
        
        # General solution:
        # x = x0 + t * B_prime
        # y = y0 - t * A_prime
        
        solutions = []
        # Find range of t
        # -bound <= x0 + t * B_prime <= bound
        # -bound <= y0 - t * A_prime <= bound
        
        # Handle B_prime == 0
        if B_prime == 0:
            if -bound <= x0 <= bound:
                # y can be anything? No, A_prime * y = C_prime - A_prime * x0 = 0?
                # If B_prime=0, then A_prime * x = C_prime. x is fixed.
                # Equation is 0*y = 0. So y is free?
                # Original: A*x = C. If solvable, x is fixed. y is free.
                for y in range(-bound, bound + 1):
                    solutions.append((x0, y))
            return solutions
            
        if A_prime == 0:
            if -bound <= y0 <= bound:
                for x in range(-bound, bound + 1):
                    solutions.append((x, y0))
            return solutions

        # t_min from x constraint
        if B_prime > 0:
            t_min_x = math.ceil((-bound - x0) / B_prime)
            t_max_x = math.floor((bound - x0) / B_prime)
        else:
            t_min_x = math.ceil((bound - x0) / B_prime)
            t_max_x = math.floor((-bound - x0) / B_prime)
            
        # t_min from y constraint
        # y = y0 - t * A_prime
        # -bound <= y0 - t * A_prime <= bound
        # -bound - y0 <= -t * A_prime <= bound - y0
        # (bound - y0) / A_prime <= t <= (y0 + bound) / A_prime  (if A_prime > 0)
        
        if A_prime > 0:
            t_min_y = math.ceil((y0 - bound) / A_prime)
            t_max_y = math.floor((y0 + bound) / A_prime)
        else:
            # A_prime < 0
            # Divide by negative flips inequality
            # t * A_prime >= y0 - bound  => t <= (y0 - bound) / A_prime
            # t * A_prime <= y0 + bound  => t >= (y0 + bound) / A_prime
            t_min_y = math.ceil((y0 + bound) / A_prime)
            t_max_y = math.floor((y0 - bound) / A_prime)
            
        t_start = max(t_min_x, t_min_y)
        t_end = min(t_max_x, t_max_y)
        
        for t in range(t_start, t_end + 1):
            x = x0 + t * B_prime
            y = y0 - t * A_prime
            if -bound <= x <= bound and -bound <= y <= bound:
                solutions.append((x, y))
                
        return solutions

    # Initialize
    initial_configs = get_initial_configs()
    best_points = None
    best_count = 0
    best_target = 0
    best_counter = None
    
    for config in initial_configs:
        # Validate and pad
        seen = set()
        valid_pts = []
        for p in config:
            if -BOUND <= p[0] <= BOUND and -BOUND <= p[1] <= BOUND and p not in seen:
                seen.add(p)
                valid_pts.append(p)
        while len(valid_pts) < 30:
            p = (random.randint(-BOUND, BOUND), random.randint(-BOUND, BOUND))
            if p not in seen:
                seen.add(p)
                valid_pts.append(p)
        
        counter, target, count = compute_areas(valid_pts)
        if count > best_count:
            best_count = count
            best_points = valid_pts
            best_target = target
            best_counter = counter
            
    current_points = list(best_points)
    current_target = best_target
    current_count = best_count
    
    # Precompute index pairs for delta calculation
    # This is constant for N=30
    j_idx, k_idx = np.triu_indices(29, k=1)
    
    # Simulated Annealing
    T = 10.0
    cooling_rate = 0.99
    no_improve = 0
    max_no_improve = 100
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            # Restart or cool down
            T = 10.0
            no_improve = 0
            max_no_improve += 50
            
            # Try to switch target if current is stuck
            # Recalculate full count
            counter, target, count = compute_areas(current_points)
            if count > best_count:
                best_count = count
                best_points = list(current_points)
                best_target = target
                best_counter = counter
                current_target = target
                current_count = count
            else:
                # If stuck, maybe try a random restart of some points
                for _ in range(5):
                    idx = random.randint(0, N-1)
                    new_pt = (random.randint(-BOUND, BOUND), random.randint(-BOUND, BOUND))
                    if new_pt not in current_points:
                        current_points[idx] = new_pt
                # Re-evaluate
                counter, target, count = compute_areas(current_points)
                if count > best_count:
                    best_count = count
                    best_points = list(current_points)
                    best_target = target
                    current_target = target
                    current_count = count
                else:
                    current_points = list(best_points)
                    current_target = best_target
                    current_count = best_count
            continue
            
        # Pick a point to move
        idx = random.randint(0, N-1)
        old_pt = current_points[idx]
        
        # Generate candidates
        candidates = set()
        
        # 1. Local random
        for _ in range(5):
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
            nx = old_pt[0] + dx
            ny = old_pt[1] + dy
            if -BOUND <= nx <= BOUND and -BOUND <= ny <= BOUND:
                candidates.add((nx, ny))
                
        # 2. Diophantine candidates
        # Pick 5 random pairs (j, k) from other points
        other_indices = [j for j in range(N) if j != idx]
        random.shuffle(other_indices)
        
        for _ in range(5):
            j = other_indices[0]
            k = other_indices[1]
            pj = current_points[j]
            pk = current_points[k]
            
            # Equation: | (xj - xk)(y - yj) - (yj - yk)(x - xj) | = A
            # dx = xj - xk, dy = yj - yk
            dx = pj[0] - pk[0]
            dy = pj[1] - pk[1]
            
            # dx * y - dy * x + (dy * xj - dx * yj) = +/- A
            # -dy * x + dx * y = +/- A - (dy * xj - dx * yj)
            
            const_term = dy * pj[0] - dx * pj[1]
            
            for sign in [1, -1]:
                C = sign * current_target - const_term
                # A_coeff * x + B_coeff * y = C
                # A_coeff = -dy
                # B_coeff = dx
                sols = solve_diophantine(-dy, dx, C, BOUND)
                # Add only a few solutions to keep candidate set small
                if sols:
                    random.shuffle(sols)
                    for s in sols[:3]:
                        candidates.add(s)
                        
        # Filter candidates
        valid_candidates = []
        for c in candidates:
            if c == old_pt:
                continue
            if c in current_points:
                continue
            if not (-BOUND <= c[0] <= BOUND and -BOUND <= c[1] <= BOUND):
                continue
            valid_candidates.append(c)
            
        if not valid_candidates:
            continue
            
        # Evaluate best candidate
        best_delta = -1000
        best_new_pt = None
        
        for new_pt in valid_candidates:
            delta = delta_count(current_points, idx, new_pt, current_target)
            if delta > best_delta:
                best_delta = delta
                best_new_pt = new_pt
                
        if best_new_pt is None:
            no_improve += 1
            continue
            
        # Acceptance
        if best_delta > 0:
            accept = True
        elif best_delta == 0:
            accept = random.random() < 0.5
        else:
            if T > 0.1:
                accept_prob = math.exp(best_delta / T)
                accept = random.random() < accept_prob
            else:
                accept = False
                
        if accept:
            current_points[idx] = best_new_pt
            current_count += best_delta
            no_improve = 0
            if current_count > best_count:
                best_count = current_count
                best_points = list(current_points)
                # Note: best_target might change, but we keep current_target for now
                # to maintain momentum. We update best_target only on restart or if we want to be strict.
                # Actually, if count increases, the target area is likely still the same or a new one.
                # Let's keep current_target to focus the search.
        else:
            no_improve += 1
            
        T *= cooling_rate
        
    # Final verification
    if len(best_points) != N or len(set(best_points)) != N:
        return [(x, y) for x in range(6) for y in range(5)]
    for p in best_points:
        if not (-BOUND <= p[0] <= BOUND and -BOUND <= p[1] <= BOUND):
            return [(x, y) for x in range(6) for y in range(5)]
            
    return best_points
