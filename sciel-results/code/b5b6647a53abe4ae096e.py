"""No-isosceles set construction with CSP/simulated annealing and local search."""
import numpy as np
import time
import random

N = 64
TIME_LIMIT = 280.0

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_points = []
    best_count = 0
    
    # Fast vectorized validity check
    def is_valid_addition(points, p):
        if not points:
            return True
        
        px, py = p
        P = np.array(points, dtype=np.int64)
        
        # Squared distances from p to all points in P
        d_p = (P[:, 0] - px)**2 + (P[:, 1] - py)**2
        
        n = len(P)
        
        # Check 1: d(p, a) == d(p, b) for any a != b
        # This is equivalent to checking if there are duplicate values in d_p
        if len(set(d_p)) != n:
            return False
            
        # Squared distances between all pairs in P
        P_norm_sq = np.sum(P**2, axis=1)
        Dot = P @ P.T
        D_sq = P_norm_sq[:, np.newaxis] + P_norm_sq[np.newaxis, :] - 2 * Dot
        
        # Check 2 and 3: d(p, a) == d(a, b) or d(p, b) == d(a, b)
        # For each pair (i, j) with i < j:
        #   d_p[i] != D_sq[i, j] and d_p[j] != D_sq[i, j]
        
        for i in range(n - 1):
            js = np.arange(i + 1, n)
            if np.any(D_sq[i, js] == d_p[i]):
                return False
            if np.any(D_sq[i, js] == d_p[js]):
                return False
                
        return True

    # Count violations in a set of points (for SA objective)
    def count_violations(pts):
        """Count number of isosceles triangles in the set."""
        n = len(pts)
        if n < 3:
            return 0
        
        P = np.array(pts, dtype=np.int64)
        P_norm_sq = np.sum(P**2, axis=1)
        Dot = P @ P.T
        D_sq = P_norm_sq[:, np.newaxis] + P_norm_sq[np.newaxis, :] - 2 * Dot
        
        violations = 0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, k + 1):
                    # Check if triangle (i,j,k) is isosceles
                    dij = D_sq[i, j]
                    dik = D_sq[i, k]
                    djk = D_sq[j, k]
                    if dij == dik or dij == djk or dik == djk:
                        violations += 1
        return violations

    # Fast violation count using vectorized operations
    def fast_count_violations(pts):
        """Count isosceles triangles efficiently."""
        n = len(pts)
        if n < 3:
            return 0
        
        P = np.array(pts, dtype=np.int64)
        P_norm_sq = np.sum(P**2, axis=1)
        Dot = P @ P.T
        D_sq = P_norm_sq[:, np.newaxis] + P_norm_sq[np.newaxis, :] - 2 * Dot
        
        # For each pair (i,j), count how many k make (i,j,k) isosceles
        violations = 0
        for i in range(n):
            for j in range(i + 1, n):
                dij = D_sq[i, j]
                # k such that D_sq[i,k] == dij or D_sq[j,k] == dij
                mask_i = (D_sq[i] == dij) & (np.arange(n) != i) & (np.arange(n) != j)
                mask_j = (D_sq[j] == dij) & (np.arange(n) != i) & (np.arange(n) != j)
                # Union of masks, but don't double count
                mask = mask_i | mask_j
                violations += np.sum(mask)
        
        # Each isosceles triangle is counted once (as the base pair)
        # But we need to be careful: an isosceles triangle has exactly one pair
        # where the two equal sides meet at a vertex. Actually, let's just count
        # all triples where any two sides are equal.
        # Simpler: just count all isosceles triangles directly
        violations = 0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    dij = D_sq[i, j]
                    dik = D_sq[i, k]
                    djk = D_sq[j, k]
                    if dij == dik or dij == djk or dik == djk:
                        violations += 1
        return violations

    # Simulated annealing to find large valid sets
    def simulated_annealing(initial_size, max_iters=5000):
        """Use SA to find a large set with few violations, then polish."""
        # Start with a random subset
        all_points = [(x, y) for x in range(N) for y in range(N)]
        random.shuffle(all_points)
        
        # Start with initial_size points
        current_pts = all_points[:initial_size]
        current_violations = fast_count_violations(current_pts)
        
        best_pts = current_pts[:]
        best_violations = current_violations
        best_count = len(current_pts)
        
        T = 10.0
        cooling_rate = 0.999
        min_T = 0.01
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if T < min_T:
                break
            
            # Random move: add, remove, or swap
            move_type = random.choice(['add', 'remove', 'swap'])
            
            new_pts = current_pts[:]
            
            if move_type == 'add' and len(new_pts) < N * N:
                # Add a random point
                p = random.choice(all_points)
                if p not in new_pts:
                    new_pts.append(p)
                else:
                    continue
            elif move_type == 'remove' and len(new_pts) > 1:
                # Remove a random point
                idx = random.randint(0, len(new_pts) - 1)
                new_pts.pop(idx)
            elif move_type == 'swap' and len(new_pts) > 0:
                # Swap a random point with another
                idx = random.randint(0, len(new_pts) - 1)
                old_pt = new_pts[idx]
                new_pt = random.choice(all_points)
                if new_pt not in new_pts:
                    new_pts[idx] = new_pt
                else:
                    continue
            else:
                continue
            
            new_violations = fast_count_violations(new_pts)
            
            # Objective: minimize violations, maximize count
            # Use a combined score
            old_score = current_violations - 0.1 * len(current_pts)
            new_score = new_violations - 0.1 * len(new_pts)
            
            delta = new_score - old_score
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_pts = new_pts
                current_violations = new_violations
                
                if current_violations == 0 and len(current_pts) > best_count:
                    best_pts = current_pts[:]
                    best_count = len(current_pts)
                elif current_violations < best_violations:
                    best_pts = current_pts[:]
                    best_violations = current_violations
                    best_count = len(current_pts)
            
            T *= cooling_rate
        
        return best_pts, best_count

    # Greedy construction
    def greedy_construct():
        pts = []
        candidates = [(x, y) for x in range(N) for y in range(N)]
        random.shuffle(candidates)
        
        for p in candidates:
            if time.time() > deadline:
                break
            if is_valid_addition(pts, p):
                pts.append(p)
        return pts

    # Local search to improve
    def local_search(initial_pts):
        pts = list(initial_pts)
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            
            # Phase 1: Try adding any valid point
            added = False
            for x in range(N):
                for y in range(N):
                    if time.time() > deadline:
                        return pts
                    p = (x, y)
                    if p not in pts and is_valid_addition(pts, p):
                        pts.append(p)
                        improved = True
                        added = True
                        break
                if added:
                    break
            
            # Phase 2: If no point could be added, try swapping
            if not improved:
                for i in range(len(pts)):
                    if time.time() > deadline:
                        return pts
                    old_pt = pts[i]
                    pts_temp = pts[:i] + pts[i+1:]
                    
                    # Try to find a replacement
                    for x in range(N):
                        for y in range(N):
                            if time.time() > deadline:
                                return pts
                            p = (x, y)
                            if p not in pts_temp and p != old_pt and is_valid_addition(pts_temp, p):
                                pts = pts_temp + [p]
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                        
        return pts

    # Phase 1: Run greedy constructions
    for trial in range(20):
        if time.time() > deadline - 60:
            break
        
        pts = greedy_construct()
        if len(pts) > best_count:
            best_points = pts
            best_count = len(pts)
            
            # Apply local search
            if time.time() < deadline - 30:
                improved_pts = local_search(pts)
                if len(improved_pts) > best_count:
                    best_points = improved_pts
                    best_count = len(improved_pts)

    # Phase 2: Run simulated annealing
    for trial in range(10):
        if time.time() > deadline - 20:
            break
        
        sa_pts, sa_count = simulated_annealing(best_count + random.randint(0, 5), max_iters=3000)
        
        if sa_count > best_count:
            best_points = sa_pts
            best_count = sa_count
            
            # Polish with local search
            if time.time() < deadline - 10:
                improved_pts = local_search(sa_pts)
                if len(improved_pts) > best_count:
                    best_points = improved_pts
                    best_count = len(improved_pts)
        
        # Also try SA from a smaller initial size to explore more
        if time.time() < deadline - 20:
            sa_pts2, sa_count2 = simulated_annealing(max(5, best_count - 10), max_iters=2000)
            if sa_count2 > best_count:
                best_points = sa_pts2
                best_count = sa_count2
                if time.time() < deadline - 10:
                    improved_pts = local_search(sa_pts2)
                    if len(improved_pts) > best_count:
                        best_points = improved_pts
                        best_count = len(improved_pts)

    # Final polish
    if best_points and time.time() < deadline - 5:
        final_pts = local_search(best_points)
        if len(final_pts) > best_count:
            best_points = final_pts
            best_count = len(final_pts)

    return best_points

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
