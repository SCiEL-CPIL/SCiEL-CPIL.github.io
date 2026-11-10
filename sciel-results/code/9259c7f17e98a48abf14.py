"""No-isosceles set: maximize points in [0,64)^2 with no three forming an isosceles triangle."""
import time
import random
import math

N = 64
TIME_LIMIT = 280  # Safety margin below 330s

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    random.seed(42)
    
    # All grid points
    all_points = [(x, y) for x in range(N) for y in range(N)]
    n_total = N * N
    
    # Precompute squared distances between all pairs (memory: 4096*4096 = 16M ints, too big)
    # Instead, compute on the fly but cache per-point distances
    
    def dist2(a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return dx * dx + dy * dy
    
    def is_valid_addition(p, pts, pts_set):
        """Check if adding p to pts creates any isosceles triangle."""
        k = len(pts)
        if k == 0:
            return True
        if k == 1:
            return True
        # For each existing point, check if p forms isosceles with it and another
        # Isosceles with vertex at p: two points at same distance from p
        # Isosceles with vertex at existing point q: dist(q,p) == dist(q,r) for some r != p
        # Isosceles with base at p-q: dist(p,q) == dist(p,r) == dist(q,r) -- this is equilateral, subset of isosceles
        
        # Check: for each pair (a,b) in pts, does p make isosceles?
        # Case 1: vertex at p: dist(p,a) == dist(p,b)
        # Case 2: vertex at a: dist(a,p) == dist(a,b)  [a is existing, p is new]
        # Case 3: vertex at b: dist(b,p) == dist(b,a)  [same as case 2 with roles swapped]
        
        # Efficient: collect distances from p to all pts
        dists_from_p = {}
        for a in pts:
            d = dist2(p, a)
            if d in dists_from_p:
                return False  # Two points at same distance from p -> isosceles with vertex p
            dists_from_p[d] = a
        
        # Now check: for each existing point a, is dist(a,p) equal to dist(a,b) for some b != p?
        # dist(a,p) is in dists_from_p keys
        for a in pts:
            dap = dist2(a, p)
            # Check if any other point b (b != a, b != p) has dist(a,b) == dap
            for b in pts:
                if b is a:
                    continue
                if dist2(a, b) == dap:
                    return False
        return True
    
    def full_validity_check(pts):
        """Verify the entire set has no isosceles triangles."""
        k = len(pts)
        if k < 3:
            return True
        for i in range(k):
            for j in range(i+1, k):
                for l in range(j+1, k):
                    a, b, c = pts[i], pts[j], pts[l]
                    d_ab = dist2(a, b)
                    d_bc = dist2(b, c)
                    d_ca = dist2(c, a)
                    if d_ab == d_bc or d_bc == d_ca or d_ca == d_ab:
                        return False
        return True
    
    # Greedy construction to get a starting set
    def greedy_construct():
        pts = []
        pts_set = set()
        # Try random orderings
        candidates = all_points[:]
        random.shuffle(candidates)
        
        for p in candidates:
            if len(pts) >= 50:  # Cap for speed
                break
            if is_valid_addition(p, pts, pts_set):
                pts.append(p)
                pts_set.add(p)
        return pts
    
    # Simulated annealing
    def simulated_annealing(initial_pts, max_time):
        pts = list(initial_pts)
        if not pts:
            return pts
        
        best_pts = list(pts)
        best_len = len(pts)
        current_temp = 10.0
        cooling_rate = 0.999
        min_temp = 0.01
        
        # Precompute some helper data
        def get_neighbors(p, pts):
            """Find points that conflict with p (would form isosceles if both present with some third)."""
            # This is expensive; instead just try random swaps/removals
            return []
        
        while time.time() - start_time < max_time:
            if current_temp < min_temp:
                break
            
            # Random move: try to add a random point, remove a random point, or swap
            move_type = random.random()
            new_pts = list(pts)
            
            if move_type < 0.4 and len(new_pts) < 60:
                # Try adding a random point
                p = random.choice(all_points)
                if p not in new_pts:
                    new_pts.append(p)
                    # Check validity
                    if not is_valid_addition(p, pts, set(pts)):
                        # Invalid, reject
                        continue
            elif move_type < 0.7:
                # Try removing a random point
                if len(new_pts) > 1:
                    idx = random.randint(0, len(new_pts) - 1)
                    new_pts.pop(idx)
            else:
                # Swap: remove one, add another
                if len(new_pts) > 0:
                    idx = random.randint(0, len(new_pts) - 1)
                    removed = new_pts.pop(idx)
                    p = random.choice(all_points)
                    if p not in new_pts:
                        new_pts.append(p)
                        # Check validity of the new point with remaining
                        remaining = [x for x in new_pts if x != p]
                        if not is_valid_addition(p, remaining, set(remaining)):
                            # Invalid
                            new_pts.remove(p)
                            new_pts.insert(idx, removed)
                            continue
            
            # Evaluate
            new_len = len(new_pts)
            if new_len > best_len:
                best_pts = list(new_pts)
                best_len = new_len
            
            # Accept/reject based on simulated annealing
            # Score = number of points
            delta_score = new_len - len(pts)
            if delta_score >= 0:
                pts = new_pts
            else:
                # Reject with probability based on temperature
                accept_prob = math.exp(delta_score / current_temp)
                if random.random() < accept_prob:
                    pts = new_pts
            
            current_temp *= cooling_rate
        
        return best_pts
    
    # Multiple restarts with greedy + SA
    best_overall = []
    best_overall_len = 0
    
    # First, try several greedy constructions
    for restart in range(20):
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        
        pts = greedy_construct()
        if len(pts) > best_overall_len:
            best_overall = pts
            best_overall_len = len(pts)
        
        # Verify
        if full_validity_check(pts):
            if len(pts) > best_overall_len:
                best_overall = pts
                best_overall_len = len(pts)
    
    # Now apply SA to the best greedy result
    if best_overall and time.time() - start_time < TIME_LIMIT - 20:
        sa_result = simulated_annealing(best_overall, TIME_LIMIT - 10)
        if full_validity_check(sa_result) and len(sa_result) > best_overall_len:
            best_overall = sa_result
            best_overall_len = len(sa_result)
    
    # Final verification
    if not full_validity_check(best_overall):
        # Fallback: return empty or minimal valid set
        return [(0, 0)]
    
    return best_overall
# EVOLVE-BLOCK-END
