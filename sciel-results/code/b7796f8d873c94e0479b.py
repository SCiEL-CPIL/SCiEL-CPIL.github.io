import time
import random
import math

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    # Lattice neighbors: (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)
    # Note: In axial coordinates (a, b), neighbors are:
    # (a+1, b), (a-1, b), (a, b+1), (a, b-1), (a+1, b-1), (a-1, b+1)
    
    # Precompute neighbors for a reasonable range
    # Points must be |a|, |b| <= 64, but we can restrict search space to a smaller blob
    # Optimal shapes are compact. Radius ~6 is sufficient.
    MAX_R = 10
    points_in_range = []
    for a in range(-MAX_R, MAX_R + 1):
        for b in range(-MAX_R, MAX_R + 1):
            # Hex distance
            if (abs(a) + abs(a + b) + abs(b)) // 2 <= MAX_R:
                points_in_range.append((a, b))
                
    # Map point to index for fast lookup? 
    # Or just use a set of tuples. Set of tuples is fast enough for N=32.
    
    def neighbors(p):
        a, b = p
        return [
            (a + 1, b), 
            (a - 1, b), 
            (a, b + 1), 
            (a, b - 1), 
            (a + 1, b - 1), 
            (a - 1, b + 1)
        ]
        
    def edge_count(S):
        # S is a set of points
        count = 0
        for p in S:
            for q in neighbors(p):
                if q in S:
                    count += 1
        return count // 2
        
    def delta_edges(S, p_remove, p_add):
        # Calculate change in edge count if we remove p_remove and add p_add
        # Assumes p_remove is in S, p_add is not in S
        # Edges lost: edges from p_remove to other points in S
        lost = 0
        for q in neighbors(p_remove):
            if q in S:
                lost += 1
                
        # Edges gained: edges from p_add to other points in S (excluding p_remove if it was a neighbor, but p_remove is gone)
        # Note: If p_add is neighbor to p_remove, that edge is lost from p_remove's side and gained from p_add's side?
        # No, we remove p_remove first, then add p_add.
        # So p_add connects to S \ {p_remove}.
        gained = 0
        for q in neighbors(p_add):
            if q in S and q != p_remove:
                gained += 1
        return gained - lost

    # Initial Seed: Hexagon of radius 2 (19 points) + 13 points from Ring 3
    # This shape is known to be very close to optimal.
    seed_points = []
    for a in range(-2, 3):
        for b in range(-2, 3):
            if (abs(a) + abs(a + b) + abs(b)) // 2 <= 2:
                seed_points.append((a, b))
                
    # Get Ring 3 points
    ring3 = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if (abs(a) + abs(a + b) + abs(b)) // 2 == 3:
                ring3.append((a, b))
                
    # Sort ring3 to pick contiguous block?
    # Let's just pick 13 random ones from ring3 for the initial seed, 
    # but a contiguous block is better.
    # Let's order ring3 by angle.
    def angle(p):
        a, b = p
        # Convert to Cartesian: x = a + b/2, y = b*sqrt(3)/2
        x = a + 0.5 * b
        y = math.sqrt(3) / 2 * b
        return math.atan2(y, x)
        
    ring3_sorted = sorted(ring3, key=angle)
    
    # Pick 13 contiguous points from the sorted ring
    # Start at index 0
    initial_set = set(seed_points)
    for i in range(13):
        initial_set.add(ring3_sorted[i])
        
    best_set = initial_set
    best_score = edge_count(best_set)
    
    # Simulated Annealing
    rng = random.Random(42)
    current_set = set(initial_set)
    current_score = best_score
    
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.9999
    
    # List of all candidate points for swapping
    # We restrict candidates to points within distance 4 of the centroid or existing points to keep search local
    # Actually, just use points_in_range but filter by distance to current set centroid?
    # For speed, let's just use a precomputed list of "nearby" points.
    # Since N=32, we can afford to check swaps against a subset.
    
    # To make it faster, we only consider swapping a point p in current_set with a point q 
    # such that q is within hex distance 3 of p.
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        improved = False
        # Try to improve
        for _ in range(100): # Local search steps per T
            if time.time() - start_time > time_limit:
                return [list(t) for t in best_set]
                
            p = rng.choice(list(current_set))
            # Find candidate q
            candidates = []
            for q in points_in_range:
                if q in current_set:
                    continue
                # Check if q is close to p
                da = p[0] - q[0]
                db = p[1] - q[1]
                if (abs(da) + abs(da + db) + abs(db)) // 2 <= 3:
                    candidates.append(q)
                    
            if not candidates:
                continue
                
            q = rng.choice(candidates)
            
            delta = delta_edges(current_set, p, q)
            
            if delta > 0 or rng.random() < math.exp(delta / T):
                current_set.discard(p)
                current_set.add(q)
                current_score += delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(current_set)
                    improved = True
                    
                    if best_score == 76:
                        return [list(t) for t in best_set]
                        
        T *= cooling_rate
        
        # If no improvement for a while, restart with a perturbation?
        # For now, just cool down.
        
    return [list(t) for t in best_set]

