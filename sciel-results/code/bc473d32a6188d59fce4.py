import time
import random
import numpy as np

N = 64
DEADLINE = time.time() + 290  # Safety margin before 300s limit

def construct():
    start_time = time.time()
    
    # Precompute all points
    all_points = [(x, y) for y in range(N) for x in range(N)]
    total_points = N * N
    
    # For efficiency, precompute squared distances between all pairs
    # dist2[i][j] = squared distance between point i and point j
    # This is a 4096x4096 matrix - might be large but manageable
    # Actually, let's be smarter: compute on the fly but cache
    
    # Greedy construction with randomization
    best_set = []
    
    # Try multiple random orderings and greedy passes
    random.seed(42)
    
    for attempt in range(200):
        if time.time() > DEADLINE:
            break
        
        # Random shuffle of points
        order = list(range(total_points))
        random.shuffle(order)
        
        # Greedy: add points if compatible
        current_set = []
        # For fast checking, maintain for each pair of points in set, 
        # the forbidden points that would form isosceles triangles
        
        # We'll use a set of forbidden point indices
        forbidden = set()
        
        for idx in order:
            if time.time() > DEADLINE:
                break
            
            p = all_points[idx]
            
            if idx in forbidden:
                continue
            
            # Check if adding p creates an isosceles triangle with any pair in current_set
            # For each pair (a, b) in current_set, check if triangle (a, b, p) is isosceles
            # This is O(k^2) where k = len(current_set)
            
            compatible = True
            k = len(current_set)
            
            # For each existing point a, check if there's another point b such that
            # |ap| = |bp| (isosceles with apex at... wait, no)
            # Triangle (a, b, p) is isosceles if:
            #   |ab| = |ap| (apex at a)
            #   |ab| = |bp| (apex at b)  
            #   |ap| = |bp| (apex at p)
            
            # More efficiently: for each existing point a, compute d_ap = |ap|^2
            # For the triangle to be isosceles with apex at a, we need another point b with |ab|^2 = d_ap
            # With apex at p, we need |ap|^2 = |bp|^2 for some b != a
            # With apex at b, we need |ab|^2 = |bp|^2
            
            # Let's just check all pairs for now - it's O(k^2) per insertion
            # But k will be small initially
            
            for i in range(k):
                a = current_set[i]
                ax, ay = a
                dx_ap = p[0] - ax
                dy_ap = p[1] - ay
                d_ap2 = dx_ap * dx_ap + dy_ap * dy_ap
                
                for j in range(i + 1, k):
                    b = current_set[j]
                    bx, by = b
                    dx_ab = bx - ax
                    dy_ab = by - ay
                    d_ab2 = dx_ab * dx_ab + dy_ab * dy_ab
                    
                    dx_bp = p[0] - bx
                    dy_bp = p[1] - by
                    d_bp2 = dx_bp * dx_bp + dy_bp * dy_bp
                    
                    # Check three conditions for isosceles
                    if d_ap2 == d_ab2 or d_bp2 == d_ab2 or d_ap2 == d_bp2:
                        compatible = False
                        break
                if not compatible:
                    break
            
            if compatible:
                current_set.append(p)
                
                # Update forbidden set: for each existing point a in current_set (before adding p),
                # find all points q such that (a, some other point b in current_set, q) would be isosceles
                # Actually, we need to forbid points that would form isosceles with ANY pair in current_set
                # This is expensive. Let's skip the forbidden set optimization for now and just do direct check.
        
        if len(current_set) > len(best_set):
            best_set = current_set[:]
        
        # Local improvement: try to add more points or swap
        # Try adding points not in current_set
        current_indices = set(id(p) for p in best_set)  # This won't work, let's use positions
        current_pos_set = set(best_set)
        
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            for p in all_points:
                if p in current_pos_set:
                    continue
                
                # Check if p can be added to best_set
                compatible = True
                k = len(best_set)
                for i in range(k):
                    a = best_set[i]
                    ax, ay = a
                    dx_ap = p[0] - ax
                    dy_ap = p[1] - ay
                    d_ap2 = dx_ap * dx_ap + dy_ap * dy_ap
                    
                    for j in range(i + 1, k):
                        b = best_set[j]
                        bx, by = b
                        dx_ab = bx - ax
                        dy_ab = by - ay
                        d_ab2 = dx_ab * dx_ab + dy_ab * dy_ab
                        
                        dx_bp = p[0] - bx
                        dy_bp = p[1] - by
                        d_bp2 = dx_bp * dx_bp + dy_bp * dy_bp
                        
                        if d_ap2 == d_ab2 or d_bp2 == d_ab2 or d_ap2 == d_bp2:
                            compatible = False
                            break
                    if not compatible:
                        break
                
                if compatible:
                    best_set.append(p)
                    current_pos_set.add(p)
                    improved = True
        
        if len(best_set) >= 15:  # Good enough, stop early
            break
    
    # Final verification and return
    return best_set
