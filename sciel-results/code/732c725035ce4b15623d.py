"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def _count_edges(points):
    """Count pairs at distance exactly 1 in the triangular lattice.
    Distance squared between (a1,b1) and (a2,b2) is a1^2 + a1*a2 + a2^2 + b1^2 + b1*b2 + b2^2 - 2*(a1*b2 + a2*b1)
    Actually, the distance in the triangular lattice with omega = exp(i*pi/3) = 1/2 + i*sqrt(3)/2:
    |a + b*omega|^2 = a^2 + a*b + b^2
    So distance squared between (a1,b1) and (a2,b2) is:
    (a1-a2)^2 + (a1-a2)*(b1-b2) + (b1-b2)^2
    """
    pts = set(points)
    count = 0
    for a, b in pts:
        # Six neighbors in triangular lattice
        for da, db in [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]:
            if (a+da, b+db) in pts:
                count += 1
    return count // 2

def _hex_spiral_order(max_r=100):
    """Generate points in hexagonal spiral order from center.
    Ring 0: (0,0)
    Ring r: points at hexagonal distance r from origin.
    """
    # Hexagonal distance: max(|a|, |b|, |a+b|) in axial coordinates? 
    # Actually for triangular lattice with basis (1,0) and omega, the hexagonal distance
    # (minimum number of unit steps) from origin to (a,b) is:
    # d = max(|a|, |b|, |a+b|) ... let me verify.
    # Actually the distance from (0,0) to (a,b) is sqrt(a^2 + a*b + b^2).
    # The "hexagonal ring" r contains points where the minimum path length is r.
    # Let's just use the Euclidean-ish distance and sort by it, breaking ties by angle.
    
    points = []
    for r in range(max_r):
        if r == 0:
            points.append((0, 0))
            continue
        # Generate all points at hexagonal distance r
        # A point (a,b) has hexagonal distance r if it can be reached in r unit steps.
        # The six directions are: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
        # Let's enumerate by walking around the hexagon.
        # Start at (r, 0) and go around.
        # The hexagon of distance r has 6*r points.
        # Corners: (r,0), (r,r), (0,r), (-r,r), (-r,0), (0,-r) ... not quite.
        # Let me think differently. Use the fact that the lattice is Z^2 with the given metric.
        # I'll just enumerate a range and filter by distance.
        pass
    
    # Simpler approach: enumerate all points in a bounding box, sort by distance from origin,
    # then by angle, and take the first 32.
    max_coord = 20
    all_points = []
    for a in range(-max_coord, max_coord+1):
        for b in range(-max_coord, max_coord+1):
            dist_sq = a*a + a*b + b*b
            all_points.append((dist_sq, a, b))
    
    # Sort by distance, then by angle for deterministic ordering
    def sort_key(x):
        dist_sq, a, b = x
        angle = math.atan2(math.sqrt(3)/2 * b, a + b/2)
        return (dist_sq, angle)
    
    all_points.sort(key=sort_key)
    return [(a, b) for _, a, b in all_points]

def construct():
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    
    # Generate a large set of candidate points in spiral order
    spiral_points = _hex_spiral_order(max_r=100)
    
    # Take the first 32 points - this should be a dense hexagonal cluster
    best_points = spiral_points[:32]
    best_count = _count_edges(best_points)
    
    # Try to improve by local search
    # Convert to set for fast lookup
    best_set = set(best_points)
    
    # Local search: try swapping each point with a nearby unoccupied point
    # Generate candidate replacement points near the cluster
    candidate_replacements = []
    for a in range(-15, 16):
        for b in range(-15, 16):
            if (a, b) not in best_set:
                candidate_replacements.append((a, b))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(best_points)):
            if time.time() > deadline:
                break
            old_pt = best_points[i]
            for new_pt in candidate_replacements:
                # Check if new_pt is not already in the set (except for the one being replaced)
                if new_pt == old_pt:
                    continue
                if new_pt in best_set:
                    continue
                # Check that new_pt is at distance >= 1 from all other points
                valid = True
                for j in range(len(best_points)):
                    if i == j:
                        continue
                    a1, b1 = new_pt
                    a2, b2 = best_points[j]
                    da, db = a1-a2, b1-b2
                    dist_sq = da*da + da*db + db*db
                    if dist_sq < 1:  # Must be at distance >= 1
                        valid = False
                        break
                if not valid:
                    continue
                # Compute new edge count
                new_set = best_set - {old_pt} | {new_pt}
                new_count = _count_edges(list(new_set))
                if new_count > best_count:
                    best_points[i] = new_pt
                    best_set = new_set
                    best_count = new_count
                    improved = True
                    break
    
    # Return as list of [a, b] pairs
    return [[a, b] for a, b in best_points]
# EVOLVE-BLOCK-END
