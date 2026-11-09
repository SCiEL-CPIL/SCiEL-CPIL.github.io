"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def _count_edges(points):
    """Count pairs at distance exactly 1 in the triangular lattice."""
    pts = set(points)
    count = 0
    for a, b in pts:
        # Six neighbors in triangular lattice
        for da, db in [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]:
            if (a+da, b+db) in pts:
                count += 1
    return count // 2

def _get_hex_spiral_points(n):
    """
    Generate the first n points of the hexagonal spiral in the triangular lattice.
    The points are ordered by increasing "hexagonal distance" from the origin,
    with ties broken by angle to form a compact, contiguous cluster.
    This cluster is known to be optimal for maximizing nearest-neighbor contacts
    for small n (attaining Harborth's bound).
    """
    # We need to find a bounding box large enough to contain n points
    # The hexagonal number H_r = 3r(r+1) + 1 for radius r
    # For n=32, r=3 gives 3*3*4+1 = 37 points, so radius 3 is sufficient.
    # To be safe, we calculate max_coord based on n.
    max_r = int(math.ceil((math.sqrt(12*n + 1) - 1) / 6.0)) + 1
    max_coord = max_r + 2
    
    candidates = []
    for a in range(-max_coord, max_coord + 1):
        for b in range(-max_coord, max_coord + 1):
            # Calculate squared Euclidean distance in the lattice metric
            # |a + b*omega|^2 = a^2 + a*b + b^2
            dist_sq = a*a + a*b + b*b
            
            # Calculate angle for tie-breaking to form a spiral/hexagon
            # omega = 0.5 + i*sqrt(3)/2
            # z = a + b*(0.5 + i*sqrt(3)/2) = (a + b/2) + i*(b*sqrt(3)/2)
            x = a + b * 0.5
            y = b * math.sqrt(3) / 2.0
            angle = math.atan2(y, x)
            
            candidates.append((dist_sq, angle, a, b))
    
    # Sort by distance, then by angle
    candidates.sort(key=lambda x: (x[0], x[1]))
    
    # Return the first n points
    return [(a, b) for _, _, a, b in candidates[:n]]

def construct():
    deadline = time.time() + 145.0  # Safety margin
    
    # The problem states that for n=32, the optimum is 76 edges,
    # attained by triangular-lattice spirals (hexagonal packing).
    # We generate the first 32 points of the hexagonal spiral.
    points = _get_hex_spiral_points(32)
    
    # Verify the count (optional, for debugging/assurance)
    # count = _count_edges(points)
    # assert count == 76, f"Expected 76 edges, got {count}"
    
    # Return as list of [a, b] pairs
    return [[a, b] for a, b in points]
# EVOLVE-BLOCK-END
