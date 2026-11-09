import math
import time

def construct():
    # Time guard
    start_time = time.time()
    deadline = 110.0  # 10 seconds margin

    # 1. Generate core hexagon (radius 2, 19 points)
    # Points (a, b) such that max(|a|, |b|, |a+b|) <= R
    # Wait, standard hexagon in axial coords (q, r) where x = q + r/2, y = r*sqrt(3)/2
    # Distance squared in axial coords (a, b) corresponding to a + b*omega:
    # d^2 = a^2 + ab + b^2
    # Hexagon of radius R: max(|a|, |b|, |a+b|) <= R? 
    # Let's check.
    # If a=2, b=0, d^2=4. dist=2.
    # If a=1, b=1, d^2=3. dist=sqrt(3).
    # The condition for a hexagon of "radius" R in terms of max distance is different.
    # Let's just use the definition: all points with d^2 <= R^2? No, that's a circle.
    # A regular hexagon on triangular lattice with side length S has vertices at distance S from center.
    # Side length S=2 (radius 2) has points:
    # Center (0,0).
    # Ring 1 (6 pts): (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1).
    # Ring 2 (12 pts): (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2)? No.
    # Let's generate points by distance from center.
    # d^2 = a^2 + ab + b^2.
    # d=0: (0,0)
    # d=1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # d=sqrt(3): (1,1), (-1,2), (-2,1), (-1,-1), (1,-2), (2,-1)
    # d=2: (2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2)
    # d=sqrt(7): (2,1), (1,2), (-2,3)...
    # d=3: (3,0)...
    
    # A "hexagon" of side length S (number of points on edge) contains S^2 points? No.
    # Side length 1 (1 point): 1 pt.
    # Side length 2 (2 points on edge): 7 pts. (Center + 6)
    # Side length 3 (3 points on edge): 19 pts. (7 + 12)
    # Side length 4 (4 points on edge): 37 pts. (19 + 18)
    
    # So for n=32, we take the 19-point hexagon (side 3) and add 13 points from the next ring (side 4 shell).
    # The next ring has 18 points.
    
    core = []
    # Generate all points with d^2 <= 4 (radius 2) -> 19 points
    for a in range(-3, 4):
        for b in range(-3, 4):
            d2 = a*a + a*b + b*b
            if d2 <= 4:
                core.append((a, b))
    
    # Check count
    if len(core) != 19:
        # Fallback to simple grid if generation fails
        return [[a, b] for a in range(6) for b in range(6)][:32]

    # Generate shell points (d^2 > 4 and d^2 <= 9? No, just the next ring)
    # The ring at "distance" 3 (side length 4) has points with max(|a|,|b|,|a+b|) == 3?
    # Let's just collect points with d^2 in (4, 7] or similar?
    # Actually, the shell of the 19-hexagon consists of points adjacent to the core but not in core.
    # Let's find neighbors of core points that are not in core.
    shell_candidates = set()
    directions = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    for p in core:
        for d in directions:
            np = (p[0]+d[0], p[1]+d[1])
            if np not in core:
                shell_candidates.add(np)
                
    shell = list(shell_candidates)
    # There should be 18 candidates.
    if len(shell) < 13:
         # Error handling
         return [[a, b] for a in range(8) for b in range(5)][:32]

    # We need to pick 13 points from the 18 to maximize edges.
    # To keep it compact, pick consecutive points on the shell.
    # Sort shell by angle around origin.
    def angle(p):
        return math.atan2(p[1], p[0])
    
    shell.sort(key=angle)
    
    best_edges = -1
    best_set = core[:]
    
    # Try all 18 rotations of taking 13 consecutive points
    n_shell = len(shell)
    for start in range(n_shell):
        selected_shell = []
        for i in range(13):
            selected_shell.append(shell[(start + i) % n_shell])
        
        current_set = core + selected_shell
        # Calculate edges
        edges = 0
        pts = set(current_set)
        for p in current_set:
            for d in directions:
                np = (p[0]+d[0], p[1]+d[1])
                if np in pts:
                    edges += 1
        edges //= 2 # Undirected
        
        if edges > best_edges:
            best_edges = edges
            best_set = current_set[:]
            
        # Check time
        if time.time() > deadline:
            break
            
    # If best_edges < 76, we might need a local search, but 76 is expected.
    # Convert to list of lists
    result = [[a, b] for a, b in best_set]
    
    # Final check: distinct, bounds
    # Bounds |a|, |b| <= 64 are easily met.
    # Distinctness is guaranteed by set logic.
    
    return result
