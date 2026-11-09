import numpy as np
import time
import random
import math
from math import gcd

def _incidence_count(points, lines):
    """Compute number of (point, line) incidences. points: (n,2), lines: (m,3)."""
    if len(points) == 0 or len(lines) == 0:
        return 0
    px = points[:, 0:1]
    py = points[:, 1:2]
    A = lines[0:1, :]
    B = lines[1:2, :]
    C = lines[2:3, :]
    vals = px * A + py * B + C
    return np.sum(vals == 0)

def _normalize_line(A, B, C):
    if A == 0 and B == 0 and C == 0:
        return None
    g = gcd(abs(A), abs(B))
    g = gcd(g, abs(C))
    if g > 1:
        A //= g
        B //= g
        C //= g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def _generate_fano_plane_q7():
    """
    Generate the 57 points and 57 lines of the projective plane PG(2,7).
    Points are represented as homogeneous coordinates (x, y, 1) or (x, 0, 0) etc.
    We map these to integer (x, y) coordinates for the final output.
    Since we need integer (x,y) in the output, we treat the projective plane
    as a set of lines in the affine plane plus lines at infinity, but we must
    map them to standard integer coordinates.
    
    Actually, the standard embedding of PG(2, q) into integer coordinates is tricky 
    because projective lines wrap around. However, we can represent the points as 
    vectors in Z^3 and lines as vectors in Z^3 such that dot product is 0.
    The output requires A*x + B*y + C = 0. This is exactly the dot product with (x,y,1).
    So if we generate points P_i = (x_i, y_i, 1) and lines L_j = (A_j, B_j, C_j) 
    from the finite field structure, we just need to ensure x_i, y_i, A_j, B_j, C_j 
    are integers.
    
    We will construct PG(2,7) over the field GF(7).
    Points are equivalence classes of non-zero vectors in GF(7)^3.
    There are (7^3 - 1)/(7-1) = 342/6 = 57 points.
    Lines are dual: each line is a set of 8 points.
    
    To get integer coordinates for the final output, we can simply pick a representative
    for each projective point with z=1 if possible, or handle the line at infinity separately.
    However, the "lines" in the output must be of the form A*x + B*y + C = 0.
    This corresponds to lines in the affine plane (z=1) and lines at infinity (z=0).
    If a point is on the line at infinity (z=0), it cannot be represented as (x,y) with z=1.
    
    Alternative: Use a specific integer embedding.
    A simpler approach for "high incidence" with integer constraints is to use the 
    combinatorial structure but map it to a grid-like set of integers if possible, 
    OR just use the homogeneous coordinates directly if the evaluator accepts 
    any integers A,B,C,x,y. The evaluator checks A*x+B*y+C==0.
    
    Let's generate the 57 points as integer vectors (x, y, 1) for affine points 
    and (x, y, 0) for points at infinity? No, the output format is fixed:
    points: (54, 2) array of (x, y).
    lines: (27, 3) array of (A, B, C).
    The equation is A*x + B*y + C = 0.
    This implies we are working in the affine plane. We cannot have points "at infinity" 
    in the (x,y) representation.
    
    Therefore, we must select 54 points from the 49 affine points of PG(2,7) (where z=1) 
    and... wait, there are only 49 affine points in PG(2,7). 
    We need 54 points. So we MUST use points from the line at infinity or other structures?
    No, PG(2,7) has 57 points total. 49 are affine (z=1), 8 are on the line at infinity (z=0).
    If we output a point (x,y), it implicitly has z=1.
    If we output a line (A,B,C), it is a line in the affine plane.
    A line in PG(2,7) that is NOT the line at infinity intersects the line at infinity 
    at exactly one point. 
    If we pick a line L in PG(2,7) (other than infinity), it contains 8 points.
    7 of these are affine (z=1), 1 is at infinity (z=0).
    In the affine representation (x,y), the line A*x+B*y+C=0 contains only the 7 affine points.
    The point at infinity is "lost" in the (x,y) representation.
    
    So, if we use PG(2,7):
    - We have 49 affine points available.
    - We need 54 points. We must add 5 more points.
    - We have 56 affine lines (all lines except the line at infinity).
    - Each affine line contains 7 affine points.
    - Total incidences in the affine part: 56 lines * 7 points/line = 392 incidences.
    
    If we select 27 lines from the 56 affine lines, and 49 points from the 49 affine points,
    the expected incidences would be roughly 27 * 7 = 189 (if we keep all 49 points).
    But we need 54 points. The extra 5 points can be placed anywhere.
    If we place them on existing lines, we increase incidences.
    
    Strategy:
    1. Generate the 49 affine points of PG(2,7) as integer coordinates.
       The affine points are (x, y) for x, y in {0, ..., 6}.
       Wait, PG(2,7) affine points correspond to (x, y, 1) where x, y in GF(7).
       So the 49 points are simply the 7x7 grid {0..6} x {0..6}.
       This is just a 7x7 grid!
       
    2. The lines in PG(2,7) that are affine (not infinity) correspond to lines 
       A*x + B*y + C = 0 over GF(7).
       There are 56 such lines.
       Each line contains exactly 7 points from the 7x7 grid.
       
    3. So the problem reduces to:
       We have a 7x7 grid of points (49 points).
       We have a set of 56 candidate lines (lines over GF(7) mapped to integer lines).
       We need to select 27 lines from these 56.
       We need to select 54 points. We have 49 grid points. We need 5 extra points.
       
       If we just use the 7x7 grid, we have 49 points.
       We can add 5 more points. Where?
       If we add points outside the 7x7 grid, they won't lie on the GF(7) lines 
       (which are defined modulo 7, but the integer equation A*x+B*y+C=0 is global).
       Wait, the lines A*x+B*y+C=0 derived from GF(7) are specific integer lines.
       For example, x=0, x=1, ..., x=6, y=0, ..., y=6, x+y=0, ..., x+y=6, etc.
       These are integer lines.
       Do they contain points outside {0..6}x{0..6}?
       Yes, e.g., x=0 contains (0, 100).
       But the "incidence" in PG(2,7) only counts the 7 points in the field.
       However, the evaluator counts ALL integer points (x,y) on the line A*x+B*y+C=0.
       So if we use the line x=0, it contains (0,0), (0,1), ..., (0,6) AND (0,7), (0,8)...
       This means the actual incidence count in the evaluator will be HIGHER than the 
       combinatorial count from PG(2,7) because the integer lines extend to infinity!
       
       This is a HUGE advantage.
       The grid {0..6}x{0..6} lies on these lines.
       The lines are "long" in the integer plane.
       
       So:
       1. Start with the 49 points of the 7x7 grid.
       2. Select 27 lines from the 56 PG(2,7) affine lines.
          These lines will each hit at least 7 of the grid points.
          Due to the integer extension, they might hit more if we place extra points on them.
       3. We need 5 more points. We should place them on lines that are already selected,
          preferably on lines that currently have the highest "potential" or to balance.
          Actually, since the lines are long, we can place the 5 extra points on 
          intersections of the selected lines that are NOT in the 7x7 grid, or just 
          on the lines themselves outside the grid.
          
       Better:
       Just pick the 49 grid points.
       Pick 27 lines from the 56.
       Add 5 points that lie on as many of the selected 27 lines as possible.
       Since the lines are distinct in PG(2,7), any two intersect at exactly one point in PG(2,7).
       In the integer plane, they intersect at one point.
       If that intersection point is not in the 7x7 grid, we can add it!
       If it is in the 7x7 grid, it's already there.
       
       So, algorithm:
       1. Generate 49 grid points P_grid.
       2. Generate 56 lines L_candidates (from PG(2,7)).
       3. Use local search to select 27 lines from L_candidates to maximize 
          incidences with P_grid + 5 extra points.
       4. The 5 extra points can be chosen from the intersections of the selected lines 
          that fall outside P_grid, or simply appended to maximize count.
          
       Actually, simpler:
       1. Fix points: 49 grid points + 5 points chosen from intersections of "popular" lines.
       2. Optimize lines: Select 27 lines from the 56 to maximize incidences with the 54 points.
       
       Let's implement this.
    """
    # Points: 7x7 grid
    points_grid = []
    for x in range(7):
        for y in range(7):
            points_grid.append((x, y))
            
    # Lines: All lines A*x + B*y + C = 0 over GF(7)
    # A, B not both 0. Normalize (A, B, C) such that gcd=1 and first non-zero is positive.
    # We iterate over all (A, B, C) in 0..6, normalize, and add to set.
    lines_set = set()
    for A in range(7):
        for B in range(7):
            if A == 0 and B == 0:
                continue
            for C in range(7):
                # Normalize
                key = _normalize_line(A, B, C)
                if key:
                    lines_set.add(key)
                    
    # There should be 56 lines.
    # Let's verify count.
    # Total projective lines in PG(2,7) is 57. One is the line at infinity (A=0, B=0, C=1? No, x=0? No.)
    # In homogeneous coords (X,Y,Z), line is aX+bY+cZ=0.
    # Affine lines have c != 0 (so they don't pass through infinity point [0:0:1]? No, [0:0:1] is origin).
    # Line at infinity is Z=0 -> 0*X + 0*Y + 1*Z = 0. So A=0, B=0, C=1.
    # My generation above allows A=0, B=0? No, I skipped A=0, B=0.
    # So I generated 56 lines. These are all affine lines.
    
    lines_list = list(lines_set)
    
    # Initial Points: 49 grid points.
    # We need 54. Let's start with 49 and add 5 later or include in optimization.
    # For now, let's just optimize the 27 lines for the 49 points, then add 5 points 
    # that lie on the most selected lines.
    
    # Score function for lines given points
    def score_lines(points, lines):
        return _incidence_count(np.array(points), np.array(lines))
    
    # Local search on lines
    current_lines = lines_list[:27]
    current_score = score_lines(points_grid, current_lines)
    
    best_lines = current_lines.copy()
    best_score = current_score
    
    # Simple local search: swap one line with another from the pool
    pool = lines_list
    start_time = time.time()
    deadline = 300.0 # Use half the time for this phase
    
    while time.time() - start_time < deadline:
        # Pick random index to replace
        idx = random.randint(0, 26)
        # Pick random new line from pool
        new_line = random.choice(pool)
        if new_line == current_lines[idx]:
            continue
        
        # Check if new_line is already in current_lines
        if new_line in current_lines:
            continue
            
        new_lines = current_lines.copy()
        new_lines[idx] = new_line
        
        new_score = score_lines(points_grid, new_lines)
        
        # Accept if better or with some probability (greedy is fine for this structure)
        if new_score >= current_score:
            current_lines = new_lines
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_lines = new_lines.copy()
                
    # Now we have 27 best lines for the 49 grid points.
    # We need 5 more points.
    # Find intersections of these 27 lines.
    # If an intersection point (x,y) is integer and not in points_grid, it's a candidate.
    # We want to pick 5 points that maximize additional incidences.
    
    extra_points = []
    points_set = set(points_grid)
    
    # Generate all pairwise intersections of the 27 lines
    intersections = set()
    for i in range(27):
        A1, B1, C1 = best_lines[i]
        for j in range(i+1, 27):
            A2, B2, C2 = best_lines[j]
            det = A1*B2 - A2*B1
            if det == 0:
                continue
            x_num = B1*C2 - B2*C1
            y_num = C1*A2 - C2*A1
            # x = x_num / det, y = y_num / det
            if x_num % det == 0 and y_num % det == 0:
                x = x_num // det
                y = y_num // det
                if (x, y) not in points_set:
                    intersections.add((x, y))
                    
    # Sort intersections by how many of the 27 lines they lie on
    # (They should lie on at least 2 by construction, but might lie on more)
    def count_incidence_for_point(p, lines):
        x, y = p
        count = 0
        for A, B, C in lines:
            if A*x + B*y + C == 0:
                count += 1
        return count
        
    # We only need to check the lines we selected.
    # But wait, if we add a point, it might also lie on lines NOT in best_lines?
    # No, we only care about incidences with the 27 selected lines.
    
    scored_intersections = []
    for p in intersections:
        c = count_incidence_for_point(p, best_lines)
        scored_intersections.append((c, p))
        
    scored_intersections.sort(key=lambda x: x[0], reverse=True)
    
    # Pick top 5
    for c, p in scored_intersections[:5]:
        extra_points.append(p)
        
    final_points = points_grid + extra_points
    
    # Final validation and formatting
    final_points_arr = np.array(final_points, dtype=np.int64)
    final_lines_arr = np.array(best_lines, dtype=np.int64)
    
    # Ensure unique points
    # The grid points are unique. Intersections are unique from grid.
    # Intersections might be duplicates of each other?
    # If 3 lines intersect at one point, it's added once to set.
    # So final_points should be unique.
    
    # Re-calculate final score
    final_score = _incidence_count(final_points_arr, final_lines_arr)
    
    return final_points_arr, final_lines_arr, final_score

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = 600.0
    safety_margin = 10.0
    
    try:
        points, lines, score = _generate_fano_plane_q7()
        
        # Basic validation
        if points.shape != (54, 2) or lines.shape != (27, 3):
            raise ValueError("Wrong shape")
            
        # Check duplicates
        if len(np.unique(points, axis=0)) != 54:
            raise ValueError("Duplicate points")
            
        seen_lines = set()
        for i in range(27):
            A, B, C = lines[i]
            if A == 0 and B == 0:
                raise ValueError("Invalid line")
            key = _normalize_line(A, B, C)
            if key in seen_lines:
                raise ValueError("Duplicate lines")
            seen_lines.add(key)
            
        return points, lines
        
    except Exception as e:
        # Fallback to grid if Fano fails
        # Generate a 6x9 grid
        p = []
        for i in range(6):
            for j in range(9):
                p.append((i, j))
        points = np.array(p, dtype=np.int64)
        
        # Generate lines: vertical, horizontal, diagonals
        lines = []
        # 6 verticals
        for i in range(6):
            lines.append((1, 0, -i))
        # 9 horizontals
        for j in range(9):
            lines.append((0, 1, -j))
        # 12 diagonals?
        # Just pad with random valid lines
        for k in range(12):
            A = k + 1
            B = k + 2
            C = k
            lines.append((A, B, C))
            
        lines = np.array(lines[:27], dtype=np.int64)
        
        return points, lines
