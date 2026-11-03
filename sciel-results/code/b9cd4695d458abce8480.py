import math
import time
import numpy as np
from itertools import product

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def _time_ok():
    return (time.time() - START_TIME) < (TIME_LIMIT - SAFETY_MARGIN)

def _axial_to_cartesian(q, r):
    # Flat-top hexagon axial to cartesian
    # x = size * sqrt(3) * (q + r/2)
    # y = size * 3/2 * r
    # For unit side length (size=1):
    x = math.sqrt(3) * (q + r / 2.0)
    y = 1.5 * r
    return x, y

def _bounding_side_length(cells):
    # Calculate the minimum side length S of a regular hexagon (centered at origin, aligned with axes)
    # that contains all cells.
    # A hexagon of side S has apothem A = S * sqrt(3)/2.
    # The condition for a point (x,y) to be inside a hexagon with apothem A and center (0,0) is:
    # max(|x|, |x/2 + y*sqrt(3)/2|, |-x/2 + y*sqrt(3)/2|) <= A
    # Wait, the normals for a hexagon with flat top are at angles 0, 60, 120, 180, 240, 300?
    # Let's check the standard hexagon definition.
    # Vertices at (S, 0), (S/2, S*sqrt(3)/2), (-S/2, S*sqrt(3)/2), (-S, 0), (-S/2, -S*sqrt(3)/2), (S/2, -S*sqrt(3)/2).
    # The supporting lines are:
    # x = S  => x <= S
    # -x = S => -x <= S
    # x/2 + y*sqrt(3)/2 = S*sqrt(3)/2 = A => x/2 + y*sqrt(3)/2 <= A
    # -x/2 - y*sqrt(3)/2 = A => -x/2 - y*sqrt(3)/2 <= A
    # -x/2 + y*sqrt(3)/2 = A => -x/2 + y*sqrt(3)/2 <= A
    # x/2 - y*sqrt(3)/2 = A => x/2 - y*sqrt(3)/2 <= A
    
    # So for each point, we compute the required apothem A_i:
    # A_i = max( |x_i|, |x_i/2 + y_i*sqrt(3)/2|, |-x_i/2 + y_i*sqrt(3)/2| )
    # The hexagon must have apothem A >= max(A_i) for all i.
    # Then S = A / (sqrt(3)/2) = 2*A / sqrt(3).
    
    max_apothem = 0.0
    sqrt3 = math.sqrt(3)
    for q, r in cells:
        x, y = _axial_to_cartesian(q, r)
        # The hexagon containing the cell must contain the entire cell.
        # The cell is a hexagon of side 1 centered at (x,y).
        # The farthest point of the cell from the origin in direction n is:
        # dist = |dot(center, n)| + apothem_cell
        # apothem_cell = sqrt(3)/2
        # So required A >= |dot(center, n)| + sqrt(3)/2.
        
        # Normals for the container hexagon (flat-top):
        # n1 = (1, 0)
        # n2 = (-1, 0)
        # n3 = (1/2, sqrt(3)/2)
        # n4 = (-1/2, -sqrt(3)/2)
        # n5 = (-1/2, sqrt(3)/2)
        # n6 = (1/2, -sqrt(3)/2)
        
        # Due to symmetry, we just need max of:
        # |x|
        # |x/2 + y*sqrt(3)/2|
        # |-x/2 + y*sqrt(3)/2|
        
        a1 = abs(x)
        a2 = abs(x/2.0 + y*sqrt3/2.0)
        a3 = abs(-x/2.0 + y*sqrt3/2.0)
        
        required_a = max(a1, a2, a3) + APOTHEM
        
        if required_a > max_apothem:
            max_apothem = required_a
            
    S = 2.0 * max_apothem / sqrt3
    return S

def _generate_all_shapes(n_cells):
    # Generate all connected sets of n_cells on the hexagonal grid
    # using DFS/BFS from origin.
    # We will generate all "free" polyhexes by fixing one cell at (0,0) and ensuring
    # no cell is in the "negative" quadrant relative to the first cell added?
    # Actually, simpler: Generate all connected sets of size N containing (0,0).
    # Then canonicalize to remove duplicates?
    # For N=11, the number of polyhexes is 10,488 (from OEIS A001168? No, A001168 is 1,1,2,7,22,81,314,1243,4908,19683,79940... wait.
    # Number of polyhexes of size n:
    # n=1: 1
    # n=2: 1
    # n=3: 2
    # n=4: 7
    # n=5: 22
    # n=6: 81
    # n=7: 314
    # n=8: 1243
    # n=9: 4908
    # n=10: 19683
    # n=11: 79940
    # 80k shapes is very manageable.
    
    shapes = set()
    
    # Directions in axial coordinates
    dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    def dfs(current_set, current_size):
        if current_size == n_cells:
            # Canonicalize the shape to avoid duplicates
            # Translate so min q, min r is at (0,0)?
            # Better: Translate so that the "lexicographically smallest" cell is at (0,0)
            # And then sort the cells.
            cells = list(current_set)
            # Find min q, then min r?
            # Just sort and take a hash of the sorted tuple of relative coords?
            # To handle rotation/reflection, we'd need more, but for bounding box,
            # we can just evaluate all connected sets containing (0,0) and take the min S.
            # Duplicates will just be evaluated multiple times, which is fine for 80k * 11 ops.
            shapes.add(frozenset(cells))
            return
            
        for dq, dr in dirs:
            new_cell = (list(current_set)[0][0] + dq, list(current_set)[0][1] + dr) # No, this is wrong.
            
        # Proper DFS expansion
        # We need to add a cell adjacent to any cell in the set.
        # To avoid infinite loops and duplicates, we can enforce an ordering?
        # Or just use a set of visited states?
        # State is the frozenset of cells.
        # For N=11, this might be slow if we don't prune.
        # However, 80k shapes is small.
        # Let's just generate all connected sets of size N containing (0,0).
        
        # Optimization: Only add cells that are "new" in some sense?
        # No, standard backtracking.
        
        # To speed up, we can restrict the search space to a bounding box?
        # Or use the "canonical" generation method.
        
        pass

    # Let's use a simpler generation:
    # Start with {(0,0)}.
    # At each step, add a cell adjacent to the current set.
    # To avoid generating the same set multiple times (via different orders),
    # we can enforce that we only add cells that are "greater" than the last added cell in some ordering?
    # No, that breaks connectivity.
    
    # Alternative: Just use a set of visited frozensets.
    # For N=11, the branching factor is roughly 6^N which is huge, but many are disconnected or duplicates.
    # Actually, the number of *connected* sets is 80k.
    # The number of *paths* generating them is larger.
    
    # Let's try a recursive generation that adds cells adjacent to the existing set,
    # and uses a set to track visited configurations.
    
    visited = set()
    results = []
    
    def gen(current_set):
        if len(current_set) == n_cells:
            if current_set not in visited:
                visited.add(current_set)
                results.append(current_set)
            return
            
        # Find all valid neighbors
        neighbors = set()
        for q, r in current_set:
            for dq, dr in dirs:
                nq, nr = q + dq, r + dr
                if (nq, nr) not in current_set:
                    neighbors.add((nq, nr))
                    
        for nb in neighbors:
            new_set = current_set | {nb}
            # Pruning: Check if the bounding box is too large?
            # We can compute a lower bound on S.
            # But for now, let's just generate.
            gen(new_set)
            
    gen(frozenset([(0,0)]))
    return results

def construct():
    if not _time_ok():
        # Fallback: simple ring
        cells = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (-2,0), (0,2), (1,2)]
        # Calculate S for this
        max_apothem = 0.0
        sqrt3 = math.sqrt(3)
        for q, r in cells:
            x, y = _axial_to_cartesian(q, r)
            a1 = abs(x)
            a2 = abs(x/2.0 + y*sqrt3/2.0)
            a3 = abs(-x/2.0 + y*sqrt3/2.0)
            required_a = max(a1, a2, a3) + APOTHEM
            if required_a > max_apothem:
                max_apothem = required_a
        S = 2.0 * max_apothem / sqrt3
        inner = [[x, y, 0.0] for x, y in [_axial_to_cartesian(q, r) for q, r in cells]]
        return inner, [0.0, 0.0], S, 0.0

    best_S = float('inf')
    best_cells = None
    
    # Generate all polyhexes of size 11
    # This might take a few seconds.
    # 80k shapes.
    
    try:
        shapes = _generate_all_shapes(N)
    except Exception:
        shapes = []
        
    if not shapes:
        # Fallback to some manual seeds
        shapes = [
            frozenset([(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (-2,0), (0,2), (1,2)]),
            frozenset([(i,0) for i in range(6)] + [(i,1) for i in range(5)])
        ]
        
    for cells in shapes:
        if not _time_ok():
            break
        S = _bounding_side_length(cells)
        if S < best_S:
            best_S = S
            best_cells = cells
            
    if best_cells is None:
        # Should not happen
        best_cells = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (-2,0), (0,2), (1,2)]
        best_S = _bounding_side_length(best_cells)
        
    # Construct output
    # Center the shape at origin?
    # The bounding side length calculation assumed center at (0,0).
    # If we shift the shape, S changes.
    # We want to minimize S over all translations.
    # For a fixed shape, the optimal translation is to center the "bounding box" of the shape?
    # Actually, the function _bounding_side_length calculates the S required if the center is at (0,0).
    # If we translate the shape by (dx, dy), the required S is determined by the new positions.
    # The minimum S for a fixed shape is achieved when the center of the hexagon is at the "center of symmetry" of the shape's bounding box?
    # Not necessarily.
    # However, for lattice packings, the optimal position is usually such that the shape is centered.
    # Let's assume the optimal center is (0,0) for the relative coordinates.
    # Wait, if I shift the shape, the relative coordinates change.
    # My _bounding_side_length function takes cells and assumes they are placed at those axial coords.
    # It calculates the S required to contain them with hexagon center at (0,0).
    # So if I find a set of cells C, the S calculated is the S for that specific placement.
    # To find the global minimum, I should also optimize the translation.
    # But for lattice packings, the translation is effectively continuous?
    # No, the cells are discrete.
    # If I fix the relative positions (the shape), I can slide the whole shape.
    # The function S(dx, dy) is the max over cells of (distance to boundary).
    # This is a convex function of dx, dy.
    # The minimum is at the center of the "bounding box" in the hexagonal metric.
    # For simplicity, let's assume the shape generated is already "centered" enough, or just take the S calculated.
    # Actually, the generation process starts at (0,0) and grows. The shape is not necessarily centered at (0,0).
    # For example, a line of 11 cells starting at (0,0) goes to (10,0).
    # The S calculated will be large.
    # If I centered it, S would be smaller.
    
    # So, for each shape, I should find the optimal center.
    # Or, better, generate shapes centered at (0,0).
    # How?
    # After generating a shape, calculate its "center of mass" or "bounding box center" and translate it.
    # Then recalculate S.
    
    # Let's refine the loop:
    best_S = float('inf')
    best_cells_centered = None
    
    for cells in shapes:
        if not _time_ok():
            break
            
        # Calculate centroid in cartesian coords
        xs = []
        ys = []
        for q, r in cells:
            x, y = _axial_to_cartesian(q, r)
            xs.append(x)
            ys.append(y)
        
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        
        # Translate cells so centroid is at (0,0)?
        # Axial coords are discrete, so we can't just subtract cx, cy.
        # We can try a few nearby integer translations.
        # Or, just evaluate S for the shape as is, and also for shapes shifted by small amounts?
        # No, the shape is fixed in axial coords.
        # The "translation" in the problem allows continuous movement.
        # So for a fixed relative configuration (shape), we can place it anywhere.
        # The optimal placement minimizes S.
        # S is the max over cells of (apothem required).
        # Apothem required for cell i at (x_i, y_i) with hex center (X, Y) is:
        # max( |x_i - X|, |...| ) + APOTHEM
        # We want to minimize max_i ( f_i(X, Y) ).
        # This is a Chebyshev center problem.
        # For N=11, we can solve this quickly using linear programming or grid search.
        # But for 80k shapes, we need it to be fast.
        
        # Approximation: The optimal center is the centroid.
        # Let's just use the centroid as the center.
        # Convert centroid back to axial? No, center is continuous.
        # So we just report the center (cx, cy) and the cells relative to it?
        # No, the output requires (x, y) of each inner hexagon.
        # So we can just output the cartesian coords of the cells relative to the optimal center.
        
        # Let's assume the optimal center is the centroid.
        # Calculate S with center at (0,0) for the *shifted* cells?
        # No, let's calculate S directly with center at (cx, cy).
        
        # Modify _bounding_side_length to take a center.
        
        sqrt3 = math.sqrt(3)
        max_apothem = 0.0
        for q, r in cells:
            x, y = _axial_to_cartesian(q, r)
            # Distance from (cx, cy)
            dx = x - cx
            dy = y - cy
            
            a1 = abs(dx)
            a2 = abs(dx/2.0 + dy*sqrt3/2.0)
            a3 = abs(-dx/2.0 + dy*sqrt3/2.0)
            
            required_a = max(a1, a2, a3) + APOTHEM
            if required_a > max_apothem:
                max_apothem = required_a
                
        S = 2.0 * max_apothem / sqrt3
        
        if S < best_S:
            best_S = S
            # Store the cartesian coords relative to center
            best_cells_centered = []
            for q, r in cells:
                x, y = _axial_to_cartesian(q, r)
                best_cells_centered.append((x - cx, y - cy))
                
    if best_cells_centered is None:
        # Fallback
        cells = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (-2,0), (0,2), (1,2)]
        xs = [_axial_to_cartesian(q,r)[0] for q,r in cells]
        ys = [_axial_to_cartesian(q,r)[1] for q,r in cells]
        cx = sum(xs)/11
        cy = sum(ys)/11
        best_cells_centered = [(x-cx, y-cy) for x,y in zip(xs,ys)]
        best_S = _bounding_side_length(cells) # Roughly
        
    inner = [[x, y, 0.0] for x, y in best_cells_centered]
    # The center of the outer hexagon is (0,0) in our local frame?
    # The problem asks for:
    # inner rows (x, y, angle_degrees)
    # outer hexagon center (x, y)
    # If I set the outer center to (0,0), then inner coords are relative to (0,0).
    # This is valid.
    
    return inner, [0.0, 0.0], float(best_S), 0.0

