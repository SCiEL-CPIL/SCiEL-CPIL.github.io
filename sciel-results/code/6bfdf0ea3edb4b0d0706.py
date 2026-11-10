"""Program for no-isosceles set using greedy randomized construction with vectorized checks."""
import numpy as np
import time

N = 64

def construct():
    deadline = time.time() + 280  # safety margin before 330s kill
    best_points = None
    best_count = 0
    
    # Pre-compute all grid points
    all_points = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int64)
    total_points = len(all_points)
    
    rng = np.random.RandomState(42)
    
    # Try multiple random orderings
    for trial in range(200):
        if time.time() > deadline - 30:
            break
        
        # Random permutation of all points
        order = rng.permutation(total_points)
        
        # Greedy construction
        selected = []
        selected_set = set()
        
        # For speed, we'll check conflicts incrementally
        # Store selected points as array for vectorized checks
        sel_array = np.empty((0, 2), dtype=np.int64)
        
        for idx in order:
            if time.time() > deadline - 10:
                break
            
            p = all_points[idx]
            px, py = int(p[0]), int(p[1])
            
            if len(selected) < 2:
                selected.append((px, py))
                sel_array = np.vstack([sel_array, [[px, py]]]) if len(sel_array) > 0 else np.array([[px, py]], dtype=np.int64)
                continue
            
            # Check if adding (px, py) creates an isosceles triangle with any pair in selected
            # We need: no A, B in selected such that:
            #   |PA|² == |PB|²  OR  |PA|² == |AB|²  OR  |PB|² == |AB|²
            
            # Vectorized check against all selected points
            # dist_p_to_selected[i] = |P - S[i]|²
            dx_p = px - sel_array[:, 0]
            dy_p = py - sel_array[:, 1]
            dist_p_sq = dx_p**2 + dy_p**2  # shape (n,)
            
            # For each pair (i, j) in selected, check if triangle P, S[i], S[j] is isosceles
            # This is O(n²) per candidate which is too slow for large n
            # Instead, check:
            # 1. Is there any i, j with dist_p_sq[i] == dist_p_sq[j]? (P equidistant from two points)
            # 2. For each i, is there any j with dist_p_sq[i] == |S[i]-S[j]|²?
            
            # Check 1: duplicate distances from P to selected points
            # If any distance appears twice, P is equidistant from two points -> isosceles with apex P
            if len(dist_p_sq) >= 2:
                # Check for duplicates
                sorted_d = np.sort(dist_p_sq)
                if np.any(sorted_d[1:] == sorted_d[:-1]):
                    continue  # Conflict: P equidistant from two selected points
            
            # Check 2: For each selected point S[i], check if |P-S[i]|² equals |S[i]-S[j]|² for any j
            # This means triangle P, S[i], S[j] has side PS[i] = S[i]S[j]
            # We need to precompute pairwise distances among selected points
            # But that's O(n²) memory. Instead, for each i, compute |S[i]-S[j]|² for all j and check if dist_p_sq[i] is in that set.
            
            # For efficiency with moderate n, do pairwise check
            n_sel = len(sel_array)
            conflict = False
            
            for i in range(n_sel):
                # Check if dist_p_sq[i] equals |S[i] - S[j]|² for any j != i
                # Also check if dist_p_sq[i] equals |S[j] - S[k]|² for any j,k (but that's covered differently)
                
                # Actually, the condition is: for some pair (A, B) in selected, 
                # |P-A|² == |P-B|² OR |P-A|² == |A-B|² OR |P-B|² == |A-B|²
                # We already checked |P-A|² == |P-B|² for all pairs via the duplicate check above.
                # Now check: for some A, B in selected, |P-A|² == |A-B|²
                # This means: for some A, B, |A-B|² == |P-A|²
                # i.e., B lies on the circle centered at A with radius |P-A|
                
                # For each A (index i), we need to check if there exists B (index j) such that |S[i]-S[j]|² == dist_p_sq[i]
                # This is: for each i, check if dist_p_sq[i] is in the set {|S[i]-S[j]|² : j != i}
                
                # Compute distances from S[i] to all other selected points
                dx_i = sel_array[i, 0] - sel_array[:, 0]
                dy_i = sel_array[i, 1] - sel_array[:, 1]
                dist_i_sq = dx_i**2 + dy_i**2
                # dist_i_sq[i] = 0, we want to check if dist_p_sq[i] is in dist_i_sq excluding index i
                # But dist_p_sq[i] > 0 (since P is not in selected), so we just check if it's in dist_i_sq
                if dist_p_sq[i] in dist_i_sq:
                    conflict = True
                    break
            
            if conflict:
                continue
            
            # No conflict, add the point
            selected.append((px, py))
            sel_array = np.vstack([sel_array, [[px, py]]])
        
        # Update best
        if len(selected) > best_count:
            best_count = len(selected)
            best_points = selected[:]
        
        # Early termination if we found a very good solution
        if best_count > 80:
            break
    
    if best_points is None:
        # Fallback: return at least some points
        return [(0, 0), (1, 0)]
    
    return best_points
