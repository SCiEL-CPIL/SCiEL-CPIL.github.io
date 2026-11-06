import numpy as np
import time
import random

def _score(points):
    """
    Vectorized calculation of the maximum multiplicity of a positive doubled area.
    points: np.array of shape (30, 2)
    Returns: max count of any specific positive doubled area.
    """
    n = len(points)
    if n < 3:
        return 0
    
    # Compute all pairwise differences
    # p[i] - p[j] for all i > j
    # We can compute the determinant for all triples (i, j, k) with i < j < k
    # det = (x_j - x_i)*(y_k - y_i) - (y_j - y_i)*(x_k - x_i)
    
    # To speed up, we can precompute differences from a fixed point, e.g., point 0?
    # No, that only covers triples containing 0.
    
    # Let's use a semi-vectorized approach.
    # For each k, compute det for all i < j < k? No, that's still O(N^3) logic but vectorized inner loops.
    
    # Better:
    # For each pair (i, j), compute the cross product with all k > j.
    # Actually, let's just do the full O(N^3) but in chunks or optimized loops.
    # 30 points -> 30*29*28/6 = 4060 triples. This is tiny.
    # The bottleneck in the previous program was likely Python loop overhead.
    # Let's try a pure Python loop with local variables for speed, or a vectorized one.
    
    # Vectorized:
    # Create arrays of indices
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    # Filter i < j < k
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    p_i = points[i_flat]
    p_j = points[j_flat]
    p_k = points[k_flat]
    
    # Doubled area = | (x_j - x_i)*(y_k - y_i) - (y_j - y_i)*(x_k - x_i) |
    # = | x_j*y_k - x_j*y_i - x_i*y_k + x_i*y_i - y_j*x_k + y_j*x_i + y_i*x_k - y_i*x_i |
    # Let's use the standard determinant formula:
    # det = x_i*(y_j - y_k) - y_i*(x_j - x_k) + (x_j*y_k - x_i*y_j)
    # Or simpler:
    # v1 = p_j - p_i
    # v2 = p_k - p_i
    # det = v1[0]*v2[1] - v1[1]*v2[0]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    dets = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    abs_dets = np.abs(dets)
    
    # Filter out zero areas
    nonzero = abs_dets[abs_dets > 0]
    
    if len(nonzero) == 0:
        return 0
        
    # Count multiplicities
    # np.unique with return_counts is efficient
    _, counts = np.unique(nonzero, return_counts=True)
    
    return int(np.max(counts))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Strategy: 3-line grid representation
    # 10 points on y=0, 10 points on y=1, 10 points on y=2
    # Initial state: x-coordinates are 0..9 for all three lines
    # This is a 10x3 grid.
    
    def init_grid():
        # x in [-5, 4] to center in [-30, 30]
        xs = list(range(-5, 5))
        return [list(xs), list(xs), list(xs)]
        
    lines = init_grid()
    
    def get_points(lines):
        pts = []
        for y_idx, xs in enumerate(lines):
            for x in xs:
                pts.append((x, y_idx))
        return pts
        
    def score_lines(lines):
        pts = get_points(lines)
        return _score(np.array(pts, dtype=np.int64))
        
    best_lines = [list(l) for l in lines]
    best_score = score_lines(best_lines)
    
    # Local Search
    # Moves:
    # 1. Swap two x-coordinates within a line
    # 2. Move a point to a new x-coordinate within bounds [-30, 30]
    # 3. Move a point to a different line (change y) - this changes the structure significantly
    
    # We will focus on moving x-coordinates first, as it preserves the 3-line structure.
    # Then try "line swaps" where we move a point from one line to another.
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        
        # Try all single-point moves
        for line_idx in range(3):
            for i in range(len(best_lines[line_idx])):
                if time.time() > deadline - 5:
                    break
                
                old_x = best_lines[line_idx][i]
                
                # Try swapping with another point in the same line
                for j in range(i + 1, len(best_lines[line_idx])):
                    if time.time() > deadline - 5:
                        break
                    # Swap i and j
                    best_lines[line_idx][i], best_lines[line_idx][j] = best_lines[line_idx][j], best_lines[line_idx][i]
                    s = score_lines(best_lines)
                    if s > best_score:
                        best_score = s
                        improved = True
                    else:
                        # Revert
                        best_lines[line_idx][i], best_lines[line_idx][j] = best_lines[line_idx][j], best_lines[line_idx][i]
                
                # Try moving to a new random x in [-30, 30]
                # To be efficient, try a few random moves
                for _ in range(5):
                    if time.time() > deadline - 5:
                        break
                    new_x = random.randint(-30, 30)
                    if new_x == old_x:
                        continue
                    # Check if new_x is already in this line
                    if new_x in best_lines[line_idx]:
                        continue
                    
                    best_lines[line_idx][i] = new_x
                    s = score_lines(best_lines)
                    if s > best_score:
                        best_score = s
                        improved = True
                    else:
                        best_lines[line_idx][i] = old_x

        # Try moving a point to a different line
        # This requires removing from one line and adding to another.
        # We need to maintain 10 points per line.
        # So we must swap a point from line A with a point from line B.
        
        for line_a in range(3):
            for line_b in range(line_a + 1, 3):
                if time.time() > deadline - 5:
                    break
                for i in range(len(best_lines[line_a])):
                    for j in range(len(best_lines[line_b])):
                        if time.time() > deadline - 5:
                            break
                        # Swap point i in line A with point j in line B
                        # This means point i moves to line B, and point j moves to line A
                        x_a = best_lines[line_a][i]
                        x_b = best_lines[line_b][j]
                        
                        # Check for collisions in the new lines
                        # x_a must not be in line_b (excluding the spot it's moving to? No, it's a swap)
                        # Wait, if we swap, x_a goes to line_b. If x_a is already in line_b (at some other index k), it's a collision.
                        # x_b goes to line_a. If x_b is already in line_a (at some other index k), it's a collision.
                        
                        # Simple check:
                        # Remove x_a from line_a, add to line_b.
                        # Remove x_b from line_b, add to line_a.
                        
                        # Let's just do it and check validity
                        temp_a = best_lines[line_a][:]
                        temp_b = best_lines[line_b][:]
                        
                        temp_a[i] = x_b
                        temp_b[j] = x_a
                        
                        if len(set(temp_a)) != 10 or len(set(temp_b)) != 10:
                            continue
                            
                        # Bounds check
                        if any(x < -30 or x > 30 for x in temp_a) or any(x < -30 or x > 30 for x in temp_b):
                            continue
                            
                        best_lines[line_a] = temp_a
                        best_lines[line_b] = temp_b
                        
                        s = score_lines(best_lines)
                        if s > best_score:
                            best_score = s
                            improved = True
                        else:
                            best_lines[line_a] = temp_a[:i] + [x_a] + temp_a[i+1:] # Revert is tricky with lists
                            # Actually, let's just revert by swapping back
                            best_lines[line_a][i] = x_a
                            best_lines[line_b][j] = x_b
                            
    # Final check
    if time.time() > deadline - 2:
        return get_points(best_lines)
        
    # Random restarts if time permits
    # Generate a few random 3-line configurations
    for _ in range(10):
        if time.time() > deadline - 5:
            break
        rand_lines = []
        for _ in range(3):
            rand_lines.append(random.sample(range(-30, 31), 10))
            
        s = score_lines(rand_lines)
        if s > best_score:
            best_score = s
            best_lines = rand_lines
            
    return get_points(best_lines)
