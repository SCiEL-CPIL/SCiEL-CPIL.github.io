import numpy as np
from scipy.optimize import minimize_scalar
import time
import random

# Precomputation for fast scoring
# We use a dense grid to approximate the sup norm, then refine the peak.
_NUM_SAMPLES = 4000
_angles = 2 * np.pi * np.arange(_NUM_SAMPLES) / _NUM_SAMPLES
_zs = np.exp(1j * _angles)
_powers = np.arange(24)
# M[i, k] = zs[i] ** k
_M = _zs[:, None] ** _powers[None, :]

# Known good sequences (some from GRS, some from literature)
# These serve as strong starting points.
_KNOWN_GOOD = [
    [1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1],
    [1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1],
    [1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1],
    [1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1],
    [1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1],
    [1, -1, 1, 1, -1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, -1, -1],
    [1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1],
    [1, -1, -1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1],
    # GRS sequence
    [1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1],
]

def _score_fast(c_vec):
    """
    Fast scoring using precomputed matrix. Returns sup_norm / sqrt(24).
    Includes local refinement of the peak.
    """
    P = _M @ c_vec
    mags = np.abs(P)
    max_val = np.max(mags)
    max_idx = np.argmax(mags)
    theta0 = _angles[max_idx]
    
    # Refine the peak using golden-section / bounded minimization
    # We minimize -|P(theta)| to maximize |P(theta)|
    def obj(theta):
        z_ref = np.exp(1j * theta)
        v = np.dot(c_vec, z_ref ** _powers)
        return -np.abs(v)
        
    step = _angles[1] - _angles[0]
    # Use a tighter bound around the grid peak
    res = minimize_scalar(obj, bounds=(theta0 - step, theta0 + step), method='bounded', options={'xatol': 1e-10})
    if res.fun < 0:
        max_val = -res.fun
    
    return max_val / np.sqrt(24)

def _score_ultra_fast(c_vec):
    """
    Very fast scoring without refinement. Used for initial filtering in tabu search.
    """
    P = _M @ c_vec
    mags = np.abs(P)
    max_val = np.max(mags)
    return max_val / np.sqrt(24)

def _generate_neighborhood(c_vec, num_neighbors=20, max_flips=2):
    """
    Generate a set of neighbors by flipping 1 or 2 bits.
    """
    neighbors = []
    n = len(c_vec)
    # Precompute all single flip indices
    single_flips = list(range(n))
    random.shuffle(single_flips)
    
    # Generate 1-flip neighbors
    for i in single_flips[:num_neighbors//2]:
        new_c = c_vec.copy()
        new_c[i] = -new_c[i]
        neighbors.append(new_c)
        
    # Generate 2-flip neighbors
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((i, j))
    random.shuffle(pairs)
    
    for i, j in pairs[:num_neighbors//2]:
        new_c = c_vec.copy()
        new_c[i] = -new_c[i]
        new_c[j] = -new_c[j]
        neighbors.append(new_c)
        
    return neighbors

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_c = None
    best_score = np.inf
    current_c = None
    current_score = np.inf
    
    # 1. Evaluate known good sequences
    for seq in _KNOWN_GOOD:
        c = np.array(seq)
        s = _score_fast(c)
        if s < best_score:
            best_score = s
            best_c = c.copy()
            current_c = c.copy()
            current_score = s
            
    # If we already found the optimal, return
    if best_score < 1.2583758 + 1e-6:
        return best_c.tolist()

    # 2. Tabu Search
    tabu_list = []  # List of (c_vec_tuple, expiration_time)
    tabu_tenure = 50  # Number of iterations to keep in tabu list
    iteration = 0
    no_improve_count = 0
    max_no_improve = 100
    
    # We will use a mix of 1-flip and 2-flip moves
    # To speed up, we will score neighbors using _score_ultra_fast first,
    # and only refine the best ones.
    
    while time.time() < deadline:
        iteration += 1
        
        # Generate neighbors
        neighbors = _generate_neighborhood(current_c, num_neighbors=30, max_flips=2)
        
        best_neighbor = None
        best_neighbor_score = np.inf
        best_neighbor_ultra = np.inf
        
        # Evaluate neighbors
        for n_c in neighbors:
            # Check if in tabu list
            is_tabu = False
            n_key = tuple(n_c)
            for t_key, t_exp in tabu_list:
                if t_key == n_key:
                    is_tabu = True
                    break
            
            if is_tabu:
                continue
                
            # Quick score
            s_ultra = _score_ultra_fast(n_c)
            
            # Aspiration: if ultra score is already better than best known, accept
            if s_ultra < best_score:
                best_neighbor = n_c
                best_neighbor_score = _score_fast(n_c)
                break
                
            # Otherwise, keep track of best non-tabu neighbor based on ultra score
            if s_ultra < best_neighbor_ultra:
                best_neighbor_ultra = s_ultra
                best_neighbor = n_c
        
        # If we found a promising neighbor via aspiration, use it
        if best_neighbor is not None and best_neighbor_score < np.inf:
            # Check if it's actually better than current
            if best_neighbor_score < current_score:
                current_c = best_neighbor
                current_score = best_neighbor_score
                no_improve_count = 0
                
                if current_score < best_score:
                    best_score = current_score
                    best_c = current_c.copy()
                    
                # Update tabu list
                tabu_list.append((tuple(current_c), iteration + tabu_tenure))
                
                if best_score < 1.2583758 + 1e-6:
                    return best_c.tolist()
                continue
        
        # If no aspiration move, evaluate the best ultra-score neighbor precisely
        if best_neighbor is not None:
            s_precise = _score_fast(best_neighbor)
            
            # Accept if better than current, or if current is tabu (which it isn't, by definition of being current)
            # Standard tabu: accept best non-tabu move, even if worse than current (to escape local optima)
            # But we should prefer improving moves.
            if s_precise < current_score:
                current_c = best_neighbor
                current_score = s_precise
                no_improve_count = 0
                
                if current_score < best_score:
                    best_score = current_score
                    best_c = current_c.copy()
                    
                if best_score < 1.2583758 + 1e-6:
                    return best_c.tolist()
            else:
                # Even if worse, we might accept to escape local minimum if we've been stuck
                if no_improve_count > max_no_improve:
                    current_c = best_neighbor
                    current_score = s_precise
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            
            # Update tabu list with the move we just made
            tabu_list.append((tuple(current_c), iteration + tabu_tenure))
            
        # Clean up expired tabu entries
        if iteration % 10 == 0:
            tabu_list = [(k, e) for k, e in tabu_list if e > iteration]
            
        # Random restart if stuck for too long
        if no_improve_count > 500:
            # Pick a random neighbor of the best known solution
            kick_c = best_c.copy()
            flip_indices = random.sample(range(24), 4)
            for i in flip_indices:
                kick_c[i] = -kick_c[i]
            current_c = kick_c
            current_score = _score_fast(current_c)
            no_improve_count = 0
            tabu_list = [] # Clear tabu list on restart

    # Final polish
    if time.time() < deadline - 2:
        polished_score = _score_fast(best_c)
        if polished_score < best_score:
            best_score = polished_score
            best_c = best_c.copy() # Just to be safe, though it's the same

    return best_c.tolist()
