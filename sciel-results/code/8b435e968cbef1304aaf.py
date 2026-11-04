"""Rewritten program: greedy 4-set packing with sign optimization for kissing number in R^11."""
import time
import numpy as np
from itertools import combinations
from collections import defaultdict

D = 11
DEADLINE = 140.0  # safety margin before 150s kill

def _all_4sets():
    """Generate all 4-element subsets of range(D)."""
    return list(combinations(range(D), 4))

def _intersection_size(s1, s2):
    """Compute intersection size of two frozensets."""
    return len(s1 & s2)

def _greedy_pack(supports, min_intersection=2, rng=None):
    """
    Greedily select supports with pairwise intersection <= min_intersection.
    Uses most-constrained-first heuristic: at each step, pick the support
    that conflicts with the fewest remaining candidates.
    """
    n = len(supports)
    if n == 0:
        return []
    
    # Precompute conflict matrix (symmetric, intersection > min_intersection means conflict)
    # But this is O(n^2) memory. For D=11, C(11,4)=330, so 330x330 is fine.
    conflict = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i+1, n):
            if len(supports[i] & supports[j]) > min_intersection:
                conflict[i, j] = True
                conflict[j, i] = True
    
    remaining = set(range(n))
    selected = []
    
    while remaining:
        if rng is not None:
            # Randomized: pick among the 10 most-constrained with some randomness
            degrees = np.array([sum(conflict[i, j] for j in remaining if j != i) for i in remaining])
            # Find the minimum degree
            min_deg = degrees.min()
            # Get all candidates with degree close to min
            candidates = [i for i, d in zip(remaining, degrees) if d <= min_deg + 1]
            idx = rng.choice(candidates)
        else:
            # Deterministic: pick the one with minimum degree
            degrees = np.array([sum(conflict[i, j] for j in remaining if j != i) for i in remaining])
            idx = remaining[np.argmin(degrees)]
        
        selected.append(idx)
        remaining.discard(idx)
        # Remove all conflicts of idx from remaining
        for j in list(remaining):
            if conflict[idx, j]:
                remaining.discard(j)
    
    return [supports[i] for i in selected]

def _assign_signs(supports, rng=None, max_iterations=200):
    """
    Given a list of 4-element supports, assign ±1 signs to each element
    to maximize the minimum pairwise distance.
    
    For two vectors with supports S_i, S_j and intersection of size s:
    dot = sum of sign products over intersection elements
    We want to minimize the maximum dot product (to maximize min distance).
    
    Use local search: start with random signs, then flip signs to reduce max dot.
    """
    n = len(supports)
    if n == 0:
        return []
    
    # Build index: which vectors contain which coordinate
    coord_to_vecs = defaultdict(list)
    for vi, sup in enumerate(supports):
        for c in sup:
            coord_to_vecs[c].append(vi)
    
    # Initialize signs: each vector gets random ±1 for its 4 coordinates
    # signs[vi] is a dict from coordinate to ±1
    signs = []
    for vi, sup in enumerate(supports):
        if rng is not None:
            s = {c: rng.choice([-1, 1]) for c in sup}
        else:
            s = {c: 1 for c in sup}
        signs.append(s)
    
    def compute_dots():
        """Compute all pairwise dot products."""
        dots = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                inter = set(supports[i]) & set(supports[j])
                d = 0
                for c in inter:
                    d += signs[i][c] * signs[j][c]
                dots[i, j] = d
                dots[j, i] = d
        return dots
    
    def max_dot(dots):
        """Get the maximum off-diagonal dot product."""
        if n <= 1:
            return 0
        return np.max(np.abs(dots[np.triu_indices(n, k=1)]))
    
    best_signs = None
    best_max_dot = float('inf')
    
    for iteration in range(max_iterations):
        dots = compute_dots()
        current_max = max_dot(dots)
        
        if current_max < best_max_dot:
            best_max_dot = current_max
            best_signs = [dict(s) for s in signs]
        
        # If we've achieved max_dot <= 2, we're optimal (can't do better for intersection<=2)
        if current_max <= 2:
            break
        
        # Find the pair with maximum |dot| and try to fix it
        triu_i, triu_j = np.triu_indices(n, k=1)
        abs_dots = np.abs(dots[triu_i, triu_j])
        max_idx = np.argmax(abs_dots)
        i, j = triu_i[max_idx], triu_j[max_idx]
        
        inter = set(supports[i]) & set(supports[j])
        if len(inter) == 0:
            continue
        
        # Try flipping signs in vector i or j to reduce the dot
        current_dot = dots[i, j]
        best_flip = None
        best_new_dot = current_dot
        
        # Try flipping each coordinate in vector i that's in the intersection
        for c in inter:
            # Flipping signs[i][c] changes dot by -2*signs[i][c]*signs[j][c]
            new_dot = current_dot - 2 * signs[i][c] * signs[j][c]
            if abs(new_dot) < abs(best_new_dot):
                best_new_dot = new_dot
                best_flip = ('i', c)
        
        # Try flipping each coordinate in vector j that's in the intersection
        for c in inter:
            new_dot = current_dot - 2 * signs[i][c] * signs[j][c]
            if abs(new_dot) < abs(best_new_dot):
                best_new_dot = new_dot
                best_flip = ('j', c)
        
        if best_flip is not None:
            vec_idx, coord = best_flip
            if vec_idx == 'i':
                signs[i][coord] *= -1
            else:
                signs[j][coord] *= -1
        else:
            # No improvement possible for this pair; try random flip
            if rng is not None:
                vec_idx = rng.choice([i, j])
                coord = rng.choice(list(supports[vec_idx]))
                signs[vec_idx][coord] *= -1
    
    # Use best signs found
    if best_signs is not None:
        signs = best_signs
    
    # Build final vectors
    vectors = []
    for vi, sup in enumerate(supports):
        v = [0] * D
        for c in sup:
            v[c] = signs[vi][c]
        vectors.append(v)
    
    return vectors

def _evaluate(vectors):
    """Compute the score: minimum pairwise distance / maximum norm."""
    if len(vectors) == 0:
        return 0, 0
    arr = np.array(vectors, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1)
    max_norm = np.max(norms)
    if max_norm == 0:
        return 0, 0
    # Compute all pairwise distances
    # ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2*v_i.v_j
    dots = arr @ arr.T
    min_dist = float('inf')
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            d2 = norms[i]**2 + norms[j]**2 - 2*dots[i, j]
            d = np.sqrt(max(0, d2))
            if d < min_dist:
                min_dist = d
    return min_dist, max_norm

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    all_supports = _all_4sets()  # 330 supports
    
    best_vectors = None
    best_score = 0
    best_k = 0
    
    # Multi-start: try different random seeds and greedy strategies
    for seed in range(50):
        if time.time() > deadline - 5:
            break
        
        rng = np.random.default_rng(seed)
        
        # Shuffle supports for diversity
        shuffled = all_supports[:]
        rng.shuffle(shuffled)
        
        # Greedy pack
        supports = _greedy_pack(shuffled, min_intersection=2, rng=rng)
        
        if len(supports) < best_k:
            continue
        
        # Assign signs
        vectors = _assign_signs(supports, rng=rng, max_iterations=100)
        
        if len(vectors) == 0:
            continue
        
        # Evaluate
        min_dist, max_norm = _evaluate(vectors)
        
        if max_norm > 0 and min_dist >= max_norm - 1e-9:
            # Valid configuration
            if len(vectors) > best_k:
                best_k = len(vectors)
                best_vectors = vectors
                best_score = min_dist / max_norm
    
    # If no valid configuration found, fall back to a simple valid one
    if best_vectors is None:
        # Use the original approach as fallback
        vectors = []
        for i, j in combinations(range(D), 2):
            v = [0] * D
            v[i], v[j] = 1, 1
            vectors.append(v)
        best_vectors = vectors
    
    return np.array(best_vectors, dtype=np.int64)
