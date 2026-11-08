"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11
import time
import numpy as np

# EVOLVE-BLOCK-START
import itertools


def construct():
    """
    Construct a kissing configuration in R^11.
    The seed program returns all weight-2 vectors (C(11,2)*4 = 220 vectors) with min dist = 2, max norm = 2.
    We try to improve upon this by finding a larger set.
    
    Strategy:
    1. Start with the known 220 weight-2 vectors.
    2. Try to add vectors of higher weight (e.g., weight 3 or 4) if they are compatible.
    3. Use a greedy approach to find additional vectors that maintain the constraint:
       min pairwise distance >= max norm.
    
    Actually, let's think about this more carefully.
    The constraint is: for all pairs (i,j), ||v_i - v_j|| >= max_k ||v_k||.
    
    If we normalize all vectors to have the same norm R, then the constraint becomes:
    ||v_i - v_j|| >= R for all i != j.
    This means the angle between any two vectors must be at least 60 degrees (since ||v_i - v_j||^2 = 2R^2 - 2R^2 cos(theta) >= R^2 => cos(theta) <= 1/2).
    
    The seed program uses vectors of norm sqrt(2). The minimum distance between two weight-2 vectors:
    - If they share no coordinates: distance = sqrt(4) = 2
    - If they share one coordinate: distance = sqrt(2) or sqrt(6) depending on signs
    - If they share both coordinates: distance = 0 or 2 or sqrt(8)
    
    Wait, let me re-examine. The vectors are of the form e_i + e_j, e_i - e_j, -e_i + e_j, -e_i - e_j.
    Two vectors e_i + e_j and e_i + e_k (sharing coordinate i, j != k):
    Difference: (0, 1, -1, 0, ...) in positions j and k, norm = sqrt(2).
    But max norm = sqrt(2). So sqrt(2) >= sqrt(2) is OK.
    
    Two vectors e_i + e_j and -e_i + e_j (sharing both coordinates, opposite sign in i):
    Difference: (2, 0, 0, ...) norm = 2. OK.
    
    Two vectors e_i + e_j and e_i - e_j (sharing i, opposite sign in j):
    Difference: (0, 2, 0, ...) norm = 2. OK.
    
    So the 220 vectors form a valid configuration with min dist = sqrt(2) = max norm.
    
    Can we add more vectors? Let's try vectors of weight 3, e.g., e_i + e_j + e_k.
    Norm = sqrt(3).
    Distance to e_i + e_j: difference has norm sqrt(1) = 1 in the k-th position. So distance = 1.
    But max norm would be sqrt(3) if we include this vector. 1 < sqrt(3), so this violates the constraint.
    
    So we can't simply add weight-3 vectors. We'd need to rescale, but the vectors must be integers.
    
    Alternative: use a different construction. The kissing number in dimension 11 is known to be 504 (or in that range).
    The Leech lattice construction gives 504 vectors in dimension 24, but we need dimension 11.
    
    Let's try a different approach: use the fact that we can have vectors with different norms, as long as the constraint holds.
    
    Actually, re-reading the task: "nonzero integer vectors in R^11 whose minimum pairwise distance is at least the maximum vector norm."
    So the constraint is: min_{i!=j} ||v_i - v_j|| >= max_k ||v_k||.
    
    The seed achieves this with all vectors having norm sqrt(2), and min distance sqrt(2).
    
    To get more vectors, we could try to use vectors with smaller norms, but they must be nonzero integer vectors, so min norm is 1.
    If we use vectors of norm 1 (the standard basis vectors and their negatives), there are 22 such vectors.
    Distance between e_i and e_j is sqrt(2) for i != j. Distance between e_i and -e_i is 2.
    Max norm = 1. Min distance = sqrt(2) >= 1. OK.
    But 22 < 220, so this is worse.
    
    What if we mix? Use the 220 weight-2 vectors (norm sqrt(2)) plus some weight-1 vectors (norm 1)?
    Max norm = sqrt(2). Min distance must be >= sqrt(2).
    Distance from e_i to e_j + e_k (j != i, k != i): if i is not in {j,k}, distance = sqrt(2). OK.
    Distance from e_i to e_i + e_j: distance = 1. But 1 < sqrt(2). Not OK.
    
    So we can't add weight-1 vectors to the weight-2 set.
    
    Let's think about this differently. The problem is equivalent to finding a code with minimum distance d and maximum norm R, where d >= R.
    
    For the seed: d = sqrt(2), R = sqrt(2).
    
    Can we find a set with larger d/R ratio? If we could get d > R, we might be able to pack more vectors.
    
    Actually, the constraint is d >= R, so the ratio d/R >= 1. The seed achieves d/R = 1.
    
    If we could achieve d/R > 1, we'd have more "room" to pack vectors. But for integer vectors, this might be hard.
    
    Let's try a completely different approach: use a random search or simulated annealing to find a larger set.
    """
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    # Start with the seed configuration: all weight-2 vectors
    base_vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                base_vectors.append(tuple(v))
    
    # Try to add more vectors using a greedy approach
    # We'll try vectors of various weights and signs
    candidate_vectors = []
    
    # Generate candidate vectors of weight 1, 2, 3, 4
    for weight in range(1, 5):
        for indices in itertools.combinations(range(D), weight):
            for signs in itertools.product((1, -1), repeat=weight):
                v = [0] * D
                for idx, sign in zip(indices, signs):
                    v[idx] = sign
                candidate_vectors.append(tuple(v))
    
    # Remove duplicates
    candidate_vectors = list(set(candidate_vectors))
    
    # Start with the base vectors
    selected = set(base_vectors)
    
    # Precompute norms
    def norm_sq(v):
        return sum(x * x for x in v)
    
    def dist_sq(v, w):
        return sum((a - b) ** 2 for a, b in zip(v, w))
    
    # Check if a new vector can be added
    def can_add(new_vec, current_set):
        new_norm_sq = norm_sq(new_vec)
        max_norm_sq = max(norm_sq(v) for v in current_set)
        required_min_dist_sq = max(new_norm_sq, max_norm_sq)
        
        for v in current_set:
            if dist_sq(new_vec, v) < required_min_dist_sq:
                return False
        return True
    
    # Try to add candidate vectors one by one
    # Sort candidates by norm (try smaller norms first, as they're easier to fit)
    candidate_vectors.sort(key=lambda v: norm_sq(v))
    
    for vec in candidate_vectors:
        if time.time() > deadline - 10:  # Safety margin
            break
        if vec not in selected and can_add(vec, selected):
            selected.add(vec)
    
    # Convert to numpy array
    result = np.array(list(selected), dtype=np.int64)
    
    return result
# EVOLVE-BLOCK-END
