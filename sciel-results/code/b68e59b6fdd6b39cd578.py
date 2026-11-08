"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, product

DEADLINE = 150.0  # evaluator kills at 150s; keep safety margin
SAFETY_MARGIN = 5.0  # seconds to stop before deadline


def _min_pairwise_dist_sq(vectors):
    """Compute the minimum pairwise squared distance among vectors."""
    if len(vectors) < 2:
        return float('inf')
    # vectors is a list of arrays or a 2D array
    if isinstance(vectors, list):
        vecs = np.array(vectors, dtype=np.int64)
    else:
        vecs = vectors
    n = len(vecs)
    # Compute all pairwise distances using broadcasting in chunks to save memory
    # For n up to a few thousand, direct computation is fine
    # dist_sq[i,j] = ||v_i - v_j||^2
    # We only need the minimum off-diagonal
    min_d2 = float('inf')
    # Use a chunked approach for large n
    chunk_size = 500
    for i_start in range(0, n, chunk_size):
        i_end = min(i_start + chunk_size, n)
        # Compute distances from vecs[i_start:i_end] to all vecs
        diff = vecs[i_start:i_end, np.newaxis, :] - vecs[np.newaxis, :, :]
        d2 = np.sum(diff ** 2, axis=2)
        # Mask diagonal
        for i in range(i_start, i_end):
            d2[i - i_start, i] = float('inf')
        chunk_min = np.min(d2)
        if chunk_min < min_d2:
            min_d2 = chunk_min
    return min_d2


def _max_norm_sq(vectors):
    """Compute the maximum squared norm among vectors."""
    if len(vectors) == 0:
        return 0
    if isinstance(vectors, list):
        vecs = np.array(vectors, dtype=np.int64)
    else:
        vecs = vectors
    norms_sq = np.sum(vecs ** 2, axis=1)
    return np.max(norms_sq)


def _is_valid(vectors):
    """Check if the configuration satisfies min pairwise dist >= max norm."""
    if len(vectors) == 0:
        return False
    min_d2 = _min_pairwise_dist_sq(vectors)
    max_n2 = _max_norm_sq(vectors)
    return min_d2 >= max_n2


def _generate_candidate_pool(max_weight=2, max_norm=4):
    """Generate a pool of candidate integer vectors with small norms."""
    pool = []
    # Include vectors with entries in {-max_weight, ..., max_weight}
    # But limit to those with norm^2 <= max_norm^2
    # To keep pool manageable, focus on sparse vectors
    # Vectors with 1 non-zero entry
    for i in range(D):
        for v in range(1, max_weight + 1):
            vec = [0] * D
            vec[i] = v
            pool.append(tuple(vec))
            vec[i] = -v
            pool.append(tuple(vec))
    # Vectors with 2 non-zero entries
    for i, j in combinations(range(D), 2):
        for a in range(1, max_weight + 1):
            for b in range(1, max_weight + 1):
                for sa in (1, -1):
                    for sb in (1, -1):
                        vec = [0] * D
                        vec[i] = sa * a
                        vec[j] = sb * b
                        if a * a + b * b <= max_norm * max_norm:
                            pool.append(tuple(vec))
    # Vectors with 3 non-zero entries (only if norm allows)
    for i, j, k in combinations(range(D), 3):
        for a in range(1, max_weight + 1):
            for b in range(1, max_weight + 1):
                for c in range(1, max_weight + 1):
                    if a * a + b * b + c * c > max_norm * max_norm:
                        continue
                    for sa in (1, -1):
                        for sb in (1, -1):
                            for sc in (1, -1):
                                vec = [0] * D
                                vec[i] = sa * a
                                vec[j] = sb * b
                                vec[k] = sc * c
                                pool.append(tuple(vec))
    return pool


def construct():
    start_time = time.time()
    
    def norm_sq(v):
        return sum(x * x for x in v)
        
    # 1. Generate the Seed Configuration (Baseline)
    # Vectors with exactly two non-zero entries of +/-1
    seed_vectors = []
    for i, j in combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                seed_vectors.append(tuple(v))
    
    # Convert to numpy array for fast distance calculations
    seed_arr = np.array(seed_vectors, dtype=np.int64)
    current_min_d2 = 4.0  # Known min dist^2 for seed
    current_max_n2 = 2.0  # Known max norm^2 for seed
    current_count = len(seed_vectors)
    
    best_vectors = seed_vectors[:]
    best_score = current_count
    
    # 2. Generate a pool of additional candidate vectors
    # We want to try to add vectors to the seed configuration.
    # The constraint is: min_pairwise_dist^2 >= max_norm^2
    # If we add a vector v, the new max_norm^2 becomes max(2, ||v||^2).
    # The new min_dist^2 becomes min(4, min_dist(v, seed), ...).
    # So we need min_dist(v, seed)^2 >= new_max_norm^2.
    
    # Let's generate candidates with norm^2 in {2, 3, 4, 5, 6}
    # Note: We already have all norm^2=2 vectors with 2 non-zeros.
    # We can add norm^2=2 vectors with 1 non-zero? No, dist to seed might be small.
    # Let's just generate a broad pool of small norm vectors.
    
    pool = []
    # 1 non-zero entry
    for i in range(D):
        for val in range(1, 4): # 1, 2, 3
            vec = [0]*D
            vec[i] = val
            pool.append(tuple(vec))
            vec[i] = -val
            pool.append(tuple(vec))
            
    # 2 non-zero entries (include those not in seed, i.e., values not just +/-1)
    for i, j in combinations(range(D), 2):
        for a in range(1, 4):
            for b in range(1, 4):
                for sa in (1, -1):
                    for sb in (1, -1):
                        vec = [0]*D
                        vec[i] = sa*a
                        vec[j] = sb*b
                        # Skip if it's already in seed (a=1, b=1)
                        if a == 1 and b == 1:
                            continue
                        pool.append(tuple(vec))
                        
    # 3 non-zero entries (small values)
    for i, j, k in combinations(range(D), 3):
        for a in range(1, 3): # 1, 2
            for b in range(1, 3):
                for c in range(1, 3):
                    # Only allow small norms to keep max_norm low
                    if a*a + b*b + c*c > 6: continue
                    for sa in (1, -1):
                        for sb in (1, -1):
                            for sc in (1, -1):
                                vec = [0]*D
                                vec[i] = sa*a
                                vec[j] = sb*b
                                vec[k] = sc*c
                                pool.append(tuple(vec))
                                
    # 4 non-zero entries (very small)
    for i, j, k, l in combinations(range(D), 4):
        for a in range(1, 2): # 1
            for b in range(1, 2):
                for c in range(1, 2):
                    for d in range(1, 2):
                        vec = [0]*D
                        vec[i] = a
                        vec[j] = b
                        vec[k] = c
                        vec[l] = d
                        # Signs
                        for sa in (1, -1):
                            for sb in (1, -1):
                                for sc in (1, -1):
                                    for sd in (1, -1):
                                        vec[i] = sa*a
                                        vec[j] = sb*b
                                        vec[k] = sc*c
                                        vec[l] = sd*d
                                        pool.append(tuple(vec))

    # Remove duplicates and vectors already in seed
    seed_set = set(seed_vectors)
    pool = list(set(pool) - seed_set)
    
    # Sort pool by norm (ascending) to try to keep max_norm low
    pool.sort(key=norm_sq)
    
    # 3. Greedy Expansion from Seed
    # We maintain:
    # - current_arr: numpy array of selected vectors
    # - current_min_d2: minimum pairwise squared distance among selected
    # - current_max_n2: maximum squared norm among selected
    
    current_arr = seed_arr.copy()
    current_min_d2 = 4.0
    current_max_n2 = 2.0
    
    # We will try to add vectors from the pool.
    # To be more robust, we can try multiple random orderings of the pool.
    rng = np.random.RandomState(42)
    
    # Try a few restarts with different shuffles
    for restart in range(5):
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
            
        # Reset state to seed
        cur_arr = seed_arr.copy()
        cur_min_d2 = 4.0
        cur_max_n2 = 2.0
        cur_count = len(seed_vectors)
        
        # Shuffle pool
        shuffled_pool = pool[:]
        rng.shuffle(shuffled_pool)
        
        # Iterate through pool
        for v in shuffled_pool:
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
                
            v_arr = np.array(v, dtype=np.int64)
            v_n2 = norm_sq(v)
            
            # New max norm
            new_max_n2 = max(cur_max_n2, v_n2)
            
            # Check distance to all current vectors
            # dist^2 = ||v - u||^2 = ||v||^2 + ||u||^2 - 2 v.u
            # We need min(dist^2) >= new_max_n2
            
            # Calculate dot products
            dots = cur_arr @ v_arr
            # Calculate norms of current vectors (we can precompute or just use the fact that we track max)
            # Actually, we need ||u||^2 for each u to compute dist^2.
            # cur_n2s = np.sum(cur_arr**2, axis=1)
            
            # To save time, we can compute dist^2 directly
            diffs = cur_arr - v_arr
            d2s = np.sum(diffs**2, axis=1)
            min_d2_to_v = np.min(d2s)
            
            # New min pairwise distance
            new_min_d2 = min(cur_min_d2, min_d2_to_v)
            
            # Check constraint
            if new_min_d2 >= new_max_n2:
                # Accept
                cur_arr = np.vstack([cur_arr, v_arr])
                cur_min_d2 = new_min_d2
                cur_max_n2 = new_max_n2
                cur_count += 1
        
        if cur_count > best_score:
            best_score = cur_count
            best_vectors = cur_arr.tolist()
    
    # 4. Also try a pure greedy from scratch with different target norms?
    # The seed is very strong. Let's stick to expanding the seed for now.
    # If we have time, we can try other strategies.
    
    if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
        pass
        
    return np.array(best_vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
