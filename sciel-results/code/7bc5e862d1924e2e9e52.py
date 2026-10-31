"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # safety margin before 300s limit

def _remaining_time():
    return _DEADLINE - time.time()

def _check_deadline():
    if time.time() >= _DEADLINE:
        return True
    return False

def _is_cap_set_check(vec, cap_set, n):
    """
    Check if adding vec to cap_set keeps it a cap set.
    A cap set has no three distinct elements a, b, c with a+b+c = 0 mod 3.
    Equivalently, no three distinct elements form an affine line.
    For a new vector v, we need to check that for any two existing vectors a, b,
    the third point c = -a - b + 2*v ... no wait.
    
    Actually, a cap set in F_3^n has no 3-term arithmetic progression.
    Three points a, b, c form a line if a + b + c = 0 (mod 3) where a, b, c are distinct.
    Equivalently, for any two points a, b in the set, the third point c = -a - b (mod 3) 
    must not be in the set (and c != a, c != b).
    
    Wait, let me be more careful. In F_3^n, three distinct points a, b, c form an affine line
    if and only if a + b + c = 0 (mod 3). This is because the line through a and b is 
    {a, b, c} where c = 2b - a = -a - b (mod 3) since 2 = -1 mod 3.
    So c = -a - b mod 3.
    
    So when adding v to an existing cap set S, we need to check that for every pair (a, b) in S,
    the point c = -a - b mod 3 is not equal to v. But we also need to check pairs involving v.
    
    Actually, the standard check: v can be added to S iff there is no pair (a, b) in S such that
    -a - b = v (mod 3), i.e., a + b + v = 0. And we also need that v is not already in S.
    
    But wait, we also need to check that v doesn't form a line with any two existing points.
    A line is {x, y, z} with x+y+z=0. If v is added, the new lines involving v are:
    {v, a, b} where a, b in S and v + a + b = 0, i.e., a + b = -v.
    So we need to check: is there a pair (a, b) in S with a + b = -v mod 3?
    
    Also, we need v not in S.
    
    For efficiency, we can precompute for the current cap set S, for each possible vector w,
    the number of pairs (a, b) in S with a + b = w. But that's expensive for large n.
    
    Alternative: for each a in S, check if -v - a is in S. This is O(|S|) per query.
    """
    neg_v = (-vec) % 3
    for a in cap_set:
        b_candidate = (neg_v - a) % 3
        if b_candidate in cap_set_set and b_candidate != a:
            return False
    return True

def priority(vector, n):
    # Uniform priority: greedy insertion in lexicographic order.
    return 0.0

def _run_cap_set_search(n, max_iters=None):
    """
    Greedy cap set construction with smart priority.
    """
    if n <= 0:
        return 0
    
    # Precompute all vectors as tuples
    # For small n, enumerate all 3^n vectors
    total_vectors = 3 ** n
    
    # Use numpy for vector operations
    all_vectors = np.array(np.unravel_index(np.arange(total_vectors), [3]*n), dtype=np.int64).T
    # all_vectors is (total_vectors, n)
    
    # Convert to list of tuples for set membership
    vec_tuples = [tuple(row) for row in all_vectors]
    
    # Priority: Hamming weight close to n/2, with hash tie-breaker
    # Weight = number of non-zero elements? Or sum of elements?
    # Let's use the sum of elements mod 3... no, let's use the actual sum.
    # Actually, for cap sets, the "middle" weights are better.
    # Let's define weight as the number of positions where the value is not 0.
    # Or just use the sum of the vector values.
    
    # Compute a priority score for each vector
    # Higher priority = better candidate
    # Use: -|weight - n/2| where weight is sum of elements
    # Add a small hash-based tiebreaker
    
    vec_sums = np.sum(all_vectors, axis=1)
    target_weight = n / 2.0
    weight_dist = np.abs(vec_sums - target_weight)
    
    # Hash tiebreaker: use a deterministic hash of the vector
    # For tie-breaking, we can use the index itself or a simple hash
    # Let's use: priority = -weight_dist + small_random
    
    # Actually, let's sort by (weight_dist, hash)
    # For hash, use a simple deterministic function
    
    # Create priority array
    # We want to process vectors in order of increasing weight_dist, with tie-breaker
    indices = np.arange(total_vectors)
    
    # Simple hash: use the vector's index in a permuted order
    # Or just use the index as tiebreaker
    # To avoid lexicographic bias, let's use a pseudo-random permutation based on index
    # But it must be deterministic
    
    # Use a simple hash function
    def vec_hash(idx):
        # Simple deterministic hash
        h = idx
        h = (h * 2654435761) % (2**32)
        return h
    
    # Create sort keys
    # Primary: weight_dist (lower is better)
    # Secondary: hash (for tie-breaking, deterministic pseudo-random)
    hashes = np.array([vec_hash(i) for i in indices], dtype=np.int64)
    
    # Sort by (weight_dist, hash)
    sort_order = np.lexsort((hashes, weight_dist))
    
    # Now greedily add vectors in this order
    cap_set = []
    cap_set_set = set()
    
    # For efficiency, we can precompute pair sums
    # But let's start with the simple O(|S|) check
    
    max_iters = max_iters or total_vectors
    iters = 0
    
    for idx in sort_order:
        if _check_deadline():
            break
        if iters >= max_iters:
            break
        
        v = all_vectors[idx]
        v_tuple = vec_tuples[idx]
        
        if v_tuple in cap_set_set:
            continue
        
        # Check if v can be added
        neg_v = (-v) % 3
        can_add = True
        for a in cap_set:
            b_candidate = (neg_v - a) % 3
            if tuple(b_candidate) in cap_set_set and tuple(b_candidate) != a:
                can_add = False
                break
        
        if can_add:
            cap_set.append(v)
            cap_set_set.add(v_tuple)
        
        iters += 1
    
    return len(cap_set)

def _run_cap_set_search_optimized(n):
    """
    Optimized greedy cap set construction.
    Uses a more efficient collision check.
    """
    if n <= 0:
        return 0
    
    total_vectors = 3 ** n
    
    if total_vectors > 2000000:  # Too large for this approach
        return _run_cap_set_search(n)
    
    all_vectors = np.array(np.unravel_index(np.arange(total_vectors), [3]*n), dtype=np.int64).T
    vec_tuples = [tuple(row) for row in all_vectors]
    
    vec_sums = np.sum(all_vectors, axis=1)
    target_weight = n / 2.0
    weight_dist = np.abs(vec_sums - target_weight)
    
    def vec_hash(idx):
        h = idx
        h = (h * 2654435761) % (2**32)
        return h
    
    hashes = np.array([vec_hash(i) for i in range(total_vectors)], dtype=np.int64)
    sort_order = np.lexsort((hashes, weight_dist))
    
    cap_set = []
    cap_set_set = set()
    
    # Precompute for faster lookup: store vectors as tuples in a set
    # For the collision check, we need: for new v, check if exists a in S such that -v-a in S
    
    for idx in sort_order:
        if _check_deadline():
            break
        
        v = all_vectors[idx]
        v_tuple = vec_tuples[idx]
        
        if v_tuple in cap_set_set:
            continue
        
        neg_v = (-v) % 3
        can_add = True
        for a in cap_set:
            b_candidate = (neg_v - a) % 3
            b_tuple = tuple(b_candidate)
            if b_tuple in cap_set_set and b_tuple != a:
                can_add = False
                break
        
        if can_add:
            cap_set.append(v)
            cap_set_set.add(v_tuple)
    
    return len(cap_set)

# The actual priority function needs to work with the greedy framework
# The evaluator calls priority(vector, n) to get a float, and adds vectors in descending order
# We need to return a priority that leads to a good cap set

# Strategy: return a priority based on Hamming weight closeness to n/2
# with a deterministic hash tiebreaker

def _compute_priority_cache(n):
    """Precompute priorities for all vectors of length n."""
    total_vectors = 3 ** n
    all_vectors = np.array(np.unravel_index(np.arange(total_vectors), [3]*n), dtype=np.int64).T
    vec_sums = np.sum(all_vectors, axis=1)
    target_weight = n / 2.0
    weight_dist = np.abs(vec_sums - target_weight)
    
    def vec_hash(idx):
        h = idx
        h = (h * 2654435761) % (2**32)
        return h
    
    hashes = np.array([vec_hash(i) for i in range(total_vectors)], dtype=np.float64)
    # Normalize hash to [0, 1]
    hashes = hashes / (2**32)
    
    # Priority: higher is better
    # We want low weight_dist and high hash (for tie-breaking)
    # Combine: priority = -weight_dist + small * hash
    # But we need to ensure the primary sort is by weight_dist
    # So: priority = -weight_dist * large_factor + hash
    
    # Actually, let's make weight_dist the dominant term
    # priority = -weight_dist + hash * 0.001
    # This way, vectors with lower weight_dist have higher priority
    # And among ties, higher hash has slightly higher priority
    
    priorities = -weight_dist + hashes * 0.001
    
    # Map from vector tuple to priority
    vec_tuples = [tuple(row) for row in all_vectors]
    priority_map = {}
    for i, vt in enumerate(vec_tuples):
        priority_map[vt] = priorities[i]
    
    return priority_map

# Cache for priority maps
_priority_cache = {}

def priority(vector, n):
    """
    Return priority for a vector. Higher priority = processed first.
    Uses Hamming weight closeness to n/2 with hash tiebreaker.
    """
    if n not in _priority_cache:
        _priority_cache[n] = _compute_priority_cache(n)
    
    vec_tuple = tuple(vector)
    return _priority_cache[n].get(vec_tuple, 0.0)

# EVOLVE-BLOCK-END
