import time
import math
import itertools
import random
import numpy as np

# EVOLVE-BLOCK-START

# Global state
_best_ordering = None
_search_done = False
_deadline = time.time() + 290  # 290s to be safe within 300s limit
_n_current = None

def _generate_all_vectors(n):
    """Generate all vectors in {0, 1, 2}^n as tuples."""
    return list(itertools.product([0, 1, 2], repeat=n))

def _vec_to_idx(vec, n):
    """Convert vector to integer index in [0, 3^n)."""
    idx = 0
    for v in vec:
        idx = idx * 3 + v
    return idx

def _idx_to_vec(idx, n):
    """Convert integer index to vector tuple."""
    vec = [0] * n
    for i in range(n - 1, -1, -1):
        vec[i] = idx % 3
        idx //= 3
    return tuple(vec)

def _simulate_greedy(ordering, n, all_vecs_idx):
    """
    Simulate greedy cap set construction with a given ordering of vector indices.
    Returns the size of the resulting cap set.
    Uses an incremental sum-set check for efficiency.
    """
    # all_vecs_idx is a list of indices corresponding to the vectors
    # ordering is a list of indices into all_vecs_idx? 
    # Let's assume ordering is a list of vector indices (0 to 3^n-1)
    
    N = 3 ** n
    # sums[i] is True if there exist a, c in current cap set such that a+c == i (mod 3)
    # We can represent sums as a boolean numpy array of size N
    sums = np.zeros(N, dtype=bool)
    
    cap_set = set()
    # To speed up iteration over cap_set, we might need a list, but set is fine for lookup.
    # However, we need to iterate over cap_set to update sums.
    # Let's keep a list of added vectors for fast iteration.
    added_vecs = []
    
    size = 0
    
    for idx in ordering:
        vec = _idx_to_vec(idx, n)
        # Target sum to check: -vec mod 3
        # target_idx = _vec_to_idx(tuple((-v) % 3 for v in vec), n)
        
        # Fast calculation of -vec mod 3 index
        # If vec is (v0, v1, ...), -vec is ((-v0)%3, ...)
        # idx calculation:
        # Let's precompute neg_idx for all possible idx?
        # Or just compute on the fly.
        
        # Optimization: Precompute neg_idx_map for all 3^n vectors
        # This is expensive for large n, but n is small.
        
        # Check if target is in sums
        # We need the index of -vec
        neg_vec = tuple((-v) % 3 for v in vec)
        target_idx = _vec_to_idx(neg_vec, n)
        
        if sums[target_idx]:
            # Conflict exists, cannot add
            continue
        
        # Add vec to cap set
        cap_set.add(idx)
        added_vecs.append(idx)
        size += 1
        
        # Update sums: add vec + a for all a in added_vecs (excluding itself? No, a and c distinct from v, but a,c can be same? 
        # Cap set condition: x,y,z distinct. 
        # If we add v, we form lines {v, a, c} where a+c = -v.
        # If a=c, then 2a = -v => v = a. But v is new, a is old. So v != a.
        # So a and c must be distinct?
        # The line is {a, c, v}. If a=c, the set is {a, v}. This is not a 3-point line.
        # So we only care about pairs (a, c) with a != c.
        # However, our `sums` array tracks if ANY pair (a, c) in S sums to target.
        # When we add v, we need to add sums v+a for all a in S.
        # These new sums correspond to pairs (v, a).
        # In the future, if we add w, we check if -w is in sums.
        # If -w = v+a, then {v, a, w} is a line.
        # Are v, a, w distinct?
        # w is new. v, a are in S.
        # If w == v, impossible (w new).
        # If w == a, impossible (w new).
        # So v, a, w are distinct.
        # So we just need to add v+a to sums.
        
        # Iterate over all a in added_vecs
        for a_idx in added_vecs:
            # sum = vec + a_vec mod 3
            # We can compute this on indices?
            # idx_add = (vec + a) mod 3 is not simply (idx_vec + idx_a) mod 3^n
            # We need to do component-wise addition.
            
            # To speed this up, precompute add_table[i][j] = index of (vec_i + vec_j) mod 3
            # This table is 3^n x 3^n. For n=8, 6561x6561 ~ 43M entries. 
            # Memory: 43M * 4 bytes ~ 172 MB. Acceptable?
            # Time to build: 43M ops. ~1s.
            # This would make the inner loop O(1) lookup instead of O(n) component-wise add.
            
            # Let's assume we have a global ADD_TABLE for the current n.
            # If not built, build it.
            
            sum_idx = ADD_TABLE[idx][a_idx]
            sums[sum_idx] = True
            
    return size

# Global cache for addition table
ADD_TABLE = None
_N_TABLE = 0

def _build_add_table(n):
    global ADD_TABLE, _N_TABLE
    if _N_TABLE == n and ADD_TABLE is not None:
        return
    N = 3 ** n
    # ADD_TABLE[i][j] = index of (vec_i + vec_j) mod 3
    # We can build this efficiently.
    # vec_i is i in base 3.
    # sum is digit-wise (d_i + d_j) % 3.
    
    # Precompute digit values for all indices?
    # Or just loop.
    # N=6561. N^2 = 43M.
    # Let's try to vectorize or optimize.
    
    # For each i, vec_i is known.
    # For each j, vec_j is known.
    
    # Actually, we can compute the index directly.
    # idx = sum(d_k * 3^(n-1-k))
    # sum_idx = sum( (d_i_k + d_j_k) % 3 * 3^(n-1-k) )
    
    # This is still O(N^2 * n).
    # 43M * 8 = 344M ops. Might be slow in Python.
    # Let's use NumPy to build it faster?
    
    # Create array of all vectors
    vecs = np.array([_idx_to_vec(i, n) for i in range(N)])
    # This is N x n array.
    
    # We want ADD_TABLE[i, j] = index of (vecs[i] + vecs[j]) % 3
    # This is hard to vectorize directly to index without loops.
    
    # Alternative: Just do the O(n) calculation in the simulation loop?
    # Inner loop runs size * N times.
    # Size ~ 1000-2000. N = 6561.
    # Total inner iterations ~ 10^7.
    # O(n) per iteration => 10^7 * 8 = 8 * 10^7 ops.
    # This is acceptable in Python if optimized (no function calls in inner loop).
    
    _N_TABLE = n
    # We will skip building the full table to save startup time and memory, 
    # and rely on the fact that n is small.
    # We will implement a fast inline addition in the simulator.
    
    # To make the simulator faster, we will precompute the digits of all vectors.
    global DIGITS
    DIGITS = np.array([_idx_to_vec(i, n) for i in range(N)], dtype=np.uint8)

DIGITS = None

def _simulate_greedy_fast(ordering, n):
    """
    Optimized simulation using precomputed digits.
    """
    global DIGITS
    N = 3 ** n
    sums = np.zeros(N, dtype=bool)
    added_indices = []
    size = 0
    
    # Precompute negative indices?
    # neg_idx[i] = index of -vec_i mod 3
    # -vec_i is (3 - d) % 3 = (-d) % 3.
    # If d=0, 0. If d=1, 2. If d=2, 1.
    # This is just 2-d if d!=0, 0 if d=0? No.
    # 0->0, 1->2, 2->1.
    # It's a permutation.
    
    # Let's just compute target_idx on the fly using DIGITS.
    
    for idx in ordering:
        # Get digits of idx
        # We need target = -vec[idx]
        # We can precompute NEG_IDX array.
        
        # Check sums[target]
        if sums[NEG_IDX[idx]]:
            continue
            
        # Add idx
        added_indices.append(idx)
        size += 1
        
        # Update sums
        # For each a in added_indices:
        #   sum_vec = vec[idx] + vec[a] mod 3
        #   sums[sum_idx] = True
        
        # To speed up, we can iterate over added_indices and compute sum_idx.
        # sum_idx = sum( (DIGITS[idx][k] + DIGITS[a][k]) % 3 * 3^(n-1-k) )
        
        # This is O(n) per a.
        # Total O(n * size^2) over the whole run? No, O(n * size * N).
        
        # Let's try to vectorize the update for a single idx.
        # We have DIGITS[added_indices] (shape k x n)
        # We have DIGITS[idx] (shape n)
        # sum_digits = (DIGITS[added_indices] + DIGITS[idx]) % 3
        # Now we need to convert these sum_digits back to indices.
        # idx_val = sum( sum_digits[:, k] * 3^(n-1-k) )
        
        # This can be done with matrix multiplication or dot product.
        # Powers = [3^(n-1), ..., 1]
        # new_indices = sum_digits @ Powers
        
        # This is O(k*n) but vectorized in NumPy.
        
        if added_indices:
            # Get digits of all added vectors
            # DIGITS is N x n.
            # added_indices is a list.
            # We can use DIGITS[added_indices]
            # But added_indices grows. 
            # We can maintain a numpy array of added indices.
            
            # For simplicity and speed in Python, let's stick to the loop 
            # but optimize the index calculation.
            
            # Precompute POWERS = [3^(n-1-i) for i in range(n)]
            # sum_idx = 0
            # for k in range(n):
            #   sum_idx += (DIGITS[idx][k] + DIGITS[a][k]) % 3 * POWERS[k]
            
            # This is still a Python loop.
            
            # Let's assume n is small enough that this is fast.
            # For n=8, 8 ops per a.
            # If size=2000, 2000*8 = 16000 ops per new vector.
            # Total vectors 6561.
            # Total ops ~ 6561 * 16000 / 2 (avg size) ~ 50M ops.
            # This might be ~5-10 seconds per simulation.
            # We have 300s. We can do ~30-50 simulations.
            
            # Let's try to implement the NumPy vectorized update.
            pass
            
    return size

# Precomputation for fast simulation
NEG_IDX = None
POWERS = None
_N_PRECOMPUTED = 0

def _precompute(n):
    global NEG_IDX, POWERS, _N_PRECOMPUTED, DIGITS
    if _N_PRECOMPUTED == n:
        return
    _N_PRECOMPUTED = n
    N = 3 ** n
    
    # Build DIGITS
    DIGITS = np.zeros((N, n), dtype=np.uint8)
    for i in range(N):
        v = i
        for k in range(n - 1, -1, -1):
            DIGITS[i, k] = v % 3
            v //= 3
            
    # Build NEG_IDX
    NEG_IDX = np.zeros(N, dtype=np.int32)
    for i in range(N):
        # -vec mod 3
        # 0->0, 1->2, 2->1
        # We can compute this from DIGITS
        neg_digits = (3 - DIGITS[i]) % 3
        # Convert back to index
        # idx = sum(neg_digits[k] * 3^(n-1-k))
        # Using POWERS
        NEG_IDX[i] = np.dot(neg_digits, POWERS)
        
    # Build POWERS
    POWERS = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int64)
    
    # Rebuild NEG_IDX with POWERS
    for i in range(N):
        neg_digits = (3 - DIGITS[i]) % 3
        NEG_IDX[i] = np.dot(neg_digits, POWERS)

def _simulate_greedy_vectorized(ordering, n):
    """
    Simulate greedy cap set using NumPy vectorization for sum updates.
    """
    global DIGITS, NEG_IDX, POWERS
    N = 3 ** n
    sums = np.zeros(N, dtype=bool)
    
    # added_indices as numpy array for fast indexing
    added_arr = np.array([], dtype=np.int32)
    size = 0
    
    for idx in ordering:
        if sums[NEG_IDX[idx]]:
            continue
            
        # Add idx
        # Update sums with idx + a for all a in added_arr
        if size > 0:
            # Get digits of idx
            d_idx = DIGITS[idx]
            # Get digits of all added
            d_added = DIGITS[added_arr]
            # Sum digits
            sum_digits = (d_added + d_idx) % 3
            # Convert to indices
            # new_indices = sum_digits @ POWERS
            new_indices = np.dot(sum_digits, POWERS)
            # Update sums
            sums[new_indices] = True
            
        # Add idx to added_arr
        added_arr = np.append(added_arr, idx)
        size += 1
        
    return size

def _local_search(n, time_limit):
    """
    Run local search to find a better ordering.
    """
    global _best_ordering
    
    all_indices = list(range(3 ** n))
    
    # Initial ordering: Random
    random.seed(42)
    current_ordering = all_indices.copy()
    random.shuffle(current_ordering)
    
    # Evaluate initial
    current_size = _simulate_greedy_vectorized(current_ordering, n)
    best_size = current_size
    best_ordering = current_ordering.copy()
    
    start_time = time.time()
    
    # Local search loop
    while time.time() < start_time + time_limit:
        # Pick two random indices to swap
        i = random.randint(0, len(current_ordering) - 1)
        j = random.randint(0, len(current_ordering) - 1)
        
        if i == j:
            continue
            
        # Swap
        current_ordering[i], current_ordering[j] = current_ordering[j], current_ordering[i]
        
        # Evaluate
        new_size = _simulate_greedy_vectorized(current_ordering, n)
        
        if new_size > current_size:
            current_size = new_size
            if current_size > best_size:
                best_size = current_size
                best_ordering = current_ordering.copy()
        else:
            # Revert
            current_ordering[i], current_ordering[j] = current_ordering[j], current_ordering[i]
            
    _best_ordering = best_ordering

def _run_offline_search(n):
    global _search_done, _deadline
    if _search_done:
        return
    
    _search_done = True
    
    # Precompute
    _precompute(n)
    
    # Time budget for search
    search_time = min(250, _deadline - time.time() - 5)
    if search_time <= 0:
        # Fallback to simple hash
        _best_ordering = list(range(3 ** n))
        return
        
    _local_search(n, search_time)

def priority(vector, n):
    global _search_done, _best_ordering, _n_current
    
    if not _search_done:
        _n_current = n
        _run_offline_search(n)
        
    if _best_ordering is None:
        return float(hash(vector))
        
    # The evaluator expects a priority value.
    # We have an ordering. We need to map vector to its rank in the ordering.
    # But wait, the evaluator sorts by priority.
    # If we return the *index* in the ordering, lower index = higher priority?
    # The contract says "descending priority".
    # So highest priority value is added first.
    # So if ordering[0] is the first vector to be added, it should have the highest priority.
    # So priority(ordering[0]) > priority(ordering[1]) > ...
    
    # We need a map from vector to its position in _best_ordering.
    # But we don't have a global map cached.
    # We can build it once.
    
    global _rank_map
    if _rank_map is None or _rank_map_n != n:
        _rank_map = {}
        _rank_map_n = n
        for rank, idx in enumerate(_best_ordering):
            # Higher priority for lower rank
            _rank_map[idx] = float(len(_best_ordering) - rank)
            
    # Convert vector to idx
    idx = _vec_to_idx(vector, n)
    return _rank_map.get(idx, 0.0)

_rank_map = None
_rank_map_n = 0

# EVOLVE-BLOCK-END
