import time
import itertools
import numpy as np

# Global state to cache the pre-computed priorities
_priority_cache = {}
_computation_done = False
_start_time = None

def _hamming_distance(v1, v2):
    """Calculate Hamming distance between two vectors."""
    return np.sum(v1 != v2)

def _precompute_priorities(n):
    """
    Pre-compute the priority of each vector using a Farthest-Point Sampling (FPS) 
    heuristic to maximize the minimum Hamming distance to the selected set, 
    while maintaining the cap set property (no affine lines).
    """
    global _priority_cache, _computation_done, _start_time
    
    if n in _priority_cache:
        return _priority_cache[n]
        
    _start_time = time.time()
    deadline = _start_time + 250  # Leave 50s buffer
    
    # Generate all vectors
    vectors = np.array(list(itertools.product([0, 1, 2], repeat=n)), dtype=int)
    N = len(vectors)
    
    selected_indices = set()
    forbidden_vectors = set()
    
    # min_dist[i] is the min Hamming distance from vectors[i] to any selected vector
    min_dist = np.full(N, np.inf)
    
    # List of indices that are still available (not selected and not forbidden)
    # We don't explicitly maintain a list, we just check conditions.
    
    priorities = [0.0] * N
    current_rank = 0.0
    
    # To speed up finding the best candidate, we can keep a list of indices
    # and filter them.
    available_indices = list(range(N))
    
    while time.time() < deadline:
        if not available_indices:
            break
            
        # Find the best candidate: max min_dist, not in forbidden
        best_idx = -1
        best_dist = -1
        
        # Optimization: If the number of available indices is large, 
        # we can sample or use a faster method. 
        # For now, linear scan is O(N). 
        # We can sort available_indices by min_dist desc once? 
        # min_dist changes every step.
        
        # Heuristic: Check the top K candidates from a previous sort?
        # Let's just do a full scan for simplicity, it's O(N).
        # N is at most ~10^6. 10^6 ops is fast in numpy if vectorized, 
        # but here we have a python loop over indices.
        # Let's use numpy to find the max efficiently.
        
        # Create a mask for available and not forbidden
        # This is tricky because forbidden is a set of tuples.
        # Converting vectors to tuples for set lookup is slow.
        # Alternative: Represent vectors as integers in base 3.
        
        # Let's switch to integer representation for faster hashing.
        # This requires regenerating or mapping.
        # Since we already have vectors as np array, let's convert to tuples once?
        # No, let's stick to numpy and use a boolean mask for forbidden.
        
        # Actually, maintaining a boolean mask `is_forbidden` of size N is better.
        
        if 'is_forbidden' not in locals():
            is_forbidden = np.zeros(N, dtype=bool)
            is_selected = np.zeros(N, dtype=bool)
            
        # Update available mask? 
        # We can just find the index of max min_dist where is_forbidden is False and is_selected is False.
        
        valid_mask = (~is_forbidden) & (~is_selected)
        if not np.any(valid_mask):
            break
            
        # Get max min_dist among valid
        # Set invalid to -1 for comparison
        valid_min_dist = np.where(valid_mask, min_dist, -1)
        best_idx = np.argmax(valid_min_dist)
        
        if valid_min_dist[best_idx] < 0:
            break
            
        u = vectors[best_idx]
        
        # Add to selected
        is_selected[best_idx] = True
        selected_indices.add(best_idx)
        
        # Update priorities
        priorities[best_idx] = -current_rank # Higher priority (less negative) for earlier selection?
        # The task says "descending priority". 
        # So the first selected should have the highest priority.
        # Let's assign priority = N - rank. Or just 1/rank?
        # Let's use 1.0 for first, 0.9 for second, etc.
        # Or simply, the order index.
        # If I return 0.0 for all, it's lex order.
        # I want the FPS order.
        # So priority[best_idx] = 1.0 / (current_rank + 1) ?
        # Or just a large number decreasing.
        priorities[best_idx] = 1000.0 - current_rank
        
        current_rank += 1
        
        # Update min_dist for all other vectors
        # min_dist[v] = min(min_dist[v], Hamming(v, u))
        # Vectorized:
        # distances = np.sum(vectors != u, axis=1)
        # min_dist = np.minimum(min_dist, distances)
        
        # But we only care about unselected, unforbidden ones?
        # No, updating all is fine, it's O(N*n).
        distances = np.sum(vectors != u, axis=1)
        min_dist = np.minimum(min_dist, distances)
        
        # Update forbidden set
        # For each existing selected vector v (including the new u? No, lines are formed by pairs)
        # If we add u, it forms lines with all previously selected vectors v.
        # The third point w satisfies u + v + w = 0 mod 3 => w = -(u+v) mod 3.
        # So for each v in selected_indices (excluding best_idx? No, including? 
        # The line is {u, v, w}. If v is in S, and we add u, we must forbid w.
        # Note: w might be u or v? 
        # If w=u, then 2u+v=0 => v=-2u=u mod 3. So u=v. But u is new, v is old. Distinct.
        # If w=v, then u+2v=0 => u=-2v=v mod 3. u=v. Distinct.
        # So w is distinct from u and v.
        
        # We need to mark w as forbidden.
        # Iterate through all previously selected vectors.
        for v_idx in selected_indices:
            if v_idx == best_idx:
                continue
            v = vectors[v_idx]
            # w = -(u + v) mod 3
            w = (- (u + v)) % 3
            # Find index of w in vectors
            # This is a search. 
            # To make this fast, we should have a mapping from vector to index.
            
            # Since we pre-computed vectors, we can create a dictionary mapping tuple->index.
            # This is expensive to create for large N, but we do it once.
            
            if 'vec_to_idx' not in locals():
                vec_to_idx = {tuple(v): i for i, v in enumerate(vectors)}
            
            w_tuple = tuple(w)
            if w_tuple in vec_to_idx:
                w_idx = vec_to_idx[w_tuple]
                is_forbidden[w_idx] = True
                
    # Assign low priority to unselected vectors
    for i in range(N):
        if priorities[i] == 0.0:
            priorities[i] = -1.0
            
    _priority_cache[n] = priorities
    _computation_done = True
    return priorities

def priority(vector, n):
    # vector is a tuple of ints
    if not _computation_done or n not in _priority_cache:
        # Trigger computation
        _precompute_priorities(n)
        
    # Get the index of this vector
    # We need to map vector to index.
    # We stored priorities in an array indexed by the order of itertools.product.
    # So we need to find the index of `vector` in that order.
    
    # Optimization: Cache the mapping for this n
    if 'vec_to_idx_cache' not in _priority_cache:
        # Re-generate or store mapping
        # We didn't store vec_to_idx in _priority_cache. Let's fix that.
        pass
        
    # Actually, let's store the mapping in the cache.
    # Let's refactor _precompute_priorities to store the mapping.
    
    # For now, let's assume the vector is passed in the same order as generated?
    # No, the evaluator passes arbitrary vectors.
    
    # We need a robust way to get the index.
    # We can compute the index directly:
    # vector = (v0, v1, ..., v_{n-1})
    # index = v0 * 3^(n-1) + v1 * 3^(n-2) + ... + v_{n-1}
    
    index = 0
    for i, val in enumerate(vector):
        index = index * 3 + val
        
    return _priority_cache[n][index]

# EVOLVE-BLOCK-START
def priority(vector, n):
    # vector is a tuple of ints
    if n not in _priority_cache:
        _precompute_priorities(n)
        
    # Compute index
    index = 0
    for val in vector:
        index = index * 3 + val
        
    return _priority_cache[n][index]
# EVOLVE-BLOCK-END
