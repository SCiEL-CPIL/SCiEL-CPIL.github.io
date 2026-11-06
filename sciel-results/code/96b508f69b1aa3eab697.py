import random
import time
import sys
import numpy as np

# Global cache for priority sets
_PRIORITY_CACHE = {}
_DEADLINE = None

def _generate_all_vectors(n):
    """Generate all 3^n vectors as a numpy array of shape (3^n, n) with dtype int8."""
    total = 3 ** n
    # Efficient generation using base-3 expansion
    # We can use a trick with broadcasting or just loop since n is small (likely <= 10-12)
    vectors = np.zeros((total, n), dtype=np.int8)
    for i in range(n):
        # Digit i changes with period 3^i
        # Pattern: 0 repeated 3^i, 1 repeated 3^i, 2 repeated 3^i, ...
        block = 3 ** i
        pattern = np.tile([0, 1, 2], (total // (3 * block)))
        # Adjust length if necessary (though total is divisible by 3*block for i<n)
        if len(pattern) < total:
            # This shouldn't happen if n is the dimension and we are generating 3^n
            pass 
        vectors[:, i] = pattern
    return vectors

def _is_cap_set_numpy(S_indices, vectors, n):
    """
    Check if a set of indices into vectors forms a cap set.
    S_indices: set of int indices
    vectors: numpy array (3^n, n)
    Returns True if cap set.
    """
    S = set(S_indices)
    for i in S:
        v_i = vectors[i]
        for j in S:
            if i == j:
                continue
            v_j = vectors[j]
            # Third point w = 2*v_j - v_i (mod 3)
            # w = (2*v_j - v_i) % 3
            w = (2 * v_j - v_i) % 3
            # Check if w is in S
            # We need to map w to an index. 
            # This is slow if we do it every time. 
            # For local search, we maintain a dictionary mapping tuple(vector) -> index.
            pass
    # This function is too slow for direct use in a loop. 
    # We will implement the logic inline in the local search with pre-computed maps.
    return True

def _local_search(initial_set, vectors, n, time_limit):
    """
    Perform local search to improve the cap set.
    initial_set: set of int indices
    vectors: numpy array (3^n, n)
    Returns the improved set of int indices.
    """
    start_time = time.time()
    S = set(initial_set)
    
    # Build a mapping from vector tuple to index for O(1) lookups
    # This is expensive to build, but we only do it once per n.
    # However, we can't easily pass this from the outer scope if we re-initialize.
    # Let's build it here or pass it.
    
    # To save time, we assume the caller provides the map or we build a global one.
    # For simplicity and speed in this function, let's build a local map if not provided?
    # No, building a map of 3^n entries takes time. 
    # Better: The caller should provide the map.
    # Let's modify the signature to accept the map.
    
    # Actually, let's just use the vectors array and compute the index on the fly?
    # Index of vector v is sum(v[i] * 3^i).
    # This is fast.
    
    pow3 = [1] * n
    for i in range(1, n):
        pow3[i] = pow3[i-1] * 3
        
    def vec_to_idx(v):
        idx = 0
        for i in range(n):
            idx += v[i] * pow3[i]
        return idx

    # Precompute forbidden points for the current set?
    # No, we update dynamically.
    
    # List of candidates to try adding
    # We will randomly pick a vector not in S.
    # If adding it creates an AP, we try to resolve.
    
    # Optimization: Keep a list of "conflicts" or just try random additions.
    
    total_vectors = 3 ** n
    indices = list(range(total_vectors))
    
    # We will iterate until time runs out.
    # Each step:
    # 1. Pick a random vector v not in S.
    # 2. Check if S U {v} is a cap set.
    #    This requires checking all pairs (a, v) in S.
    #    For each a, w = 2v - a. If w in S, conflict.
    #    Complexity: O(|S|) per check.
    # 3. If conflict, find a vector a in S that conflicts with v.
    #    Try removing a.
    #    Check if (S - {a}) U {v} is a cap set.
    #    This is also O(|S|).
    #    If valid, accept.
    
    # To speed up, we can pre-compute for each v in S, the set of vectors that would conflict with it?
    # That's too much memory.
    
    # Let's stick to the simple O(|S|) check.
    # For n=10, |S| ~ 5000. 5000 ops per check.
    # 300s / (5000 ops * 10ns) ~ 600,000 iterations. This is feasible.
    
    while time.time() - start_time < time_limit:
        # Pick a random candidate
        v_idx = random.choice(indices)
        if v_idx in S:
            continue
        
        v = vectors[v_idx]
        
        # Check if adding v creates a conflict
        conflict_a = None
        for a_idx in S:
            a = vectors[a_idx]
            # w = 2v - a
            w = (2 * v - a) % 3
            w_idx = vec_to_idx(w)
            if w_idx in S:
                conflict_a = a_idx
                break
        
        if conflict_a is None:
            # Safe to add
            S.add(v_idx)
        else:
            # Try to swap: remove conflict_a, add v
            # Check if (S - {conflict_a}) U {v} is a cap set
            # We need to check if v conflicts with any other a in S (excluding conflict_a)
            # And if removing conflict_a breaks any existing APs? 
            # No, removing a point never creates an AP.
            # So we only need to check if v conflicts with any a in S \ {conflict_a}.
            
            valid_swap = True
            for a_idx in S:
                if a_idx == conflict_a:
                    continue
                a = vectors[a_idx]
                w = (2 * v - a) % 3
                w_idx = vec_to_idx(w)
                if w_idx in S and w_idx != v_idx: # w_idx != v_idx is always true if v not in S
                    # Wait, w_idx could be conflict_a?
                    # If w_idx == conflict_a, then the AP is v, a, conflict_a.
                    # Since we are removing conflict_a, this AP is broken.
                    # So if w_idx == conflict_a, it's OK.
                    if w_idx != conflict_a:
                        valid_swap = False
                        break
            
            if valid_swap:
                S.remove(conflict_a)
                S.add(v_idx)
                
    return S

def priority(vector, n):
    global _DEADLINE
    
    if _DEADLINE is None:
        # Set deadline to 320s from start of first call
        # But we don't know when the "program" started.
        # We assume the first call is near the start.
        _DEADLINE = time.time() + 320.0

    if n not in _PRIORITY_CACHE:
        start_time = time.time()
        # Time budget for search: leave 10s for safety
        time_budget = min(300.0 - (time.time() - start_time), 310.0)
        
        # Generate vectors
        vectors = _generate_all_vectors(n)
        
        # 1. Generate a good initial seed using algebraic construction
        # For n, a cap set can be found in a linear subspace.
        # Simple construction: 
        # If n is even, use the set of vectors with sum 0 and sum of squares 0?
        # Actually, the largest known cap sets in F_3^n are often constructed via 
        # "linear codes" or "sum-free sets".
        
        # A very simple and robust seed:
        # Take all vectors with exactly 2 non-zero coordinates? No, that's not linear.
        
        # Let's use a greedy approach for the SEED only, but very fast?
        # Or use a known small cap set and extend?
        
        # Better: Use the "linear" construction.
        # A set S is a cap set if it contains no 3-term AP.
        # The set {x : sum(x_i) = 0} is not necessarily a cap set.
        # Example: 1, 1, 1 -> sum 3=0. 1, 1, 1 is not an AP? 
        # AP: a, b, c. a+c=2b.
        # If sum(a)=0, sum(c)=0, then sum(2b)=0 => 2*sum(b)=0 => sum(b)=0.
        # So the hyperplane sum(x)=0 contains only vectors that are part of APs entirely within the hyperplane.
        # It does NOT guarantee it is a cap set.
        
        # However, the set of vectors with weight 0 (just 0) is a cap set.
        # The set of vectors with weight 1? {e_i}. 
        # AP: e_i, e_j, e_k. e_i + e_k = 2e_j.
        # If i!=j!=k, e_i+e_k != 2e_j.
        # If i=j=k, e_i+e_i = 2e_i. AP: e_i, e_i, e_i (degenerate).
        # Usually distinct elements.
        # So all unit vectors form a cap set?
        # e1, e2, e3. e1+e3 = (1,0,1). 2e2 = (0,2,0). Not equal.
        # So yes, the set of all vectors with Hamming weight 1 is a cap set.
        # Size: n.
        
        # We can do better.
        # Vectors with weight 2?
        # e1+e2, e1+e3, e2+e3.
        # (e1+e2) + (e2+e3) = e1 + 2e3. 2(e1+e3) = 2e1 + 2e3.
        # e1 + 2e3 != 2e1 + 2e3 (unless e1 = 2e1 => e1=0).
        # So weight 2 vectors are also a cap set?
        # Generally, vectors with weight 1 or 2 form a cap set?
        # Let's check weight 1 and 2 together.
        # e1, e1+e2.
        # e1 + (e1+e2) = 2e1+e2.
        # 2 * ? 
        # If we have e1, e2, e1+e2.
        # e1 + (e1+e2) = 2e1 + e2.
        # 2e2 = (0, 2, 0).
        # 2e1+e2 = (2, 1, 0). Not equal.
        
        # It is known that the set of vectors with weight 1 or 2 is a cap set in F_3^n.
        # Size: n + n(n-1)/2 = n(n+1)/2.
        # For n=10, size = 55.
        # This is a very weak seed.
        
        # Better seed: Random Greedy for the first 5 seconds to get a decent seed, then Local Search.
        # The current program does Random Greedy for 20s. 
        # Let's do Random Greedy for 5s to get a seed, then Local Search for the rest.
        
        # Implement a fast random greedy
        indices = list(range(3**n))
        random.shuffle(indices)
        S_seed = set()
        seed_time_limit = 5.0
        seed_start = time.time()
        
        # Precompute powers of 3 for index conversion
        pow3 = [1] * n
        for i in range(1, n):
            pow3[i] = pow3[i-1] * 3
            
        def vec_to_idx(v):
            idx = 0
            for i in range(n):
                idx += v[i] * pow3[i]
            return idx

        # Fast greedy
        # We need to check conflicts quickly.
        # For each v, check against S_seed.
        # This is O(|S|) per v.
        # 5 seconds is enough for a decent seed.
        
        for idx in indices:
            if time.time() - seed_start > seed_time_limit:
                break
            if idx in S_seed:
                continue
            v = vectors[idx]
            conflict = False
            for a_idx in S_seed:
                a = vectors[a_idx]
                w = (2 * v - a) % 3
                w_idx = vec_to_idx(w)
                if w_idx in S_seed:
                    conflict = True
                    break
            if not conflict:
                S_seed.add(idx)
        
        # Now run Local Search with the remaining time
        remaining_time = time_budget - (time.time() - start_time) - 5.0 # 5s for seed
        if remaining_time < 10:
            remaining_time = 10 # Minimum search time
        
        S_final = _local_search(S_seed, vectors, n, remaining_time)
        
        # Convert final set to a set of tuples for the cache
        # The priority function expects `vector` (a tuple) to be in the cache.
        # So we must store the tuples, not the indices.
        S_tuples = set()
        for idx in S_final:
            S_tuples.add(tuple(vectors[idx]))
            
        _PRIORITY_CACHE[n] = S_tuples

    return 1.0 if vector in _PRIORITY_CACHE[n] else 0.0
