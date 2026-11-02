"""Kissing number in dimension 11: maximize number of integer vectors with min pairwise distance >= max norm."""
import time
import numpy as np

D = 11
DEADLINE = 115.0  # seconds, safety margin under 120s

def construct():
    t_start = time.time()
    
    # We want to maximize k such that for all pairs i != j: ||v_i - v_j|| >= max_j ||v_j||
    # Equivalently: for all pairs i != j: ||v_i - v_j||^2 >= max_j ||v_j||^2
    # And ||v_i||^2 >= 1 (nonzero)
    
    # Strategy: Use vectors of a fixed squared norm m. Then the constraint becomes:
    # For all i != j: ||v_i - v_j||^2 >= m
    # ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2 v_i · v_j = 2m - 2 v_i · v_j >= m
    # So v_i · v_j <= m/2
    
    # For integer vectors, v_i · v_j is integer, so v_i · v_j <= floor(m/2)
    
    # Candidate pool: all nonzero integer vectors with ||v||^2 = m
    # For small m, we can enumerate efficiently.
    
    # Let's try m = 2 (vectors with two ±1 entries and rest 0)
    # This gives the seed: C(11,2) * 4 = 55 * 4 = 220 vectors
    # Dot product between two such vectors: they can share 0, 1, or 2 positions.
    # If they share 2 positions with same signs: dot = 2. floor(2/2) = 1. Conflict!
    # So we can't just take all of them.
    
    # For m = 3: vectors with either (three ±1) or (one ±sqrt(3))... no, integer only.
    # So three ±1 entries. C(11,3) * 8 = 165 * 8 = 1320 vectors.
    # Dot product: can be -3,-2,-1,0,1,2,3. floor(3/2) = 1. So dot must be <= 1.
    # Conflict if dot >= 2, i.e., share 2 positions with same signs.
    
    # For m = 4: vectors with four ±1 entries. C(11,4) * 16 = 330 * 16 = 5280 vectors.
    # floor(4/2) = 2. Conflict if dot >= 3, i.e., share 3+ positions with same signs.
    
    # For m = 5: five ±1 entries. C(11,5) * 32 = 462 * 32 = 14784 vectors.
    # floor(5/2) = 2. Conflict if dot >= 3.
    
    # For m = 6: six ±1 entries. C(11,6) * 64 = 462 * 64 = 29568 vectors.
    # floor(6/2) = 3. Conflict if dot >= 4.
    
    # Larger m gives more candidates but harder constraints. Let's try m=4 first.
    
    def generate_vectors_for_m(m):
        """Generate all nonzero integer vectors in R^11 with squared norm m.
        For simplicity, we only consider vectors with entries in {-1, 0, 1}."""
        if m <= 0:
            return []
        
        vectors = []
        
        # Only consider vectors with entries in {-1, 0, 1} for now
        # Number of nonzero entries k, with k <= m and k <= D
        # Sum of squares = k = m (since each nonzero is ±1)
        # So we need exactly m nonzero entries, each ±1
        
        if m > D:
            return []
        
        from itertools import combinations, product
        
        # Choose m positions out of D
        for positions in combinations(range(D), m):
            # Assign signs
            for signs in product([1, -1], repeat=m):
                v = [0] * D
                for pos, sign in zip(positions, signs):
                    v[pos] = sign
                vectors.append(v)
        
        return vectors
    
    # Start with m=2 (the seed), then try to improve
    # Actually, let's directly try m=3, 4, 5, 6 and see which gives the largest valid set
    
    # For each m, we need to find the maximum subset of vectors with pairwise dot <= floor(m/2)
    # This is a maximum independent set problem on a conflict graph.
    
    # For efficiency, let's use a greedy approach with randomization.
    
    def greedy_max_independent(vectors, threshold, time_limit=30):
        """Find a large independent set using greedy random restarts.
        Conflict: dot product > threshold."""
        n = len(vectors)
        if n == 0:
            return []
        
        # Precompute dot products? Too expensive for large n.
        # Use greedy: repeatedly pick a random vector, remove all conflicting ones.
        
        best_set = []
        deadline = time.time() + time_limit
        
        for restart in range(50):
            if time.time() > deadline:
                break
            
            remaining = set(range(n))
            chosen = []
            
            # Random order
            order = list(remaining)
            np.random.shuffle(order)
            
            for idx in order:
                if time.time() > deadline:
                    break
                if idx not in remaining:
                    continue
                
                chosen.append(idx)
                v = vectors[idx]
                
                # Remove all conflicting vectors
                to_remove = []
                for other in remaining:
                    if other == idx:
                        continue
                    dot = np.dot(v, vectors[other])
                    if dot > threshold:
                        to_remove.append(other)
                
                for r in to_remove:
                    remaining.discard(r)
            
            if len(chosen) > len(best_set):
                best_set = chosen
                if len(best_set) == n:
                    break
        
        return [vectors[i] for i in best_set]
    
    # Try different m values
    best_result = None
    best_score = 0
    
    for m in [2, 3, 4, 5, 6]:
        if time.time() > DEADLINE:
            break
        
        vectors = generate_vectors_for_m(m)
        if len(vectors) == 0:
            continue
        
        threshold = m // 2
        
        print(f"m={m}: {len(vectors)} candidates, threshold={threshold}")
        
        # For large candidate sets, greedy may be slow.
        # Limit the time per m value.
        remaining_time = DEADLINE - time.time()
        alloc_time = min(remaining_time / (6 - m), 20.0)
        
        if alloc_time < 5:
            continue
        
        result = greedy_max_independent(vectors, threshold, time_limit=alloc_time)
        
        if len(result) > best_score:
            best_score = len(result)
            best_result = result
            print(f"  Found set of size {len(result)}")
    
    if best_result is None or len(best_result) == 0:
        # Fallback: return the seed construction
        vectors = []
        for i, j in combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    return np.array(best_result, dtype=np.int64)

from itertools import combinations
