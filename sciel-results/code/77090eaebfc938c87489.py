# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that
    their maximum squared norm is less than their minimum pairwise squared distance.
    
    Strategy:
    1. Start with all vectors of squared norm 2 (two non-zero entries, each ±1).
       There are C(11,2)*4 = 220 such vectors.
       For any two such vectors, the squared distance is 4 - 2*dot(v,w).
       The dot product can be -2, -1, 0, 1, 2.
       Squared distance = 4 - 2*dot.
       Max norm squared = 2.
       We need min squared distance >= max norm squared = 2.
       So 4 - 2*dot >= 2 => 2*dot <= 2 => dot <= 1.
       Also need points distinct (no zero vector, and min dist > 0 which is implied by >= 2).
       So we need to select a subset where no two have dot product 2.
       Dot product 2 happens when v = w (same vector). Since we use distinct vectors, 
       wait, dot product 2 also happens if v and w share both non-zero positions with same signs? 
       No, if v has non-zeros at i,j with signs (a,b) and w has non-zeros at i,j with signs (a,b), dot=2.
       If signs are different, dot is 0 or -2.
       If they share one position, dot is ±1.
       If they share no positions, dot is 0.
       So the only constraint for distinct vectors from this pool is that we don't pick the same vector twice.
       Wait, is that true?
       Let v = e_i + e_j. w = e_i + e_j. dot = 2. dist^2 = 4-4=0. Invalid.
       Let v = e_i + e_j. w = e_i - e_j. dot = 1 - 1 = 0. dist^2 = 4. Valid.
       Let v = e_i + e_j. w = e_i + e_k. dot = 1. dist^2 = 4-2=2. Valid (2>=2).
       So actually, ANY set of distinct vectors from the norm-2 pool is valid?
       Let's check:
       Max norm squared = 2.
       Min squared distance:
       If v, w are distinct vectors from the pool.
       If they share 0 indices: dot=0, dist^2=4.
       If they share 1 index: dot=±1, dist^2=4∓2 -> 2 or 6.
       If they share 2 indices:
         If same signs: v=w (not distinct).
         If one sign diff: dot=0, dist^2=4.
         If both sign diff: dot=-2, dist^2=8.
       So for any distinct v, w from the norm-2 pool, dist^2 >= 2.
       And max norm^2 = 2.
       Condition: min dist^2 >= max norm^2 => 2 >= 2. Satisfied.
       
       So we can just return ALL 220 vectors!
       Is there a catch? "no point is the zero vector" - satisfied.
       "k >= 2" - satisfied.
       "second dimension is 11" - satisfied.
       
       Can we do better?
       What about norm-4 vectors (4 non-zero entries, each ±1)?
       Number of such vectors: C(11,4) * 2^4 = 330 * 16 = 5280.
       Max norm squared = 4.
       We need min squared distance >= 4.
       dist^2 = ||v||^2 + ||w||^2 - 2 dot(v,w) = 4 + 4 - 2 dot = 8 - 2 dot.
       We need 8 - 2 dot >= 4 => 2 dot <= 4 => dot <= 2.
       So we need to select a subset of norm-4 vectors such that no two have dot product > 2.
       Dot product of two norm-4 vectors:
       They share k common indices.
       The dot product is sum of products of signs at common indices.
       Max dot product is 4 (if v=w).
       Next max is 2?
       If they share 3 indices:
         Signs can match in 2, differ in 1 -> dot = 2 - 1 = 1.
         Signs can match in 3 -> dot = 3.
         Signs can match in 1 -> dot = 1 - 2 = -1.
         Signs can match in 0 -> dot = -3.
       If they share 2 indices:
         Max dot = 2.
       If they share 1 index:
         Max dot = 1.
       If they share 0 indices:
         Max dot = 0.
       
       So the constraint is: no two vectors can have dot product 3 or 4.
       Dot product 4 means v=w.
       Dot product 3 means they share 3 indices with matching signs, and the 4th index is different (or not shared? no, if they share 3 indices, the 4th is different).
       Wait, if they share 3 indices, they differ in 1 index.
       Let v have non-zeros at A (size 4). w have non-zeros at B (size 4).
       If |A ∩ B| = 3, let A = {1,2,3,4}, B = {1,2,3,5}.
       Dot = v1w1 + v2w2 + v3w3.
       Max dot = 3 (if v_i = w_i for i=1,2,3).
       So we must avoid pairs with |A ∩ B| = 3 and matching signs on the intersection.
       
       This is a large combinatorial problem. Greedy selection might work well.
       Pool size 5280.
       Constraint: dot <= 2.
       
       Let's try to construct a large set using a greedy approach or a known code.
       The Leech lattice or similar constructions might yield high kissing numbers.
       However, the task asks for integer coordinates.
       
       Known result: Kissing number in 11 dimensions is not known exactly, but bounds are known.
       The seed program gives 2.
       The norm-2 pool gives 220.
       The norm-4 pool might give more.
       
       Let's implement a greedy selection from the norm-4 pool.
       Time limit is 600s.
       5280 vectors.
       Checking all pairs is 5280^2 / 2 ~ 14 million. This is fast in Python if vectorized or optimized.
       But greedy selection:
       Start with empty set S.
       For each v in pool (randomized or sorted):
         If for all w in S, dot(v,w) <= 2:
           Add v to S.
       
       This is O(N^2) in the worst case, where N is the size of the final set.
       If the final set is large (e.g., 1000), this is 1000 * 5280 ~ 5 million operations per step? No.
       For each candidate v, we check against current S.
       Sum of |S| over all steps.
       If we pick 1000 points, the last point is checked against 999.
       Total checks ~ N * (N/2) = N^2/2.
       If N=1000, 500k checks. Very fast.
       If N=2000, 2M checks. Fast.
       
       Let's try this.
       
       We should also consider mixing norm-2 and norm-4 vectors?
       Norm-2 vectors have norm^2=2.
       Norm-4 vectors have norm^2=4.
       Max norm^2 would be 4.
       So min dist^2 must be >= 4.
       For v (norm 2) and w (norm 4):
       dist^2 = 2 + 4 - 2 dot = 6 - 2 dot.
       Need 6 - 2 dot >= 4 => 2 dot <= 2 => dot <= 1.
       Dot product of norm-2 and norm-4 vector:
       Norm-2 has 2 non-zeros. Norm-4 has 4 non-zeros.
       Max dot is 2 (if the 2 non-zeros of v are in w and match signs).
       So we need dot <= 1.
       This means we cannot have v's non-zeros be a subset of w's non-zeros with matching signs.
       
       This adds complexity. Maybe stick to one norm class first.
       Norm-2 gives 220.
       Norm-4 might give more.
       
       Let's implement greedy for Norm-4.
       
       To maximize the set, randomization with multiple restarts might help.
       
       Also, we can use a faster check using numpy.
       
       Let's write the code.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    d = 11
    
    # Generate norm-4 vectors
    # 4 non-zero entries, each ±1
    # C(11,4) = 330 combinations of indices
    # 2^4 = 16 sign combinations
    indices = list(itertools.combinations(range(d), 4))
    pool = []
    for idx in indices:
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(d, dtype=np.int64)
            for i, s in zip(idx, signs):
                v[i] = s
            pool.append(v)
    
    pool = np.array(pool)
    n_pool = len(pool)
    
    # Precompute dot products? n_pool^2 is ~28 million. 28M * 8 bytes = 224 MB.
    # This is acceptable.
    # But let's be careful with memory.
    # Alternatively, compute on the fly.
    
    # Greedy selection with random restarts
    best_set = []
    
    # Try a few random orderings
    np.random.seed(42)
    
    for restart in range(5):
        if time.time() > deadline:
            break
            
        order = np.random.permutation(n_pool)
        selected = []
        selected_indices = []
        
        for idx in order:
            if time.time() > deadline:
                break
                
            v = pool[idx]
            
            # Check against all selected
            valid = True
            for s_idx in selected_indices:
                w = pool[s_idx]
                dot = np.dot(v, w)
                if dot > 2:
                    valid = False
                    break
            
            if valid:
                selected.append(v)
                selected_indices.append(idx)
                
        if len(selected) > len(best_set):
            best_set = selected
    
    # If best_set is empty (shouldn't happen), fallback to norm-2
    if not best_set:
        # Generate norm-2 vectors
        indices2 = list(itertools.combinations(range(d), 2))
        pool2 = []
        for idx in indices2:
            for signs in itertools.product([1, -1], repeat=2):
                v = np.zeros(d, dtype=np.int64)
                for i, s in zip(idx, signs):
                    v[i] = s
                pool2.append(v)
        best_set = pool2
    
    points = np.array(best_set)
    
    # Verify and round
    points = np.around(points).astype(np.int64)
    
    return points

# EVOLVE-BLOCK-END
