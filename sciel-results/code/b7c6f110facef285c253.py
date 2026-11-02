import numpy as np
import itertools
import time
from collections import defaultdict

def construct():
    """
    Constructs a set of nonzero integer vectors in R^11 such that the minimum 
    pairwise distance is at least the maximum vector norm.
    
    Strategy: 
    1. Start with all weight-2 vectors (220 vectors of norm sqrt(2)).
    2. Try to add weight-4 vectors (norm 2) that are compatible with the current set.
       Compatibility: for u (weight-2, norm^2=2) and v (weight-4, norm^2=4):
       ||u-v||^2 >= max(2,4) = 4
       ||u||^2 + ||v||^2 - 2<u,v> >= 4
       2 + 4 - 2<u,v> >= 4
       6 - 2<u,v> >= 4
       2<u,v> <= 2
       <u,v> <= 1
       
       Also, for two weight-4 vectors u, v (both norm^2=4):
       ||u-v||^2 >= 4
       4+4-2<u,v> >= 4
       8-2<u,v> >= 4
       <u,v> <= 2
       
       And for a weight-2 u and weight-4 v where we already have all weight-2:
       We need <u,v> <= 1 for all weight-2 u in our set.
       
       Actually, let's reconsider. The condition is:
       min_{i!=j} ||v_i - v_j|| >= max_i ||v_i||
       
       If we mix norms, this is tricky. Let's stick to a single norm class for simplicity first,
       or ensure the condition holds.
       
       Let's try a different approach: Use only weight-2 vectors but try to find a larger
       set? No, we already have ALL weight-2 vectors (220). They all satisfy the condition.
       Can we add more?
       
       What if we use weight-1 vectors? Norm 1.
       Condition: min dist >= max norm.
       If we have weight-1 (norm 1) and weight-2 (norm sqrt(2)), max norm is sqrt(2).
       Dist between weight-1 e_i and weight-2 (e_j + e_k):
       If i not in {j,k}: ||e_i - (e_j+e_k)||^2 = 1 + 2 - 0 = 3. sqrt(3) > sqrt(2). OK.
       If i in {j,k}, say i=j: ||e_j - (e_j+e_k)||^2 = ||-e_k||^2 = 1. 1 < sqrt(2). FAIL.
       
       So we cannot mix weight-1 and weight-2 if the weight-2 vector shares an index.
       
       Let's try to find a large set of vectors all of the SAME norm.
       Weight-2 gives 220.
       Weight-4: C(11,4)*2^4 = 330 * 16 = 5280 candidates.
       Condition for two weight-4 vectors u, v: <u,v> <= 2.
       
       Let's use a greedy approach to build a large independent set in the conflict graph
       of weight-4 vectors.
       
       But wait, the seed program already has 220. The inspirations mention scores > 2.
       Score 2.1273 means k ~ 2.1273 * 220 = 468? Or is the seed score 1.0 = 220?
       Yes, seed is 220. So 2.1273 * 220 = 468.
       
       Let's try to find a large set of weight-4 vectors.
       
       Actually, let's look at the "conflict graph" idea.
       For weight-4 vectors, we want <u,v> <= 2.
       The maximum possible dot product for distinct weight-4 vectors is 4 (if same).
       If they share 3 indices: <u,v> can be 3, 1, -1, -3.
       If they share 2 indices: <u,v> can be 2, 0, -2.
       If they share 1 index: <u,v> can be 1, -1.
       If they share 0 indices: <u,v> = 0.
       
       So conflicts (where <u,v> > 2) happen when:
       - They share 3 indices and the signs match on those 3 (dot product 3 or more).
         Specifically, if they share 3 indices, let the shared part be S.
         u = S + a*e_i, v = S + b*e_j where i!=j.
         <u,v> = ||S||^2 + 0 = 3. (Since a,b are +/-1 and i!=j, the cross terms vanish).
         Wait, if u and v share 3 indices, they must have the same values on those indices?
         No, they are vectors. If they share 3 indices, it means the support intersection has size 3.
         Let support(u) ∩ support(v) = {i,j,k}.
         u_i, u_j, u_k are +/-1. v_i, v_j, v_k are +/-1.
         <u,v> = u_i v_i + u_j v_j + u_k v_k + (terms from non-shared indices which are 0).
         The non-shared indices contribute 0.
         So <u,v> = sum of 3 terms, each +/-1.
         Possible values: 3, 1, -1, -3.
         We need <u,v> <= 2. So 3 is a conflict.
         This happens if u_i=v_i, u_j=v_j, u_k=v_k.
       
       - They share 2 indices: <u,v> = u_i v_i + u_j v_j. Values: 2, 0, -2.
         2 is OK (<=2). So no conflict from sharing 2 indices.
       
       - They share 4 indices: They are identical. Not distinct.
       
       So the ONLY conflicts between distinct weight-4 vectors are those that share exactly 3 indices
       AND have the same signs on those 3 indices.
       
       This is a very sparse conflict graph!
       Total weight-4 vectors: 5280.
       For each vector, how many conflicts?
       Fix a weight-4 vector v with support {1,2,3,4} and signs (1,1,1,1).
       A conflict w must share exactly 3 indices with v and have the same signs on those 3.
       The 4th index of w must be different from the 4th index of v?
       No, w has support of size 4.
       If w shares 3 indices with v, say {1,2,3}, and same signs.
       Then w's support is {1,2,3, j} where j != 4 (since v's support is {1,2,3,4}, and if j=4, w=v).
       Also j must be such that w is a valid weight-4 vector.
       The index j can be any of the remaining 11-4=7 indices.
       And the sign of w_j can be +1 or -1.
       So for each of the 4 choices of which index to drop from v's support,
       there are 7 choices for the new index, and 2 choices for its sign.
       Total conflicts per vector: 4 * 7 * 2 = 56.
       
       This is a graph with 5280 nodes and average degree 56.
       We want the Maximum Independent Set (MIS).
       
       We can use a greedy algorithm or a simple branch and bound.
       Given the time limit (120s), a greedy approach might be fast enough to get a large set.
       Or we can use a simple heuristic: sort by degree (ascending) and greedily add.
       
       Let's implement a greedy MIS for weight-4 vectors.
       
       Steps:
       1. Generate all 5280 weight-4 vectors.
       2. Compute the conflict graph (adjacency matrix or list).
       3. Greedy selection:
          - Start with empty set S.
          - While candidates remain:
            - Pick a vertex v with minimum degree in the remaining graph?
              Or just pick the first one?
              Minimum degree is often better for MIS.
            - Add v to S.
            - Remove v and all its neighbors from the candidate set.
       
       Let's try this.
       
       However, 5280 nodes is small enough to build the adjacency matrix (5280x5280 ~ 27M entries, fits in memory).
       But building it naively might be slow in Python.
       We can use bitsets or just lists.
       
       Actually, let's just use a simple greedy:
       - Compute degrees.
       - Sort by degree.
       - Iterate, if not removed and no conflict with already selected, select.
       
       Wait, the "remove neighbors" step is important.
       
       Let's try to improve on 220.
       If we get a large independent set of weight-4 vectors, say size 2000, that's a huge win.
       
       Let's code the greedy MIS for weight-4.
       
       Also, we can combine with weight-2?
       If we have a set S4 of weight-4 vectors, can we add weight-2 vectors?
       Condition: <u,v> <= 1 for u in S2, v in S4.
       And <u,u'> <= 1 for u,u' in S2. (Always true for distinct weight-2).
       And <v,v'> <= 2 for v,v' in S4. (True if independent set).
       
       So we can try to add weight-2 vectors that are compatible with S4.
       For a weight-2 vector u, it conflicts with a weight-4 vector v if <u,v> > 1.
       <u,v> is the sum of products on shared indices.
       u has 2 non-zero entries. v has 4.
       If they share 2 indices: <u,v> can be 2, 0, -2.
       If <u,v> = 2, conflict.
       This happens if u and v have the same signs on the 2 shared indices.
       
       So a weight-2 vector u is compatible with S4 if for all v in S4,
       it is NOT the case that (|supp(u) ∩ supp(v)| == 2 AND signs match).
       
       This is a constraint satisfaction problem.
       
       Let's first just find the MIS of weight-4.
       """
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    D = 11
    
    # Generate all weight-4 vectors
    # Represent each vector by its support (tuple of 4 indices) and signs (tuple of 4 +/-1)
    # Index mapping: (support, signs) -> index
    vectors = []
    vec_to_idx = {}
    
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            idx = len(vectors)
            vectors.append((support, signs))
            key = (support, signs)
            vec_to_idx[key] = idx
            
    n_vecs = len(vectors)
    
    # Build conflict graph
    # Conflict if share 3 indices and same signs on those 3.
    # For each vector, find conflicts.
    # To speed up, we can group by support?
    # Or just iterate pairs? 5280^2 / 2 = 14M pairs. Might be too slow in pure Python.
    
    # Better: For each vector, generate its conflicts explicitly.
    # As derived: 4 (choices of which index to keep 3 of) * 7 (choices of new index) * 2 (signs) = 56 conflicts.
    
    conflicts = [set() for _ in range(n_vecs)]
    
    # Map from (3-tuple of indices, 3-tuple of signs) to list of vectors having that property
    # This allows us to find vectors sharing 3 indices and signs quickly.
    # Key: (i1, i2, i3, s1, s2, s3) where i1<i2<i3
    # Value: list of vector indices
    
    triplet_map = defaultdict(list)
    
    for idx, (support, signs) in enumerate(vectors):
        # Generate all 4 triplets from the 4 indices
        for k in range(4):
            # Triplet excluding index k
            trip_sup = tuple(support[j] for j in range(4) if j != k)
            trip_sig = tuple(signs[j] for j in range(4) if j != k)
            key = (trip_sup, trip_sig)
            triplet_map[key].append(idx)
            
    # Now for each vector, find conflicts
    for idx, (support, signs) in enumerate(vectors):
        for k in range(4):
            trip_sup = tuple(support[j] for j in range(4) if j != k)
            trip_sig = tuple(signs[j] for j in range(4) if j != k)
            key = (trip_sup, trip_sig)
            # All vectors in triplet_map[key] share this triplet
            # They are conflicts if they are not idx itself
            for other_idx in triplet_map[key]:
                if other_idx != idx:
                    conflicts[idx].add(other_idx)
                    
    # Now we have the conflict graph.
    # Greedy MIS:
    # 1. Compute degrees.
    # 2. Sort by degree.
    # 3. Greedy selection.
    
    degrees = [len(conflicts[i]) for i in range(n_vecs)]
    
    # Sort indices by degree
    order = sorted(range(n_vecs), key=lambda i: degrees[i])
    
    selected = []
    removed = [False] * n_vecs
    
    for idx in order:
        if removed[idx]:
            continue
        # Check if idx has any conflict with already selected vertices
        # We can just check against the selected list?
        # Or maintain a set of selected and check conflicts.
        # Since we remove neighbors when we select, if we do that properly,
        # we don't need to check against selected, just against removed.
        # But the standard greedy is:
        #   Pick v.
        #   Add v to solution.
        #   Mark v and all neighbors as removed.
        
        selected.append(idx)
        removed[idx] = True
        for neighbor in conflicts[idx]:
            removed[neighbor] = True
            
    # Check time
    if time.time() > deadline:
        # Return what we have
        pass
        
    # Convert selected indices to vectors
    result_vectors = []
    for idx in selected:
        support, signs = vectors[idx]
        v = [0] * D
        for i, s in zip(support, signs):
            v[i] = s
        result_vectors.append(v)
        
    # Try to add weight-2 vectors
    # Generate all weight-2 vectors
    w2_vectors = []
    for support in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for i, s in zip(support, signs):
                v[i] = s
            w2_vectors.append(v)
            
    # For each weight-2 vector, check compatibility with all selected weight-4 vectors
    # <u,v> <= 1
    # u is weight-2, v is weight-4.
    # <u,v> is sum over shared indices.
    # Max 2 shared indices.
    # If 2 shared, check if signs match (dot=2) -> conflict.
    # If 1 shared, dot is +/-1 -> OK.
    # If 0 shared, dot=0 -> OK.
    
    compatible_w2 = []
    for u in w2_vectors:
        ok = True
        u_supp = [i for i in range(D) if u[i] != 0]
        for idx in selected:
            v_supp, v_signs = vectors[idx]
            # Check intersection
            # u has 2 non-zero, v has 4.
            # Find shared indices
            shared = []
            for i in u_supp:
                if i in v_supp:
                    shared.append(i)
            if len(shared) == 2:
                # Check signs
                # u[i] and v[i]
                # v[i] is determined by v_supp and v_signs
                # Find position of i in v_supp
                pos1 = v_supp.index(shared[0])
                pos2 = v_supp.index(shared[1])
                if u[shared[0]] == v_signs[pos1] and u[shared[1]] == v_signs[pos2]:
                    ok = False
                    break
        if ok:
            compatible_w2.append(u)
            
    result_vectors.extend(compatible_w2)
    
    if not result_vectors:
        # Fallback to weight-2 only
        for support in itertools.combinations(range(D), 2):
            for signs in itertools.product([1, -1], repeat=2):
                v = [0] * D
                for i, s in zip(support, signs):
                    v[i] = s
                result_vectors.append(v)
                
    result = np.array(result_vectors, dtype=np.int64)
    return result
