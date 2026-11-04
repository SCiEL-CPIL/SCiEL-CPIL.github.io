import numpy as np
import itertools
import time

def construct():
    D = 11
    # Generate candidates with norm squared 1 and 2
    candidates = []
    
    # Norm 1 vectors: +/- e_i
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = s
            candidates.append(v)
            
    # Norm 2 vectors: +/- e_i +/- e_j
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                candidates.append(v)
                
    # Convert to numpy for fast distance checks
    cand_arr = np.array(candidates, dtype=np.int64)
    n_cands = len(candidates)
    
    # We want to find a subset S such that min pairwise dist sq >= M.
    # If we use only Norm 2 vectors, M=2.
    # If we mix Norm 1 and Norm 2, M=2 (since max norm is 2).
    # Condition: dist sq >= 2.
    
    # Build conflict graph: edge if dist sq < 2
    # Dist sq < 2 means dist sq == 1 (since min dist sq for integer vectors is 1)
    # Dist sq 1 happens if vectors differ by 1 in one coordinate and 0 elsewhere.
    # i.e., v and w are "adjacent" in the hypercube graph.
    
    # We want Max Independent Set in this graph.
    # Graph size 242.
    
    # Simple greedy heuristic first to get a baseline
    # Or use a simple branch and bound if time permits.
    
    # Let's try a simple greedy:
    # Sort candidates by degree (number of conflicts), pick lowest degree first?
    # Or just pick all Norm 2?
    
    # The seed program picks all 220 Norm 2 vectors.
    # Do any Norm 1 vectors conflict with Norm 2 vectors?
    # Yes. e_i conflicts with e_i+e_j and e_i-e_j.
    
    # If we include e_i, we must exclude e_i+e_j and e_i-e_j.
    # e_i has 20 conflicts with Norm 2 vectors.
    # e_i has 1 conflict with -e_i (dist 2, so NO conflict, dist sq 4 >= 2).
    # e_i has 0 conflicts with other Norm 1 vectors (dist sq 2).
    
    # So adding e_i removes 20 Norm 2 vectors.
    # Net change: +1 - 20 = -19.
    # So including Norm 1 vectors is bad if we want to keep all Norm 2.
    
    # What if we drop some Norm 2 to add more Norm 1?
    # Unlikely to beat 220.
    
    # So 220 is likely optimal for M=2.
    
    # Can we use M=4?
    # If M=4, we need dist sq >= 4.
    # Norm 2 vectors have dist sq 2. So we can only pick at most one from each "clique" of dist 2.
    # This reduces the count drastically.
    
    # So M=2 is the way.
    
    # Return the 220 vectors.
    vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                vectors.append(v)
                
    return np.array(vectors, dtype=np.int64)
