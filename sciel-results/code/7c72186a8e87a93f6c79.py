"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = 120.0 - 10.0  # safety margin

    # Phase 1: Generate all norm-sqrt(2) vectors (the seed set)
    # These are vectors with exactly two non-zero entries, each +/-1.
    vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                vectors.append(v)
    vectors = np.array(vectors, dtype=np.int64)

    # Phase 2: Greedily add vectors of norm-sqrt(5) (one entry 2, one entry 1)
    # Condition: For any existing vector u and new vector v, ||u - v||^2 >= max(||u||^2, ||v||^2)
    # ||u||^2 = 2 for seed vectors, ||v||^2 = 5 for new vectors.
    # So we need ||u - v||^2 >= 5.
    # ||u - v||^2 = ||u||^2 + ||v||^2 - 2 u.v = 7 - 2 u.v
    # So 7 - 2 u.v >= 5  =>  2 u.v <= 2  =>  u.v <= 1.
    
    # Generate candidate norm-sqrt(5) vectors.
    candidates = []
    # Case A: One entry is 2, another is 1 (or -1)
    for i, j in itertools.combinations(range(D), 2):
        for s2 in (1, -1):
            for s1 in (1, -1):
                v = [0] * D
                v[i] = 2 * s2
                v[j] = 1 * s1
                candidates.append(v)
                # Also swap which index gets 2
                v2 = [0] * D
                v2[i] = 1 * s1
                v2[j] = 2 * s2
                candidates.append(v2)
    # Case B: One entry is -2, another is 1 (or -1) - covered above by s2, s1 signs
    # Are there other norm-sqrt(5) integer vectors?
    # 5 = 4 + 1 (done)
    # 5 = 1 + 4 (done)
    # 5 = 0 + 5 (no integer sqrt)
    # So only these.
    
    # Remove duplicates (though the generation above might produce some? No, i!=j and specific assignments)
    # Actually, for a fixed pair (i,j), (2, 1) and (1, 2) are distinct vectors.
    # For a fixed pair (i,j) and signs, v[i]=2*s2, v[j]=1*s1 is distinct from v[i]=1*s1', v[j]=2*s2'.
    # So no duplicates within this generation for distinct (i,j).
    
    # Check compatibility with existing vectors
    # u.v for u in vectors, v in candidates.
    # We can speed this up using matrix multiplication.
    # vectors shape (220, 11)
    # candidates shape (N_cand, 11)
    # dot products shape (220, N_cand)
    
    # But we need to add them one by one, because adding one candidate might conflict with another candidate
    # if we were adding multiple? No, the constraint is pairwise between ALL selected vectors.
    # If we add v1 and v2, they must also satisfy ||v1 - v2||^2 >= max(||v1||^2, ||v2||^2) = 5.
    # ||v1 - v2||^2 = 5 + 5 - 2 v1.v2 = 10 - 2 v1.v2 >= 5  =>  2 v1.v2 <= 5  =>  v1.v2 <= 2.5  =>  v1.v2 <= 2.
    
    # So we need a set of candidates such that:
    # 1. Each candidate has dot product <= 1 with all current vectors (norm sqrt(2)).
    # 2. Any two candidates have dot product <= 2.
    
    # Let's filter candidates first against the base set.
    if len(candidates) > 0:
        cand_arr = np.array(candidates, dtype=np.int64)
        # Compute dot products between vectors (base) and cand_arr
        # base shape (220, 11), cand shape (N, 11)
        dots_base = vectors @ cand_arr.T  # shape (220, N)
        
        # A candidate is valid if all dot products with base vectors are <= 1.
        valid_mask = np.all(dots_base <= 1, axis=0)  # shape (N,)
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) > 0:
            valid_cands = cand_arr[valid_indices]
            
            # Now we need to select a subset of valid_cands such that pairwise dot products <= 2.
            # This is a maximum clique problem on the graph where edges exist if dot product <= 2.
            # Or rather, a maximum independent set on the conflict graph where edges exist if dot product > 2.
            
            # Since N is small (at most 2 * 11 * 10 * 2 * 2 = 880? 
            # Combinations of 2 from 11 is 55.
            # For each pair, 4 sign combos for (2,1) and 4 for (1,2)?
            # My loop:
            # for i,j in comb(11,2): 55
            # for s2 in (1,-1): 2
            # for s1 in (1,-1): 2
            # v1: v[i]=2*s2, v[j]=1*s1
            # v2: v[i]=1*s1, v[j]=2*s2
            # Total 55 * 2 * 2 * 2 = 440 candidates?
            # Let's just compute the size.
            # Actually, let's just use a greedy or simple search if N is small enough.
            # 440 is small.
            
            # Compute pairwise dot products among valid_cands.
            nc = len(valid_cands)
            if nc > 0:
                dots_cand = valid_cands @ valid_cands.T
                
                # We want to select the largest subset where all pairwise dots <= 2.
                # Greedy: Sort candidates by something? Or just try all?
                # Since 440 is too large for brute force 2^440, we need a heuristic.
                # However, the conflict is sparse?
                # Let's try a greedy approach: 
                # 1. Compute degree in conflict graph (number of other cands with dot > 2).
                # 2. Pick the one with lowest degree, add it, remove conflicts.
                
                # Conflict matrix: True if dot > 2.
                # Note diagonal is 5 (self dot), which is > 2, so we must be careful.
                # We want to pick indices i, j, k... such that for any distinct pair, dot <= 2.
                
                # Let's use a simple greedy:
                # 1. Calculate how many conflicts each candidate has with other valid candidates.
                # 2. Sort by ascending conflict count.
                # 3. Iterate, if candidate doesn't conflict with already selected, select it.
                
                # To make it deterministic and potentially better, we can do a few passes with different orderings
                # or use a local search. But given time, let's try a simple greedy first.
                
                # Conflict count for each candidate j: number of i != j such that dots_cand[i,j] > 2.
                # Mask out self.
                np.fill_diagonal(dots_cand, 2) # Set self to non-conflicting value for counting, or just ignore diagonal.
                conflicts = (dots_cand > 2)
                conflict_counts = np.sum(conflicts, axis=1)
                
                # Sort indices by conflict count
                order = np.argsort(conflict_counts)
                
                selected = []
                selected_dots = [] # Not needed if we just check against selected list
                
                # We can optimize by keeping a bitmask or just a list of selected indices.
                # Since nc <= 440, and selected will be small, checking against list is O(S) per candidate.
                # Total O(N*S).
                
                for idx in order:
                    if time.time() > deadline:
                        break
                    # Check if candidate idx conflicts with any already selected
                    ok = True
                    for sel_idx in selected:
                        if dots_cand[idx, sel_idx] > 2:
                            ok = False
                            break
                    if ok:
                        selected.append(idx)
                
                if len(selected) > 0:
                    # Add these vectors to the main list
                    new_vectors = valid_cands[selected]
                    vectors = np.vstack([vectors, new_vectors])
    
    # Phase 3: Check for norm-sqrt(10) vectors?
    # 10 = 4+4+2? No, 10 = 9+1. So one entry 3, one entry 1.
    # ||v||^2 = 10.
    # Condition against base (norm 2): ||u-v||^2 >= 10.
    # 4 + 10 - 2 u.v >= 10 => 4 - 2 u.v >= 0 => u.v <= 2.
    # Condition against norm-sqrt(5) vectors (norm 5): ||u-v||^2 >= 10.
    # 5 + 10 - 2 u.v >= 10 => 5 - 2 u.v >= 0 => u.v <= 2.5 => u.v <= 2.
    # Condition against other norm-sqrt(10) vectors: ||u-v||^2 >= 10.
    # 10 + 10 - 2 u.v >= 10 => 10 - 2 u.v >= 0 => u.v <= 5.
    
    # So we can add norm-sqrt(10) vectors if:
    # - dot with all base <= 2
    # - dot with all added norm-sqrt(5) <= 2
    # - dot with other added norm-sqrt(10) <= 5
    
    # Let's generate these.
    # 10 = 9 + 1.
    candidates10 = []
    for i, j in itertools.combinations(range(D), 2):
        for s3 in (1, -1):
            for s1 in (1, -1):
                v = [0] * D
                v[i] = 3 * s3
                v[j] = 1 * s1
                candidates10.append(v)
                v2 = [0] * D
                v2[i] = 1 * s1
                v2[j] = 3 * s3
                candidates10.append(v2)
    
    if len(candidates10) > 0:
        cand10_arr = np.array(candidates10, dtype=np.int64)
        
        # Check against base vectors
        dots_base10 = vectors @ cand10_arr.T
        # Wait, 'vectors' now includes the added sqrt(5) vectors?
        # My code structure: I added sqrt(5) to 'vectors' in phase 2.
        # So 'vectors' is the current full set.
        # The condition is dot <= 2 for ALL current vectors?
        # For base (norm 2): dot <= 2.
        # For sqrt(5) (norm 5): dot <= 2.
        # So yes, dot <= 2 for all current vectors.
        
        valid_mask10 = np.all(dots_base10 <= 2, axis=0)
        valid_indices10 = np.where(valid_mask10)[0]
        
        if len(valid_indices10) > 0:
            valid_cands10 = cand10_arr[valid_indices10]
            nc10 = len(valid_cands10)
            
            if nc10 > 0:
                dots_cand10 = valid_cands10 @ valid_cands10.T
                # Condition: dot <= 5 for distinct pairs.
                
                # Greedy selection again.
                np.fill_diagonal(dots_cand10, 5)
                conflicts10 = (dots_cand10 > 5)
                conflict_counts10 = np.sum(conflicts10, axis=1)
                order10 = np.argsort(conflict_counts10)
                
                selected10 = []
                for idx in order10:
                    if time.time() > deadline:
                        break
                    ok = True
                    for sel_idx in selected10:
                        if dots_cand10[idx, sel_idx] > 5:
                            ok = False
                            break
                    if ok:
                        selected10.append(idx)
                
                if len(selected10) > 0:
                    new_vectors10 = valid_cands10[selected10]
                    vectors = np.vstack([vectors, new_vectors10])

    return vectors

# EVOLVE-BLOCK-END
