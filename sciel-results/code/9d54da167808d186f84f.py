import itertools
import numpy as np
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Generate candidate vectors
    # Weight 2: (i, j) with signs
    # Weight 3: (i, j, k) with signs
    
    candidates = []
    
    # Weight 2
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                candidates.append(tuple(v))
                
    # Weight 3
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i], v[j], v[k] = si, sj, sk
                    candidates.append(tuple(v))
                    
    # Convert to numpy arrays for faster distance checks?
    # Or just keep as tuples and use dot product logic.
    # Dot product of two sparse vectors is fast.
    
    def is_compatible(v1, v2, max_norm_sq):
        # v1, v2 are tuples
        # Condition: ||v1 - v2||^2 >= max_norm_sq
        # ||v1||^2 + ||v2||^2 - 2*v1.v2 >= max_norm_sq
        # Since we mix norms, we need to be careful.
        # Actually, the constraint is: min pairwise dist >= MAX norm.
        # Let R = max(norm(v) for v in set).
        # This is tricky because R changes as we add vectors.
        # Strategy: 
        # If we only use vectors of norm sqrt(2) and sqrt(3), 
        # and we ensure dist >= sqrt(3) (norm of W3), 
        # then any pair of W2s must be dist >= sqrt(3).
        # W2-W2 max dist is 2 (sqrt(4)). Min dist is sqrt(2).
        # If we include W3, R = sqrt(3).
        # W2-W2 dist must be >= sqrt(3) ~ 1.732.
        # W2-W2 min dist is sqrt(2) ~ 1.414. 
        # So we CANNOT have arbitrary W2s if W3s are present.
        # We must filter W2s based on the W3s chosen.
        
        # Simpler approach: 
        # 1. Select a set of W3 vectors S3 such that dist(v, v') >= sqrt(3) for all v,v' in S3.
        #    This implies v.v' <= (3-3)/2 = 0? No.
        #    ||v-v'||^2 >= 3 => 3+3 - 2v.v' >= 3 => v.v' <= 1.5 => v.v' <= 1.
        # 2. Select a set of W2 vectors S2 such that:
        #    a. dist(u, u') >= sqrt(3) for all u,u' in S2.
        #       u.u' <= (2+2-3)/2 = 0.5 => u.u' <= 0.
        #       So W2s must be orthogonal or anti-parallel?
        #       u=(1,1,0), u'=(1,-1,0) dot 0. OK.
        #       u=(1,1,0), u'=(1,1,0) same.
        #       u=(1,1,0), u'=(1,0,1) dot 1. 1 > 0.5. FAIL.
        #       So W2s must have dot product <= 0.
        #    b. dist(u, w) >= sqrt(3) for all u in S2, w in S3.
        #       2+3 - 2u.w >= 3 => u.w <= 1.
        
        # This is a complex constraint. 
        # Let's just use the raw distance check against the current MAX norm.
        
        pass

    # Given the complexity, let's stick to a single norm class to ensure validity easily, 
    # or use a conservative check.
    
    # Re-evaluating: The seed uses W2 only. R=sqrt(2). 
    # If we add W3, R becomes sqrt(3). 
    # This invalidates many W2 pairs.
    # So it's better to try to find a large set of W3s alone, 
    # or a large set of W2s alone (we have 220).
    # Can we find >220 W3s?
    # My estimate was ~180.
    # Can we find >220 W4s?
    # W4 norm 2. R=2.
    # W4-W4 dist >= 2 => 4+4 - 2dot >= 4 => dot <= 2.
    # W4s with dot <= 2.
    # Pool size 5280.
    # Conflict if dot >= 3.
    # Degree of conflict?
    # Likely lower relative density.
    
    # Let's try Greedy on W4s.
    
    candidates_w4 = []
    for i, j, k, l in itertools.combinations(range(D), 4):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = [0] * D
                        v[i], v[j], v[k], v[l] = si, sj, sk, sl
                        candidates_w4.append(tuple(v))
                        
    best_set = []
    
    # Greedy on W4
    random.shuffle(candidates_w4)
    
    current_set = []
    current_max_norm_sq = 0
    
    for v in candidates_w4:
        if time.time() > deadline:
            break
            
        v_np = np.array(v)
        v_norm_sq = np.dot(v_np, v_np)
        
        if v_norm_sq > current_max_norm_sq:
            # New max norm. Need to check against ALL existing.
            new_max_sq = v_norm_sq
            valid = True
            for u in current_set:
                u_np = np.array(u)
                dist_sq = np.dot(v_np - u_np, v_np - u_np)
                if dist_sq < new_max_sq:
                    valid = False
                    break
            if valid:
                current_set.append(v)
                current_max_norm_sq = new_max_sq
        else:
            # Same or lower norm. Check against existing.
            # Dist >= current_max_norm_sq
            valid = True
            for u in current_set:
                u_np = np.array(u)
                dist_sq = np.dot(v_np - u_np, v_np - u_np)
                if dist_sq < current_max_norm_sq:
                    valid = False
                    break
            if valid:
                current_set.append(v)
                
    if len(current_set) > len(best_set):
        best_set = current_set
        
    # Try W3 as well?
    # W3 greedy
    candidates_w3 = []
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i], v[j], v[k] = si, sj, sk
                    candidates_w3.append(tuple(v))
                    
    random.shuffle(candidates_w3)
    current_set = []
    current_max_norm_sq = 0
    
    for v in candidates_w3:
        if time.time() > deadline:
            break
        v_np = np.array(v)
        v_norm_sq = np.dot(v_np, v_np)
        
        if v_norm_sq > current_max_norm_sq:
            new_max_sq = v_norm_sq
            valid = True
            for u in current_set:
                u_np = np.array(u)
                dist_sq = np.dot(v_np - u_np, v_np - u_np)
                if dist_sq < new_max_sq:
                    valid = False
                    break
            if valid:
                current_set.append(v)
                current_max_norm_sq = new_max_sq
        else:
            valid = True
            for u in current_set:
                u_np = np.array(u)
                dist_sq = np.dot(v_np - u_np, v_np - u_np)
                if dist_sq < current_max_norm_sq:
                    valid = False
                    break
            if valid:
                current_set.append(v)
                
    if len(current_set) > len(best_set):
        best_set = current_set
        
    # Return best
    if not best_set:
        # Fallback to seed
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    return np.array(best_set, dtype=np.int64)

# EVOLVE-BLOCK-END
