import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin
    
    n = 12
    m = 40
    num_masks = 1 << n
    masks = np.arange(num_masks)
    
    # Precompute comparability matrix C
    i_col = masks[:, None]
    j_row = masks[None, :]
    C_bool = (i_col & j_row) == i_col
    C_bool = C_bool & (i_col != j_row)
    C = C_bool.astype(np.uint8)
    
    best_score = 0
    best_family = None
    
    def get_score(fam):
        if len(fam) < 2:
            return 0
        # Score is sum of C[i, j] for i != j in fam
        # Which is sum of S[i] for i in fam
        # Or just sum of sub_C
        sub_C = C[np.ix_(fam, fam)]
        return int(sub_C.sum())
    
    def greedy_init():
        fam = []
        in_fam = np.zeros(num_masks, dtype=bool)
        S = np.zeros(num_masks, dtype=np.int32)
        T = np.zeros(num_masks, dtype=np.int32)
        
        for _ in range(m):
            # Maximize S[b] + T[b]
            score_b = S + T
            best_b = np.argmax(score_b)
            fam.append(best_b)
            in_fam[best_b] = True
            S += C[:, best_b]
            T += C[best_b, :]
            
        return fam
    
    def local_search(fam):
        in_fam = np.zeros(num_masks, dtype=bool)
        for x in fam:
            in_fam[x] = True
        
        S = C[:, fam].sum(axis=1).astype(np.int32)
        T = C[fam, :].sum(axis=0).astype(np.int32)
        
        improved = True
        while improved:
            improved = False
            best_gain = 0
            best_a = -1
            best_b = -1
            
            for a in fam:
                # Constant part for a
                const_a = S[a] + T[a]
                
                # Variable part for b: S[b] + T[b] - C[b, a] - C[a, b]
                # We want to maximize this over b not in fam
                var_b = S + T - C[:, a] - C[a, :]
                var_b[in_fam] = -10000
                
                b = np.argmax(var_b)
                gain = var_b[b] - const_a
                
                if gain > best_gain:
                    best_gain = gain
                    best_a = a
                    best_b = b
            
            if best_gain > 0:
                # Swap best_a with best_b
                in_fam[best_a] = False
                in_fam[best_b] = True
                fam[fam.index(best_a)] = best_b
                
                # Update S and T
                S += C[:, best_b] - C[:, best_a]
                T += C[best_b, :] - C[best_a, :]
                
                improved = True
                
                if time.time() > deadline:
                    return fam
        
        return fam
    
    # Restart 0: Greedy
    fam = greedy_init()
    fam = local_search(fam)
    score = get_score(fam)
    if score > best_score:
        best_score = score
        best_family = fam.copy()
        
    # Random restarts
    random_state = np.random.RandomState(42)
    while time.time() < deadline:
        fam = random_state.choice(num_masks, size=m, replace=False)
        fam = local_search(fam)
        score = get_score(fam)
        if score > best_score:
            best_score = score
            best_family = fam.copy()
            
    result = []
    for idx in best_family:
        mask = masks[idx]
        subset = [i for i in range(n) if (mask >> i) & 1]
        result.append(subset)
        
    return result
