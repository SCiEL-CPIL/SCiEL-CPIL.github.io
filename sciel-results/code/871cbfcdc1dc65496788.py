"""Rewritten program: maximize a 4-uniform family on [16] with no three
distinct sets whose three pairwise intersection sizes are all equal
(including the empty-core t=0 case)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def construct():
    t0 = time.time()
    DEADLINE = 55.0  # safety margin well under the 60 s wall clock

    # Universe: all 4-subsets of [16]
    SETS = list(combinations(range(1, 17), 4))
    N = len(SETS)  # 1820

    # Binary incidence matrix M (N x 16); M@M.T gives intersection sizes.
    M = np.zeros((N, 16), dtype=np.int8)
    for idx, tup in enumerate(SETS):
        for x in tup:
            M[idx, x - 1] = 1

    I = (M @ M.T).astype(np.int8)
    np.fill_diagonal(I, -1)  # so diagonal never equals a real intersection size

    def fast_greedy(order):
        """Optimized greedy: checks candidate against current family F (size ~k)
        rather than scanning all N items."""
        F = []
        for i in order:
            if time.time() > DEADLINE:
                break
            
            if len(F) < 2:
                F.append(i)
                continue
            
            # Check if i can be added to F
            # We need to ensure no j, k in F such that I[i,j] == I[i,k] == I[j,k]
            
            F_arr = np.array(F)
            I_iF = I[i, F_arr]
            
            # We only need to check unique intersection sizes t between i and F
            # For each such t, if there are at least 2 elements in F with that intersection,
            # we must check if any pair among them also has intersection t.
            
            unique_ts = np.unique(I_iF)
            valid = True
            for t in unique_ts:
                mask = (I_iF == t)
                count = mask.sum()
                if count < 2:
                    continue
                
                # Get the indices in F that have intersection t with i
                # We need to check the submatrix of I for these indices
                # I_FF_sub = I[np.ix_(F_arr[mask], F_arr[mask])]
                # if np.any(I_FF_sub == t):
                #     valid = False
                #     break
                
                # Optimization: 
                # If t is 0, 1, 2, 3.
                # We can pre-compute nothing, but we can use the fact that I is symmetric.
                # Let's just do the check.
                sub_indices = F_arr[mask]
                if len(sub_indices) < 2:
                    continue
                sub_I = I[np.ix_(sub_indices, sub_indices)]
                if np.any(sub_I == t):
                    valid = False
                    break
            
            if valid:
                F.append(i)
        
        return F

    def is_valid(F):
        """Vectorized check that F contains no forbidden triple."""
        n = len(F)
        if n < 3:
            return True
        F_arr = np.array(F)
        J = I[np.ix_(F_arr, F_arr)]
        for a in range(n):
            for b in range(a + 1, n):
                t = J[a, b]
                # Check if any c != a,b has J[a,c]==t and J[b,c]==t
                # (J[a] == t) & (J[b] == t) will be True for a and b themselves.
                # We need to exclude a and b.
                mask = (J[a] == t) & (J[b] == t)
                # Set a and b to False
                mask[a] = False
                mask[b] = False
                if np.any(mask):
                    return False
        return True

    try:
        # Deterministic first pass
        best = fast_greedy(list(range(N)))
        
        rng = np.random.default_rng(12345)
        # Run many restarts
        while time.time() < DEADLINE - 2.0:
            order = rng.permutation(N)
            F = fast_greedy(order)
            if len(F) > len(best):
                best = F
                # Optional: try to improve best further with local search? 
                # For now, just keep best.
                
        # Final safety validation
        if not is_valid(best):
            # Fallback to a simple valid set
            best = [0]
            
    except Exception:
        return [list(SETS[0])]

    if len(best) < 1:
        best = [0]
    best = best[:250]
    return [list(SETS[i]) for i in best]
# EVOLVE-BLOCK-END
