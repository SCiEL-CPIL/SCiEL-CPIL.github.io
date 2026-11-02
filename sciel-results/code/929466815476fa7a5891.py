import itertools
import time
import numpy as np
from scipy.sparse import csr_matrix

def construct(q, n, radius):
    """
    Construct a q-ary covering code of length n and radius r.
    Uses Vectorized Greedy Set Cover followed by Vectorized Local Search.
    """
    # Fallback for non-target parameters
    if q != 3 or n != 8 or radius != 1:
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Specific optimization for q=3, n=8, radius=1
    N = q ** n  # 6561
    start_time = time.time()
    TIME_LIMIT = 115

    # Precompute neighbors
    pow3 = [1] * n
    for i in range(1, n):
        pow3[i] = pow3[i-1] * 3

    neighbors = [[] for _ in range(N)]
    for i in range(N):
        neighbors[i].append(i)
        w = i
        for p in range(n):
            p3 = pow3[p]
            digit = (w // p3) % 3
            base = w - digit * p3
            for new_d in range(3):
                if new_d == digit:
                    continue
                j = base + new_d * p3
                neighbors[i].append(j)

    # Convert neighbors to numpy arrays for faster indexing
    neighbors_np = [np.array(neigh, dtype=np.int32) for neigh in neighbors]
    
    # Build sparse adjacency matrix A
    row_idx = []
    col_idx = []
    for i in range(N):
        for j in neighbors_np[i]:
            row_idx.append(i)
            col_idx.append(j)
    
    A = csr_matrix((np.ones(len(row_idx)), (row_idx, col_idx)), shape=(N, N))

    # Precompute all words for output
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]

    # 1. Greedy Initialization (Vectorized)
    uncovered = np.ones(N, dtype=bool)
    in_code = np.zeros(N, dtype=bool)
    
    gain = A @ uncovered
    
    while np.any(uncovered):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        best_c = np.argmax(gain)
        if gain[best_c] == 0:
            break
            
        in_code[best_c] = True
        
        # Update uncovered
        prev_uncovered = uncovered.copy()
        uncovered[neighbors_np[best_c]] = False
        
        # Update gain
        # newly_covered = prev_uncovered & ~uncovered
        # gain -= A[:, newly_covered] @ np.ones(np.sum(newly_covered))
        newly_covered = prev_uncovered & ~uncovered
        if np.any(newly_covered):
             # A[:, newly_covered] is a sparse matrix slice
             # Multiplying by ones is equivalent to summing rows
             # gain -= A[:, newly_covered].sum(axis=1).A1
             # But A is sparse.
             # Let's just use the dense vector product if N is small?
             # No, sparse is fine.
             # gain -= (A[:, newly_covered] @ np.ones(np.sum(newly_covered))).A1
             
             # Faster:
             # gain[newly_covered_indices] -= ... no, it affects all neighbors.
             
             # Let's stick to the full recalculation if it's fast enough?
             # A @ uncovered is ~1ms?
             # 500 iterations * 1ms = 0.5s. Acceptable.
             gain = A @ uncovered
             
    # Ensure validity
    if not np.all(uncovered == False):
        for i in range(N):
            if uncovered[i]:
                in_code[i] = True
                uncovered[neighbors_np[i]] = False

    # 2. Local Search
    count_cover = np.zeros(N, dtype=np.int32)
    # Initialize count_cover
    # We can do: count_cover = (A @ in_code).astype(np.int32)
    count_cover = (A @ in_code).astype(np.int32)
    
    improved = True
    while improved and (time.time() - start_time < TIME_LIMIT - 5):
        improved = False
        
        # Pass 1: Simple Removal
        current_code = np.where(in_code)[0]
        for c in current_code:
            if np.min(count_cover[neighbors_np[c]]) >= 2:
                in_code[c] = False
                count_cover[neighbors_np[c]] -= 1
                improved = True
        
        # Pass 2: Swap
        critical_mask = count_cover == 1
        
        if not np.any(critical_mask):
            break
            
        gain_crit = A @ critical_mask
        gain_crit[in_code] = 0
        
        K = 100
        if N > K:
            top_indices = np.argpartition(gain_crit, -K)[-K:]
        else:
            top_indices = np.arange(N)
            
        top_indices = top_indices[gain_crit[top_indices] > 0]
        
        for c_new in top_indices:
            if in_code[c_new]:
                continue
                
            added_mask = np.zeros(N, dtype=bool)
            added_mask[neighbors_np[c_new]] = True
            
            removals = []
            for c_old in np.where(in_code)[0]:
                nb = neighbors_np[c_old]
                crit_nb = nb[count_cover[nb] == 1]
                
                if len(crit_nb) == 0:
                    continue
                
                if np.all(added_mask[crit_nb]):
                    removals.append(c_old)
                    
            if removals:
                in_code[c_new] = True
                count_cover[neighbors_np[c_new]] += 1
                
                for c_old in removals:
                    in_code[c_old] = False
                    count_cover[neighbors_np[c_old]] -= 1
                
                improved = True
                break
        
    return [all_words[i] for i in np.where(in_code)[0]]
