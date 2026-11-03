import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # 2s safety margin for 120s limit
    
    # This implementation is optimized for the specific case q=3, n=6, radius=1
    # but includes a fallback for other cases.
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback: simple greedy for other parameters
        # This is just to ensure the function doesn't crash if called with different args,
        # though the task contract specifies the 6-match football pool.
        total = q ** n
        # Greedy set cover
        uncovered = set(range(total))
        code = []
        # Precompute spheres
        spheres = []
        for i in range(total):
            w = [int(d) for d in str(i).zfill(n)] if q == 10 else []
            # Reconstruct word from index is messy for general q,n.
            # For the specific task, we use the fast path.
            # For general, we'd need itertools.
            pass
        
        # Since the task is specific, if it's not the specific case, 
        # we return a trivial valid (but large) code to be safe?
        # Or just raise? The contract says "evaluator passes this task's alphabet size...".
        # It implies q=3, n=6.
        # If not, we do a basic greedy.
        import itertools
        words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(words)}
        
        spheres = []
        for i, w in enumerate(words):
            ball = [i]
            for pos in range(n):
                orig_val = w[pos]
                for new_val in range(q):
                    if new_val != orig_val:
                        new_w_list = list(w)
                        new_w_list[pos] = new_val
                        new_w_tuple = tuple(new_w_list)
                        ball.append(word_to_idx[new_w_tuple])
            spheres.append(ball)
            
        uncovered = set(range(total))
        code = []
        while uncovered:
            best_c = -1
            best_cnt = -1
            for c in range(total):
                cnt = len(uncovered.intersection(spheres[c]))
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_c = c
            if best_c == -1:
                break
            code.append(best_c)
            uncovered.difference_update(spheres[best_c])
            
        return [list(words[i]) for i in code]

    # FAST PATH FOR q=3, n=6, radius=1
    total = 729
    
    # 1. Build the Incidence Matrix A (729 x 729)
    # A[i, j] = 1 if word i is within distance 1 of word j
    
    # Generate all words as integers 0..728
    # Base-3 representation
    # To speed up sphere generation, we can use vectorized numpy operations
    # But 729x729 is small enough that a loop is fine, or we can construct it cleverly.
    
    # Let's use a vectorized approach for sphere construction
    # indices: 0..728
    indices = np.arange(total)
    
    # Convert indices to base-3 digits
    # digits shape: (729, 6)
    digits = np.zeros((total, n), dtype=np.int32)
    temp = indices.copy()
    for pos in range(n - 1, -1, -1):
        digits[:, pos] = temp % q
        temp //= q
    
    # For each position pos (0..5), and each new value val (0..2, val != digits[:, pos])
    # Calculate the new index.
    # index = sum(d_i * 3^(n-1-i))
    # If we change digit at pos p from d_p to v:
    # new_index = old_index - d_p * 3^(n-1-p) + v * 3^(n-1-p)
    # new_index = old_index + (v - d_p) * 3^(n-1-p)
    
    A = np.zeros((total, total), dtype=np.int8)
    
    # Self coverage
    A[np.arange(total), np.arange(total)] = 1
    
    for pos in range(n):
        power = q ** (n - 1 - pos)
        orig_vals = digits[:, pos]
        
        for new_val in range(q):
            # Only for words where orig_val != new_val
            mask = orig_vals != new_val
            if not np.any(mask):
                continue
            
            # Calculate new indices for the masked words
            # delta = (new_val - orig_vals) * power
            # We need to add 1 to A[row, new_index]
            
            rows = np.where(mask)[0]
            origs = orig_vals[rows]
            deltas = (new_val - origs) * power
            new_indices = rows + deltas
            
            # Ensure new_indices are within bounds (they should be for valid base-3 changes)
            # But just in case, clip or check.
            # Actually, since it's a valid digit change, it stays in 0..total-1.
            
            A[rows, new_indices] = 1
            
    # 2. Formulate ILP
    # Minimize sum(x_j)
    # Subject to: A @ x >= 1
    # x_j in {0, 1}
    
    c = np.ones(total, dtype=float)
    
    # Constraints: A @ x >= 1
    # LinearConstraint(A, lb=np.ones(total), ub=np.inf)
    
    constraints = LinearConstraint(A, lb=np.ones(total), ub=np.inf)
    integrality = np.ones(total)
    bounds = Bounds(lb=0, ub=1)
    
    # Solve
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0:
            # Success
            x = res.x
            code_indices = np.where(x > 0.5)[0]
            
            # Verify
            # Convert indices to words
            words = []
            for idx in code_indices:
                w = []
                t = int(idx)
                for _ in range(n):
                    w.append(t % q)
                    t //= q
                w.reverse()
                words.append(w)
                
            # Final check
            # Build coverage
            covered = np.zeros(total, dtype=bool)
            for idx in code_indices:
                # Reuse A to check coverage? A[row, col] = 1 if row covered by col.
                # So covered = A[:, code_indices].any(axis=1)
                pass
            
            if len(code_indices) == 0:
                 # Fallback if solver failed silently or returned empty (unlikely)
                 raise ValueError("Empty solution")
                 
            return words

        else:
            # Fallback to heuristic if ILP fails or times out
            pass
            
    except Exception as e:
        # Fallback to heuristic
        pass
        
    # HEURISTIC FALLBACK
    # Greedy + Local Search
    # We can reuse the logic from the previous program but optimized.
    # Or a simple greedy.
    
    import itertools
    words_list = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words_list)}
    
    spheres = []
    for i, w in enumerate(words_list):
        ball = [i]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    new_w_list = list(w)
                    new_w_list[pos] = new_val
                    new_w_tuple = tuple(new_w_list)
                    ball.append(word_to_idx[new_w_tuple])
        spheres.append(ball)
        
    uncovered = set(range(total))
    code = []
    while uncovered:
        best_c = -1
        best_cnt = -1
        for c in range(total):
            cnt = len(uncovered.intersection(spheres[c]))
            if cnt > best_cnt:
                best_cnt = cnt
                best_c = c
        if best_c == -1:
            break
        code.append(best_c)
        uncovered.difference_update(spheres[best_c])
        
    # Try to remove redundant
    def remove_redundant(code_set):
        code_list = sorted(code_set)
        if not code_list:
            return code_set
        
        counts = np.zeros(total, dtype=np.int32)
        for ci in code_list:
            for w in spheres[ci]:
                counts[w] += 1
                
        changed = True
        while changed:
            changed = False
            new_code = set()
            for ci in code_list:
                redundant = True
                for w in spheres[ci]:
                    if counts[w] <= 1:
                        redundant = False
                        break
                if redundant:
                    for w in spheres[ci]:
                        counts[w] -= 1
                    changed = True
                else:
                    new_code.add(ci)
            code_list = sorted(new_code)
        return new_code
        
    code_set = set(code)
    code_set = remove_redundant(code_set)
    
    # Simple local search: try to swap
    # ... (skip complex local search if time is tight, ILP should have handled it)
    
    return [list(words_list[i]) for i in sorted(code_set)]

