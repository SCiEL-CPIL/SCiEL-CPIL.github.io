import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(q, n, radius):
    """
    Constructs a covering code of length n over alphabet size q with covering radius radius.
    Uses Integer Linear Programming (ILP) for small spaces to find optimal or near-optimal codes.
    Falls back to a greedy algorithm for larger spaces.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    total_words = q ** n
    
    # If the space is too large for ILP, use greedy
    if total_words > 5000:
        return _greedy_construct(q, n, radius, start_time, deadline)

    # Helper to convert integer to list of digits
    def int_to_word(idx, n, q):
        word = [0] * n
        for i in range(n):
            word[i] = idx % q
            idx //= q
        return word

    # Helper to convert list of digits to integer
    def word_to_int(word, q):
        val = 0
        for i in range(len(word) - 1, -1, -1):
            val = val * q + word[i]
        return val

    # Precompute the incidence matrix
    # A[c, w] = 1 if word w is within radius of center c
    # Constraint: For each word w, sum_{c} A[c, w] * x_c >= 1
    # In scipy milp, constraints are A_ub @ x <= b_ub or LinearConstraint(A, lb, ub)
    # We want sum_{c in N(w)} x_c >= 1.
    # This is equivalent to -sum_{c in N(w)} x_c <= -1.
    
    # Let's build the matrix M where M[w, c] = 1 if c covers w.
    # Then the constraint is M @ x >= 1.
    # scipy milp uses LinearConstraint(A, lb, ub) meaning lb <= A @ x <= ub.
    # So we can use A = M, lb = 1, ub = inf.
    
    M = np.zeros((total_words, total_words), dtype=np.int8)
    
    # Generate balls
    # For each center c, find all words w within radius.
    # For efficiency, we can iterate over all words and check distance, 
    # but generating the ball directly is faster for small radius.
    
    # For radius 1, ball size is 1 + n*(q-1).
    # For general radius, we can use itertools.combinations for positions and product for values.
    
    import itertools
    
    for c_idx in range(total_words):
        c_word = int_to_word(c_idx, n, q)
        
        # Generate all words in the ball of radius 'radius' around c_word
        # This is equivalent to choosing up to 'radius' positions to change, 
        # and changing them to any value in 0..q-1 except the original.
        
        # Optimization: For small n and q, just iterate all words and check distance?
        # 729 * 729 * 6 ~ 3 million ops. Very fast.
        # Let's just check distance for all pairs to build M.
        # Actually, iterating all pairs is O(N^2 * n).
        # N=729, n=6. 729*729*6 = 3.1M. This is fine.
        
        pass # We will fill M in a nested loop or vectorized way.

    # Vectorized distance calculation might be tricky with mixed types, 
    # but let's stick to the pair loop for clarity and correctness first, 
    # or generate neighbors.
    
    # Generating neighbors is O(N * BallSize).
    # BallSize for R=1, n=6, q=3 is 1 + 6*2 = 13.
    # 729 * 13 = 9477 ops. Much faster.
    
    for c_idx in range(total_words):
        c_word = int_to_word(c_idx, n, q)
        
        # Center itself
        M[c_idx, c_idx] = 1
        
        if radius >= 1:
            # Single position changes
            for pos in range(n):
                orig_val = c_word[pos]
                for new_val in range(q):
                    if new_val != orig_val:
                        c_word[pos] = new_val
                        w_idx = word_to_int(c_word, q)
                        M[w_idx, c_idx] = 1
                        c_word[pos] = orig_val
                        
        if radius >= 2:
            # Two position changes
            # For n=6, this is C(6,2)=15 pairs.
            # For each pair, (q-1)^2 = 4 variations.
            # 15 * 4 = 60 neighbors.
            # Total ball size 1 + 12 + 60 = 73.
            # 729 * 73 = 53k ops. Still fine.
            for pos1 in range(n):
                for pos2 in range(pos1 + 1, n):
                    orig1 = c_word[pos1]
                    orig2 = c_word[pos2]
                    for new1 in range(q):
                        if new1 == orig1: continue
                        for new2 in range(q):
                            if new2 == orig2: continue
                            c_word[pos1] = new1
                            c_word[pos2] = new2
                            w_idx = word_to_int(c_word, q)
                            M[w_idx, c_idx] = 1
                            c_word[pos1] = orig1
                            c_word[pos2] = orig2
                    # restore
                    c_word[pos1] = orig1
                    c_word[pos2] = orig2

    # Define the ILP problem
    # Minimize sum(x)
    # Subject to M @ x >= 1
    # x in {0, 1}
    
    c_obj = np.ones(total_words)
    
    # Symmetry breaking: Force x[0] = 1 (the all-zero word is in the code)
    # This reduces the search space significantly.
    # We can add a constraint x[0] == 1.
    # Or simply set the lower bound of x[0] to 1.
    
    # Bounds for x
    lb = np.zeros(total_words)
    ub = np.ones(total_words)
    lb[0] = 1.0 # Force 000...0 to be selected
    
    # Integrality
    integrality = np.ones(total_words)
    
    # Constraints: M @ x >= 1
    # LinearConstraint(A, lb, ub)
    A_con = M
    lb_con = np.ones(total_words)
    ub_con = np.full(total_words, np.inf)
    
    constraints = LinearConstraint(A_con, lb_con, ub_con)
    bounds = Bounds(lb, ub)
    
    try:
        res = milp(c=c_obj, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0: # Optimal
            # Extract solution
            x = res.x
            # Due to floating point, threshold at 0.5
            selected_indices = [i for i in range(total_words) if x[i] > 0.5]
            
            # Convert to words
            result = [int_to_word(idx, n, q) for idx in selected_indices]
            return result
        else:
            # If ILP fails or times out, fallback to greedy
            pass
            
    except Exception as e:
        print(f"ILP failed: {e}")
        
    # Fallback to greedy
    return _greedy_construct(q, n, radius, start_time, deadline)

def _greedy_construct(q, n, radius, start_time, deadline):
    """
    Fallback greedy algorithm with redundancy removal.
    """
    total_words = q ** n
    
    def int_to_word(idx, n, q):
        word = [0] * n
        for i in range(n):
            word[i] = idx % q
            idx //= q
        return word

    def word_to_int(word, q):
        val = 0
        for i in range(len(word) - 1, -1, -1):
            val = val * q + word[i]
        return val

    # Precompute neighbors
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        w = int_to_word(i, n, q)
        neighbors[i].append(i)
        if radius >= 1:
            for pos in range(n):
                original_val = w[pos]
                for new_val in range(q):
                    if new_val != original_val:
                        w[pos] = new_val
                        neighbor_idx = word_to_int(w, q)
                        neighbors[i].append(neighbor_idx)
                w[pos] = original_val
        if radius >= 2:
            for pos1 in range(n):
                for pos2 in range(pos1 + 1, n):
                    orig1 = w[pos1]
                    orig2 = w[pos2]
                    for new1 in range(q):
                        if new1 == orig1: continue
                        for new2 in range(q):
                            if new2 == orig2: continue
                            w[pos1] = new1
                            w[pos2] = new2
                            neighbor_idx = word_to_int(w, q)
                            neighbors[i].append(neighbor_idx)
                            w[pos1] = orig1
                            w[pos2] = orig2
                    w[pos1] = orig1
                    w[pos2] = orig2

    covered = [False] * total_words
    uncovered_set = set(range(total_words))
    codewords = []
    
    # Seed with 000...0
    seed_idx = 0
    codewords.append(seed_idx)
    for nb in neighbors[seed_idx]:
        if not covered[nb]:
            covered[nb] = True
            if nb in uncovered_set:
                uncovered_set.remove(nb)
                
    while uncovered_set:
        if time.time() > deadline:
            break

        best_center = -1
        best_count = -1

        for candidate in range(total_words):
            count = 0
            for nb in neighbors[candidate]:
                if nb in uncovered_set:
                    count += 1

            if count > best_count:
                best_count = count
                best_center = candidate

        if best_center == -1:
            break

        codewords.append(best_center)
        for nb in neighbors[best_center]:
            if nb in uncovered_set:
                uncovered_set.remove(nb)

    # Post-processing: remove redundant codewords
    # This is O(K^2 * BallSize). K is ~73, BallSize ~13. 73^2 * 13 ~ 70k. Fast.
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        for i in range(len(codewords)):
            if time.time() > deadline:
                break
            c_idx = codewords[i]
            redundant = True
            for w_idx in neighbors[c_idx]:
                covered_by_other = False
                for j in range(len(codewords)):
                    if j == i:
                        continue
                    if w_idx in neighbors[codewords[j]]:
                        covered_by_other = True
                        break
                if not covered_by_other:
                    redundant = False
                    break

            if redundant:
                codewords.pop(i)
                changed = True
                break

    result = [int_to_word(c, n, q) for c in codewords]
    return result
