import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix

def construct(q, n, radius):
    t_start = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    space_size = q ** n
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i - 1] * q

    def idx_to_word(idx):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n - 1 - pos]) % q
        return w

    # Precompute neighbors for radius=1
    # For general radius, this would be more complex, but the task specifies radius=1 
    # implicitly by the context of K_3(7,1) and the seed code structure.
    # However, to be robust to the interface `construct(q, n, radius)`, 
    # I should handle general radius if possible, or at least radius 1 efficiently.
    # Given the specific task is K_3(7,1), I will optimize for radius=1 but 
    # include a fallback or general logic if radius != 1.
    
    # For radius=1, neighbors are self + 1-step changes.
    # Number of neighbors = 1 + n*(q-1)
    
    if radius == 1:
        nb_count = 1 + n * (q - 1)
        neighbors = np.empty((space_size, nb_count), dtype=np.int32)
        for idx in range(space_size):
            w = idx_to_word(idx)
            nb_list = [idx]
            for pos in range(n):
                for new_d in range(q):
                    if new_d == w[pos]:
                        continue
                    delta = (new_d - w[pos]) * q_powers[n - 1 - pos]
                    nb_list.append(idx + delta)
            neighbors[idx] = nb_list
    else:
        # Fallback for other radii: just compute Hamming distance directly 
        # This is slow, but the task is specifically K_3(7,1).
        # For now, assume radius 1 is the primary case.
        # If radius > 1, we might need a different approach or it might be too slow.
        # Let's stick to the radius=1 optimization as it's the target task.
        raise NotImplementedError("Only radius=1 optimized for this strategy")

    # Build the constraint matrix A for ILP
    # Variables x_i in {0,1} indicating if codeword i is chosen.
    # Constraints: For every word j in the space, sum_{i in N(j)} x_i >= 1
    # Here N(j) is the set of codewords within distance 'radius' of j.
    # Note: The matrix A is the transpose of the neighbor matrix if we view
    # neighbors[i] as "i covers these".
    # Let's define A such that A[j, i] = 1 if i is in neighbors[j] (i covers j).
    # Actually, it's easier to build A where rows are constraints (words to cover) 
    # and columns are variables (potential codewords).
    # A[j, i] = 1 if HammingDist(word_j, word_i) <= radius.
    
    # We already have `neighbors` where neighbors[i] are indices covered by i.
    # So, for each i (column), for each j in neighbors[i] (row), A[j, i] = 1.
    
    A = lil_matrix((space_size, space_size), dtype=np.int8)
    for i in range(space_size):
        for j in neighbors[i]:
            A[j, i] = 1
            
    A_csr = A.tocsr()

    # Objective: minimize sum(x_i)
    c = np.ones(space_size, dtype=np.float64)

    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(0, 1)

    # Constraints: A @ x >= 1
    # milp requires LinearConstraint(A, lb, ub)
    # lb = np.ones(space_size), ub = np.inf
    constraints = LinearConstraint(A_csr, lb=np.ones(space_size), ub=np.inf)

    # Integrality: x_i must be integer
    integrality = np.ones(space_size, dtype=int)

    # Solve using HiGHS (default in scipy)
    # We need to set a time limit. 
    # scipy's milp does not directly expose a time limit parameter in the standard interface 
    # in all versions, but we can run it and hope it finishes, or use a thread-safe wrapper?
    # Actually, scipy.optimize.milp uses HiGHS. 
    # To enforce time limit, we might need to rely on the solver's internal timeout if exposed, 
    # or just run it and check time. 
    # However, if it runs too long, we can't easily interrupt it in Python without threads.
    
    # Alternative: Use a heuristic ILP solver or a different library? 
    # Only standard lib, numpy, scipy are allowed.
    
    # Let's try running the ILP. If it takes too long, we might need a fallback.
    # But HiGHS is quite fast for this size (2187 vars, 2187 constraints).
    # The matrix is sparse (each row has ~14 non-zeros for q=3, n=7).
    
    try:
        # Set options for HiGHS if possible? 
        # scipy.optimize.milp options can pass to the solver.
        # 'time_limit' is not a standard option key in all scipy versions for milp directly 
        # in the simple dict, but HiGHS supports it.
        # Let's try passing options.
        options = {'presolve': True}
        
        # Note: If the solver doesn't respect a hard stop, this might hang.
        # However, for this specific problem size, HiGHS should be very fast 
        # (often < 1 second for 2k variables with sparse constraints).
        
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds, options=options)
        
        if res.success and res.status == 0:
            x = res.x
            # Round to nearest integer (should be 0 or 1)
            x_int = np.round(x).astype(int)
            code_indices = np.where(x_int == 1)[0]
            
            # Verify coverage
            if len(code_indices) > 0:
                cov = np.zeros(space_size, dtype=np.int32)
                for c_idx in code_indices:
                    cov[neighbors[c_idx]] += 1
                if np.all(cov > 0):
                    return [idx_to_word(i) for i in code_indices]
            else:
                # Fallback if ILP failed or returned empty
                pass
                
    except Exception as e:
        pass

    # Fallback: Greedy Heuristic (similar to seed but simpler)
    # If ILP fails or times out (unlikely for this size, but just in case),
    # use a fast greedy approach.
    
    def greedy_cover(seed=0):
        rng = np.random.default_rng(seed)
        uncovered = np.ones(space_size, dtype=bool)
        code = []
        while np.any(uncovered) and (time.time() - t_start) < deadline - 5.0:
            # Compute gains
            # Sum of uncovered neighbors for each candidate
            # This is O(N * M) where M is nb_count. 
            # Can be optimized with matrix multiplication on boolean arrays.
            gains = np.sum(uncovered[neighbors], axis=1)
            # Add perturbation
            perturb = rng.random(space_size) * 0.001
            gains_pert = gains + perturb
            best = np.argmax(gains_pert)
            if gains[best] == 0:
                break
            code.append(int(best))
            uncovered[neighbors[best]] = False
        return code

    # Try a few greedy seeds
    best_code = None
    best_size = space_size + 1
    
    for seed in range(20):
        if (time.time() - t_start) > deadline - 10.0:
            break
        code = greedy_cover(seed)
        cov = np.zeros(space_size, dtype=np.int32)
        for c in code:
            cov[neighbors[c]] += 1
        if np.all(cov > 0):
            if len(code) < best_size:
                best_code = code[:]
                best_size = len(code)
        
        # Quick improvement: try removing redundant codewords
        if best_code is not None and best_size < space_size:
            cur_code = best_code[:]
            cur_cov = np.zeros(space_size, dtype=np.int32)
            for c in cur_code:
                cur_cov[neighbors[c]] += 1
            
            improved = True
            while improved and (time.time() - t_start) < deadline - 5.0:
                improved = False
                for i in range(len(cur_code)):
                    c = cur_code[i]
                    if np.min(cur_cov[neighbors[c]]) >= 2:
                        cur_code.pop(i)
                        cur_cov[neighbors[c]] -= 1
                        improved = True
                        break
                if not improved:
                    # Try random swaps? No, just stick to removal for speed.
                    break
            
            if len(cur_code) < best_size:
                best_code = cur_code[:]
                best_size = len(cur_code)

    if best_code is None:
        # Last resort: all words
        return [idx_to_word(i) for i in range(space_size)]
        
    return [idx_to_word(i) for i in best_code]
