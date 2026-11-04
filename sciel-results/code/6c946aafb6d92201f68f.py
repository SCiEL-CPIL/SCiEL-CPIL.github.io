import numpy as np
import itertools
import time
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(q, n, radius):
    start_time = time.time()
    # The evaluator kills after 150s. We need to be safe. 
    # Let's aim to finish in 100s to be very safe.
    deadline = start_time + 100.0
    
    # 1. Generate all words and map them to indices
    words = list(itertools.product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # 2. Precompute neighbors for each word (the sets S_j)
    # For radius 1, a word covers itself and all words at Hamming distance 1.
    # Number of neighbors = 1 + n * (q - 1)
    k = radius
    # We only implement radius 1 for now as per task contract K_3(7,1)
    # But let's make it slightly general for radius 1 specifically.
    # For radius > 1, the number of neighbors grows combinatorially, 
    # but for n=7, q=3, r=1, it's 15.
    
    neighbors = []
    for i, w in enumerate(words):
        neigh_set = set([i])
        if radius == 1:
            for pos in range(n):
                orig_val = w[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    new_w_list = list(w)
                    new_w_list[pos] = val
                    new_w = tuple(new_w_list)
                    neigh_set.add(word_to_idx[new_w])
        # Note: If radius > 1, we would need to generate all words within that distance.
        # Given the specific task is K_3(7,1), we assume radius=1.
        neighbors.append(neigh_set)
        
    # 3. Heuristic: Greedy Set Cover to get an initial feasible solution
    # This is important for ILP solvers to start from a good bound.
    uncovered = np.ones(N, dtype=bool)
    heuristic_code = []
    
    # Precompute neighbor lists for faster access in heuristic
    neighbor_lists = [list(neigh) for neigh in neighbors]
    
    while np.any(uncovered):
        best_idx = -1
        best_count = 0
        # Find word that covers most uncovered points
        for j in range(N):
            # Count how many neighbors of j are uncovered
            count = np.sum([uncovered[nb] for nb in neighbor_lists[j]])
            if count > best_count:
                best_count = count
                best_idx = j
        heuristic_code.append(best_idx)
        for nb in neighbor_lists[best_idx]:
            uncovered[nb] = False
            
        if time.time() > deadline - 5:
            break

    # 4. Formulate ILP
    # Variables: x_j (0/1) for each codeword j
    # Objective: min sum(x_j)
    c = np.ones(N)
    
    # Constraints: For every point i, sum(x_j for j in neighbors_of_i) >= 1
    # We need to build the constraint matrix A.
    # A is N x N. A[i, j] = 1 if point i is covered by codeword j.
    # Note: neighbors[j] contains points covered by j.
    # So A[i, j] = 1 if i in neighbors[j].
    # This is the transpose of the "word covers point" relation if we think of rows as words.
    # Let's build A directly.
    
    # Efficiently build sparse A? scipy.optimize.milp accepts dense arrays or sparse.
    # N=2187, N^2 ~ 4.7M floats. This is ~37MB. Dense is fine.
    A = np.zeros((N, N), dtype=float)
    
    for j in range(N):
        for i in neighbors[j]:
            A[i, j] = 1.0
            
    constraints = LinearConstraint(A, lb=1, ub=np.inf)
    integrality = np.ones(N)
    bounds = Bounds(0, 1)
    
    # Prepare initial solution x0 from heuristic
    x0 = np.zeros(N)
    for idx in heuristic_code:
        x0[idx] = 1.0
        
    # Verify heuristic is valid before passing to MILP (optional but good for debugging)
    # Actually, if the heuristic broke early, it might not be valid.
    # If not valid, we shouldn't pass x0 or we should pass None.
    # Let's check validity of heuristic.
    is_valid_heuristic = True
    if time.time() < deadline - 5:
        counts = np.zeros(N, dtype=int)
        for idx in heuristic_code:
            counts[neighbors[idx]] += 1 # This is slow if using sets. 
            # Let's use neighbor_lists
            pass # Skip detailed check, just check if any count is 0? 
            # Actually, let's just compute coverage properly for x0.
    
    # Recalculate coverage for x0 properly
    coverage = np.zeros(N, dtype=int)
    for idx in heuristic_code:
        for nb in neighbor_lists[idx]:
            coverage[nb] += 1
    
    if np.any(coverage == 0):
        x0 = None # Heuristic failed to cover everything, don't use as start point
    else:
        x0 = x0 # Keep the good start point

    # 5. Solve ILP
    # We have about 90 seconds left.
    time_left = deadline - time.time() - 2
    if time_left < 1:
        # Not enough time, return heuristic if valid, else all words
        if np.any(coverage == 0):
             # Emergency fallback: add all words not covered? 
             # Or just return the heuristic which is invalid? No, must be valid.
             # Let's finish the greedy quickly
             uncovered = (coverage == 0)
             while np.any(uncovered):
                 best_idx = -1
                 best_count = 0
                 for j in range(N):
                     if x0[j] > 0: continue # Already picked
                     count = np.sum([uncovered[nb] for nb in neighbor_lists[j]])
                     if count > best_count:
                         best_count = count
                         best_idx = j
                 if best_idx == -1: break
                 x0[best_idx] = 1
                 heuristic_code.append(best_idx)
                 for nb in neighbor_lists[best_idx]:
                     coverage[nb] += 1
                     if coverage[nb] == 1:
                         uncovered[nb] = False
                 if time.time() > deadline - 1:
                     break
             # Final check
             if np.any(coverage == 0):
                 # Absolute fallback
                 return [list(words[i]) for i in range(N)]
             return [list(words[i]) for i in range(N) if x0[i] > 0]

    try:
        res = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
            x0=x0,
            options={
                "time_limit": int(time_left),
                "mip_rel_gap": 0.0, # Seek optimal
                "presolve": True,
                "parallel": True
            }
        )
        
        if res.x is not None:
            # Round to nearest integer (should be exactly 0 or 1)
            x_opt = np.round(res.x).astype(int)
            
            # Verify the solution
            # Check if x_opt is binary
            if not np.all((x_opt == 0) | (x_opt == 1)):
                # Should not happen, but fallback to rounding
                x_opt[x_opt > 0.5] = 1
                x_opt[x_opt <= 0.5] = 0
                
            # Check coverage
            valid = True
            coverage_check = np.zeros(N, dtype=int)
            # Efficient coverage check using matrix multiplication?
            # A is N x N. x_opt is N x 1.
            # covered_counts = A.T @ x_opt ? No.
            # A[i, j] is 1 if point i is covered by word j.
            # So (A @ x_opt)[i] is the number of selected words covering point i.
            counts = A @ x_opt
            if np.any(counts < 1):
                valid = False
            
            if valid:
                code_indices = np.where(x_opt == 1)[0]
                # Ensure distinctness (implied by x_opt being binary)
                return [list(words[i]) for i in code_indices]
            else:
                # If ILP returned an invalid solution (rare), fall back to heuristic
                pass
    except Exception as e:
        # Handle any exceptions from solver
        pass

    # 6. Fallback: Return the heuristic solution (fixed if needed)
    # Re-verify heuristic coverage
    final_coverage = np.zeros(N, dtype=int)
    final_code_indices = []
    for idx in heuristic_code:
        final_code_indices.append(idx)
        for nb in neighbor_lists[idx]:
            final_coverage[nb] += 1
    
    if np.any(final_coverage == 0):
         # If heuristic was invalid, we must fix it.
         # This is a last resort.
         uncovered = (final_coverage == 0)
         while np.any(uncovered) and time.time() < deadline - 1:
             best_idx = -1
             best_count = 0
             for j in range(N):
                 if j in final_code_indices: continue
                 count = np.sum([uncovered[nb] for nb in neighbor_lists[j]])
                 if count > best_count:
                     best_count = count
                     best_idx = j
             if best_idx == -1: break
             final_code_indices.append(best_idx)
             for nb in neighbor_lists[best_idx]:
                 final_coverage[nb] += 1
                 if final_coverage[nb] == 1:
                     uncovered[nb] = False
         
         if np.any(final_coverage == 0):
             # Absolute fallback
             return [list(words[i]) for i in range(N)]
             
    return [list(words[i]) for i in final_code_indices]
