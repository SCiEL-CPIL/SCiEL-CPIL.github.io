import itertools
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    num_words = len(all_words)
    
    # Precompute balls (coverage sets) efficiently using vectorization
    # Create a matrix of all words
    words_array = np.array(all_words)
    
    # Compute pairwise Hamming distances
    # dist_matrix[i, j] = number of positions where words_array[i] != words_array[j]
    # We can compute this by broadcasting
    # dist_matrix = np.sum(words_array[:, None, :] != words_array[None, :, :], axis=2)
    # This creates a (num_words, num_words, n) array which might be large but for 729x729x6 it's ~3M elements, which is fine.
    dist_matrix = np.sum(words_array[:, None, :] != words_array[None, :, :], axis=2)
    
    # Incidence matrix A: A[i, j] = 1 if dist(i, j) <= radius
    A = (dist_matrix <= radius).astype(int)
    
    # Time check
    if time.time() > deadline:
        return _fallback_construct(q, n, radius)

    # Formulate ILP
    # Minimize sum(x_i)
    c = np.ones(num_words)
    
    # Constraints: For every word i, sum of x_j for j in ball(i) >= 1
    # A x >= 1
    lb = np.ones(num_words)
    ub = np.full(num_words, np.inf)
    
    constraints = LinearConstraint(A, lb, ub)
    
    # Integrality: all variables are binary (0 or 1)
    integrality = np.ones(num_words)
    
    # Bounds for variables
    var_bounds = Bounds(0, 1)
    
    solve_start = time.time()
    try:
        # Use a tighter gap if possible, but respect time limit
        # For this size, HiGHS (default in scipy) should be fast.
        res = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=var_bounds,
            options={
                'time_limit': 120, # Hard limit inside solver
                'mip_rel_gap': 1e-6,
                'verbose': 0
            }
        )
    except Exception:
        return _fallback_construct(q, n, radius)
    
    if time.time() > deadline:
        return _fallback_construct(q, n, radius)

    if res.status == 0 or res.status == 1: # Optimal or Feasible
        x = res.x
        # Extract codewords
        indices = [i for i in range(num_words) if x[i] > 0.5]
        codewords = [list(all_words[i]) for i in indices]
        
        # Post-processing: Remove redundant codewords
        # A codeword is redundant if all words it covers are also covered by other selected codewords.
        # This is a greedy post-optimization step.
        if codewords:
            # Build coverage sets for selected codewords
            # selected_cover[i] = set of indices covered by codeword i
            # We can reuse A. A[j, i] == 1 means word j is covered by codeword i.
            # Let's create a list of sets for the selected indices.
            selected_set = set(indices)
            
            # For each selected codeword, check if it is redundant
            # A codeword idx is redundant if for every word j such that A[j, idx]==1, 
            # there exists another selected codeword idx2 != idx such that A[j, idx2]==1.
            
            # To speed this up, we can iterate and remove.
            # However, for 73 words, checking redundancy for each is O(73 * 729 * 73) which is manageable.
            # Actually, we can be smarter.
            # Let's just do a simple greedy removal if we have time.
            
            remaining = list(indices)
            # Sort by degree (number of words covered) descending? 
            # Usually removing high-degree nodes is riskier if they are unique, but if they are redundant, they are covered by others.
            # Let's just iterate.
            
            # Optimization: Precompute union of coverage of all *other* selected nodes for each node?
            # That's expensive.
            
            # Simpler: 
            # For each node u in remaining:
            #   is_redundant = True
            #   for each word v covered by u:
            #      if no other node w in remaining (w != u) covers v:
            #          is_redundant = False
            #          break
            #   if is_redundant: remove u
            
            # Given the small size (73 nodes), this is fast.
            new_remaining = []
            # We need to check against the *current* set of remaining nodes.
            # This is a greedy set cover minimization on the subgraph induced by the solution.
            
            # Let's implement a simple loop.
            # We want to remove as many as possible.
            # Order matters. Removing a node might make another redundant.
            # We can repeat until no more can be removed.
            
            # To avoid O(N^2 * M) complexity in Python, we can use bitsets or numpy.
            # But N=73, M=729. 73^2 * 729 ~ 4M ops. Python is slow.
            # Let's use numpy for the redundancy check.
            
            # Matrix R where R[i, j] = 1 if codeword j covers word i. (This is just columns of A)
            # We have selected indices.
            # Let's extract the submatrix of A corresponding to selected codewords.
            # A_sel = A[:, indices]
            
            # We want to find a minimal subset of columns of A_sel such that the union of their 1s is all 1s.
            # This is again a set cover, but on a smaller instance.
            # We can just use the same ILP solver on this smaller instance?
            # Or a greedy heuristic.
            
            # Let's try a greedy heuristic for redundancy removal.
            # It's safe because we only remove if coverage is preserved.
            
            # For simplicity and safety within time limits, we might skip this if the ILP already returned a minimal solution.
            # ILP solvers often return minimal solutions if the objective is sum(x).
            # However, if it hit the time limit (status 1), it might not be minimal.
            # Let's add a lightweight check.
            
            # If we have time, do a greedy removal.
            if time.time() < deadline - 5.0:
                # Greedy removal
                # We represent coverage of each selected codeword as a set of indices.
                # Or better, use the matrix A.
                # Let's keep a list of active indices.
                active = list(indices)
                changed = True
                while changed and time.time() < deadline - 2.0:
                    changed = False
                    for i, idx in enumerate(active):
                        # Check if idx is redundant given active[0:i] + active[i+1:]
                        # Union of coverage of others
                        # If A[:, idx] is subset of sum(A[:, others], axis=1) >= 1
                        
                        # Vectorized check:
                        # others_mask = np.ones(len(active), dtype=bool)
                        # others_mask[i] = False
                        # others_indices = np.array([active[j] for j in range(len(active)) if j != i])
                        # covered_by_others = np.sum(A[:, others_indices], axis=1) > 0
                        # covered_by_idx = A[:, idx] == 1
                        # if np.all(covered_by_idx[covered_by_idx] <= covered_by_others[covered_by_idx]):
                        #    # Redundant
                        #    active.pop(i)
                        #    changed = True
                        #    break # Restart loop since indices shifted
                        
                        # This inner loop is O(N^2 * M). With N=73, it's fine.
                        
                        # To be faster, we can just check if there is any word covered ONLY by idx.
                        # count_covered = np.sum(A[:, active], axis=1)
                        # If min(count_covered[covered_by_idx]) == 1, then idx is not redundant.
                        # If min(count_covered[covered_by_idx]) > 1, idx might be redundant?
                        # No, if a word is covered by 2 nodes, removing one is fine.
                        # If a word is covered by 1 node (idx), idx is essential.
                        
                        # So: idx is redundant IF NO word is covered ONLY by idx.
                        # i.e., For all j where A[j, idx]==1, there exists k != idx such that A[j, k]==1.
                        
                        # Let's compute counts for all active nodes.
                        # But counts change as we remove nodes.
                        
                        # Let's just do the simple check for each candidate.
                        # It's 73 iterations max in the inner loop.
                        
                        others_indices = [active[k] for k in range(len(active)) if k != i]
                        if not others_indices:
                            continue
                        covered_by_others = np.sum(A[:, others_indices], axis=1) > 0
                        covered_by_idx = A[:, idx] == 1
                        
                        # If there is any word covered by idx but NOT by others, idx is essential.
                        # essential_mask = covered_by_idx & ~covered_by_others
                        # if np.any(essential_mask):
                        #    continue
                        # else:
                        #    active.pop(i)
                        #    changed = True
                        #    break
                        
                        if not np.any(covered_by_idx & ~covered_by_others):
                            active.pop(i)
                            changed = True
                            break
                            
                if active != indices:
                    codewords = [list(all_words[i]) for i in active]
        
        return codewords
    else:
        return _fallback_construct(q, n, radius)

def _fallback_construct(q, n, radius):
    # The original seed program's logic
    # Direct sum of the ternary tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    if n == 4:
        return tetracode
    if n < 4:
        # Handle small n recursively or brute force if needed, but task is n=6
        # For n=6, this won't be called unless ILP fails
        pass
        
    # If n > 4, extend tetracode
    # The seed used: [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    # This yields 9 * 3**(n-4) words.
    # For n=6, this is 9 * 9 = 81 words.
    
    if n >= 4:
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    
    # Fallback for n < 4 (not expected for this task)
    # Simple greedy or brute force
    all_words = list(itertools.product(range(q), repeat=n))
    selected = []
    covered = set()
    for w in all_words:
        if w not in covered:
            selected.append(list(w))
            # Add all words within radius to covered
            for v in all_words:
                dist = sum(1 for a, b in zip(w, v) if a != b)
                if dist <= radius:
                    covered.add(v)
    return selected

# EVOLVE-BLOCK-END
