import numpy as np
import itertools
import time
import random
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def construct(q, n, radius):
    start_time = time.time()
    # Leave margin for Python overhead, local search, and safety
    deadline = start_time + 140.0 

    # 1. Generate all words and map to indices
    words = list(itertools.product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # 2. Precompute neighbors (words within Hamming distance <= radius)
    # For q=3, n=7, radius=1, each word has 1 + 7*(3-1) = 15 neighbors.
    neighbors = []
    for i, w in enumerate(words):
        neigh = set([w])
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                neigh.add(tuple(new_w))
        neigh_indices = [word_to_idx[ng] for ng in neigh]
        neighbors.append(neigh_indices)
        
    # Helper: Compute coverage counts for a given list of codeword indices
    def compute_counts(code_list):
        counts = np.zeros(N, dtype=int)
        for idx in code_list:
            for nb in neighbors[idx]:
                counts[nb] += 1
        return counts

    # Helper: Local search to remove redundant codewords
    def local_search_remove(code_list):
        # Convert to list of indices
        code_indices = list(code_list)
        counts = compute_counts(code_indices)
        
        changed = True
        # Iterate until no more removals possible or time is up
        while changed and (time.time() < deadline - 5.0):
            changed = False
            i = 0
            while i < len(code_indices):
                idx = code_indices[i]
                # Check if idx is redundant: all its neighbors are covered by other codewords
                # i.e., for all nb in neighbors[idx], counts[nb] > 1
                is_redundant = True
                for nb in neighbors[idx]:
                    if counts[nb] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove idx
                    code_indices.pop(i)
                    for nb in neighbors[idx]:
                        counts[nb] -= 1
                    changed = True
                else:
                    i += 1
        return code_indices

    # 3. Greedy Set Cover (Fast Initial Solution)
    # This provides a valid upper bound and a fallback solution.
    uncovered = np.ones(N, dtype=bool)
    greedy_code = []
    
    while np.any(uncovered):
        best_idx = -1
        best_count = 0
        # Find word that covers most uncovered points
        for j in range(N):
            # Count how many neighbors of j are uncovered
            # Optimization: Use numpy sum over the neighbor indices
            count = np.sum(uncovered[neighbors[j]])
            if count > best_count:
                best_count = count
                best_idx = j
        
        greedy_code.append(best_idx)
        uncovered[neighbors[best_idx]] = False
        
        # Safety check for greedy loop
        if time.time() > deadline - 60:
            break

    # Run local search on greedy solution
    greedy_code = local_search_remove(greedy_code)
    
    # If we are running out of time, return the best we have
    if time.time() > deadline - 10:
        return [list(words[i]) for i in greedy_code]

    # 4. ILP Formulation
    # Build sparse incidence matrix A
    # A[i, j] = 1 if codeword j covers word i
    rows = []
    cols = []
    data = []
    for i in range(N):
        for j in neighbors[i]:
            rows.append(i)
            cols.append(j)
            data.append(1)
            
    A = csr_matrix((data, (rows, cols)), shape=(N, N))
    
    c = np.ones(N)
    constraints = LinearConstraint(A, lb=1, ub=np.inf)
    integrality = np.ones(N)
    bounds = Bounds(0, 1)
    
    best_code = greedy_code
    best_len = len(greedy_code)
    
    try:
        # Solve ILP
        # Set time limit to leave room for local search and return
        ilp_time_limit = max(10, int(deadline - (time.time() - start_time) - 15))
        
        res = milp(c=c, 
                   constraints=constraints, 
                   integrality=integrality, 
                   bounds=bounds, 
                   options={
                       "time_limit": ilp_time_limit,
                       "mip_rel_gap": 0.0  # Ask for optimal if possible
                   })
        
        if res.x is not None:
            # Extract solution
            # res.x might be float, convert to int
            x = res.x
            code_indices = [int(i) for i in np.where(x > 0.5)[0]]
            
            # Verify validity just in case of numerical issues
            counts = compute_counts(code_indices)
            if np.all(counts >= 1):
                # Apply local search to polish
                polished_code = local_search_remove(code_indices)
                if len(polished_code) < best_len:
                    best_code = polished_code
                    best_len = len(polished_code)
            else:
                # If ILP returned an invalid solution (shouldn't happen, but safe)
                # Fall back to greedy
                pass
                
    except Exception:
        # If MILP fails, stick with greedy
        pass

    # 5. Final Polish with Local Search on the best candidate
    # We might have more time if ILP finished early
    if time.time() < deadline - 5:
        final_code = local_search_remove(best_code)
        # One more pass to be sure
        final_code = local_search_remove(final_code)
        best_code = final_code

    return [list(words[i]) for i in best_code]
