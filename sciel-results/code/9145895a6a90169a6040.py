import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    total_space = q ** n
    
    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * q + v
        return idx

    def idx_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        w.reverse()
        return w

    all_words = [idx_to_word(i) for i in range(total_space)]
    
    # Precompute neighbors as bitsets (Python integers)
    # neighbor[i] has bit j set if dist(i, j) <= radius
    neighbors = [0] * total_space
    for i in range(total_space):
        w = all_words[i]
        mask = 1 << i
        neighbors[i] |= mask
        for pos in range(n):
            orig_val = w[pos]
            for val in range(q):
                if val != orig_val:
                    nw = w[:]
                    nw[pos] = val
                    j = word_to_idx(nw)
                    neighbors[i] |= (1 << j)
    
    # Greedy Set Cover
    covered = 0
    code = []
    remaining = set(range(total_space))
    
    # Optimization: Pre-sort or just iterate? 
    # Iterating all remaining each time is O(N^2). 
    # N=2187. 2187^2 = 4.7M. 
    # Each step involves popcount.
    # 200 steps * 2187 candidates * (bitwise ops) is fine.
    
    while remaining:
        if time.time() > deadline - 40:
            break
        best_idx = -1
        max_gain = 0
        for i in remaining:
            gain = (neighbors[i] & ~covered).bit_count()
            if gain > max_gain:
                max_gain = gain
                best_idx = i
        if best_idx == -1:
            break
        code.append(best_idx)
        covered |= neighbors[best_idx]
        remaining.remove(best_idx)
        
    # Local Search
    # We represent code as a set for O(1) lookup
    code_set = set(code)
    
    # Helper to calculate coverage of all except one
    # This is expensive if done naively.
    # Let's maintain a list and recompute if needed, or just recompute in the loop.
    # Given small size, recomputing is fine.
    
    improved = True
    while improved and time.time() < deadline - 20:
        improved = False
        
        # Pass 1: Deletion
        for i in range(len(code)):
            w_i = code[i]
            # Calculate coverage of all other words
            others_covered = 0
            for j in range(len(code)):
                if j != i:
                    others_covered |= neighbors[code[j]]
            
            # Check if w_i is redundant
            if (neighbors[w_i] & ~others_covered) == 0:
                # Remove w_i
                code.pop(i)
                code_set.remove(w_i)
                improved = True
                break # Restart deletion pass
        
        # Pass 2: Swap
        if not improved:
            for i in range(len(code)):
                w_i = code[i]
                others_covered = 0
                for j in range(len(code)):
                    if j != i:
                        others_covered |= neighbors[code[j]]
                
                unique_i = neighbors[w_i] & ~others_covered
                if unique_i == 0:
                    # Should have been caught in deletion
                    continue
                
                # Find a word w_j not in code that covers unique_i
                # We want to swap w_i with w_j
                # Condition: unique_i is subset of neighbors[w_j]
                # i.e., (unique_i & ~neighbors[w_j]) == 0
                
                # To make the swap useful, we might prefer w_j that covers more "new" stuff?
                # Or just any valid swap?
                # Any valid swap maintains coverage.
                # Let's pick the first one found to keep it fast.
                # Or the one with max gain?
                # Let's try max gain among valid swaps.
                
                best_j = -1
                max_new_gain = -1
                for j in range(total_space):
                    if j in code_set:
                        continue
                    if (unique_i & ~neighbors[j]) == 0:
                        # Valid swap
                        # Gain: how many new words does j cover compared to others?
                        # new_gain = (neighbors[j] & ~others_covered).bit_count()
                        # We already know it covers unique_i.
                        # Let's just take the first valid one to speed up, 
                        # or the one with max popcount(neighbors[j] & ~others_covered)
                        gain = (neighbors[j] & ~others_covered).bit_count()
                        if gain > max_new_gain:
                            max_new_gain = gain
                            best_j = j
                
                if best_j != -1:
                    # Perform swap
                    code[i] = best_j
                    code_set.remove(w_i)
                    code_set.add(best_j)
                    improved = True
                    break # Restart deletion pass
    
    # ILP Solve
    time_limit = max(1.0, deadline - time.time() - 5.0)
    
    if time_limit > 2.0:
        # Build sparse matrix for ILP
        # A[i, j] = 1 if j in neighbors[i]
        # We can reuse the neighbor logic or build from the bitsets.
        # Building from bitsets is faster than iterating lists.
        row_idx = []
        col_idx = []
        data = []
        for i in range(total_space):
            mask = neighbors[i]
            # Extract set bits
            # This is O(popcount)
            while mask:
                # Get lowest set bit
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                row_idx.append(i)
                col_idx.append(j)
                data.append(1.0)
                mask ^= lsb
                
        A = csr_matrix((data, (row_idx, col_idx)), shape=(total_space, total_space))
        c = np.ones(total_space)
        lb = np.ones(total_space)
        ub = np.full(total_space, np.inf)
        constraints = LinearConstraint(A, lb, ub)
        integrality = np.ones(total_space)
        bounds = Bounds(lb=np.zeros(total_space), ub=np.ones(total_space))
        
        # Provide initial solution to ILP
        # x0 is the indicator vector of the current code
        x0 = np.zeros(total_space)
        for idx in code:
            x0[idx] = 1.0
            
        try:
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds, 
                       options={"time_limit": time_limit, "presolve": True, "mip_rel_gap": 0.0, "mip_start": x0})
            
            if res.status == 0 or res.status == 2:
                if res.x is not None:
                    x_opt = res.x
                    # Check feasibility
                    prod = A @ x_opt
                    if np.all(prod >= 1.0 - 1e-6):
                        code_indices = [i for i in range(total_space) if x_opt[i] > 0.5]
                        return [all_words[i] for i in code_indices]
        except Exception:
            pass
            
    return [all_words[i] for i in code]
