import time
import random
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Pre-process clauses into 0-based indices and signs
    # literal is 1-based signed. 
    # var = abs(l) - 1
    # sign = 1 if l > 0 else 0 (True for positive, False for negative? 
    # Let's use 1 for True, 0 for False.
    # Literal l is satisfied if assignment[var] == (l > 0)
    
    pre_clauses = []
    for c in clauses:
        pc = []
        for l in c:
            var = abs(l) - 1
            is_pos = 1 if l > 0 else 0
            pc.append((var, is_pos))
        pre_clauses.append(pc)
        
    # Spectral Initialization
    def spectral_init():
        # Build graph where edges connect variables in same clause
        # Weight by frequency
        adj = np.zeros((n, n))
        for c in pre_clauses:
            for i in range(len(c)):
                for j in range(i+1, len(c)):
                    v1, s1 = c[i]
                    v2, s2 = c[j]
                    adj[v1, v2] += 1
                    adj[v2, v1] += 1
        
        # Compute Laplacian
        degree = adj.sum(axis=1)
        L = np.diag(degree) - adj
        
        # Find Fiedler vector (2nd smallest eigenvalue)
        try:
            # k=3 to ensure we get the 2nd smallest (index 1)
            # sigma=0 to find smallest eigenvalues
            vals, vecs = eigsh(L, k=3, sigma=0, which='LM')
            # Sort by eigenvalue
            idx = np.argsort(vals)
            fiedler_vec = vecs[:, idx[1]]
            
            # Map to assignment
            assignment = np.zeros(n, dtype=bool)
            # If value > 0, assign True, else False
            # Note: eigenvector sign is arbitrary, but consistent within run
            assignment = fiedler_vec > 0
            return assignment
        except Exception:
            # Fallback to random if spectral fails
            return np.random.rand(n) < 0.5

    def count_unsat(assignment):
        count = 0
        for c in pre_clauses:
            sat = False
            for var, is_pos in c:
                if assignment[var] == (is_pos == 1):
                    sat = True
                    break
            if not sat:
                count += 1
        return count

    def weighted_walksat(assignment, max_flips, deadline):
        # Weights for literals
        # Index: var * 2 + (1 - is_pos) ? 
        # Let's use two arrays: weight_true[var], weight_false[var]
        # Or just a dict or array of size 2*n
        # weight[2*var] for False, weight[2*var+1] for True
        weights = np.ones(2 * n)
        
        current_unsat = count_unsat(assignment)
        
        for _ in range(max_flips):
            if time.time() > deadline:
                break
            
            if current_unsat == 0:
                return assignment, 0
            
            # Find an unsatisfied clause
            # Randomly pick from unsatisfied ones
            # To avoid scanning all clauses every time, we can maintain a list, 
            # but for N=50, scanning is fast enough.
            unsat_clauses = []
            for idx, c in enumerate(pre_clauses):
                sat = False
                for var, is_pos in c:
                    if assignment[var] == (is_pos == 1):
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(idx)
            
            if not unsat_clauses:
                return assignment, 0
                
            c_idx = random.choice(unsat_clauses)
            c = pre_clauses[c_idx]
            
            # Calculate gain for each literal in c
            best_gain = -1
            best_var = -1
            best_is_pos = -1
            candidates = []
            
            for var, is_pos in c:
                # If we flip this literal, it becomes satisfied.
                # How many other unsatisfied clauses contain this literal (with same polarity)?
                # And how many satisfied clauses contain this literal (with same polarity) where it is the ONLY true literal?
                
                # This requires checking all clauses.
                # Optimization: Pre-compute for each literal the list of clauses it appears in.
                # But for N=50, M is small, direct check is fine.
                
                gain = 0
                # 1. Satisfy other unsatisfied clauses
                for u_idx in unsat_clauses:
                    u_c = pre_clauses[u_idx]
                    for u_var, u_is_pos in u_c:
                        if u_var == var and u_is_pos == is_pos:
                            gain += 1
                            break
                            
                # 2. Break satisfied clauses
                # Check all clauses to see if any are satisfied ONLY by this literal
                for s_idx, s_c in enumerate(pre_clauses):
                    # If this clause is currently satisfied
                    sat_count = 0
                    for s_var, s_is_pos in s_c:
                        if assignment[s_var] == (s_is_pos == 1):
                            sat_count += 1
                    if sat_count == 1:
                        # Check if the satisfying literal is the one we are flipping
                        for s_var, s_is_pos in s_c:
                            if assignment[s_var] == (s_is_pos == 1):
                                if s_var == var and s_is_pos == is_pos:
                                    gain -= 1
                                break
                            
                # Add weight bias
                # Weight of the literal we are flipping (currently False, so we look at weight of True state? 
                # No, weights usually accumulate for the literal that is "wanted".
                # Let's just use the gain.
                
                if gain > best_gain:
                    best_gain = gain
                    candidates = [(var, is_pos)]
                elif gain == best_gain:
                    candidates.append((var, is_pos))
            
            # Choose move
            if random.random() < 0.6 and candidates:
                var, is_pos = random.choice(candidates)
            else:
                var, is_pos = random.choice(c)
                
            # Flip
            assignment[var] = not assignment[var]
            
            # Update weights
            # Increase weight of the literal we just made True? 
            # Or increase weight of literals in the unsatisfied clause?
            # Standard WalkSAT doesn't use weights. Minisat does.
            # Let's increase weight of the literal that was in the unsatisfied clause.
            # The literal is (var, is_pos). Since it was unsatisfied, it was False.
            # So we are setting it to True.
            # We want to remember that "True" for this var is good?
            weights[2*var + (1 if is_pos else 0)] += 0.1
            # Decay other weights slightly?
            # weights *= 0.99 (too slow for numpy vector op every step? N=50 is small)
            # Let's skip decay for now, or do it every 100 steps.
            
            # Recalculate unsat count (expensive but N is small)
            # Optimization: Only recalculate affected clauses.
            # For simplicity, recalculate all.
            current_unsat = count_unsat(assignment)
            
        return assignment, current_unsat

    # Main Loop
    deadline = time.time() + 280
    best_assignment = None
    best_unsat = float('inf')
    
    # Try spectral init first
    assignment = spectral_init()
    assignment, unsat = weighted_walksat(assignment, 10000, deadline)
    if unsat == 0:
        return assignment.tolist()
    if unsat < best_unsat:
        best_unsat = unsat
        best_assignment = assignment.copy()
        
    # Try random inits
    for i in range(50):
        if time.time() > deadline:
            break
        assignment = np.random.rand(n) < 0.5
        assignment, unsat = weighted_walksat(assignment, 10000, deadline)
        if unsat == 0:
            return assignment.tolist()
        if unsat < best_unsat:
            best_unsat = unsat
            best_assignment = assignment.copy()
            
    if best_assignment is not None:
        return best_assignment.tolist()
    return [False] * n
