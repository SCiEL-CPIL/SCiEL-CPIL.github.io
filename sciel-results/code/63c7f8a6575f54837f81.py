import time
from math import gcd
import random
import sys
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    N = 128
    elements = list(range(1, N + 1))
    
    # 1. Precompute GCD matrix
    G = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            G[i][j] = gcd(i, j)
            G[j][i] = G[i][j]

    # 2. Identify all "bad triples" (a, b, c) with a < b < c such that
    #    gcd(a,b) == gcd(a,c) == gcd(b,c)
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = G[a][b]
            for c in range(b + 1, N + 1):
                if G[a][c] == g_ab and G[b][c] == g_ab:
                    bad_triples.append((a, b, c))
    
    T = len(bad_triples)
    
    # 3. Attempt to solve using MILP (scipy.optimize.milp)
    # Maximize sum(x_i)
    # Subject to: x_a + x_b + x_c <= 2 for all (a,b,c) in bad_triples
    # x_i in {0, 1}
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        # Objective: minimize -sum(x_i)
        c = -np.ones(N)
        
        # Constraints: A x <= b
        # We have T constraints. Each row has 3 non-zeros (1, 1, 1).
        # Using sparse matrix for efficiency.
        from scipy.sparse import lil_matrix
        A = lil_matrix((T, N), dtype=int)
        b = np.full(T, 2, dtype=int)
        
        for i, (a, b_val, c_val) in enumerate(bad_triples):
            A[i, a-1] = 1
            A[i, b_val-1] = 1
            A[i, c_val-1] = 1
            
        A = A.tocsr()
        
        # Bounds: 0 <= x_i <= 1
        bounds = Bounds(lb=0, ub=1)
        
        # Integrality: all variables are integers
        integrality = np.ones(N)
        
        # Solve
        # time_limit is in seconds. We have ~80s left.
        remaining_time = deadline - time.time()
        if remaining_time < 5:
            raise TimeoutError("Not enough time for MILP")
            
        res = milp(
            c=c,
            constraints=LinearConstraint(A, ub=2),
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": remaining_time - 2.0, "verbose": 0}
        )
        
        if res.status == 0 or res.status == 2: # 0=optimal, 2=feasible (if not optimal but found a solution)
            # res.x contains the solution
            # Note: res.x might be floats like 0.99999, so round them.
            x = np.round(res.x).astype(int)
            S = [i + 1 for i in range(N) if x[i] == 1]
            
            # Verify validity just in case
            if S:
                # Quick check
                S_set = set(S)
                valid = True
                # This check is O(T) which is fine
                for a, b, c in bad_triples:
                    if a in S_set and b in S_set and c in S_set:
                        valid = False
                        break
                if valid:
                    return S
            
    except Exception:
        pass # Fall through to heuristic if MILP fails or times out
        
    # 4. Fallback Heuristic: Greedy + Local Search (similar to current but tuned)
    # Since MILP might fail, we need a robust fallback.
    # Use the previous logic but optimized.
    
    # Precompute triples_by_elem for fallback
    triples_by_elem = [[] for _ in range(N)]
    for i, t in enumerate(bad_triples):
        for x in t:
            triples_by_elem[x - 1].append(i)
            
    # Greedy Hitting Set for R
    R = set()
    active_triples = set(range(T))
    active_count = [0] * N
    for t_idx in active_triples:
        for x in bad_triples[t_idx]:
            active_count[x - 1] += 1
            
    while active_triples:
        if time.time() > deadline - 10.0:
            break
            
        best_elem = -1
        max_count = -1
        for x in range(N):
            if x not in R:
                if active_count[x] > max_count:
                    max_count = active_count[x]
                    best_elem = x
                    
        if best_elem == -1:
            break
            
        R.add(best_elem)
        
        triples_to_remove = []
        for t_idx in triples_by_elem[best_elem]:
            if t_idx in active_triples:
                triples_to_remove.append(t_idx)
                
        for t_idx in triples_to_remove:
            active_triples.remove(t_idx)
            for x in bad_triples[t_idx]:
                if x - 1 != best_elem:
                    active_count[x - 1] -= 1
                    
    # Local Search on R (Minimize |R|)
    # We use a simple swap-based local search that maintains validity.
    # State: R (set of removed indices)
    # We try to remove an element from R. If it causes a violation (uncovered triple), 
    # we must add another element to cover it. This is a "move" that keeps validity.
    
    current_R = set(R)
    
    # Helper to check if removing x keeps validity
    # If we remove x, any triple containing x that has NO other element in R becomes uncovered.
    # We need to add at least one element from each such uncovered triple.
    
    def get_uncovered_triples_if_remove(x, R_set):
        """Returns list of triples that become uncovered if x is removed from R_set"""
        uncovered = []
        for t_idx in triples_by_elem[x]:
            a, b, c = bad_triples[t_idx]
            others = [i-1 for i in (a,b,c) if i-1 != x]
            if others[0] not in R_set and others[1] not in R_set:
                uncovered.append(t_idx)
        return uncovered

    # Local Search Loop
    # We try to swap x in R with y not in R, or just remove x if possible.
    # Actually, just removing x is only possible if no triple becomes uncovered.
    # If a triple becomes uncovered, we must add one of its elements.
    # This is complex. Let's use a simpler "Random Restart + Greedy" if time permits.
    
    # Given time constraints, if MILP failed, the greedy result is likely decent.
    # Let's do a few random restarts of the greedy if time allows.
    
    best_S = [i + 1 for i in range(N) if i not in current_R]
    
    # Try to improve with simple hill climbing on R size
    # If we can remove an element from R without creating a violation, do it.
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for x in list(current_R):
            if time.time() > deadline - 5.0:
                break
            unc = get_uncovered_triples_if_remove(x, current_R)
            if not unc:
                current_R.remove(x)
                improved = True
                break # Restart loop after modification
        
    S = [i + 1 for i in range(N) if i not in current_R]
    
    if not S:
        return [1]
        
    return S

# EVOLVE-BLOCK-END
