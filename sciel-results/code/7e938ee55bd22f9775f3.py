import time
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin for 60s limit
    
    N = 256
    max_sum = 2 * N
    
    # 1. Precompute squares
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 1):
        squares.add(i * i)
    
    # 2. Filter vertices: 2*v must not be a square (a=b case)
    valid_vertices = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_vertices.append(v)
            
    n_vars = len(valid_vertices)
    if n_vars == 0:
        return [1]
        
    # Map vertex value to index
    v_to_idx = {v: i for i, v in enumerate(valid_vertices)}
    
    # 3. Build adjacency list and edge list
    # Edge exists if vi + vj is a square
    edges = []
    neighbors = {v: set() for v in valid_vertices}
    for i in range(n_vars):
        vi = valid_vertices[i]
        for j in range(i + 1, n_vars):
            vj = valid_vertices[j]
            if vi + vj in squares:
                edges.append((i, j))
                neighbors[vi].add(vj)
                neighbors[vj].add(vi)
                
    n_edges = len(edges)
    
    # If no edges, all valid vertices can be chosen
    if n_edges == 0:
        return valid_vertices[:]
        
    # --- Strategy 1: Fast Heuristic (Greedy + Local Search) ---
    # This provides a strong baseline quickly.
    
    # Greedy: Sort by degree ascending
    vs_sorted = sorted(valid_vertices, key=lambda v: len(neighbors[v]))
    
    S_heuristic = set()
    for v in vs_sorted:
        if neighbors[v].isdisjoint(S_heuristic):
            S_heuristic.add(v)
            
    # Local Search: Try to improve by swapping
    # We can try to add a vertex v not in S, and remove conflicting vertices from S.
    # If |S| - |conflicts| + 1 > |S|, i.e., if |conflicts| < 1, we add.
    # If |conflicts| == 1, we swap (size unchanged, but might allow further additions later).
    # Let's do a simple hill climbing:
    # 1. Try adding any v not in S.
    # 2. If conflicts, remove all conflicts and add v.
    # 3. If new size > old size, accept.
    # 4. Repeat until no improvement.
    
    improved = True
    while improved:
        improved = False
        for v in valid_vertices:
            if v in S_heuristic:
                continue
            conflicts = neighbors[v] & S_heuristic
            # If we add v, we must remove conflicts.
            # New size = |S| - |conflicts| + 1
            # We accept if new size > |S|, i.e., |conflicts| < 1.
            # This only adds isolated vertices.
            # To allow swaps, we need a more sophisticated move.
            # Let's try: For each v not in S, for each c in conflicts:
            # Remove c, add v. Size unchanged.
            # Then check if we can add more.
            # This is complex. Let's stick to a simpler "kick" strategy or just rely on MILP.
            pass
    
    # Actually, a better local search:
    # Randomly remove a vertex, then greedily re-add vertices.
    # Or, try all single-vertex additions/deletions.
    
    # Given the small size, let's just use the Greedy result as a fallback.
    # The MILP is the primary engine.
    
    # --- Strategy 2: Exact MILP ---
    
    # Build A matrix (n_edges x n_vars)
    A = lil_matrix((n_edges, n_vars))
    for k, (i, j) in enumerate(edges):
        A[k, i] = 1
        A[k, j] = 1
        
    A = A.tocsr()
    
    # Objective: maximize sum(x) -> minimize -sum(x)
    c = -np.ones(n_vars)
    
    # Constraints:
    # 1. x_i + x_j <= 1 for all edges
    # 2. 0 <= x_i <= 1
    
    constraints = [
        LinearConstraint(A, ub=np.ones(n_edges)),
        LinearConstraint(np.eye(n_vars), lb=0, ub=1)
    ]
    
    integrality = np.ones(n_vars)
    bounds = Bounds(lb=0, ub=1)
    
    best_solution = list(S_heuristic)
    best_score = len(best_solution)
    
    try:
        # Check time before running MILP
        if time.time() > deadline - 5.0:
            raise TimeoutError("Not enough time for MILP")
            
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0:
            x = res.x
            x = np.round(x)
            
            selected_vertices = []
            for i in range(n_vars):
                if x[i] > 0.5:
                    selected_vertices.append(valid_vertices[i])
                    
            # Verify solution
            if _is_valid(selected_vertices, squares):
                if len(selected_vertices) > best_score:
                    best_solution = selected_vertices
                    best_score = len(best_solution)
        else:
            # MILP failed, rely on heuristic
            pass
            
    except Exception:
        # Fallback to heuristic if MILP fails or times out
        pass
        
    # Final verification
    if not _is_valid(best_solution, squares):
        # If something went wrong, return a single valid vertex
        for v in valid_vertices:
            if _is_valid([v], squares):
                return [v]
        return [valid_vertices[0]]
        
    return best_solution

def _is_valid(subset, squares):
    s = set(subset)
    for a in s:
        # Check a=a
        if 2 * a in squares:
            return False
        for b in s:
            if a < b:
                if a + b in squares:
                    return False
    return True
