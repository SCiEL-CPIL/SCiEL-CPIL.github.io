import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 10s safety margin
    
    # 1. Initialize Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    # Adjacency matrix
    A = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                A[i, j] = 1
                A[j, i] = 1
                
    # Helper to get neighbors
    def get_neighbors(i):
        return np.where(A[i] == 1)[0]
        
    # Find all K4s and I4s in the current graph
    def find_k4s(A, n):
        k4s = []
        for i in range(n):
            for j in range(i+1, n):
                if A[i,j]:
                    for k in range(j+1, n):
                        if A[i,k] and A[j,k]:
                            for l in range(k+1, n):
                                if A[i,l] and A[j,l] and A[k,l]:
                                    k4s.append((i,j,k,l))
        return k4s
        
    def find_i4s(A, n):
        i4s = []
        for i in range(n):
            for j in range(i+1, n):
                if not A[i,j]:
                    for k in range(j+1, n):
                        if not A[i,k] and not A[j,k]:
                            for l in range(k+1, n):
                                if not A[i,l] and not A[j,l] and not A[k,l]:
                                    i4s.append((i,j,k,l))
        return i4s
        
    k4s = find_k4s(A, n)
    i4s = find_i4s(A, n)
    
    best_n = n
    
    # Build constraint lists: for each vertex, which K4s/I4s it participates in
    # This helps with backtracking search for valid S
    
    def try_add_vertex(A, n, k4s, i4s, deadline):
        """Try to find a valid neighbor set S for a new vertex using constraint propagation."""
        # For each vertex v, track:
        # - k4_forbidden[v]: list of K4s containing v (if v in S, can't have all 4 in S)
        # - i4_forbidden[v]: list of I4s containing v (if v not in S, can't have all 4 not in S)
        
        # Use backtracking with constraint propagation
        # Assign values to vertices: 0 = not in S (in T), 1 = in S
        
        assignment = [-1] * n  # -1 = unassigned, 0 = T, 1 = S
        
        # Count constraints: for each K4, how many of its vertices are assigned to S
        # For each I4, how many of its vertices are assigned to T
        
        def is_valid_partial(assignment):
            """Check if partial assignment is still extendable."""
            # Check K4s: if 3 vertices are in S and the 4th is unassigned, the 4th must be in T
            # If all 4 are in S, invalid
            for k4 in k4s:
                s_count = 0
                unassigned_count = 0
                for v in k4:
                    if assignment[v] == 1:
                        s_count += 1
                    elif assignment[v] == -1:
                        unassigned_count += 1
                if s_count == 4:
                    return False
                if s_count == 3 and unassigned_count == 0:
                    # All 4 assigned, 3 in S, 1 in T - this is fine
                    pass
            
            # Check I4s: if 3 vertices are in T and the 4th is unassigned, the 4th must be in S
            for i4 in i4s:
                t_count = 0
                unassigned_count = 0
                for v in i4:
                    if assignment[v] == 0:
                        t_count += 1
                    elif assignment[v] == -1:
                        unassigned_count += 1
                if t_count == 4:
                    return False
                if t_count == 3 and unassigned_count == 0:
                    # All 4 assigned, 3 in T, 1 in S - this is fine
                    pass
            
            return True
        
        def backtrack(idx):
            """Backtracking search for valid assignment."""
            if time.time() > deadline - 2:
                return None
            
            if idx == n:
                return assignment[:]
            
            # Try assigning 1 (in S) first, then 0 (in T)
            # Use heuristic: try value that violates fewer constraints
            for val in [1, 0]:
                assignment[idx] = val
                if is_valid_partial(assignment):
                    result = backtrack(idx + 1)
                    if result is not None:
                        return result
                assignment[idx] = -1
            
            return None
        
        # Simple backtracking may be too slow for large n
        # Instead, use a randomized approach with constraint checking
        
        # Randomized local search: start with random assignment, fix violations
        for attempt in range(200):
            if time.time() > deadline - 2:
                break
            
            # Start with random assignment
            assignment = [random.choice([0, 1]) for _ in range(n)]
            
            # Iteratively fix violations
            improved = True
            iterations = 0
            while improved and iterations < 50:
                improved = False
                iterations += 1
                
                # Check K4 violations: K4 fully in S
                for k4 in k4s:
                    if all(assignment[v] == 1 for v in k4):
                        # Flip one of the vertices to 0
                        v = random.choice(k4)
                        assignment[v] = 0
                        improved = True
                        break
                
                if improved:
                    continue
                
                # Check I4 violations: I4 fully in T
                for i4 in i4s:
                    if all(assignment[v] == 0 for v in i4):
                        # Flip one of the vertices to 1
                        v = random.choice(i4)
                        assignment[v] = 1
                        improved = True
                        break
            
            # Check if final assignment is valid
            valid = True
            for k4 in k4s:
                if all(assignment[v] == 1 for v in k4):
                    valid = False
                    break
            if valid:
                for i4 in i4s:
                    if all(assignment[v] == 0 for v in i4):
                        valid = False
                        break
            
            if valid:
                return assignment
        
        return None
    
    # Try to add vertices
    while time.time() < deadline:
        # Try to add a new vertex
        added = False
        
        S_mask = try_add_vertex(A, n, k4s, i4s, deadline)
        
        if S_mask is not None:
            # If valid, add the vertex
            new_n = n + 1
            new_A = np.zeros((new_n, new_n), dtype=np.int8)
            new_A[:n, :n] = A
            new_A[:n, n] = S_mask
            new_A[n, :n] = S_mask
            
            A = new_A
            n = new_n
            best_n = n
            added = True
            
            # Update K4s and I4s
            # New K4s must include the new vertex n-1
            new_k4s = k4s.copy()
            new_i4s = i4s.copy()
            
            # Find new K4s involving n-1
            # A new K4 is {n-1, i, j, k} where i,j,k are in S and form a K3 in G[S]
            S_vertices = [v for v in range(n-1) if S_mask[v] == 1]
            for i, j, k in combinations(S_vertices, 3):
                if A[i,j] and A[j,k] and A[i,k]:
                    new_k4s.append((i, j, k, n-1))
                    
            # Find new I4s involving n-1
            # A new I4 is {n-1, i, j, k} where i,j,k are in T and form an independent set in G[T]
            T_vertices = [v for v in range(n-1) if S_mask[v] == 0]
            for i, j, k in combinations(T_vertices, 3):
                if not A[i,j] and not A[j,k] and not A[i,k]:
                    new_i4s.append((i, j, k, n-1))
                    
            k4s = new_k4s
            i4s = new_i4s
        else:
            # If we can't add a vertex, try a few random edge flips to escape local optimum
            found_escape = False
            for _ in range(20):
                if time.time() > deadline - 5:
                    break
                i, j = random.sample(range(n), 2)
                old_ij = A[i,j]
                A[i,j] = 1 - A[i,j]
                A[j,i] = A[i,j]
                
                # Recompute K4s and I4s
                new_k4s = find_k4s(A, n)
                new_i4s = find_i4s(A, n)
                
                # Check if graph is still valid (no K5, no I5)
                valid = True
                # Check for K5: any 5-clique
                # This is expensive, so do a quick check
                # For now, trust that small changes won't create K5 easily
                # But we should verify
                
                # Quick check: if any K4 now has a 5th vertex connected to all 4
                for k4 in new_k4s:
                    a, b, c, d = k4
                    for v in range(n):
                        if v in k4:
                            continue
                        if A[a,v] and A[b,v] and A[c,v] and A[d,v]:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    # Check for I5
                    for i4 in new_i4s:
                        a, b, c, d = i4
                        for v in range(n):
                            if v in i4:
                                continue
                            if not A[a,v] and not A[b,v] and not A[c,v] and not A[d,v]:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    k4s = new_k4s
                    i4s = new_i4s
                    found_escape = True
                    break
                else:
                    # Revert
                    A[i,j] = old_ij
                    A[j,i] = old_ij
            
            if not found_escape:
                break # If we can't add and can't escape, stop
            
    # Convert to edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if A[i,j] == 1:
                edges.append((i,j))
                
    return n, edges
