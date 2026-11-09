<<<<<<< SEARCH
"""SAT solver for planted 3-SAT instances."""
# EVOLVE-BLOCK-START
import time
import random
import sys
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    start_time = time.time()
    deadline = start_time + 290.0  # 10s margin before 300s limit
    
    # Convert to 0-based internal representation
    clauses_internal = []
    for c in clauses:
        clause_int = []
        for lit in c:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            clause_int.append((var, sign))
        clauses_internal.append(clause_int)
    
    # Build variable-to-clause index
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses_internal):
        for var, sign in clause:
            var_to_clauses[var].append((ci, sign))
    
    # Precompute clause data as numpy arrays for efficient WalkSAT
    clause_vars = np.array([[v for v, s in c] for c in clauses_internal], dtype=np.int32)
    clause_signs = np.array([[s for v, s in c] for c in clauses_internal], dtype=np.int8)
=======
"""SAT solver for planted 3-SAT instances."""
# EVOLVE-BLOCK-START
import time
import random
import sys
import numpy as np
from collections import deque


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    start_time = time.time()
    deadline = start_time + 290.0  # 10s margin before 300s limit
    
    # Convert to 0-based internal representation
    clauses_internal = []
    for c in clauses:
        clause_int = []
        for lit in c:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            clause_int.append((var, sign))
        clauses_internal.append(clause_int)
    
    # Build variable-to-clause index
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses_internal):
        for var, sign in clause:
            var_to_clauses[var].append((ci, sign))
    
    # Build variable adjacency for connected components
    var_adj = [[] for _ in range(n)]
    for clause in clauses_internal:
        vars_in_clause = [v for v, s in clause]
        for i in range(len(vars_in_clause)):
            for j in range(i + 1, len(vars_in_clause)):
                u, w = vars_in_clause[i], vars_in_clause[j]
                var_adj[u].append(w)
                var_adj[w].append(u)
    
    # Find connected components
    visited = [False] * n
    components = []
    for start_var in range(n):
        if not visited[start_var]:
            comp = []
            queue = deque([start_var])
            visited[start_var] = True
            while queue:
                v = queue.popleft()
                comp.append(v)
                for neighbor in var_adj[v]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            components.append(comp)
    
    # Map each variable to its component index
    var_to_comp = [-1] * n
    for ci, comp in enumerate(components):
        for v in comp:
            var_to_comp[v] = ci
    
    # Group clauses by component
    comp_clauses = [[] for _ in components]
    for ci, clause in enumerate(clauses_internal):
        # All variables in a clause belong to the same component
        comp_idx = var_to_comp[clause[0][0]]
        comp_clauses[comp_idx].append(ci)
    
    # Precompute clause data as numpy arrays for efficient WalkSAT
    clause_vars = np.array([[v for v, s in c] for c in clauses_internal], dtype=np.int32)
    clause_signs = np.array([[s for v, s in c] for c in clauses_internal], dtype=np.int8)
>>>>>>> REPLACE

