import itertools
import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

def construct():
    n = 16
    all_edges = list(itertools.combinations(range(n), 2))
    m_total = len(all_edges)
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    
    # 1. Precompute Q3 patterns on K8 (vertex labels 0..7)
    q3_edges_local = []
    for i in range(8):
        for j in range(i+1, 8):
            xor = i ^ j
            if xor & (xor - 1) == 0 and xor != 0:
                q3_edges_local.append((i, j))
                
    q3_patterns_set = set()
    for perm in itertools.permutations(range(8)):
        edges = []
        for u, v in q3_edges_local:
            pu, pv = perm[u], perm[v]
            if pu > pv:
                pu, pv = pv, pu
            edges.append((pu, pv))
        edges.sort()
        q3_patterns_set.add(tuple(edges))
    q3_patterns = list(q3_patterns_set)
    
    # 2. Precompute all 8-subsets and their corresponding Q3 edge index sets
    all_8_subsets = list(itertools.combinations(range(n), 8))
    
    # Map: subset_index -> list of 840 frozensets of global edge indices
    subset_q3_constraints = []
    for subset in all_8_subsets:
        patterns_for_subset = []
        for pattern in q3_patterns:
            edge_indices = []
            for u, v in pattern:
                gu = subset[u]
                gv = subset[v]
                if gu > gv:
                    gu, gv = gv, gu
                edge_indices.append(edge_to_idx[(gu, gv)])
            patterns_for_subset.append(frozenset(edge_indices))
        subset_q3_constraints.append(patterns_for_subset)
        
    # 3. Setup ILP
    # Start with a small random set of constraints
    random.seed(42)
    active_constraint_indices = set() # indices into a global list of all possible constraints?
    # It's easier to just store the frozensets directly in active_constraints
    active_constraints = []
    seen_constraints = set()
    
    def add_random_constraints(count):
        for _ in range(count):
            sub_idx = random.randint(0, len(all_8_subsets) - 1)
            pat_idx = random.randint(0, len(q3_patterns) - 1)
            # Get the frozenset
            fs = subset_q3_constraints[sub_idx][pat_idx]
            if fs not in seen_constraints:
                seen_constraints.add(fs)
                active_constraints.append(fs)

    add_random_constraints(2000)
    
    best_solution = None
    start_time = time.time()
    timeout = 115
    
    c = np.ones(m_total)
    integrality = np.ones(m_total)
    bounds = Bounds(lb=0, ub=1)
    
    while time.time() - start_time < timeout:
        # Build A
        if not active_constraints:
            break
        A = lil_matrix((len(active_constraints), m_total), dtype=np.float64)
        b = np.ones(len(active_constraints))
        
        for i, fs in enumerate(active_constraints):
            for e_idx in fs:
                A[i, e_idx] = 1.0
        A = A.tocsr()
        
        constraint = LinearConstraint(A, ub=b)
        
        try:
            res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
            if not res.success:
                # If failed, maybe reduce constraints?
                # For now, just break or continue
                if len(active_constraints) > 100:
                    active_constraints.pop()
                    continue
                else:
                    break
            else:
                x = res.x
                edge_set_indices = set()
                for i in range(m_total):
                    if x[i] > 0.5:
                        edge_set_indices.add(i)
                        
                # Check for Q3
                violation_fs = None
                for sub_idx, patterns in enumerate(subset_q3_constraints):
                    for fs in patterns:
                        if fs.issubset(edge_set_indices):
                            violation_fs = fs
                            break
                    if violation_fs:
                        break
                
                if violation_fs:
                    if violation_fs not in seen_constraints:
                        seen_constraints.add(violation_fs)
                        active_constraints.append(violation_fs)
                    else:
                        # Should not happen
                        pass
                else:
                    # Valid
                    best_solution = [all_edges[i] for i in edge_set_indices]
                    break
        except Exception:
            break
            
    if best_solution:
        return best_solution
    else:
        return [[a, b] for a in range(3) for b in range(3, 16)]
