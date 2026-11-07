import time
import sys
from random import Random
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np

def construct():
    """
    Solve the minimum clique partition problem for three fixed split graphs.
    
    Strategy:
    1. Generate the split graph.
    2. Formulate an ILP to maximize the number of size-3 cliques {v,u,w} 
       (which cover 2 cross edges and 1 internal edge) while ensuring 
       the internal edges are covered.
    3. If full internal coverage is not possible or not optimal, 
       cover remaining internal edges with one large clique.
    4. Cover remaining cross edges with size-2 cliques.
    """
    
    deadline = time.time() + 145.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        
        # Build adjacency: for each v in I, list of K-neighbors
        nbrs_of_I = {}
        for v in range(k, n):
            nbrs = []
            for u in range(k):
                if rng.random() < p:
                    nbrs.append(u)
            nbrs_of_I[v] = nbrs
        
        # All internal edges in K
        all_internal = set()
        for u in range(k):
            for w in range(u + 1, k):
                all_internal.add((u, w))
        
        I_vertices = list(range(k, n))
        num_I = n - k
        num_K = k
        
        if num_K < 2:
            # No internal edges possible
            cliques = []
            for v in I_vertices:
                for u in nbrs_of_I[v]:
                    cliques.append([v, u])
            return cliques

        # Map internal edges to indices for ILP
        edge_list = list(all_internal)
        edge_to_idx = {e: i for i, e in enumerate(edge_list)}
        num_internal_edges = len(edge_list)
        
        # For each v in I, determine edges in G_v (subgraph of K induced by N(v))
        # and map them to global edge indices
        v_edge_indices = {}
        for v in I_vertices:
            nbrs = nbrs_of_I[v]
            if len(nbrs) < 2:
                continue
            indices = []
            for i in range(len(nbrs)):
                for j in range(i + 1, len(nbrs)):
                    u, w = nbrs[i], nbrs[j]
                    e = (u, w) if u < w else (w, u)
                    if e in edge_to_idx:
                        indices.append(edge_to_idx[e])
            v_edge_indices[v] = indices

        # Variables: x[v, e_idx] for each v in I and e in E(G_v)
        # We flatten these into a single vector
        var_map = [] # list of (v, e_idx)
        var_index = 0
        v_var_ranges = {}
        for v in I_vertices:
            if v not in v_edge_indices:
                continue
            indices = v_edge_indices[v]
            if not indices:
                continue
            start = var_index
            for e_idx in indices:
                var_map.append((v, e_idx))
                var_index += 1
            v_var_ranges[v] = (start, var_index)
        
        num_vars = var_index
        
        if num_vars == 0:
            # No triangles possible
            cliques = []
            for v in I_vertices:
                for u in nbrs_of_I[v]:
                    cliques.append([v, u])
            if all_internal:
                cliques.append(list(range(num_K)))
            return cliques

        # Objective: Maximize sum(x)
        # c = -1 for all vars (minimizing -sum is maximizing sum)
        c = -np.ones(num_vars)
        
        # Constraints:
        # 1. Matching constraints: For each v, for each u in N(v), sum of x[v, e] for e incident to u <= 1
        #    We need to build the matrix A for these constraints.
        #    Number of such constraints: sum_{v} |N(v)|
        
        # 2. Cover constraints (optional for first run): For each internal edge e, sum of x[v, e] for v containing e >= 1
        
        # We will solve two ILPs:
        # ILP 1: Maximize sum(x) subject to matching and cover constraints.
        # ILP 2: Maximize sum(x) subject to matching constraints only.
        
        def solve_ilp(require_cover):
            if time.time() > deadline - 5:
                return None, None
            
            # Build matching constraints matrix
            # Each row corresponds to a (v, u) pair where u in N(v)
            # A[row, col] = 1 if var col is x[v, e] and e is incident to u
            
            # Pre-compute for each v, for each u in N(v), which vars are incident
            # This is a bit expensive to build naively, but sizes are small.
            
            rows = []
            row_v_u = [] # (v, u) for each row
            
            for v in I_vertices:
                if v not in v_var_ranges:
                    continue
                nbrs = nbrs_of_I[v]
                # Map u to list of var indices incident to u in G_v
                u_to_vars = {u: [] for u in nbrs}
                start, end = v_var_ranges[v]
                indices = v_edge_indices[v]
                for k_idx, e_idx in enumerate(indices):
                    var_idx = start + k_idx
                    e = edge_list[e_idx]
                    u1, u2 = e
                    u_to_vars[u1].append(var_idx)
                    u_to_vars[u2].append(var_idx)
                
                for u in nbrs:
                    if u_to_vars[u]:
                        rows.append(u_to_vars[u])
                        row_v_u.append((v, u))
            
            num_match_constraints = len(rows)
            
            # Build A and b for matching constraints (<= 1)
            if num_match_constraints > 0:
                A_match = np.zeros((num_match_constraints, num_vars))
                for i, vars_list in enumerate(rows):
                    for var_idx in vars_list:
                        A_match[i, var_idx] = 1.0
                b_match_ub = np.ones(num_match_constraints)
                b_match_lb = np.zeros(num_match_constraints)
            else:
                A_match = None
                b_match_ub = None
                b_match_lb = None

            # Build cover constraints (>= 1) if required
            if require_cover:
                # For each internal edge e, sum x[v, e] >= 1
                # We need to know which vars correspond to e
                e_to_vars = {e_idx: [] for e_idx in range(num_internal_edges)}
                for var_idx, (v, e_idx) in enumerate(var_map):
                    e_to_vars[e_idx].append(var_idx)
                
                num_cover_constraints = num_internal_edges
                A_cover = np.zeros((num_cover_constraints, num_vars))
                for e_idx in range(num_internal_edges):
                    if e_to_vars[e_idx]:
                        for var_idx in e_to_vars[e_idx]:
                            A_cover[e_idx, var_idx] = 1.0
                b_cover_lb = np.ones(num_cover_constraints)
                b_cover_ub = np.full(num_cover_constraints, num_vars) # Max possible sum
            else:
                A_cover = None
                b_cover_lb = None
                b_cover_ub = None

            # Combine constraints
            constraints = []
            if A_match is not None:
                constraints.append(LinearConstraint(A_match, b_match_lb, b_match_ub))
            if A_cover is not None:
                constraints.append(LinearConstraint(A_cover, b_cover_lb, b_cover_ub))
            
            bounds = Bounds(0, 1)
            
            try:
                res = milp(c=c, constraints=constraints, bounds=bounds, integrality=np.ones(num_vars))
                if res.status == 0:
                    return res.x, res.fun
                else:
                    return None, None
            except Exception:
                return None, None

        # Solve ILP 1: With cover
        x1, obj1 = solve_ilp(True)
        
        # Solve ILP 2: Without cover
        x2, obj2 = solve_ilp(False)
        
        # Calculate costs
        # Cost = sum(d_v) - T + (1 if not full cover else 0)
        sum_d_v = sum(len(nbrs_of_I[v]) for v in I_vertices)
        
        best_x = None
        best_cost = float('inf')
        best_full_cover = False
        
        if x1 is not None:
            T1 = -obj1
            cost1 = sum_d_v - T1
            if cost1 < best_cost:
                best_cost = cost1
                best_x = x1
                best_full_cover = True
        elif x1 is None and x2 is None:
            # Fallback: no triangles
            pass

        if x2 is not None:
            T2 = -obj2
            cost2 = sum_d_v - T2 + 1 # Assume not full cover unless we verify
            # Check if x2 actually covers all edges? 
            # If x2 is the max T, it might or might not cover.
            # If it covers, cost is sum_d_v - T2.
            # Let's check coverage for x2
            covered_edges = set()
            for var_idx, (v, e_idx) in enumerate(var_map):
                if x2[var_idx] > 0.5:
                    covered_edges.add(e_idx)
            
            if covered_edges == set(range(num_internal_edges)):
                cost2 = sum_d_v - T2
            else:
                cost2 = sum_d_v - T2 + 1
                
            if cost2 < best_cost:
                best_cost = cost2
                best_x = x2
                best_full_cover = (covered_edges == set(range(num_internal_edges)))

        if best_x is None:
            # Fallback: no triangles at all
            cliques = []
            for v in I_vertices:
                for u in nbrs_of_I[v]:
                    cliques.append([v, u])
            if all_internal:
                cliques.append(list(range(num_K)))
            return cliques

        # Construct cliques from best_x
        cliques = []
        covered_cross = set()
        covered_internal = set()
        
        # Map var_idx to (v, e_idx)
        # We already have var_map
        # Group by v
        v_cliques = {v: [] for v in I_vertices}
        
        for var_idx, (v, e_idx) in enumerate(var_map):
            if best_x[var_idx] > 0.5:
                e = edge_list[e_idx]
                u, w = e
                v_cliques[v].append((u, w))
                covered_cross.add((v, u))
                covered_cross.add((v, w))
                covered_internal.add(e)
        
        # Add triangle cliques
        for v in I_vertices:
            for u, w in v_cliques[v]:
                cliques.append([v, u, w])
        
        # Add single edge cliques for uncovered cross edges
        for v in I_vertices:
            for u in nbrs_of_I[v]:
                if (v, u) not in covered_cross:
                    cliques.append([v, u])
        
        # Add big clique for remaining internal edges
        if not best_full_cover:
            remaining_verts = set()
            for e in all_internal:
                if e not in covered_internal:
                    remaining_verts.add(e[0])
                    remaining_verts.add(e[1])
            if len(remaining_verts) >= 2:
                cliques.append(list(remaining_verts))
        else:
            # Even if best_full_cover is True from ILP 1, we should double check?
            # ILP 1 enforced cover. So covered_internal should be all.
            # But what if x1 was None and we fell back to x2 which covered?
            # Then best_full_cover is set correctly.
            pass

        # Ensure all are Python int lists
        final_cliques = []
        for c in cliques:
            final_cliques.append([int(x) for x in c])
        
        return final_cliques
    
    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline - 2:
            # Emergency: return empty (invalid but avoids timeout)
            result.append([])
            continue
        result.append(instance(*spec))
    
    return result
