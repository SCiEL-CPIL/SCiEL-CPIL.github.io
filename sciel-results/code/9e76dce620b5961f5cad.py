import time
import random
import numpy as np
from itertools import product, combinations
from scipy.optimize import linprog

DIMENSIONS = [3, 5, 7]

def construct():
    """
    Find rank-minimal exact tensor decomposition for 3x5 * 5x7 matrix multiplication.
    
    Strategy:
    - Enumerate all useful "pattern" terms (structured ways to cover index subsets at fixed rank cost)
    - Use integer programming to select the minimum-rank combination of patterns
      that covers all 105 tensor indices exactly once
    - Patterns include: Strassen 2x2x2 (cost 7), various rectangular sub-blocks, single terms
    """
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS  # 3, 5, 7
    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21
    total_terms = n * m * p  # 105

    # Each tensor index (i, j, k) must be covered exactly once
    # A "pattern" is defined by:
    #   - the set of (i,j,k) indices it covers
    #   - the rank cost (number of scalar multiplications)
    #   - the U, V, W entries for each rank component

    # Generate patterns:
    # 1. Single term: covers {(i,j,k)}, cost 1
    # 2. 2x2x2 Strassen: covers 8 indices, cost 7
    # 3. 2x1x1, 1x2x1, 1x1x2: trivial, cost = product of dims (no saving)
    # 4. 2x2x1: covers 4 indices, cost 4 (no saving over schoolbook)
    # Actually, only Strassen-type decompositions save rank.
    # Let's also consider 3x3x3 if n,m,p >= 3, but n=3 so only one 3x3x3 possible if m,p>=3
    # For 3x3x3: rank 23 (Bini et al.) vs schoolbook 27, saves 4
    
    # Let's focus on 2x2x2 Strassen blocks and single terms, plus check for 3x3x3
    
    patterns = []  # list of (indices_set, cost, decomposition)
    
    # Add all single-term patterns
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = (i, j, k)
                u = {i * m + j: 1}
                v = {j * p + k: 1}
                w = {k * n + i: 1}
                patterns.append((frozenset([idx]), 1, [([u], [v], [w])]))
    
    # Add all 2x2x2 Strassen patterns
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            indices = frozenset(
                                (i, j, k)
                                for i in (i1, i2)
                                for j in (j1, j2)
                                for k in (k1, k2)
                            )
                            # Strassen decomposition
                            a11 = i1 * m + j1
                            a12 = i1 * m + j2
                            a21 = i2 * m + j1
                            a22 = i2 * m + j2
                            b11 = j1 * p + k1
                            b12 = j1 * p + k2
                            b21 = j2 * p + k1
                            b22 = j2 * p + k2
                            c11 = k1 * n + i1
                            c12 = k2 * n + i1
                            c21 = k1 * n + i2
                            c22 = k2 * n + i2
                            
                            decomp = [
                                ([{a11: 1, a22: 1}], [{b11: 1, b22: 1}], [{c11: 1, c22: 1}]),
                                ([{a21: 1, a22: 1}], [{b11: 1}], [{c21: 1, c22: -1}]),
                                ([{a11: 1}], [{b12: 1, b22: -1}], [{c12: 1, c22: 1}]),
                                ([{a22: 1}], [{b21: 1, b11: -1}], [{c11: 1, c21: 1}]),
                                ([{a11: 1, a12: 1}], [{b22: 1}], [{c11: -1, c12: 1}]),
                                ([{a21: 1, a11: -1}], [{b11: 1, b12: 1}], [{c22: 1}]),
                                ([{a12: 1, a22: -1}], [{b21: 1, b22: 1}], [{c11: 1}]),
                            ]
                            patterns.append((indices, 7, decomp))
    
    # Add 3x3x3 pattern if applicable (n=3, so i1=0,i2=2; need m>=3, p>=3)
    if n >= 3 and m >= 3 and p >= 3:
        i1, i2 = 0, 2
        for j1 in range(m - 2):
            for j2 in range(j1 + 2, m):
                for k1 in range(p - 2):
                    for k2 in range(k1 + 2, p):
                        indices = frozenset(
                            (i, j, k)
                            for i in range(3)
                            for j in range(j1, j2 + 1)
                            for k in range(k1, k2 + 1)
                        )
                        # Use known rank-23 decomposition for 3x3x3
                        # This is complex; let's skip for now and rely on 2x2x2 + singles
                        # We can add this later if needed
                        pass
    
    # Now we have a set covering problem:
    # Minimize sum(cost_p * x_p) subject to: each index covered exactly once
    # x_p in {0, 1}
    
    num_patterns = len(patterns)
    num_indices = total_terms
    
    # Index mapping
    index_to_id = {}
    for idx in range(n):
        for jdx in range(m):
            for kdx in range(p):
                index_to_id[(idx, jdx, kdx)] = len(index_to_id)
    
    # Build constraint matrix: A[i][p] = 1 if pattern p covers index i
    # We want A @ x = 1 (each index covered exactly once)
    # Minimize c @ x
    
    # For ILP, use scipy's linprog with integer bounds or use a different approach
    # Since the problem is small enough, let's try a greedy + local search approach
    
    # Greedy: repeatedly pick the pattern with best cost-per-new-index ratio
    best_solution = None
    best_cost = total_terms  # schoolbook baseline
    
    # Try multiple greedy strategies
    for trial in range(200):
        if time.time() > deadline - 30:
            break
        
        remaining = set(range(num_indices))
        selected = []
        current_cost = 0
        
        # Shuffle patterns for diversity
        pattern_order = list(range(num_patterns))
        random.shuffle(pattern_order)
        
        # Greedy with priority: prefer Strassen blocks that cover many remaining indices
        while remaining:
            best_pat = None
            best_ratio = float('inf')
            best_covered = 0
            
            for p_idx in pattern_order:
                if time.time() > deadline - 30:
                    break
                pat_indices, cost, decomp = patterns[p_idx]
                # Convert to index IDs
                pat_ids = frozenset(index_to_id[idx] for idx in pat_indices)
                covered = len(pat_ids & remaining)
                if covered == 0:
                    continue
                # Ratio: cost per newly covered index
                ratio = cost / covered
                if ratio < best_ratio or (ratio == best_ratio and covered > best_covered):
                    best_ratio = ratio
                    best_pat = p_idx
                    best_covered = covered
            
            if best_pat is None:
                # Shouldn't happen since single terms always available
                break
            
            pat_indices, cost, decomp = patterns[best_pat]
            pat_ids = frozenset(index_to_id[idx] for idx in pat_indices)
            selected.append(best_pat)
            remaining -= pat_ids
            current_cost += cost
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_solution = selected[:]
            if best_cost <= 84:  # 12 Strassen blocks * 7 = 84, plus some singles
                break
    
    # Local search: try swapping one pattern for another to reduce cost
    if best_solution is not None:
        improved = True
        while improved and time.time() < deadline - 20:
            improved = False
            # Compute current coverage
            covered_indices = set()
            for p_idx in best_solution:
                pat_indices, _, _ = patterns[p_idx]
                for idx in pat_indices:
                    covered_indices.add(index_to_id[idx])
            
            # Try removing one pattern and re-covering
            for remove_idx in range(len(best_solution)):
                if time.time() > deadline - 20:
                    break
                p_remove = best_solution[remove_idx]
                pat_remove_indices, cost_remove, _ = patterns[p_remove]
                remove_ids = frozenset(index_to_id[idx] for idx in pat_remove_indices)
                
                # Indices that become uncovered
                new_uncovered = set()
                for idx_id in remove_ids:
                    # Check if covered by another selected pattern
                    covered_by_other = False
                    for p_idx in best_solution:
                        if p_idx == p_remove:
                            continue
                        pat_indices, _, _ = patterns[p_idx]
                        if idx_id in frozenset(index_to_id[idx] for idx in pat_indices):
                            covered_by_other = True
                            break
                    if not covered_by_other:
                        new_uncovered.add(idx_id)
                
                if not new_uncovered:
                    # Removing this pattern doesn't uncover anything (redundant)
                    new_solution = best_solution[:remove_idx] + best_solution[remove_idx + 1:]
                    new_cost = best_cost - cost_remove
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_solution = new_solution
                        improved = True
                        break
                else:
                    # Try to cover new_uncovered with one or two patterns
                    # First try single pattern
                    for p_add in range(num_patterns):
                        if p_add == p_remove:
                            continue
                        pat_add_indices, cost_add, _ = patterns[p_add]
                        add_ids = frozenset(index_to_id[idx] for idx in pat_add_indices)
                        if new_uncovered <= add_ids:
                            # Check no conflicts (add_ids shouldn't overlap with other covered)
                            conflict = False
                            other_covered = set()
                            for p_idx in best_solution:
                                if p_idx == p_remove:
                                    continue
                                pat_indices, _, _ = patterns[p_idx]
                                other_covered |= frozenset(index_to_id[idx] for idx in pat_indices)
                            if add_ids & other_covered:
                                conflict = True
                            if not conflict:
                                new_solution = best_solution[:remove_idx] + [p_add] + best_solution[remove_idx + 1:]
                                new_cost = best_cost - cost_remove + cost_add
                                if new_cost < best_cost:
                                    best_cost = new_cost
                                    best_solution = new_solution
                                    improved = True
                                    break
                    if improved:
                        break
    
    # Fallback: if no solution found, use schoolbook
    if best_solution is None:
        best_solution = [i for i in range(num_patterns) if patterns[i][1] == 1]  # single terms only
        best_cost = total_terms
    
    # Build U, V, W from selected patterns
    U_cols = []
    V_cols = []
    W_cols = []
    
    for p_idx in best_solution:
        pat_indices, cost, decomp = patterns[p_idx]
        for u_dict, v_dict, w_dict in decomp:
            U_cols.append([0] * U_rows)
            V_cols.append([0] * V_rows)
            W_cols.append([0] * W_rows)
            for idx, val in u_dict.items():
                U_cols[-1][idx] = val
            for idx, val in v_dict.items():
                V_cols[-1][idx] = val
            for idx, val in w_dict.items():
                W_cols[-1][idx] = val
    
    rank = len(U_cols)
    
    # Convert to matrix format
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    for c in range(rank):
        for r in range(U_rows):
            U[r][c] = U_cols[c][r]
        for r in range(V_rows):
            V[r][c] = V_cols[c][r]
        for r in range(W_rows):
            W[r][c] = W_cols[c][r]
    
    return {"U": U, "V": V, "W": W}
