import time
from math import gcd
import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds
from fractions import Fraction
from sympy import isprime

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin against 60s limit
    
    N = 256
    all_nodes = list(range(2, N + 1))
    n_total = len(all_nodes)
    node_to_idx = {v: i for i, v in enumerate(all_nodes)}
    
    # Precompute conflict masks: conflict if lcm(a,b) <= 256
    # lcm(a,b) = a*b/gcd(a,b) <= 256  <=>  a*b <= 256*gcd(a,b)
    conflict_mask = [0] * n_total
    for i in range(n_total):
        a = all_nodes[i]
        mask = 0
        for j in range(n_total):
            if i == j:
                continue
            b = all_nodes[j]
            if a * b <= 256 * gcd(a, b):
                mask |= (1 << j)
        conflict_mask[i] = mask
    
    # Number-theoretic reduction: fix primes p where 2*p > 256 (i.e., p > 128)
    # These primes conflict with all their multiples k*p (k>=2), but since 2*p>256,
    # the only conflicts are with multiples that don't exist in [2,256].
    # Actually, for p > 128, the only multiple in range is p itself, so no conflicts.
    # For primes p where 2*p <= 256 but 3*p > 256 (i.e., 86 < p <= 128),
    # p conflicts with 2p but not with other numbers.
    
    # Key insight: If we include a prime p, we cannot include any multiple kp (k>=2).
    # The reciprocal gain of including p is 1/p.
    # If we exclude p, we might be able to include some multiples kp.
    
    # Let's identify "forced" primes: primes p such that 1/p > sum(1/(kp) for all valid multiples)
    # This is complex, so let's use a simpler heuristic.
    
    # Strategy: For each prime p, compute the "opportunity cost" of including p
    # vs excluding it and including its multiples.
    
    # First, let's identify which numbers are multiples of which primes
    prime_multiples = {}  # p -> list of indices of multiples of p
    for i in range(n_total):
        v = all_nodes[i]
        if isprime(v):
            prime_multiples[v] = [i]
    
    # For each prime, find its multiples in [2, 256]
    for p in list(prime_multiples.keys()):
        for k in range(2, 257 // p + 1):
            m = p * k
            if 2 <= m <= 256:
                idx = node_to_idx[m]
                if idx not in prime_multiples[p]:
                    prime_multiples[p].append(idx)
    
    # Determine which primes to fix as included
    # A prime p should be included if 1/p > sum(1/(p*k) for k in valid multiples that could be included)
    # This is an approximation; let's just include all primes > 128 (no conflicts)
    # and decide on others via local search.
    
    # Fixed included: primes p where 2*p > 256 (p > 128)
    fixed_included = set()
    fixed_excluded = set()
    
    for p in range(2, 257):
        if isprime(p) and 2 * p > 256:
            fixed_included.add(p)
    
    # For primes p where 2*p <= 256, we'll let the solver decide
    # But we can pre-exclude multiples of fixed_included primes
    for p in fixed_included:
        for k in range(2, 257 // p + 1):
            m = p * k
            if 2 <= m <= 256:
                fixed_excluded.add(m)
    
    # Build reduced problem: only consider nodes not in fixed_excluded
    # and mark fixed_included as always selected
    active_nodes = []
    active_indices = []
    for i in range(n_total):
        v = all_nodes[i]
        if v in fixed_excluded:
            continue
        if v in fixed_included:
            continue  # Already fixed, don't need to decide
        active_nodes.append(v)
        active_indices.append(i)
    
    n_active = len(active_nodes)
    
    if n_active == 0:
        result = list(fixed_included)
        return result if result else [256]
    
    # Build conflict masks for active nodes
    active_conflict = [0] * n_active
    for i in range(n_active):
        orig_i = active_indices[i]
        mask = 0
        for j in range(n_active):
            if i == j:
                continue
            orig_j = active_indices[j]
            if (conflict_mask[orig_i] >> orig_j) & 1:
                mask |= (1 << j)
        active_conflict[i] = mask
    
    # Also check conflicts with fixed_included
    fixed_mask = 0
    for v in fixed_included:
        orig_idx = node_to_idx[v]
        for i in range(n_active):
            orig_i = active_indices[i]
            if (conflict_mask[orig_i] >> orig_idx) & 1:
                fixed_mask |= (1 << i)
    
    # Exclude active nodes that conflict with fixed_included
    valid_active = []
    valid_indices = []
    for i in range(n_active):
        if (fixed_mask >> i) & 1:
            continue
        valid_active.append(active_nodes[i])
        valid_indices.append(active_indices[i])
    
    n_valid = len(valid_active)
    
    if n_valid == 0:
        result = list(fixed_included)
        return result if result else [256]
    
    # Rebuild conflict masks for valid active nodes
    valid_conflict = [0] * n_valid
    for i in range(n_valid):
        orig_i = valid_indices[i]
        mask = 0
        for j in range(n_valid):
            if i == j:
                continue
            orig_j = valid_indices[j]
            if (conflict_mask[orig_i] >> orig_j) & 1:
                mask |= (1 << j)
        valid_conflict[i] = mask
    
    # Conflict counts for valid nodes
    conflict_count = [bin(valid_conflict[i]).count('1') for i in range(n_valid)]
    
    # Multiple greedy orderings
    scores = [(1.0 / valid_active[i]) / (1 + conflict_count[i]) for i in range(n_valid)]
    order1 = sorted(range(n_valid), key=lambda i: scores[i], reverse=True)
    order2 = sorted(range(n_valid), key=lambda i: 1.0 / valid_active[i], reverse=True)
    order3 = sorted(range(n_valid), key=lambda i: conflict_count[i])
    scores4 = [(1.0 / valid_active[i]) / (1 + conflict_count[i])**0.5 for i in range(n_valid)]
    order4 = sorted(range(n_valid), key=lambda i: scores4[i], reverse=True)
    
    best_heuristic_subset = []
    best_heuristic_value = 0.0
    
    for order in [order1, order2, order3, order4]:
        sel = []
        used = 0
        for idx in order:
            if used & valid_conflict[idx]:
                continue
            sel.append(valid_active[idx])
            used |= (1 << idx)
        val = sum(1.0 / v for v in sel)
        if val > best_heuristic_value:
            best_heuristic_value = val
            best_heuristic_subset = sel[:]
    
    # Local search on reduced problem
    def local_search(selected, deadline):
        selected = selected[:]
        used = 0
        for v in selected:
            if v in valid_active:
                idx = valid_active.index(v)
                used |= (1 << idx)
        
        improved = True
        while improved and (time.time() - start_time) < deadline:
            improved = False
            
            # 1. Try adding nodes
            for i in range(n_valid):
                if (used >> i) & 1:
                    continue
                if used & valid_conflict[i]:
                    continue
                selected.append(valid_active[i])
                used |= (1 << i)
                improved = True
            
            # 2. Try 1-swap
            non_selected = [i for i in range(n_valid) if not ((used >> i) & 1)]
            for s_idx, s in enumerate(selected):
                if s not in valid_active:
                    continue
                s_i = valid_active.index(s)
                used_without_s = used & ~(1 << s_i)
                best_n = None
                best_gain = 0
                for n_i in non_selected:
                    n_val = valid_active[n_i]
                    if n_val >= s:
                        continue
                    if used_without_s & valid_conflict[n_i]:
                        continue
                    gain = 1.0 / n_val - 1.0 / s
                    if gain > best_gain:
                        best_gain = gain
                        best_n = n_i
                if best_n is not None:
                    selected[s_idx] = valid_active[best_n]
                    used = used_without_s | (1 << best_n)
                    improved = True
                    non_selected = [i for i in range(n_valid) if not ((used >> i) & 1)]
                    break
        
        return selected
    
    # Run local search from heuristic
    best_subset = local_search(best_heuristic_subset, deadline - 15)
    best_value = sum(1.0 / v for v in best_subset)
    
    # Build MILP on reduced problem
    n_vars = n_valid
    c = -np.array([1.0 / x for x in valid_active])
    
    row_indices = []
    col_indices = []
    data = []
    constraint_idx = 0
    
    for i in range(n_vars):
        mask = valid_conflict[i]
        j = i + 1
        while j < n_vars:
            if (mask >> j) & 1:
                row_indices.append(constraint_idx)
                col_indices.append(i)
                data.append(1)
                row_indices.append(constraint_idx)
                col_indices.append(j)
                data.append(1)
                constraint_idx += 1
            j += 1
    
    num_constraints = constraint_idx
    
    if num_constraints > 0:
        A = sp.csr_matrix((data, (row_indices, col_indices)), shape=(num_constraints, n_vars))
        lb = np.zeros(num_constraints)
        ub = np.ones(num_constraints)
        constraint = LinearConstraint(A, lb, ub)
        bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
        integrality = np.ones(n_vars)
        
        x0 = np.zeros(n_vars)
        for v in best_subset:
            if v in valid_active:
                x0[valid_active.index(v)] = 1.0
        
        remaining = deadline - (time.time() - start_time) - 5
        if remaining > 1:
            try:
                res = milp(c=c, constraints=[constraint], integrality=integrality, bounds=bounds,
                           options={"time_limit": max(1, remaining), "mip_gap": 0.0})
                
                if res.x is not None:
                    milp_subset = [valid_active[i] for i in range(n_vars) if res.x[i] > 0.5]
                    if milp_subset:
                        milp_value = sum(1.0 / v for v in milp_subset)
                        if milp_value > best_value:
                            best_subset = milp_subset
                            best_value = milp_value
            except Exception:
                pass
    
    # Final polish: local search on the best solution
    if (time.time() - start_time) < deadline - 3:
        polished = local_search(best_subset, deadline - 2)
        val = sum(1.0 / v for v in polished)
        if val > best_value:
            best_subset = polished
            best_value = val
    
    # Combine with fixed_included
    final_subset = list(set(best_subset) | fixed_included)
    
    if not final_subset:
        return [256]
    
    return final_subset
