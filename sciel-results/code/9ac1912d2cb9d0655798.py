import time
import math
from math import gcd, lcm
from fractions import Fraction
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import scipy.sparse as sp

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    numbers = list(range(2, 257))
    n = len(numbers)
    
    # Precompute conflict graph
    # conflict[i] is a set of indices j such that lcm(numbers[i], numbers[j]) <= 256
    conflict = [set() for _ in range(n)]
    for i in range(n):
        a = numbers[i]
        for j in range(i + 1, n):
            b = numbers[j]
            # lcm(a, b) = a * b // gcd(a, b)
            g = gcd(a, b)
            l = a * b // g
            if l <= 256:
                conflict[i].add(j)
                conflict[j].add(i)
    
    # Weights for objective function: 1/x
    weights = [1.0 / numbers[i] for i in range(n)]
    
    # MILP Formulation
    # Variables: x_i in {0, 1}
    # Maximize sum(weights[i] * x_i)
    # Constraints: x_i + x_j <= 1 for all (i, j) in conflicts
    #               x_i >= 0, x_i <= 1
    #               sum(x_i) >= 1 (non-empty, though likely satisfied)
    
    # Build constraint matrix A_ub such that A_ub @ x <= b_ub
    # Each conflict (i, j) gives a row [0...1...1...0] with 1 at i and j, b_ub = 1
    
    conflict_pairs = []
    for i in range(n):
        for j in conflict[i]:
            if i < j:
                conflict_pairs.append((i, j))
    
    m_constraints = len(conflict_pairs)
    
    if m_constraints == 0:
        # No conflicts, take all numbers with largest reciprocals (smallest numbers)
        # Actually, if no conflicts, we take all numbers.
        return numbers
    
    # Create sparse matrix for constraints
    # We need to map variable index to column index
    # Variables are x_0 to x_{n-1} corresponding to numbers[0] to numbers[n-1]
    
    rows = []
    cols = []
    vals = []
    
    for idx, (i, j) in enumerate(conflict_pairs):
        rows.extend([idx, idx])
        cols.extend([i, j])
        vals.extend([1, 1])
    
    A_ub = sp.csr_matrix((vals, (rows, cols)), shape=(m_constraints, n))
    b_ub = np.ones(m_constraints)
    
    # Bounds for variables: 0 <= x_i <= 1
    lb = np.zeros(n)
    ub = np.ones(n)
    
    # Objective: maximize c^T x  => minimize -c^T x
    c = np.array([-weights[i] for i in range(n)])
    
    # Integrality: all variables are integers (binary)
    integrality = np.ones(n)  # 1 means integer
    
    try:
        res = milp(
            c=c,
            constraints=LinearConstraint(A_ub, -np.inf, b_ub),
            integrality=integrality,
            bounds=Bounds(lb, ub)
        )
        
        if res.status == 0 and res.success:
            # Extract solution
            sol = res.x
            selected_indices = [i for i in range(n) if sol[i] > 0.5]
            
            # Verify non-empty and valid
            if selected_indices:
                # Double check validity just in case
                valid = True
                sel_set = set(selected_indices)
                for i in selected_indices:
                    if conflict[i].intersection(sel_set):
                        valid = False
                        break
                
                if valid:
                    result = [numbers[i] for i in selected_indices]
                    return result
            
        # If MILP failed or returned empty/invalid, fall through to fallback
        pass
    except Exception as e:
        pass # Fallback to greedy/local search
    
    # Fallback: Greedy + Local Search
    # Sort by reciprocal descending (number ascending)
    order = sorted(range(n), key=lambda i: numbers[i])
    
    selected_indices = []
    selected_set = set()
    
    for i in order:
        if not conflict[i].intersection(selected_set):
            selected_indices.append(i)
            selected_set.add(i)
    
    current_sum = sum(Fraction(1, numbers[i]) for i in selected_indices)
    
    # Simple local search: try swaps
    improved = True
    while improved and (time.time() - start_time) < deadline - 5.0:
        improved = False
        # Try adding
        for i in order:
            if i not in selected_set:
                if not conflict[i].intersection(selected_set):
                    selected_set.add(i)
                    selected_indices.append(i)
                    current_sum += Fraction(1, numbers[i])
                    improved = True
                    break
        
        if improved:
            continue
        
        # Try swapping
        members = list(selected_set)
        for m in members:
            for i in order:
                if i in selected_set:
                    continue
                # Check if i is compatible with (selected_set - {m})
                temp_set = selected_set - {m}
                if not conflict[i].intersection(temp_set):
                    new_sum = current_sum - Fraction(1, numbers[m]) + Fraction(1, numbers[i])
                    if new_sum > current_sum:
                        selected_set.discard(m)
                        selected_set.add(i)
                        # Update selected_indices (remove m, add i)
                        selected_indices = [i2 for i2 in selected_indices if i2 != m]
                        selected_indices.append(i)
                        current_sum = new_sum
                        improved = True
                        break
            if improved:
                break
    
    result = [numbers[i] for i in selected_set]
    if not result:
        result = [256]
        
    return result
