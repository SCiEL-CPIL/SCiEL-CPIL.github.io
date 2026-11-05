"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Build all possible (m, r) pairs
    all_pairs = []
    for m in range(20, 201):
        for r in range(m):
            all_pairs.append((m, r))
    
    n = len(all_pairs)
    
    # Precompute weights: 1/m
    weights = [1.0 / m for m, r in all_pairs]
    
    # Sort by weight descending (largest 1/m first) for better B&B pruning
    order = sorted(range(n), key=lambda i: -weights[i])
    sorted_pairs = [all_pairs[i] for i in order]
    sorted_weights = [weights[i] for i in order]
    # Map from sorted index to original index
    orig_idx = [order[i] for i in range(n)]
    
    # Build incompatibility: two pairs (m1,r1) and (m2,r2) are incompatible
    # if r1 % gcd(m1,m2) == r2 % gcd(m1,m2)
    # For each sorted index i, compute set of sorted indices j > i that are incompatible
    
    # Precompute gcd table for moduli 20..200
    moduli = list(range(20, 201))
    gcd_table = {}
    for a in moduli:
        for b in moduli:
            if a <= b:
                gcd_table[(a, b)] = math.gcd(a, b)
    
    # Build incompatibility list: for each sorted position i, list of sorted positions j > i incompatible
    # Use bitmasks or lists
    incompat = [set() for _ in range(n)]
    
    for i in range(n):
        m1, r1 = sorted_pairs[i]
        for j in range(i + 1, n):
            m2, r2 = sorted_pairs[j]
            g = gcd_table.get((m1, m2)) if m1 <= m2 else gcd_table.get((m2, m1))
            if r1 % g == r2 % g:
                incompat[i].add(j)
                incompat[j].add(i)
    
    # Convert to sorted lists for efficiency
    for i in range(n):
        incompat[i] = sorted(incompat[i])
    
    # Branch and bound
    best_score = 0.0
    best_solution = []
    
    def branch_and_bound(pos, current_score, selected, remaining):
        nonlocal best_score, best_solution
        
        # Time check
        if time.time() > deadline:
            return
        
        # Upper bound: current_score + sum of weights of all remaining nodes
        # (relaxation: ignore incompatibility constraints)
        if not remaining:
            if current_score > best_score:
                best_score = current_score
                best_solution = list(selected)
            return
        
        # Tighter upper bound: sum of top weights from remaining
        # Sort remaining by weight descending and take prefix sum
        # For speed, precompute suffix sums of sorted_weights
        # But remaining is a subset, so we need to compute
        # Quick bound: current_score + sum of weights of remaining
        rem_sum = sum(sorted_weights[i] for i in remaining)
        if current_score + rem_sum <= best_score + 1e-12:
            return
        
        # Try to pick the best remaining node
        # Choose the node with highest weight from remaining
        best_node = max(remaining, key=lambda i: sorted_weights[i])
        
        # Option 1: include best_node
        # Find remaining nodes compatible with best_node
        new_remaining = [j for j in remaining if j != best_node and j not in incompat[best_node]]
        selected.append(best_node)
        branch_and_bound(pos + 1, current_score + sorted_weights[best_node], selected, new_remaining)
        selected.pop()
        
        # Option 2: exclude best_node
        new_remaining2 = [j for j in remaining if j != best_node]
        branch_and_bound(pos + 1, current_score, selected, new_remaining2)
    
    # Initial call
    all_indices = list(range(n))
    branch_and_bound(0, 0.0, [], all_indices)
    
    if not best_solution:
        # Fallback: greedy
        selected = []
        used = set()
        for i in range(n):
            m, r = sorted_pairs[i]
            # Check compatibility with all selected
            compatible = True
            for j in selected:
                m2, r2 = sorted_pairs[j]
                g = gcd_table.get((m, m2)) if m <= m2 else gcd_table.get((m2, m))
                if r % g == r2 % g:
                    compatible = False
                    break
            if compatible:
                selected.append(i)
        best_solution = selected
        best_score = sum(sorted_weights[i] for i in selected)
    
    # Convert back to original format
    result = [[sorted_pairs[i][0], sorted_pairs[i][1]] for i in best_solution]
    return result

# EVOLVE-BLOCK-END
