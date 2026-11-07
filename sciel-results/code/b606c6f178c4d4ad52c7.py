"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt

N = 256
DEADLINE_SECONDS = 55.0  # safety margin within 60s (or 90s) limit

def _precompute():
    """Precompute weights, coprime masks, and initial structures."""
    weights = [0] + list(range(1, N + 1))  # 1-indexed, weight[i] = i
    
    # coprime_mask[i] is a bitset where bit j is set if gcd(i, j) == 1
    coprime_mask = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if i == j:
                continue
            if gcd(i, j) == 1:
                mask |= (1 << j)
        coprime_mask[i] = mask
    
    # Total weight of all vertices
    total_weight = sum(weights[1:])
    
    return weights, coprime_mask, total_weight

def _greedy_lower_bound(weights, coprime_mask):
    """Get a good lower bound using greedy heuristic."""
    # Greedy: repeatedly pick the vertex with highest weight that is coprime to all selected
    selected = set()
    remaining = set(range(1, N + 1))
    
    while remaining:
        # Pick vertex with max weight that is coprime to all in selected
        best_v = None
        best_w = -1
        for v in remaining:
            if v in selected:
                continue
            # Check if v is coprime to all selected
            ok = True
            for s in selected:
                if not (coprime_mask[v] >> s) & 1:
                    ok = False
                    break
            if ok:
                if weights[v] > best_w:
                    best_w = weights[v]
                    best_v = v
        if best_v is None:
            break
        selected.add(best_v)
        remaining.discard(best_v)
    
    return sum(weights[v] for v in selected), selected

def _branch_and_bound(weights, coprime_mask, deadline):
    """
    Exact branch-and-bound for Maximum Weight Independent Set.
    Vertices: 1..N
    Edge between i,j if gcd(i,j) > 1 (non-coprime)
    We want max weight independent set in this graph = max weight set of pairwise-coprime ints.
    
    Using bitset representation for candidate set.
    """
    start_time = time.time()
    
    # Precompute: for each vertex, the mask of vertices that are NON-coprime to it (its neighbors in conflict graph)
    # conflict_mask[i] has bit j set if gcd(i,j) > 1
    conflict_mask = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if i != j and gcd(i, j) > 1:
                mask |= (1 << j)
        conflict_mask[i] = mask
    
    # Greedy lower bound
    lb, _ = _greedy_lower_bound(weights, coprime_mask)
    best_solution = None
    best_weight = lb
    
    # Candidate set as bitset: all vertices 1..N
    all_candidates = 0
    for i in range(1, N + 1):
        all_candidates |= (1 << i)
    
    # Upper bound: sum of weights of all candidates
    def sum_weights(mask):
        s = 0
        m = mask
        while m:
            lowbit = m & (-m)
            idx = lowbit.bit_length() - 1
            s += weights[idx]
            m ^= lowbit
        return s
    
    # Sort vertices by weight descending for better pruning
    # We'll use a recursive B&B with candidate mask
    
    # For efficiency, precompute weight sums for masks is too expensive.
    # Instead, use a simpler upper bound: sum of weights in candidate mask
    
    # Branch and bound
    # State: candidate mask, current weight
    # At each step, pick a vertex v from candidates, branch on include or exclude
    
    # Better approach: use a priority queue or recursive with ordering by weight
    
    # Let's use a recursive approach with memoization-like pruning
    # Key insight: the candidate set is determined by which vertices we've excluded
    
    def bnb(candidates, current_weight, selected_mask):
        nonlocal best_weight, best_solution, start_time
        
        if time.time() - start_time > DEADLINE_SECONDS:
            return
        
        if candidates == 0:
            if current_weight > best_weight:
                best_weight = current_weight
                best_solution = selected_mask
            return
        
        # Upper bound: current_weight + sum of weights in candidates
        ub = current_weight + sum_weights(candidates)
        if ub <= best_weight:
            return
        
        # Pick a vertex to branch on
        # Heuristic: pick the vertex with highest weight in candidates
        m = candidates
        best_v = None
        best_w = -1
        while m:
            lowbit = m & (-m)
            idx = lowbit.bit_length() - 1
            if weights[idx] > best_w:
                best_w = weights[idx]
                best_v = idx
            m ^= lowbit
        
        v = best_v
        bit_v = 1 << v
        
        # Branch 1: Include v
        # New candidates: those in candidates that are coprime to v (and not v itself)
        new_candidates_include = candidates & coprime_mask[v] & ~bit_v
        bnb(new_candidates_include, current_weight + weights[v], selected_mask | bit_v)
        
        # Branch 2: Exclude v
        new_candidates_exclude = candidates & ~bit_v
        bnb(new_candidates_exclude, current_weight, selected_mask)
    
    bnb(all_candidates, 0, 0)
    
    # Extract solution
    if best_solution is None:
        # Fallback to primes
        result = [1] + [a for a in range(2, N + 1) if all(a % d for d in range(2, isqrt(a) + 1))]
    else:
        result = []
        m = best_solution
        while m:
            lowbit = m & (-m)
            idx = lowbit.bit_length() - 1
            result.append(idx)
            m ^= lowbit
        result.sort()
    
    return result, best_weight

def construct():
    start = time.time()
    
    weights, coprime_mask, total_weight = _precompute()
    
    # Try exact B&B with deadline
    try:
        result, weight = _branch_and_bound(weights, coprime_mask, DEADLINE_SECONDS)
    except Exception:
        # Fallback to prime-only solution
        result = [1] + [a for a in range(2, N + 1) if all(a % d for d in range(2, isqrt(a) + 1))]
        weight = sum(result)
    
    # Ensure result is a list of Python ints
    result = [int(x) for x in result]
    
    return result
# EVOLVE-BLOCK-END
