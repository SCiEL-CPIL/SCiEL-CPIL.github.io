"""Rewritten program: Maximum weight independent set on 3-AP conflict graph via branch-and-bound."""

import math
from fractions import Fraction

def construct():
    n = 256
    # Vertices are numbers 1..n. Edge between i and j if there exists k such that
    # {i,j,k} forms a nonconstant 3-term AP. That is, i+j=2k, i+k=2j, or j+k=2i.
    # Equivalently, for each pair (i,j) with i<j, they conflict if:
    #   2*j - i is in [1,n] and != i,j  (k = 2j-i, so i,j,k is AP with j middle? No: i, j, k with j-i = k-j => k=2j-i)
    #   2*i - j is in [1,n] and != i,j  (k = 2i-j, so k,i,j is AP with i middle)
    #   i+j is even and (i+j)//2 is in [1,n] and != i,j  (i, m, j is AP with m middle)
    
    # Build conflict graph
    n_vertices = n
    # conflicts[v] = set of vertices that conflict with v
    conflicts = [set() for _ in range(n_vertices + 1)]
    
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            # Check if i and j can be part of a 3-AP with some k
            # Case 1: i, j, k with j - i = k - j => k = 2j - i
            k1 = 2 * j - i
            if 1 <= k1 <= n and k1 != i and k1 != j:
                conflicts[i].add(j)
                conflicts[j].add(i)
            # Case 2: k, i, j with i - k = j - i => k = 2i - j
            k2 = 2 * i - j
            if 1 <= k2 <= n and k2 != i and k2 != j:
                conflicts[i].add(j)
                conflicts[j].add(i)
            # Case 3: i, m, j with m - i = j - m => m = (i+j)/2
            if (i + j) % 2 == 0:
                m = (i + j) // 2
                if 1 <= m <= n and m != i and m != j:
                    conflicts[i].add(j)
                    conflicts[j].add(i)
    
    # Weights: reciprocal of vertex value
    weights = [0.0] * (n_vertices + 1)
    for v in range(1, n_vertices + 1):
        weights[v] = 1.0 / v
    
    # For exact rational arithmetic in the final answer, we'll track selected vertices
    # and compute the exact sum at the end.
    
    # Branch and bound for maximum weight independent set
    # We use a greedy fractional relaxation for upper bound:
    # At each node, we have a set of candidates (vertices not yet decided)
    # and a set of already selected vertices.
    # Upper bound = sum of weights of all candidates that don't conflict with selected
    #   plus a fractional relaxation: sort candidates by weight/degree, take greedily.
    
    # For efficiency, we'll use a recursive B&B with memoization-like pruning.
    # Given the size (256 vertices), we need strong pruning.
    
    # Let's use a different approach: 
    # 1. Order vertices by decreasing weight (1/1 > 1/2 > 1/3 > ...)
    # 2. Use branch-and-bound where at each step we decide to include or exclude the next vertex
    # 3. Upper bound: sum of weights of remaining candidates that are compatible with current selection
    
    # Precompute for each vertex its conflict set
    # Use bitsets for efficiency
    from bitarray import bitarray
    
    # Actually, let's use Python's built-in integers as bitsets
    # vertex v corresponds to bit (v-1)
    
    # conflicts_bit[v] = integer with bits set for all conflicts of v
    conflicts_bit = [0] * (n_vertices + 1)
    for v in range(1, n_vertices + 1):
        mask = 0
        for u in conflicts[v]:
            mask |= (1 << (u - 1))
        conflicts_bit[v] = mask
    
    # Order vertices by decreasing weight (i.e., increasing value)
    # So vertex 1 has highest weight 1/1, vertex 256 has lowest weight 1/256
    order = list(range(1, n_vertices + 1))
    # Already in order of decreasing weight since weight = 1/v
    
    # Branch and bound
    best_weight = 0.0
    best_set = []
    
    # We'll use iterative deepening or just recursive with good pruning
    # Given 256 vertices, recursion depth 256 is fine.
    
    # For upper bound, we compute the sum of weights of all vertices in the candidate set
    # that are compatible with the current selection.
    
    # Let's define:
    # current_mask: bitset of selected vertices
    # candidate_mask: bitset of vertices still available (not selected, not excluded)
    
    # Upper bound: sum of weights of vertices in candidate_mask that don't conflict with current_mask
    # This is a valid upper bound (relaxation: ignore conflicts among candidates)
    
    # To make it tighter, we can use a greedy fractional bound:
    # Sort candidates by weight, then greedily pick vertices that don't conflict with each other
    # and with current selection, summing their weights. This gives a tighter upper bound.
    
    def greedy_bound(candidate_mask, current_mask):
        """Compute an upper bound using greedy independent set on candidates."""
        # Get list of candidate vertices
        candidates = []
        m = candidate_mask
        while m:
            # Get lowest set bit
            lb = m & (-m)
            idx = lb.bit_length() - 1
            v = idx + 1
            # Check if v conflicts with current_mask
            if (conflicts_bit[v] & current_mask) == 0:
                candidates.append(v)
            m ^= lb
        
        # Sort by weight descending (i.e., by value ascending)
        candidates.sort(key=lambda x: x)  # Smaller x means larger weight 1/x
        
        # Greedy independent set
        bound = 0.0
        used_mask = current_mask
        for v in candidates:
            if (conflicts_bit[v] & used_mask) == 0:
                bound += weights[v]
                used_mask |= (1 << (v - 1))
        return bound
    
    # Recursive branch and bound
    import sys
    sys.setrecursionlimit(10000)
    
    def bb(current_mask, candidate_mask, current_weight):
        nonlocal best_weight, best_set
        
        # Pruning: if current_weight + upper_bound <= best_weight, prune
        if current_weight + greedy_bound(candidate_mask, current_mask) <= best_weight + 1e-15:
            return
        
        # If no candidates left, update best
        if candidate_mask == 0:
            if current_weight > best_weight:
                best_weight = current_weight
                # Recover the set
                best_set = []
                m = current_mask
                while m:
                    lb = m & (-m)
                    idx = lb.bit_length() - 1
                    best_set.append(idx + 1)
                    m ^= lb
            return
        
        # Pick the next vertex to branch on: the one with highest weight among candidates
        # that is compatible with current selection
        next_v = None
        max_w = -1
        m = candidate_mask
        while m:
            lb = m & (-m)
            idx = lb.bit_length() - 1
            v = idx + 1
            if (conflicts_bit[v] & current_mask) == 0:
                if weights[v] > max_w:
                    max_w = weights[v]
                    next_v = v
            m ^= lb
        
        if next_v is None:
            # No compatible candidates
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = []
                m2 = current_mask
                while m2:
                    lb2 = m2 & (-m2)
                    idx2 = lb2.bit_length() - 1
                    best_set.append(idx2 + 1)
                    m2 ^= lb2
            return
        
        # Branch 1: include next_v
        new_current = current_mask | (1 << (next_v - 1))
        new_candidate = candidate_mask & ~(1 << (next_v - 1))
        # Remove all conflicts of next_v from candidates
        new_candidate &= ~conflicts_bit[next_v]
        bb(new_current, new_candidate, current_weight + weights[next_v])
        
        # Branch 2: exclude next_v
        new_candidate2 = candidate_mask & ~(1 << (next_v - 1))
        bb(current_mask, new_candidate2, current_weight)
    
    # Initial call
    full_mask = (1 << n_vertices) - 1
    bb(0, full_mask, 0.0)
    
    return best_set

