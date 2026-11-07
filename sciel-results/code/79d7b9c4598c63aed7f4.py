"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    N = 256
    nodes = list(range(1, N + 1))
    
    # Build conflict graph: adjacency list where edge exists if gcd(a, b) > 1
    # We only care about edges between distinct nodes
    adj = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if gcd(i, j) > 1:
                adj[i].append(j)
                adj[j].append(i)
    
    # Sort nodes by weight (value) in descending order to help pruning
    # We will index them 0 to N-1 in this sorted order for the solver
    sorted_nodes = sorted(nodes, reverse=True)
    # Map original node value to index in sorted_nodes
    val_to_idx = {v: i for i, v in enumerate(sorted_nodes)}
    
    # Rebuild adjacency in terms of indices
    M = len(sorted_nodes)
    adj_idx = [set() for _ in range(M)]
    for i in range(M):
        for j in range(i + 1, M):
            u = sorted_nodes[i]
            v = sorted_nodes[j]
            if gcd(u, v) > 1:
                adj_idx[i].add(j)
                adj_idx[j].add(i)
                
    # Precompute suffix sums of weights for upper bound calculation
    # weights[i] is the value of the node at index i
    weights = [sorted_nodes[i] for i in range(M)]
    suffix_sum = [0] * (M + 1)
    for i in range(M - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + weights[i]
        
    best_sum = 0
    best_set = []
    
    # Branch and Bound
    # state: index in sorted_nodes, current set of selected indices
    # We use a recursive approach with explicit stack or recursion.
    # Given M=256, recursion depth is fine, but branching factor can be high.
    # We prune if current_sum + suffix_sum[i] <= best_sum
    
    def solve(i, current_sum, current_set):
        nonlocal best_sum, best_set
        
        if time.time() > deadline:
            return
            
        if i == M:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = current_set.copy()
            return
            
        # Upper bound: current sum + sum of all remaining weights
        if current_sum + suffix_sum[i] <= best_sum:
            return
            
        node_val = sorted_nodes[i]
        
        # Option 1: Exclude node i
        # We can skip checking if including it is impossible due to conflicts?
        # No, we just recurse.
        solve(i + 1, current_sum, current_set)
        
        # Option 2: Include node i
        # Check if node i is compatible with all nodes in current_set
        # Since we process in descending weight order, and we only add to the end,
        # we just need to check against current_set.
        compatible = True
        for j in current_set:
            if j in adj_idx[i]:
                compatible = False
                break
                
        if compatible:
            current_set.append(i)
            solve(i + 1, current_sum + node_val, current_set)
            current_set.pop()
            
    solve(0, 0, [])
    
    # Convert indices back to values
    result = [sorted_nodes[i] for i in best_set]
    
    # Ensure distinct integers, Python int types
    # The problem allows 1. 
    # The result is a list of ints.
    return result
# EVOLVE-BLOCK-END
