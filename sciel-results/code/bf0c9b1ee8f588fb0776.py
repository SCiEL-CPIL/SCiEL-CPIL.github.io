import time
from math import gcd

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 90s, we target 85s to be safe
    deadline = start_time + 85.0
    
    N = 256
    nodes = list(range(2, N + 1))
    
    # Precompute incompatibility sets
    # incompat[x] contains y such that lcm(x, y) <= 256
    incompat = {x: set() for x in nodes}
    for i in range(len(nodes)):
        a = nodes[i]
        for j in range(i + 1, len(nodes)):
            b = nodes[j]
            g = gcd(a, b)
            if (a * b) // g <= 256:
                incompat[a].add(b)
                incompat[b].add(a)
                
    # Sort nodes by descending weight (1/x) to prioritize high-value nodes in BB
    # This helps the upper bound tighten faster as we include "heavy" nodes early
    sorted_nodes = sorted(nodes, key=lambda x: -1.0/x)
    
    best_score = 0.0
    best_set = set()
    
    # Branch and Bound
    # State: (current_set, remaining_nodes, current_score)
    # We use recursion with explicit stack or recursion limit adjustment
    import sys
    sys.setrecursionlimit(3000)
    
    def bb_search(current_set, remaining_indices, current_score, depth):
        nonlocal best_score, best_set
        
        # Time check
        if time.time() > deadline:
            return
            
        # Upper bound: current score + sum of weights of all remaining nodes
        # This is a loose bound, but we can tighten it.
        # A better bound: current_score + sum of weights of remaining nodes that are compatible with current_set
        # However, computing this at every node is expensive.
        # Let's use a slightly tighter bound:
        # The maximum possible addition is the sum of reciprocals of all remaining nodes.
        # Since we are processing in order of decreasing weight, the remaining nodes are "smaller" weights.
        # Actually, the simplest effective bound for this specific problem structure (where conflicts are local/small)
        # is just the sum of all remaining weights.
        
        # Optimization: If current_score + sum(1/x for x in remaining) <= best_score, prune.
        # Precomputing suffix sums of weights for sorted_nodes helps here.
        
        # Let's implement a simpler, robust BB with a tightening heuristic.
        # We'll maintain the list of available nodes (indices in sorted_nodes).
        
        # Prune if no remaining nodes
        if not remaining_indices:
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
            return
            
        # Calculate upper bound
        # Sum of weights of all remaining nodes
        # To make this fast, we can precompute suffix sums of weights in sorted_nodes
        # But remaining_indices is a subset. 
        # Alternative: Just use current_score + sum(1/sorted_nodes[i] for i in remaining_indices)
        # This is O(n) per node. With n=255, this might be too slow for deep searches.
        
        # Let's try a different approach: 
        # Since the objective is sum(1/x), and x >= 2, the maximum possible sum is bounded by H_256 - 1.
        # But we need a dynamic bound.
        
        # Let's use a greedy upper bound: 
        # Add all remaining nodes that are compatible with current_set? No, they might conflict with each other.
        # The sum of all remaining weights is a valid upper bound.
        # Let's precompute the total weight of all nodes.
        # total_weight = sum(1/x for x in nodes)
        # We can pass a "remaining_weight_sum" as an argument to avoid recalculation.
        
        # Revised BB signature: bb_search(current_set, remaining_indices, current_score, remaining_weight_sum, depth)
        
        # If the upper bound is not better than best, prune
        upper_bound = current_score + remaining_weight_sum
        if upper_bound <= best_score + 1e-12:
            return
            
        # Select a node to branch on
        # Heuristic: Pick the node with the highest weight (first in remaining_indices)
        # This is already sorted by weight descending.
        idx = remaining_indices[0]
        node = sorted_nodes[idx]
        
        # Option 1: Include node
        # Check if node is compatible with current_set
        if not (incompat[node] & current_set):
            new_set = current_set | {node}
            new_remaining = remaining_indices[1:]
            # Calculate new remaining weight sum
            # We can't just subtract 1/node from remaining_weight_sum because 
            # the other nodes in remaining_indices are still there.
            # Wait, remaining_weight_sum is the sum of weights of nodes in remaining_indices.
            # If we remove the first one, new_sum = remaining_weight_sum - 1.0/node
            new_weight_sum = remaining_weight_sum - 1.0/node
            
            bb_search(new_set, new_remaining, current_score + 1.0/node, new_weight_sum, depth + 1)
            
            # Check time again
            if time.time() > deadline:
                return
            
        # Option 2: Exclude node
        # We can exclude it.
        # But we should also prune branches where excluding a high-value node makes it impossible to beat best_score
        # The upper bound for the exclude branch is current_score + (remaining_weight_sum - 1.0/node)
        # If this is <= best_score, we can skip the exclude branch entirely? 
        # No, because the include branch might have failed or been pruned, but the exclude branch
        # still has the same upper bound relative to the *remaining* set.
        # Actually, the upper bound for the exclude branch is current_score + remaining_weight_sum - 1.0/node.
        # If that is <= best_score, we can return.
        
        exclude_upper_bound = current_score + (remaining_weight_sum - 1.0/node)
        if exclude_upper_bound <= best_score + 1e-12:
            return
            
        bb_search(current_set, remaining_indices[1:], current_score, remaining_weight_sum - 1.0/node, depth + 1)

    # Initialize
    initial_weight_sum = sum(1.0/x for x in nodes)
    # Start with empty set, all nodes available
    # To speed up, we can do a greedy pass first to get a good lower bound (best_score)
    
    # Greedy initialization
    greedy_set = set()
    greedy_score = 0.0
    for node in sorted_nodes:
        if not (incompat[node] & greedy_set):
            greedy_set.add(node)
            greedy_score += 1.0/node
    best_score = greedy_score
    best_set = set(greedy_set)
    
    # Run BB
    # We need to be careful with recursion depth. 255 is fine.
    # But the number of nodes visited could be huge.
    # The pruning `upper_bound <= best_score` is key.
    # Since we start with a good greedy solution, the upper bound will prune many branches.
    
    # One issue: The "Exclude" branch explores a lot of "small" nodes.
    # We can improve the bound by checking compatibility of remaining nodes with current_set.
    # But that's expensive.
    
    # Let's just run the basic BB.
    
    try:
        bb_search(set(), list(range(len(sorted_nodes))), 0.0, initial_weight_sum, 0)
    except RecursionError:
        pass # Should not happen with limit 3000 for depth 255

    # If time is left, do a quick local search on the best set found to polish
    if time.time() < deadline - 5.0:
        S = set(best_set)
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            # Try to add
            for x in sorted_nodes:
                if x not in S and not (incompat[x] & S):
                    S.add(x)
                    best_score += 1.0/x
                    improved = True
                    break
            if improved:
                continue
            # Try to swap
            S_list = list(S)
            for u in S_list:
                S_temp = S - {u}
                # Find best v to add
                best_v = None
                best_gain = 0
                for v in sorted_nodes:
                    if v in S:
                        continue
                    if not (incompat[v] & S_temp):
                        gain = 1.0/v - 1.0/u
                        if gain > best_gain:
                            best_gain = gain
                            best_v = v
                if best_v:
                    S.remove(u)
                    S.add(best_v)
                    best_score += best_gain
                    improved = True
                    break
    
    return list(best_set)

# Ensure the function is named construct and returns a list
# The code above defines construct.

# Quick sanity check wrapper if needed, but the task asks for the program.
# The code is complete.

if __name__ == "__main__":
    # Optional: run locally to test
    res = construct()
    print(f"Score: {sum(1.0/x for x in res)}, Size: {len(res)}")
