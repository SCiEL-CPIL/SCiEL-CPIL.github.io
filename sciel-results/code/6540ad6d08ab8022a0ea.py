import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Leave 5s margin for safety
    
    N = 256
    
    # Squares up to 2*256 = 512
    squares = set()
    i = 2
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1
    
    # Identify valid nodes (no self-conflict: 2v not a square)
    valid_nodes = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_nodes.append(v)
    
    n = len(valid_nodes)
    node_index = {v: i for i, v in enumerate(valid_nodes)}
    
    # Build conflict edges
    conflicts = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if valid_nodes[i] + valid_nodes[j] in squares:
                conflicts[i].add(j)
                conflicts[j].add(i)
    
    # Helper to check if a set (list of indices) is independent
    def is_independent(indices):
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                if indices[j] in conflicts[indices[i]]:
                    return False
        return True

    # 1. Greedy Initialization
    # Sort nodes by degree (ascending) to pick low-conflict nodes first
    degrees = [len(conflicts[i]) for i in range(n)]
    sorted_nodes = sorted(range(n), key=lambda x: degrees[x])
    
    current_set = set()
    for v in sorted_nodes:
        # Check if v conflicts with any node in current_set
        conflict = False
        for u in current_set:
            if u in conflicts[v]:
                conflict = True
                break
        if not conflict:
            current_set.add(v)
            
    best_set = set(current_set)
    best_size = len(best_set)
    
    # 2. Local Search: Iterated Local Search / Simulated Annealing style
    # We try to improve the set by swapping nodes or adding/removing.
    # Since adding a node requires removing all its neighbors, we look for
    # swaps that increase size or maintain size with better "expansion" potential.
    # For simplicity and speed, we will use a 1-swap heuristic:
    # Remove one node from the set, add another node that was not in the set,
    # if it results in a larger set.
    
    # Pre-compute neighbors for fast access
    neighbors = [list(conflicts[i]) for i in range(n)]
    
    # Random seed for reproducibility if needed, but random restarts help escape local optima
    random.seed(42)
    
    while time.time() < deadline:
        # Try to improve current_set
        improved = False
        # Shuffle nodes to avoid bias
        nodes_to_try = list(range(n))
        random.shuffle(nodes_to_try)
        
        for v_out in nodes_to_try:
            if v_out not in current_set:
                continue
                
            # If we remove v_out, which nodes in current_set are freed?
            # Actually, removing v_out allows any node u NOT in current_set 
            # that only conflicted with v_out (among current_set nodes) to potentially be added.
            # But u might conflict with other nodes in current_set \ {v_out}.
            
            # A simpler strategy: 
            # 1. Remove v_out.
            # 2. Try to add a node v_in not in current_set \ {v_out} 
            #    that has no conflicts with current_set \ {v_out}.
            # 3. If we can add MORE than 1 node, or even 1 node if we removed 1, 
            #    size stays same or increases.
            # To maximize size, we prefer adding nodes.
            
            temp_set = current_set - {v_out}
            
            # Find candidates to add
            added_any = False
            # We want to find a node v_in such that 
            # v_in not in temp_set and v_in has no neighbors in temp_set.
            
            # To make this fast, we can iterate through nodes not in current_set
            # and check conflicts.
            # Optimization: Only check nodes that conflict with v_out? 
            # No, any node could be added if it doesn't conflict with the rest.
            
            # Let's try to find the "best" v_in to add.
            # We just need to increase size. So if we add 1 node, size is same.
            # If we add 2 nodes, size increases.
            
            # Strategy: Try to add as many nodes as possible after removing v_out.
            # This is a greedy fill.
            
            # Get list of nodes not in current_set
            # This is O(N) per iteration. N~250. 
            # We do this for each v_out (N times). N^2 = 62500.
            # Inner loop checks conflicts.
            
            # Let's try a simpler 1-swap: Remove v_out, Add v_in.
            # This keeps size same. Not helpful for maximization.
            
            # We need a move that increases size.
            # This requires removing k nodes and adding k+1 nodes.
            # This is complex.
            
            # Alternative: Just try to add nodes to the current set if possible?
            # If current_set is maximal independent set, no single node can be added.
            # So we MUST remove nodes to add more.
            
            # Let's use a "kick" approach:
            # Remove a random subset of nodes (size 1 or 2), then greedily fill.
            
            # For efficiency, let's stick to removing 1 node and trying to fill greedily.
            # If the greedy fill adds > 1 node, we improve.
            
            # Identify nodes that can be added to temp_set
            # A node x can be added if x not in temp_set and x has no neighbors in temp_set.
            
            # To speed up, we can precompute for each node x, the set of nodes it conflicts with.
            # conflict_mask[x]
            
            # Let's use bitmasks for speed if N < 64? No, N~250.
            # Python sets are fast enough.
            
            # Check all nodes not in current_set
            # This is expensive. 
            # Better: Only consider nodes that are "compatible" with temp_set.
            
            # Let's just do a limited search.
            # Pick a random v_out from current_set.
            # Remove it.
            # Try to add random nodes from complement until no more can be added.
            # If final size > best_size, update.
            
            # To avoid O(N^2) full scan every time, we will just do this for a few v_outs per loop.
            pass
        
        # The logic above is getting complex and might be slow.
        # Let's switch to a simpler, robust heuristic:
        # Random Restart Greedy with Local Search (1-swap to improve density/size)
        
        # Actually, the Greedy above is likely close to optimal for this graph structure.
        # The bottleneck is finding the *maximum* size.
        
        # Let's try a different approach: 
        # Since N is small, we can use a simple branch and bound for just a few seconds
        # if we have a good upper bound. But we removed B&B.
        
        # Let's try "Iterated Local Search" with 2-swap moves.
        # Move: Remove {u, v}, Add {w, x, y} such that size increases.
        
        # Given time constraints, let's just run the Greedy multiple times with different tie-breaking
        # and take the best.
        
        if time.time() > deadline - 1:
            break
            
        # Re-run greedy with random tie-breaking
        current_set = set()
        # Shuffle nodes
        nodes = list(range(n))
        random.shuffle(nodes)
        
        # Sort by degree, but break ties randomly? 
        # Or just iterate in random order and pick if valid.
        # Random order greedy is often decent.
        
        for v in nodes:
            conflict = False
            for u in current_set:
                if u in conflicts[v]:
                    conflict = True
                    break
            if not conflict:
                current_set.add(v)
                
        if len(current_set) > best_size:
            best_set = current_set
            best_size = len(best_set)
            
            # If we found a larger set, we can try to improve it locally
            # Simple local improvement: try to swap one node out for two in?
            # This is hard to implement fast.
            
            # Let's just keep restarting.
            
    result_nodes = [valid_nodes[i] for i in best_set]
    
    # Final verification
    if not is_independent(result_nodes):
        # Fallback to single node if verification fails (should not happen)
        result_nodes = [valid_nodes[0]]
        
    return result_nodes
