import random
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 60.0 - 5.0  # 5s safety margin
    
    # The optimal set is likely a subset of {51, ..., 100}
    # Let's verify this assumption briefly.
    # If we include x < 51, we need to ensure no subset of the rest sums to 200-x.
    # Given the density of sums in {51..100}, adding small numbers creates more conflicts.
    # We will focus on finding the Maximum Independent Set in the hypergraph of triplets summing to 200 within {51..100}.
    
    nodes = list(range(51, 101))
    
    # Find all triplets (a, b, c) with a < b < c summing to 200
    triplets = []
    n = len(nodes)
    # Map value to index for faster lookup if needed, but n is small
    for i in range(n):
        a = nodes[i]
        for j in range(i + 1, n):
            b = nodes[j]
            c = 200 - a - b
            if c > b and c <= 100:
                # Check if c is in nodes (it is, since c > b >= 52 > 51)
                triplets.append((i, j, nodes.index(c)))
                
    # We want to maximize the number of selected nodes such that no triplet is fully selected.
    # This is the Maximum Independent Set in a 3-uniform hypergraph.
    
    best_set = set()
    best_size = 0
    
    # Local Search / Greedy with Randomization
    # We can run this for a few seconds
    while time.time() < deadline:
        # Initialize with all nodes
        selected = set(range(n))
        
        # Remove nodes to break all triplets
        # Greedy: Find a violating triplet, remove a random node from it
        # To improve, we can prioritize removing nodes that are in many triplets
        # Let's use a simple random removal first, then local improvement
        
        random.shuffle(triplets) # Shuffling edges doesn't help much, shuffling nodes does
        
        # Better approach:
        # 1. Start with all nodes.
        # 2. While there is a triplet fully in selected:
        #      Pick one such triplet.
        #      Remove a random node from it.
        
        # To make it faster and better, we can use a "conflict-driven" removal.
        # But simple random restarts with local search is robust.
        
        # Let's try a more structured greedy:
        # Sort nodes by some heuristic?
        # Actually, let's just use a fast local search.
        
        current_selected = set(range(n))
        
        # Precompute for each node, the list of triplets it belongs to
        # node_to_triplets = {i: [] for i in range(n)}
        # for t in triplets:
        #     for idx in t:
        #         node_to_triplets[idx].append(t)
        
        # This is small enough to just check on the fly or recompute.
        
        # Improvement loop
        improved = True
        while improved:
            improved = False
            # Find all violating triplets
            violating = [t for t in triplets if all(x in current_selected for x in t)]
            if not violating:
                break
            
            # Pick a random violating triplet
            t = random.choice(violating)
            
            # Pick a node to remove from t
            # Heuristic: remove the one that breaks the most other potential triplets?
            # Or just random.
            # Let's try removing the node with the highest "degree" in the remaining hypergraph
            # Degree = number of triplets in violating list that contain this node?
            # Or just global degree.
            
            # Simple: random choice
            node_to_remove = random.choice(t)
            current_selected.remove(node_to_remove)
            improved = True
            
        # After breaking all conflicts, try to add back nodes
        # Keep a list of removed nodes
        # We lost track of which were removed in the loop above.
        # Let's restart the logic for a cleaner implementation.
        
        # RESTART CLEANER LOGIC
        selected = set(range(n))
        removed = []
        
        # Phase 1: Greedy Removal
        while True:
            # Find a violating triplet
            viol = None
            for t in triplets:
                if all(x in selected for x in t):
                    viol = t
                    break
            if not viol:
                break
            
            # Remove a node from viol
            # Choose the node that appears in the most other violating triplets?
            # Let's just pick one randomly to allow diversity
            node = random.choice(viol)
            selected.remove(node)
            removed.append(node)
            
        # Phase 2: Local Search - Try to add back removed nodes
        # Shuffle removed nodes
        random.shuffle(removed)
        for node in removed:
            # Check if adding node creates a violation
            # A violation is a triplet fully in selected U {node}
            # This means a triplet containing node, where the other 2 are in selected
            can_add = True
            for t in triplets:
                if node in t:
                    other_two = [x for x in t if x != node]
                    if all(x in selected for x in other_two):
                        can_add = False
                        break
            if can_add:
                selected.add(node)
                
        # Phase 3: Local Search - Try to swap?
        # If we can't add a removed node, can we remove a selected node to allow it?
        # Only if it increases size (it doesn't, 1 for 1).
        # So we just want max size.
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = set(selected)
            
    # Convert indices back to values
    result = [nodes[i] for i in best_set]
    
    # If no set found (should not happen), return a safe default
    if not result:
        return [100]
        
    return result
# EVOLVE-BLOCK-END
