"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    import time
    import numpy as np
    from collections import defaultdict, deque
    
    deadline = time.time() + 80  # 80s safety margin for 90s limit
    
    # Parse DAG structure
    # dag format: dict with 'num_vertices', 'parents', 'outputs', 'inputs'
    n = dag.get('num_vertices', 0)
    if n == 0:
        return []
    
    parents = dag.get('parents', [])
    outputs = set(dag.get('outputs', []))
    inputs = set(dag.get('inputs', []))
    
    # Build children adjacency
    children = defaultdict(list)
    for v in range(n):
        for p in parents[v]:
            children[p].append(v)
    
    # Compute topological order (already given by vertex indices)
    # Compute usage counts: how many times each vertex is needed
    usage_count = [0] * n
    for v in range(n):
        usage_count[v] = len(children[v])  # number of children that need it
    
    # Naive schedule: process in topological order, load each parent when needed,
    # compute, store result, delete parents when done
    # Count naive transfers
    naive_transfers = 0
    for v in range(n):
        if v in inputs:
            continue  # inputs are already in fast memory (blue)
        # For non-input vertices, we need to load parents
        for p in parents[v]:
            if p not in inputs:
                naive_transfers += 1  # load parent
        # Store the result if it's needed by multiple children or is an output
        if v in outputs or len(children[v]) > 1:
            naive_transfers += 1  # store result
        # Actually, let's think more carefully
        # Naive: load all parents, compute, if needed later store it
        # But the exact naive schedule definition matters for scoring
    
    # Let me implement a better strategy:
    # 1. Keep track of what's in fast memory (red values)
    # 2. Process vertices in topological order
    # 3. For each vertex, ensure its parents are in fast memory
    # 4. Compute the vertex
    # 5. Decide whether to keep it in fast memory or store it back
    
    # Strategy: Greedy with lookahead
    # - Keep values in fast memory that will be used soon
    # - Evict values that won't be used for a while
    
    # First, compute for each vertex the next time it will be needed
    # and the distance to next use
    
    # Let's use a simpler but effective approach:
    # Process in topological order
    # Maintain a set of values currently in fast memory (red)
    # For each vertex:
    #   - Load any parents not in fast memory
    #   - Compute the vertex
    #   - If fast memory is full, evict a value that won't be needed soonest
    #   - Keep the computed value if it will be needed again
    
    # Compute next_use: for each vertex, when is it next needed as a parent
    # Since we process in topological order, we can precompute this
    
    # For each vertex v, find the earliest child that uses v
    # Actually, we need to track which vertices are "active" (needed in future)
    
    # Let me implement a clean greedy algorithm:
    
    # State: set of vertices currently in fast memory (as red values)
    # fast_mem = set()
    # Initially, all inputs are in fast memory (blue, so they don't consume slots? 
    # Wait - "Inputs start blue" - blue means they're in slow memory but accessible?
    # Re-reading: "every red value consumes one fast-memory slot"
    # So blue values are in slow memory, red values are in fast memory
    # "Inputs start blue" - so inputs are NOT in fast memory initially
    # We need to load them when needed
    
    # Actually, let me re-read: "Inputs start blue, all outputs must finish blue"
    # Blue = in slow memory (or not in fast memory)
    # Red = in fast memory
    # So inputs are NOT in fast memory initially, we need to load them
    
    # Let me implement the algorithm:
    
    fast_mem = set()  # vertices currently in fast memory (red)
    operations = []
    transfers = 0
    
    # Precompute: for each vertex, the list of children that need it
    # and track which children have been processed
    
    # For eviction policy: evict the vertex with the farthest next use
    # Precompute next_use for each vertex
    
    # Since we process in topological order, for vertex v, its children are all > v
    # The "next use" of v is the smallest child index that hasn't been processed yet
    
    # Let's track remaining children for each vertex
    remaining_children = [list(children[v]) for v in range(n)]
    
    for v in range(n):
        if time.time() > deadline:
            break
            
        # Ensure all parents are in fast memory
        for p in parents[v]:
            if p not in fast_mem:
                # Load p from slow memory
                operations.append(['load', p])
                fast_mem.add(p)
                transfers += 1
                # Check if we're over budget
                if len(fast_mem) > fast_memory_budget:
                    # Need to evict something
                    # Find the vertex with the farthest next use
                    # (excluding current vertex v and its parents that are still needed)
                    # Actually, we just loaded p, so we might need to evict
                    # Let's find the vertex to evict: the one with the largest next use
                    # or the one that won't be used again
                    
                    # Find candidate for eviction: not v, not a parent of v that's still needed
                    # Actually, we can evict any vertex that won't be needed soon
                    # Let's evict the one with the farthest next use
                    
                    # Build list of vertices in fast_mem with their next use
                    candidates = []
                    for u in fast_mem:
                        if u == v:
                            continue
                        # Find next use of u
                        next_use = float('inf')
                        for child in remaining_children[u]:
                            if child > v:  # child not yet processed
                                next_use = min(next_use, child)
                            else:
                                # child already processed, shouldn't happen in topo order
                                pass
                        candidates.append((next_use, u))
                    
                    if candidates:
                        # Evict the one with the largest next use (least urgent)
                        candidates.sort(key=lambda x: -x[0])
                        evict_u = candidates[0][1]
                        operations.append(['store', evict_u])
                        fast_mem.discard(evict_u)
                        transfers += 1
                    
                    # If still over budget, this is a problem
                    # But we just added one, so if budget is 0, we can't keep anything
                    # Let's handle this more carefully later
        
        # Compute v
        operations.append(['compute', v])
        
        # Add v to fast memory if it will be needed again or is an output
        if v in outputs or len(remaining_children[v]) > 0:
            fast_mem.add(v)
            # Check budget again
            while len(fast_mem) > fast_memory_budget:
                # Evict the least urgent
                candidates = []
                for u in fast_mem:
                    if u == v:
                        continue  # don't evict the just-computed value
                    next_use = float('inf')
                    for child in remaining_children[u]:
                        if child > v:
                            next_use = min(next_use, child)
                    candidates.append((next_use, u))
                
                if not candidates:
                    break
                
                candidates.sort(key=lambda x: -x[0])
                evict_u = candidates[0][1]
                operations.append(['store', evict_u])
                fast_mem.discard(evict_u)
                transfers += 1
        
        # Remove v from remaining_children of its parents
        # Actually, we should mark v as processed for its parents
        # Let's update remaining_children for parents
        for p in parents[v]:
            if v in remaining_children[p]:
                remaining_children[p].remove(v)
    
    # Ensure all outputs are in fast memory (blue means in slow memory, but they must finish blue)
    # Wait - "all outputs must finish blue" - so outputs should NOT be in fast memory at the end?
    # Or does "finish blue" mean they're available?
    # Re-reading: "Inputs start blue, all outputs must finish blue"
    # I think blue means in slow memory, so we need to store outputs back
    # But that doesn't make sense for a pebbling problem...
    # Let me assume "finish blue" means the outputs are correctly computed and available
    # For safety, let's store all outputs at the end
    
    for v in outputs:
        if v in fast_mem:
            operations.append(['store', v])
            fast_mem.discard(v)
            transfers += 1
    
    return operations
# EVOLVE-BLOCK-END
