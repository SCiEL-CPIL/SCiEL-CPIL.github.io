def construct():
    import collections
    
    # State: (residue, mask)
    # We want to find the min number of steps (denominators) to reach each residue.
    # Since denominators must be distinct, we track the mask of used denominators.
    
    # dist[r][mask] = min steps to reach residue r using exactly the denominators in mask?
    # No, we just need min steps to reach r.
    # But we need to know *which* denominators to output.
    
    # BFS from (0, 0)
    # Queue: (residue, mask, steps, path)
    # We stop expanding a node if we've already found a shorter or equal path to (residue, mask)?
    # Actually, we just need the shortest path to each residue.
    # But different masks can lead to the same residue.
    # We want the shortest path (min steps).
    
    # Let's use Dijkstra or BFS since edge weights are 1.
    # State: (residue, mask)
    # Transitions: add a new denominator d not in mask.
    # New residue: (residue + inv(d)) % 101
    # New mask: mask | (1 << (d-1))
    
    # We want to find the min steps for each residue.
    # We can just do BFS on the state space.
    
    # To save memory, we can just track best_steps[residue] and best_mask[residue]
    # But wait, a path with more steps might allow a different continuation?
    # No, we are just looking for the shortest path to the target.
    # Once we reach a residue, the shortest path to it is fixed.
    # We don't need to continue from it to find a shorter path to it.
    
    # However, we need to output the *subset*.
    # So we store the mask for the best path to each residue.
    
    best_mask = [None] * 101
    best_steps = [1000] * 101
    
    # BFS Queue
    q = collections.deque()
    q.append((0, 0, 0)) # residue, mask, steps
    best_mask[0] = 0
    best_steps[0] = 0
    
    # We need to handle the case where we reach a residue with the same steps but different mask?
    # It doesn't matter, any mask of min size is fine.
    
    while q:
        r, mask, steps = q.popleft()
        
        # If we have already found a shorter path to this residue, skip
        # (This is handled by checking best_steps before enqueueing or here)
        
        # Try adding each denominator d
        for d in range(1, 11):
            bit = 1 << (d-1)
            if mask & bit:
                continue
            
            new_r = (r + pow(d, -1, 101)) % 101
            new_mask = mask | bit
            new_steps = steps + 1
            
            if new_steps < best_steps[new_r]:
                best_steps[new_r] = new_steps
                best_mask[new_r] = new_mask
                q.append((new_r, new_mask, new_steps))
                
    # Now best_steps[r] is the min size for residue r.
    # Max row length is max(best_steps[1:]) (excluding 0? No, 0 is a residue too? 
    # "residues modulo 101" usually 0..100.
    # The task says "row r containing ... sum to r".
    # So r=0 must be represented.
    # Empty set sums to 0. But "nonempty lists".
    # So we need a non-empty subset summing to 0.
    # My BFS starts at 0 with empty mask.
    # If we find a non-empty subset summing to 0, best_steps[0] will be updated?
    # No, I initialized best_steps[0]=0.
    # I should initialize best_steps to infinity, and start with neighbors.
    # Or, handle r=0 specially.
    
    # Let's restart BFS logic to ensure non-empty for r=0.
    
    pass
