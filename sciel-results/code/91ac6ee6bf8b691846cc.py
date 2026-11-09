import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit (and 90s kill)
    
    # can_reach[s] is True if a subset of selected numbers sums to s
    can_reach = [False] * 201
    can_reach[0] = True
    
    selected = []
    
    # Iterate from largest to smallest
    for x in range(100, 0, -1):
        # Check if we should stop due to time
        if time.time() > deadline:
            break
            
        # Check if adding x would create a sum of 200
        # This happens if there is a subset summing to 200 - x
        if can_reach[200 - x]:
            # Adding x would create a subset summing to 200, so skip x
            continue
            
        # Accept x
        selected.append(x)
        
        # Update reachable sums
        # We must iterate backwards or use a copy to avoid using x multiple times
        # Since we are adding x to existing sums, we need to update based on the state BEFORE adding x
        # But we are checking 200-x first.
        # If we update in place, we might use the new sums to add x again? No, we only add x once.
        # However, updating in place from 200 down to 0 ensures we don't use x multiple times in the sum calculation.
        # Wait, the update is: new_reachable[s+x] = reachable[s]
        # If we iterate s from 200 down to 0:
        # If s+x <= 200: can_reach[s+x] = can_reach[s+x] or can_reach[s]
        # This is correct for 0/1 knapsack style updates.
        
        for s in range(200 - x, -1, -1):
            if can_reach[s]:
                can_reach[s + x] = True
                
    return selected
