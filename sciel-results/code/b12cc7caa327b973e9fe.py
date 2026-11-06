import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    N = 256
    
    # Precompute APs
    ap_list = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(0, max_a):
            ap_list.append((a, a+d, a+2*d, a+3*d))
            
    if not ap_list:
        return [0] * N
        
    num_aps = len(ap_list)
    
    # Store APs as numpy arrays for faster access? 
    # Or just a list of tuples.
    # Let's use a list of tuples for simplicity, 
    # but we need to check mono-ness fast.
    
    # Map index -> list of AP indices
    ap_map = [[] for _ in range(N)]
    for idx, ap in enumerate(ap_list):
        for pos in ap:
            ap_map[pos].append(idx)
    
    # Initial random coloring
    colors = np.random.randint(0, 2, N)
    
    # Precompute is_mono
    is_mono = np.zeros(num_aps, dtype=bool)
    mono_count = 0
    for k, (a, b, c, d) in enumerate(ap_list):
        if colors[a] == colors[b] == colors[c] == colors[d]:
            is_mono[k] = True
            mono_count += 1
            
    best_count = mono_count
    best_colors = colors.copy()
    
    # Simulated Annealing
    T = 1.0
    cooling_rate = 0.999
    min_T = 1e-4
    
    # Estimate iterations per second to adjust loop
    # We will just run until deadline
    
    while time.time() < deadline:
        if T < min_T:
            # Restart with higher T? Or just stop? 
            # Maybe do a few more greedy steps?
            # Let's just continue with min_T
            pass
            
        i = random.randint(0, N - 1)
        
        # Calculate delta
        delta = 0
        affected_aps = ap_map[i]
        
        # Store old states to revert if needed
        # We only need to revert if we reject.
        # Since we check one by one, we can compute delta first, 
        # then decide, then apply.
        
        # To compute delta, we need to know if the AP was mono 
        # and if it will be mono after flip.
        
        # Flip colors[i] mentally
        new_color = 1 - colors[i]
        
        for k in affected_aps:
            a, b, c, d = ap_list[k]
            # Check if it was mono
            was_mono = is_mono[k]
            
            # Check if it will be mono
            # The AP contains i. 
            # We need to check the other 3 colors and new_color.
            # Identify which position i is in.
            # This is inefficient. 
            # Better: just check all 4, replacing the one at i with new_color.
            
            # Optimization: 
            # If was_mono is False, it becomes mono only if all 4 become equal.
            # If was_mono is True, it becomes non-mono (since we flip one, 
            # and the other 3 were equal to the old color, so they are not equal to new_color).
            # Wait, if was_mono is True, colors[a]==colors[b]==colors[c]==colors[d]==old_color.
            # New color is 1-old_color.
            # So the AP will definitely NOT be mono.
            # So delta = -1.
            
            # If was_mono is False:
            # It becomes mono if new_color == colors[o1] == colors[o2] == colors[o3].
            
            if was_mono:
                delta -= 1
            else:
                # Check if the other 3 are equal to new_color
                # Find the indices of the other 3
                # We can do this by checking the tuple
                # a,b,c,d
                # If i==a, check b,c,d
                # If i==b, check a,c,d
                # etc.
                
                # This branch is taken less often if we start with random?
                # Random mono probability is 1/8. So was_mono is True 12.5% of the time.
                
                # To avoid if/else for position, we can just construct the new set?
                # No, just check:
                # We can store the AP as a tuple and use a helper?
                # Or just:
                # If i == a: c1, c2, c3 = colors[b], colors[c], colors[d]
                # ...
                
                # Let's optimize this check.
                # We know the AP indices.
                # Let's just check the 3 others.
                # We can precompute for each AP and each position in it, the indices of the others?
                # That's 4 * num_aps * 3 integers. ~ 4 * 10000 * 3 = 120k ints. Fine.
                # But ap_list is just tuples.
                
                # Let's just do the check inline.
                # Since i is in the AP, we can find the others.
                # Actually, we can just check:
                # if new_color == colors[a] == colors[b] == colors[c] == colors[d] is not possible 
                # because colors[i] is still old.
                
                # Let's just do:
                # other_colors = [colors[j] for j in (a,b,c,d) if j != i]
                # if all(c == new_color for c in other_colors): delta += 1
                
                # This list comprehension is slow.
                
                # Alternative:
                # We can precompute `others[k]` as a list of 3 lists?
                # Or `others[k][pos]`?
                
                # Let's restructure precomputation.
                # For each AP k, store the 4 indices.
                # And for each position p in the AP, store the indices of the other 3.
                
                # This will be done in the next version if this is too slow.
                # For now, let's try to just run and see.
                # But we have a deadline.
                
                # Let's use a faster check.
                # We know i is one of a,b,c,d.
                # We can just check the 3 others manually.
                # It's a bit of code but faster.
                
                # Let's assume the check is fast enough for ~1M iterations.
                
                pass # Placeholder for logic below

        # This inner loop logic needs to be concrete and fast.
        # I will implement the precomputation of "other indices" to make the check O(1) without branching on position.
        
        # Re-doing precomputation in my head for the final code:
        # ap_others[k] = [ [idx1, idx2, idx3], [idx1, idx2, idx3], ... ] for each of the 4 positions?
        # No, we just need the 3 others for the specific i.
        # Since i is in the AP, we can find which slot it is.
        
        # Actually, simpler:
        # Store APs as (a,b,c,d).
        # In the loop:
        # if i == a: others = (b,c,d)
        # elif i == b: others = (a,c,d)
        # ...
        # This is 4 comparisons.
        
        # Let's write the code with this logic.
        
        # If accepted, update colors and is_mono.
        
    return best_colors.tolist()
