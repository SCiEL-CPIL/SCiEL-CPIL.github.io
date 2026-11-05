import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    N = 128
    max_colors = 3  # Try 3 colors first
    
    # Precompute all 4-APs
    # A 4-AP is (a, a+d, a+2d, a+3d) with a+3d < N
    # Store as list of tuples (a, d)
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            aps.append((a, d))
            
    num_aps = len(aps)
    
    # Map position to list of AP indices containing it
    pos_to_aps = [[] for _ in range(N)]
    for i, (a, d) in enumerate(aps):
        pos = a
        for _ in range(4):
            pos_to_aps[pos].append(i)
            pos += d
            
    # Initialize colors randomly
    colors = [random.randint(0, max_colors - 1) for _ in range(N)]
    
    # Helper to calculate distinct colors count for an AP
    def get_distinct_colors(a, d):
        c1 = colors[a]
        c2 = colors[a + d]
        c3 = colors[a + 2 * d]
        c4 = colors[a + 3 * d]
        return len({c1, c2, c3, c4})
        
    # Calculate initial bad APs
    # bad_aps[i] = 1 if distinct colors < 3, else 0
    bad_aps = [0] * num_aps
    bad_count = 0
    for i, (a, d) in enumerate(aps):
        dc = get_distinct_colors(a, d)
        if dc < 3:
            bad_aps[i] = 1
            bad_count += 1
            
    if bad_count == 0:
        return colors
        
    # Local Search
    # We want to minimize bad_count.
    # Move: pick a bad AP, pick a position in it, change color.
    
    # To speed up, we can prioritize APs that are "very bad" (1 color) vs "less bad" (2 colors)?
    # Or just random.
    
    # We will use a simple hill climbing / simulated annealing hybrid.
    # If we get stuck, we do random restarts or larger jumps.
    
    best_colors = colors[:]
    best_bad_count = bad_count
    
    # Precompute positions for each AP for faster access
    ap_positions = [[a, a+d, a+2*d, a+3*d] for a, d in aps]
    
    # Main loop
    while time.time() < deadline:
        if bad_count == 0:
            return colors
            
        # Pick a random bad AP
        # Instead of building a list of bad AP indices every time (O(N)), 
        # we can just pick a random AP and check if it's bad.
        # Or maintain a list of bad AP indices.
        
        # Let's maintain a list of bad AP indices for efficiency if bad_count is small.
        # But bad_count might be large initially.
        # Let's just pick random APs until we find a bad one.
        
        found_bad_ap = -1
        for _ in range(100): # Try to find a bad AP
            idx = random.randint(0, num_aps - 1)
            if bad_aps[idx]:
                found_bad_ap = idx
                break
                
        if found_bad_ap == -1:
            # If we couldn't find a bad AP in 100 tries, bad_count is very small.
            # Re-scan to find one.
            for i in range(num_aps):
                if bad_aps[i]:
                    found_bad_ap = i
                    break
            if found_bad_ap == -1:
                return colors # Should be 0
                
        a, d = aps[found_bad_ap]
        positions = ap_positions[found_bad_ap]
        
        # Pick a position to change
        pos_idx = random.choice(positions)
        old_color = colors[pos_idx]
        
        # Try all other colors
        new_color = -1
        best_delta = 0 # We want to decrease bad_count, so delta < 0 is good
        
        for c in range(max_colors):
            if c == old_color:
                continue
                
            # Calculate delta in bad_count
            # Only APs containing pos_idx are affected
            delta = 0
            for ap_idx in pos_to_aps[pos_idx]:
                # Check status before
                old_bad = bad_aps[ap_idx]
                
                # Simulate change
                # We need to know the distinct colors of this AP
                # Efficiently:
                # If ap_idx == found_bad_ap, we know it's bad.
                # But other APs might be good or bad.
                
                # To avoid full recalculation, we can just recalc distinct colors for affected APs.
                # Since N is small, recalculation is cheap.
                
                aa, dd = aps[ap_idx]
                # Get current colors
                c0 = colors[aa]
                c1 = colors[aa + dd]
                c2 = colors[aa + 2 * dd]
                c3 = colors[aa + 3 * dd]
                
                # Apply change
                if aa == pos_idx: c0 = c
                elif aa + dd == pos_idx: c1 = c
                elif aa + 2 * dd == pos_idx: c2 = c
                elif aa + 3 * dd == pos_idx: c3 = c
                
                new_dc = len({c0, c1, c2, c3})
                new_bad = 1 if new_dc < 3 else 0
                
                delta += new_bad - old_bad
                
            if delta < best_delta:
                best_delta = delta
                new_color = c
                
        # If no improvement, try a random change to escape local minima
        if new_color == -1:
            # Random move
            possible_colors = [c for c in range(max_colors) if c != old_color]
            new_color = random.choice(possible_colors)
            
            # Calculate actual delta for this random move
            delta = 0
            for ap_idx in pos_to_aps[pos_idx]:
                old_bad = bad_aps[ap_idx]
                aa, dd = aps[ap_idx]
                c0 = colors[aa]
                c1 = colors[aa + dd]
                c2 = colors[aa + 2 * dd]
                c3 = colors[aa + 3 * dd]
                
                if aa == pos_idx: c0 = new_color
                elif aa + dd == pos_idx: c1 = new_color
                elif aa + 2 * dd == pos_idx: c2 = new_color
                elif aa + 3 * dd == pos_idx: c3 = new_color
                
                new_dc = len({c0, c1, c2, c3})
                new_bad = 1 if new_dc < 3 else 0
                delta += new_bad - old_bad
            
        # Apply change
        colors[pos_idx] = new_color
        
        # Update bad_aps and bad_count
        for ap_idx in pos_to_aps[pos_idx]:
            aa, dd = aps[ap_idx]
            c0 = colors[aa]
            c1 = colors[aa + dd]
            c2 = colors[aa + 2 * dd]
            c3 = colors[aa + 3 * dd]
            new_dc = len({c0, c1, c2, c3})
            new_bad = 1 if new_dc < 3 else 0
            
            if new_bad != bad_aps[ap_idx]:
                bad_count += new_bad - bad_aps[ap_idx]
                bad_aps[ap_idx] = new_bad
                
        if bad_count < best_bad_count:
            best_bad_count = bad_count
            best_colors = colors[:]
            
        # Optional: If stuck for a long time, restart or expand color space
        # For now, we rely on the random moves to escape.
        
    # If we didn't find a solution with 3 colors, try 4?
    # The task says "Minimize the number of labels". 
    # If 3 is impossible, we should return a 4-coloring if found.
    # But the local search above only uses 3.
    # If best_bad_count > 0, we should try 4 colors.
    
    if best_bad_count > 0:
        # Restart with 4 colors
        max_colors = 4
        colors = [random.randint(0, max_colors - 1) for _ in range(N)]
        bad_aps = [0] * num_aps
        bad_count = 0
        for i, (a, d) in enumerate(aps):
            dc = get_distinct_colors(a, d)
            if dc < 3:
                bad_aps[i] = 1
                bad_count += 1
                
        best_colors = colors[:]
        best_bad_count = bad_count
        
        while time.time() < deadline:
            if bad_count == 0:
                return colors
                
            found_bad_ap = -1
            for _ in range(100):
                idx = random.randint(0, num_aps - 1)
                if bad_aps[idx]:
                    found_bad_ap = idx
                    break
            if found_bad_ap == -1:
                for i in range(num_aps):
                    if bad_aps[i]:
                        found_bad_ap = i
                        break
                if found_bad_ap == -1:
                    return colors
                    
            a, d = aps[found_bad_ap]
            positions = ap_positions[found_bad_ap]
            pos_idx = random.choice(positions)
            old_color = colors[pos_idx]
            
            new_color = -1
            best_delta = 0
            
            for c in range(max_colors):
                if c == old_color:
                    continue
                delta = 0
                for ap_idx in pos_to_aps[pos_idx]:
                    old_bad = bad_aps[ap_idx]
                    aa, dd = aps[ap_idx]
                    c0 = colors[aa]
                    c1 = colors[aa + dd]
                    c2 = colors[aa + 2 * dd]
                    c3 = colors[aa + 3 * dd]
                    
                    if aa == pos_idx: c0 = c
                    elif aa + dd == pos_idx: c1 = c
                    elif aa + 2 * dd == pos_idx: c2 = c
                    elif aa + 3 * dd == pos_idx: c3 = c
                    
                    new_dc = len({c0, c1, c2, c3})
                    new_bad = 1 if new_dc < 3 else 0
                    delta += new_bad - old_bad
                    
                if delta < best_delta:
                    best_delta = delta
                    new_color = c
                    
            if new_color == -1:
                possible_colors = [c for c in range(max_colors) if c != old_color]
                new_color = random.choice(possible_colors)
                delta = 0
                for ap_idx in pos_to_aps[pos_idx]:
                    old_bad = bad_aps[ap_idx]
                    aa, dd = aps[ap_idx]
                    c0 = colors[aa]
                    c1 = colors[aa + dd]
                    c2 = colors[aa + 2 * dd]
                    c3 = colors[aa + 3 * dd]
                    
                    if aa == pos_idx: c0 = new_color
                    elif aa + dd == pos_idx: c1 = new_color
                    elif aa + 2 * dd == pos_idx: c2 = new_color
                    elif aa + 3 * dd == pos_idx: c3 = new_color
                    
                    new_dc = len({c0, c1, c2, c3})
                    new_bad = 1 if new_dc < 3 else 0
                    delta += new_bad - old_bad
                    
            colors[pos_idx] = new_color
            for ap_idx in pos_to_aps[pos_idx]:
                aa, dd = aps[ap_idx]
                c0 = colors[aa]
                c1 = colors[aa + dd]
                c2 = colors[aa + 2 * dd]
                c3 = colors[aa + 3 * dd]
                new_dc = len({c0, c1, c2, c3})
                new_bad = 1 if new_dc < 3 else 0
                if new_bad != bad_aps[ap_idx]:
                    bad_count += new_bad - bad_aps[ap_idx]
                    bad_aps[ap_idx] = new_bad
                    
            if bad_count < best_bad_count:
                best_bad_count = bad_count
                best_colors = colors[:]
                
        return best_colors

    return best_colors
