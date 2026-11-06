import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Generate initial K32 coloring (Greenwood-Gleason / Clebsch)
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    def get_initial_32():
        log, x = {}, 1
        for e in range(15):
            log[x] = e
            x = mul(x, 2)
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                if i // 16 == j // 16:
                    colors.append(log[(i % 16) ^ (j % 16)] % 3)
                else:
                    colors.append(3)
        return colors

    current_n = 32
    current_colors = get_initial_32()
    best_n = 32
    best_colors = current_colors[:]
    
    # Helper: get index in flat list for edge (i, j) with i < j
    def edge_idx(i, j, n):
        if i > j:
            i, j = j, i
        # Number of edges in rows 0 to i-1
        # Row k has n-1-k edges
        # Sum_{k=0}^{i-1} (n-1-k) = i(n-1) - i(i-1)/2
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    # Helper: compute threats for a given coloring and n
    # Returns list of threats per edge? Or just total?
    # For local search, we need to know if a flip creates a threat.
    # Let's maintain a global threat count and an array of threats per edge.
    # threats[e] = number of w such that C(i,w)==C(j,w)==C(i,j) for edge e=(i,j).
    
    def compute_threats(colors, n):
        threats = [0] * (n * (n - 1) // 2)
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[edge_idx(i, j, n)]
                count = 0
                for k in range(n):
                    if k == i or k == j:
                        continue
                    # Check if (i,k) and (j,k) have color c
                    ik_idx = edge_idx(i, k, n)
                    jk_idx = edge_idx(j, k, n)
                    if colors[ik_idx] == c and colors[jk_idx] == c:
                        count += 1
                threats[edge_idx(i, j, n)] = count
        return threats

    # Initial threats should be 0
    current_threats = compute_threats(current_colors, current_n)
    
    # Helper: try to flip edge (i, j) to color c_new
    # Returns (new_threats_delta, new_threats_list) or None if invalid?
    # We want to minimize total threats.
    # Let's implement a function to get the change in total threats if we flip (i, j) to c_new.
    
    def get_flip_delta(i, j, c_new, colors, threats, n):
        """
        Calculate change in total threats if we change color of (i, j) to c_new.
        Returns delta_threats.
        """
        idx = edge_idx(i, j, n)
        c_old = colors[idx]
        if c_old == c_new:
            return 0
        
        # Current threats for edge (i, j) is threats[idx].
        # These threats are removed.
        delta = -threats[idx]
        
        # Add new threats for color c_new
        new_threats_count = 0
        for k in range(n):
            if k == i or k == j:
                continue
            ik_idx = edge_idx(i, k, n)
            jk_idx = edge_idx(j, k, n)
            if colors[ik_idx] == c_new and colors[jk_idx] == c_new:
                new_threats_count += 1
        
        delta += new_threats_count
        return delta

    # Helper: perform a valid local search step (keep threats at 0)
    # Returns True if a flip was made, False otherwise
    def local_search_step(colors, threats, n):
        # Find an edge with 0 threats
        # Randomly pick an edge
        # Try all 4 colors
        # If any color keeps threats at 0, flip to it (randomly chosen among valid ones)
        
        # To speed up, we can just pick a random edge and try to change it.
        # If it fails, try another.
        
        edges = list(range(n * (n - 1) // 2))
        random.shuffle(edges)
        
        for e in edges:
            # Decode e to (i, j)
            # This is slow. Let's just iterate i, j.
            pass
            
        # Better: iterate i, j
        for i in range(n):
            for j in range(i + 1, n):
                idx = edge_idx(i, j, n)
                c_old = colors[idx]
                valid_colors = []
                for c_new in range(4):
                    if c_new == c_old:
                        continue
                    delta = get_flip_delta(i, j, c_new, colors, threats, n)
                    if delta == 0:
                        valid_colors.append(c_new)
                if valid_colors:
                    c_new = random.choice(valid_colors)
                    # Apply flip
                    colors[idx] = c_new
                    # Update threats array
                    # We need to update threats for all edges involved in triangles with (i, j)
                    # This is complex. Let's just recompute threats if we do many flips?
                    # No, recompute is O(N^3).
                    # Let's update threats incrementally.
                    
                    # Remove old threats for (i, j)
                    threats[idx] = 0
                    
                    # Add new threats for (i, j)
                    new_count = 0
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        ik_idx = edge_idx(i, k, n)
                        jk_idx = edge_idx(j, k, n)
                        if colors[ik_idx] == c_new and colors[jk_idx] == c_new:
                            new_count += 1
                            # This triangle (i, j, k) is now monochromatic.
                            # It contributes to threats of (i, j), (i, k), (j, k).
                            # We already counted for (i, j).
                            # We must increment threats[ik_idx] and threats[jk_idx].
                            threats[ik_idx] += 1
                            threats[jk_idx] += 1
                    threats[idx] = new_count
                    
                    # Wait, the logic above for updating threats[ik_idx] is wrong.
                    # If (i, j, k) becomes monochromatic with color c_new:
                    # 1. Edge (i, j) gains a threat from k. (Handled by threats[idx] = new_count)
                    # 2. Edge (i, k) gains a threat from j. (threats[ik_idx] += 1)
                    # 3. Edge (j, k) gains a threat from i. (threats[jk_idx] += 1)
                    
                    # What about the OLD threats?
                    # The old color c_old might have formed triangles.
                    # We need to REMOVE those contributions.
                    
                    # This incremental update is tricky.
                    # Given N <= 62, N^3 = 238k.
                    # If we recompute threats every 100 flips, it's fast enough?
                    # 100 flips * O(N^2) for delta check? No, delta check is O(N).
                    # So 100 flips is O(N^2) = 3800 ops.
                    # Recompute is O(N^3) = 238k ops.
                    # So recompute is 60x slower.
                    # But incremental update is error-prone.
                    
                    # Let's try a simpler approach:
                    # Just recompute threats after a batch of flips?
                    # Or, since we only care about VALID colorings (threats=0),
                    # we can just check validity after each flip?
                    # Check validity: sum(threats) == 0.
                    # If we maintain threats correctly, sum is O(N^2).
                    
                    # Let's stick to the incremental update but be careful.
                    
                    # Actually, let's just return True and let the caller recompute if needed?
                    # No, we need threats for the next step.
                    
                    # Let's simplify:
                    # We will NOT maintain threats incrementally in the local search.
                    # Instead, we will use a "soft" penalty function and minimize it.
                    # Objective: Total Threats.
                    # We want to reach 0.
                    # We use Simulated Annealing on Total Threats.
                    # This allows us to temporarily increase threats to escape local minima.
                    
                    return True
        return False

    # Revised Strategy: Simulated Annealing on Total Threats
    # State: colors list.
    # Energy: Total Threats (sum of threats array).
    # Move: Flip a random edge to a random color.
    # Delta Energy: Change in total threats.
    
    def get_total_threats_delta(i, j, c_new, colors, threats, n):
        idx = edge_idx(i, j, n)
        c_old = colors[idx]
        if c_old == c_new:
            return 0
        
        # Current threats for this edge are removed
        delta = -threats[idx]
        
        # New threats for this edge are added
        new_threats_count = 0
        for k in range(n):
            if k == i or k == j:
                continue
            ik_idx = edge_idx(i, k, n)
            jk_idx = edge_idx(j, k, n)
            if colors[ik_idx] == c_new and colors[jk_idx] == c_new:
                new_threats_count += 1
                # This new threat also affects neighbors
                # But for TOTAL threats, we just count the triangle once?
                # No, "Total Threats" as defined by sum(threats[e]) counts each triangle 3 times.
                # Triangle (i, j, k) contributes 1 to threats[(i, j)], 1 to threats[(i, k)], 1 to threats[(j, k)].
                # So if we form a new triangle, total threats increases by 3.
                # If we break an old triangle, total threats decreases by 3.
                
                # Wait, my `delta` calculation above only accounted for the edge (i, j).
                # I need to account for the other two edges.
                
                # Let's re-evaluate.
                # Old state: Edge (i, j) had color c_old.
                # If c_old formed a triangle with k, it contributed 1 to threats[(i, j)].
                # And it contributed 1 to threats[(i, k)] and 1 to threats[(j, k)].
                # So removing c_old removes 3 from total threats (if it was a triangle).
                
                # New state: Edge (i, j) has color c_new.
                # If c_new forms a triangle with k, it adds 1 to threats[(i, j)], 1 to threats[(i, k)], 1 to threats[(j, k)].
                # So adding c_new adds 3 to total threats (if it forms a triangle).
                
                # So delta_total = -3 * (old_triangles_broken) + 3 * (new_triangles_formed).
                
                # How many old triangles are broken?
                # For each k, if colors[ik_idx]==c_old and colors[jk_idx]==c_old, then (i, j, k) was a triangle.
                # How many new triangles are formed?
                # For each k, if colors[ik_idx]==c_new and colors[jk_idx]==c_new, then (i, j, k) is a new triangle.
                
                pass
        
        # Let's implement this correctly.
        old_tri = 0
        new_tri = 0
        for k in range(n):
            if k == i or k == j:
                continue
            ik_idx = edge_idx(i, k, n)
            jk_idx = edge_idx(j, k, n)
            c_ik = colors[ik_idx]
            c_jk = colors[jk_idx]
            
            if c_ik == c_old and c_jk == c_old:
                old_tri += 1
            if c_ik == c_new and c_jk == c_new:
                new_tri += 1
        
        return 3 * (new_tri - old_tri)

    # Now we can do SA properly.
    
    def try_extend(n, colors, threats):
        """
        Try to extend K_n to K_{n+1}.
        Returns (success, new_colors, new_threats)
        """
        new_n = n + 1
        # New colors list
        # Old edges are at the same indices?
        # No, the index formula depends on N.
        # We need to map old colors to new indices.
        
        # This is a pain.
        # Alternative: Keep the 2D matrix representation internally?
        # Or just rebuild the flat list.
        
        # Let's rebuild.
        # Old N = n. New N = n+1.
        # Edges (i, j) with i, j < n:
        # Old idx: i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
        # New idx: i * (new_n - 1) - i * (i - 1) // 2 + (j - i - 1)
        
        new_colors = [0] * (new_n * (new_n - 1) // 2)
        
        # Copy old edges
        for i in range(n):
            for j in range(i + 1, n):
                old_idx = edge_idx(i, j, n)
                new_idx = edge_idx(i, j, new_n)
                new_colors[new_idx] = colors[old_idx]
        
        # Initialize new edges (n, k) for k < n randomly
        for k in range(n):
            new_idx = edge_idx(k, n, new_n)
            new_colors[new_idx] = random.randint(0, 3)
        
        # Compute threats for new coloring
        new_threats = compute_threats(new_colors, new_n)
        
        # Now we want to minimize threats to 0.
        # We only care about edges involving the new vertex n?
        # No, changing an old edge might help.
        # But changing old edges is risky (might break validity of old part).
        # However, if we use SA, we can allow temporary invalidity.
        
        # Let's run a short SA to minimize threats.
        # Temperature starts high, cools down.
        # We want to reach 0 threats.
        
        current_energy = sum(new_threats)
        T = 1.0
        T_min = 0.01
        steps = 2000
        step_size = 1.0 / steps
        
        for s in range(steps):
            if time.time() > deadline - 5:
                break
            if current_energy == 0:
                return True, new_colors, new_threats
            
            # Pick random edge
            e = random.randint(0, len(new_colors) - 1)
            # Decode e to (i, j)
            # This is slow. Let's precompute a list of (i, j) for the current N?
            # Or just iterate.
            # For N=50, 1225 edges.
            # Decoding:
            # i = 0
            # remaining = e
            # while remaining >= new_n - 1 - i:
            #    remaining -= new_n - 1 - i
            #    i += 1
            # j = i + 1 + remaining
            
            i = 0
            rem = e
            while rem >= new_n - 1 - i:
                rem -= new_n - 1 - i
                i += 1
            j = i + 1 + rem
            
            c_old = new_colors[e]
            c_new = random.randint(0, 3)
            if c_old == c_new:
                continue
                
            delta = get_total_threats_delta(i, j, c_new, new_colors, new_threats, new_n)
            
            if delta < 0 or random.random() < pow(2.718, -delta / T):
                # Accept
                new_colors[e] = c_new
                # Update threats
                # We need to update new_threats array.
                # This is the same incremental update problem.
                # Let's just recompute threats every 100 steps?
                # Or use the delta to update the SUM, but we need the array for future deltas.
                # If we don't update the array, future deltas are wrong.
                
                # Let's recompute threats every 50 steps.
                if s % 50 == 0:
                    new_threats = compute_threats(new_colors, new_n)
                    current_energy = sum(new_threats)
                else:
                    # Approximate energy update
                    current_energy += delta
                    # This is dangerous if array is stale.
                    # But for SA, it might be okay if we recompute periodically.
                    
            T -= step_size
            if T < T_min:
                T = T_min
        
        if sum(new_threats) == 0:
            return True, new_colors, new_threats
        
        return False, new_colors, new_threats

    # Main Loop
    while current_n < 62 and time.time() < deadline - 10:
        success, new_colors, new_threats = try_extend(current_n, current_colors, current_threats)
        if success:
            current_n += 1
            current_colors = new_colors
            current_threats = new_threats
            if current_n > best_n:
                best_n = current_n
                best_colors = current_colors[:]
        else:
            # Failed to extend.
            # Try to perturb current coloring to escape local minimum.
            # Run SA on current_n to minimize threats (should be 0).
            # If it finds a different valid coloring, maybe it's easier to extend.
            
            # Let's just restart the extension with a different random seed?
            # Or perturb the new vertex edges.
            
            # For now, just break and return best found.
            # To improve, we could try multiple random initializations for the new vertex.
            
            # Let's try a few more times with different random initializations for the new vertex.
            found = False
            for attempt in range(10):
                # Re-initialize new vertex edges randomly
                # This is done inside try_extend.
                # Let's just call try_extend again.
                success, new_colors, new_threats = try_extend(current_n, current_colors, current_threats)
                if success:
                    current_n += 1
                    current_colors = new_colors
                    current_threats = new_threats
                    if current_n > best_n:
                        best_n = current_n
                        best_colors = current_colors[:]
                    found = True
                    break
                if time.time() > deadline - 5:
                    break
            
            if not found:
                # Stuck.
                # Try to perturb the existing coloring.
                # We can do a random walk on valid colorings of K_current_n.
                # This is complex.
                # Let's just break.
                break

    return {'n': best_n, 'colors': best_colors}

# Test
# print(construct())
