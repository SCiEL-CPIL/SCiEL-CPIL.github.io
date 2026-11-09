import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s margin
    
    n = 128
    # Precompute all 4-term APs
    aps_list = []
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            aps_list.append((a - 1, a - 1 + d, a - 1 + 2 * d, a - 1 + 3 * d))
    
    aps = np.array(aps_list, dtype=np.int32)  # shape (num_aps, 4)
    num_aps = len(aps)
    
    # Map index to list of AP indices containing it
    idx_to_aps = [[] for _ in range(n)]
    for i, ap in enumerate(aps_list):
        for idx in ap:
            idx_to_aps[idx].append(i)
    
    # Convert to numpy arrays for faster access
    idx_to_aps_arr = [np.array(lst, dtype=np.int32) for lst in idx_to_aps]
    
    # Precompute for each index i, for each position p in 0..3, which APs contain i at position p
    # This helps in fast delta computation
    # For a given index i, we have a list of (ap_idx, pos_in_ap)
    idx_to_aps_pos = [[] for _ in range(n)]
    for i, ap in enumerate(aps_list):
        for pos, idx in enumerate(ap):
            idx_to_aps_pos[idx].append((i, pos))
    
    # Convert to arrays: for each index, two arrays: ap_indices and positions
    idx_to_aps_pos_arr = []
    for i in range(n):
        if idx_to_aps_pos[i]:
            ap_idxs = np.array([x[0] for x in idx_to_aps_pos[i]], dtype=np.int32)
            positions = np.array([x[1] for x in idx_to_aps_pos[i]], dtype=np.int32)
            idx_to_aps_pos_arr.append((ap_idxs, positions))
        else:
            idx_to_aps_pos_arr.append((np.array([], dtype=np.int32), np.array([], dtype=np.int32)))

    def count_bad_colors(colors):
        """Count APs with <= 2 distinct colors using vectorization."""
        c0 = colors[aps[:, 0]]
        c1 = colors[aps[:, 1]]
        c2 = colors[aps[:, 2]]
        c3 = colors[aps[:, 3]]
        
        # A 4-tuple has <= 2 distinct colors iff it has >= 2 matching pairs
        matches = ((c0 == c1).astype(np.int8) + 
                   (c0 == c2).astype(np.int8) + 
                   (c0 == c3).astype(np.int8) + 
                   (c1 == c2).astype(np.int8) + 
                   (c1 == c3).astype(np.int8) + 
                   (c2 == c3).astype(np.int8))
        
        bad = matches >= 2
        return int(np.sum(bad))

    def compute_delta_fast(colors, i, new_color):
        """Compute delta in bad count when changing colors[i] to new_color."""
        ap_idxs, positions = idx_to_aps_pos_arr[i]
        m = len(ap_idxs)
        if m == 0:
            return 0
        
        # Get the relevant APs
        sub_aps = aps[ap_idxs]  # shape (m, 4)
        
        # Get current colors for these APs
        c0 = colors[sub_aps[:, 0]]
        c1 = colors[sub_aps[:, 1]]
        c2 = colors[sub_aps[:, 2]]
        c3 = colors[sub_aps[:, 3]]
        
        # Determine which position in each AP is index i
        pos_mask_0 = (positions == 0)
        pos_mask_1 = (positions == 1)
        pos_mask_2 = (positions == 2)
        pos_mask_3 = (positions == 3)
        
        # Old bad status
        old_matches = ((c0 == c1).astype(np.int8) + 
                       (c0 == c2).astype(np.int8) + 
                       (c0 == c3).astype(np.int8) + 
                       (c1 == c2).astype(np.int8) + 
                       (c1 == c3).astype(np.int8) + 
                       (c2 == c3).astype(np.int8))
        old_bad = old_matches >= 2
        
        # New colors: replace the position corresponding to i with new_color
        nc0 = np.where(pos_mask_0, new_color, c0)
        nc1 = np.where(pos_mask_1, new_color, c1)
        nc2 = np.where(pos_mask_2, new_color, c2)
        nc3 = np.where(pos_mask_3, new_color, c3)
        
        new_matches = ((nc0 == nc1).astype(np.int8) + 
                       (nc0 == nc2).astype(np.int8) + 
                       (nc0 == nc3).astype(np.int8) + 
                       (nc1 == nc2).astype(np.int8) + 
                       (nc1 == nc3).astype(np.int8) + 
                       (nc2 == nc3).astype(np.int8))
        new_bad = new_matches >= 2
        
        return int(np.sum(new_bad) - np.sum(old_bad))

    def greedy_with_lookahead(k, look_depth=4):
        """
        Greedy assignment with look-ahead.
        Assigns colors to positions 0..n-1 in order.
        For each position, tries colors 0..k-1.
        For each color, simulates assigning random colors to the next 'look_depth' positions
        and checks if this leads to a "dead-end" (i.e., high local cost or infeasibility).
        Actually, a simpler look-ahead: check if the chosen color creates a situation
        where a future AP *must* be bad.
        
        Let's refine: 
        The constraint is on 4-term APs. 
        When we assign color to x, we complete APs where x is the last element (x = a+3d).
        We also create "partial" APs where x is the first, second, or third element.
        
        Look-ahead strategy:
        For candidate color c at position x:
        1. Check all APs where x is the last element (fully determined). If any is bad, reject c.
        2. For APs where x is not the last, we have partial info.
           We can estimate the "pressure" on future colors.
           
        A simpler effective look-ahead:
        Assign color c to x.
        Then, for the next L positions (x+1, ..., x+L), try to assign them greedily 
        using only colors 0..k-1, but prioritize colors that don't create new bad APs 
        involving already assigned positions.
        If we get stuck (no valid color for some next position without creating a bad AP 
        with previously assigned positions), then the choice c was bad.
        
        However, "valid color" in the lookahead is tricky because we haven't assigned 
        all positions yet.
        
        Alternative Look-ahead:
        Just check for immediate conflicts with "almost complete" APs.
        If x is the 3rd element of an AP (a, a+d, a+2d=x, a+3d), then the last element 
        is x+d. If we assign c to x, and the first two are c1, c2, then the last element 
        must be such that {c1, c2, c, c_last} has >= 3 colors.
        If c1==c2==c, then c_last must be != c. (1 constraint)
        If c1==c2!=c, then c_last can be anything? No. {c1, c1, c, c_last}. 
           If c_last == c1, colors are {c1, c}. 2 colors. Bad.
           If c_last == c, colors are {c1, c}. 2 colors. Bad.
           So c_last must be different from both c1 and c. (2 constraints)
        If c1!=c2, c1!=c, c2!=c (all distinct so far), then any c_last works?
           {c1, c2, c, c_last}. Already 3 distinct. So yes, any c_last works.
           
        So, for each AP where x is the 3rd element, we can determine the set of forbidden 
        colors for the 4th element (x+d).
        
        Similarly, if x is the 2nd element (a, a+d=x, a+2d, a+3d), we have 2 unknowns.
        If x is the 1st element (a=x, a+d, a+2d, a+3d), we have 3 unknowns.
        
        The look-ahead can be:
        For each candidate color c at x:
          1. Check completed APs (x is 4th). Reject if bad.
          2. For APs where x is 3rd, compute forbidden colors for x+d.
          3. For APs where x is 2nd, compute constraints on (x+d, x+2d).
          4. For APs where x is 1st, compute constraints on (x+d, x+2d, x+3d).
          
          Then, for the next few positions (x+1, ..., x+look_depth), check if there exists 
          a color assignment that satisfies all these constraints.
          
          This is essentially a small CSP check.
          
          To keep it fast, we can just check if the number of forbidden colors for any 
          immediate next position exceeds k.
          
          Let's implement a simple version:
          For candidate c at x:
            - Check completed APs.
            - For each AP where x is 3rd (i.e., next pos is x+d), add forbidden colors to a set F[x+d].
            - For each AP where x is 2nd (next pos x+d), we have constraints on x+d and x+2d.
              This is more complex. Let's just handle the 3rd -> 4th case for now, 
              as it's the most restrictive immediate constraint.
              
          If for any position y in x+1..x+look_depth, |F[y]| == k, then no color is available 
          for y, so c is a bad choice.
          
          We also need to consider that y might be constrained by multiple APs.
          
          Let's refine the lookahead to include the 2nd element case as well, by simply 
          checking if the constraints are satisfiable for the next 1 step.
          
          Actually, let's just use the "stuck" check:
          Simulate the next L steps. For each step, pick the color with the fewest conflicts.
          If at any step no color is available (all k colors are forbidden by constraints 
          from previously assigned positions in this simulation), then the initial choice 
          was bad.
          
          This is a heuristic.
        """
        colors = np.full(n, -1, dtype=np.int32)
        
        for x in range(n):
            best_c = -1
            best_score = float('inf')
            
            for c in range(k):
                # Check completed APs (x is the 4th element)
                valid = True
                # Find APs where x is the 4th element: a+3d = x => a = x-3d
                for d in range(1, (x // 3) + 1):
                    a = x - 3 * d
                    if a < 0:
                        break
                    ap = (a, a + d, a + 2 * d, x)
                    c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], c
                    # Check if <= 2 distinct colors
                    if len({c0, c1, c2, c3}) <= 2:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Look-ahead: simulate next 'look_depth' positions
                # We need to track constraints on future positions.
                # For simplicity, we'll just check if choosing c creates a "tight" situation.
                # Let's count how many future positions would have all colors forbidden.
                
                # Create a copy of colors for simulation
                sim_colors = colors.copy()
                sim_colors[x] = c
                
                # Constraints on future positions
                # We'll use a dictionary: pos -> set of forbidden colors
                forbidden = {pos: set() for pos in range(x + 1, min(x + look_depth + 1, n))}
                
                # Process APs where x is 1st, 2nd, or 3rd
                # x is 3rd: a+2d = x => a = x-2d. 4th is x+d.
                for d in range(1, (x // 2) + 1):
                    a = x - 2 * d
                    if a < 0:
                        break
                    next_pos = x + d
                    if next_pos < n and next_pos in forbidden:
                        c0, c1, c2 = colors[a], colors[a + d], c
                        # Forbidden colors for c3 such that {c0, c1, c2, c3} has <= 2 colors
                        # If c0==c1==c2, c3 must be != c0
                        if c0 == c1 == c2:
                            forbidden[next_pos].add(c0)
                        # If c0==c1!=c2, c3 must be != c0 and != c2
                        elif c0 == c1 != c2:
                            forbidden[next_pos].add(c0)
                            forbidden[next_pos].add(c2)
                        elif c0 == c2 != c1:
                            forbidden[next_pos].add(c0)
                            forbidden[next_pos].add(c1)
                        elif c1 == c2 != c0:
                            forbidden[next_pos].add(c1)
                            forbidden[next_pos].add(c0)
                        # If all distinct, no restriction
                        # If two are same, handled above.
                        
                # x is 2nd: a+d = x => a = x-d. 3rd is x+d, 4th is x+2d.
                # This constrains x+d and x+2d jointly.
                # For a simple lookahead, we might just add constraints to x+d.
                # If c0 (at a) is known, and c1 (at x) is c.
                # Then {c0, c, c2, c3} must have >= 3 colors.
                # If c0 == c, then c2, c3 must provide a 3rd color.
                # This is complex. Let's skip detailed joint constraints for now 
                # and rely on the 3rd->4th and completed checks.
                
                # Check if any position in the lookahead window is completely blocked
                score = 0
                for pos in range(x + 1, min(x + look_depth + 1, n)):
                    if pos in forbidden:
                        # Number of available colors
                        avail = k - len(forbidden[pos])
                        if avail <= 0:
                            score = float('inf')
                            break
                        # Prefer colors that leave more options
                        score += max(0, 3 - avail)  # Heuristic penalty
                
                if score < best_score:
                    best_score = score
                    best_c = c
            
            if best_c == -1:
                # No valid color found, fallback to random
                best_c = random.randint(0, k - 1)
            
            colors[x] = best_c
            
        return colors

    best_solution = None
    best_cost = float('inf')
    
    # Try k = 3, 4, 5...
    for k in range(3, 10):
        if time.time() > deadline - 10:
            break
            
        # Local search for this k
        best_k_solution = None
        best_k_cost = float('inf')
        
        # 1. Try Greedy with Lookahead first
        if time.time() < deadline - 5:
            g_colors = greedy_with_lookahead(k, look_depth=6)
            g_cost = count_bad_colors(g_colors)
            if g_cost < best_k_cost:
                best_k_cost = g_cost
                best_k_solution = g_colors.copy()
            
            if best_k_cost == 0:
                return [int(c) for c in best_k_solution]
        
        # 2. Simulated Annealing
        max_restarts = 80
        restart_count = 0
        while time.time() < deadline - 5 and restart_count < max_restarts:
            restart_count += 1
            
            # Random initialization
            colors = np.random.randint(0, k, n).astype(np.int32)
            current_cost = count_bad_colors(colors)
            
            if current_cost == 0:
                best_k_solution = colors.copy()
                best_k_cost = 0
                break
                
            if current_cost < best_k_cost:
                best_k_cost = current_cost
                best_k_solution = colors.copy()
            
            # Simulated Annealing with adaptive parameters
            T = 3.0
            T_min = 0.01
            cooling = 0.9997
            no_improve = 0
            max_no_improve = 800
            max_iters = 30000
            iter_count = 0
            
            for iter_count in range(max_iters):
                if time.time() > deadline - 5:
                    break
                    
                if current_cost == 0:
                    best_k_solution = colors.copy()
                    best_k_cost = 0
                    break
                    
                # Pick random index
                i = np.random.randint(n)
                old_color = colors[i]
                new_color = np.random.randint(0, k)
                if new_color == old_color:
                    continue
                    
                # Calculate delta cost
                delta = compute_delta_fast(colors, i, new_color)
                
                if delta < 0:
                    colors[i] = new_color
                    current_cost += delta
                    no_improve = 0
                else:
                    # Accept with probability exp(-delta/T)
                    if random.random() < np.exp(-delta / T):
                        colors[i] = new_color
                        current_cost += delta
                    else:
                        no_improve += 1
                        if no_improve > max_no_improve:
                            break
                            
                # Cool down
                T *= cooling
                if T < T_min:
                    T = T_min

            if current_cost < best_k_cost:
                best_k_cost = current_cost
                best_k_solution = colors.copy()
                
            if best_k_cost == 0:
                break

        # Final polish: exact local search (only accept improvements)
        if best_k_solution is not None and best_k_cost > 0 and time.time() < deadline - 2:
            colors = best_k_solution.copy()
            current_cost = best_k_cost
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                for i in range(n):
                    old_color = colors[i]
                    for new_color in range(k):
                        if new_color == old_color:
                            continue
                        delta = compute_delta_fast(colors, i, new_color)
                        if delta < 0:
                            colors[i] = new_color
                            current_cost += delta
                            improved = True
                            break
                    if time.time() > deadline - 2:
                        break
            if current_cost < best_k_cost:
                best_k_cost = current_cost
                best_k_solution = colors.copy()

        if best_k_cost == 0:
            # Found a valid solution with k colors
            # Convert to list of ints
            return [int(c) for c in best_k_solution]
            
    # Fallback: Greedy construction similar to seed
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color);break
    return colors
