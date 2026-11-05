"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    N = 128
    # Precompute all 4-term arithmetic progressions (a, a+d, a+2d, a+3d) with 0 <= a, a+3d < N
    all_taps = []
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 1 - 3 * d
        for a in range(max_a + 1):
            all_taps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Map each position to the indices of taps it participates in
    pos_taps = [[] for _ in range(N)]
    for i, tap in enumerate(all_taps):
        for p in tap:
            pos_taps[p].append(i)

    def is_valid_coloring(colors):
        """Check if every 4-term progression uses at least 3 distinct colors."""
        for a, b, c, d in all_taps:
            if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                return False
        return True

    def greedy_construct(max_colors, rng=None):
        """
        Greedy left-to-right construction.
        Tries to assign the smallest valid color in 0..max_colors-1 to each position.
        Returns None if stuck.
        """
        if rng is None:
            rng = random.Random()
        
        colors = [-1] * N
        
        for pos in range(N):
            # Determine forbidden colors based on already assigned neighbors in taps
            # If a tap has 3 assigned colors and they are all the same, we are stuck (impossible to reach 3 distinct).
            # If a tap has 3 assigned colors and 2 are same, 1 different, the 4th must be different from the pair?
            # Let S be the set of colors of the 3 assigned neighbors.
            # We need len(S U {c}) >= 3.
            # If |S| == 3: any c works.
            # If |S| == 2: c must not be in S.
            # If |S| == 1: impossible (max distinct would be 2).
            
            forbidden_mask = 0
            stuck = False
            
            for ti in pos_taps[pos]:
                a, b, c, d = all_taps[ti]
                pts = [a, b, c, d]
                others = [p for p in pts if p != pos]
                
                # Check if all others are assigned
                assigned_colors = []
                all_assigned = True
                for p in others:
                    if colors[p] == -1:
                        all_assigned = False
                        break
                    assigned_colors.append(colors[p])
                
                if not all_assigned:
                    continue
                
                s = set(assigned_colors)
                if len(s) == 1:
                    stuck = True
                    break
                elif len(s) == 2:
                    for c in s:
                        forbidden_mask |= (1 << c)
                # len(s) == 3: no restriction
            
            if stuck:
                return None
            
            # Find smallest valid color
            found = False
            for c in range(max_colors):
                if not (forbidden_mask & (1 << c)):
                    colors[pos] = c
                    found = True
                    break
            
            if not found:
                return None
        
        # Final validation
        if not is_valid_coloring(colors):
            return None
        
        return colors

    def try_remove_color(colors, target_color, time_limit=2.0):
        """
        Attempts to remove target_color by recoloring its occurrences to other existing colors.
        Uses backtracking with forward checking.
        """
        start_t = time.time()
        positions = [i for i, c in enumerate(colors) if c == target_color]
        if not positions:
            return None
        
        available_colors = [c for c in set(colors) if c != target_color]
        if not available_colors:
            return None
        
        new_colors = colors[:]
        
        # Heuristic: order positions by number of constraints (most constrained first)
        # Or just simple order. Let's try sorting by degree in tap graph.
        # Degree is len(pos_taps[i]).
        positions.sort(key=lambda i: len(pos_taps[i]), reverse=True)
        
        def is_local_valid(pos, c):
            """Check if assigning c to pos is valid with respect to all taps where pos is involved and others are assigned."""
            for ti in pos_taps[pos]:
                a, b, c_tap, d_tap = all_taps[ti]
                pts = [a, b, c_tap, d_tap]
                others = [p for p in pts if p != pos]
                
                vals = [c]
                for p in others:
                    vals.append(new_colors[p])
                
                if len(set(vals)) < 3:
                    return False
            return True

        def backtrack(idx):
            if time.time() - start_t > time_limit:
                return False
            
            if idx == len(positions):
                # Full check just in case (though local checks should suffice if we check all involved taps)
                return True
            
            pos = positions[idx]
            
            # Try available colors
            # Optimization: shuffle available colors? Or keep deterministic?
            # Deterministic is safer for reproducibility, but random might escape local minima.
            # Let's try available colors in order.
            
            for c in available_colors:
                if c == new_colors[pos]:
                    continue
                
                # Tentative assignment
                new_colors[pos] = c
                
                if is_local_valid(pos, c):
                    if backtrack(idx + 1):
                        return True
                
                # Undo
                new_colors[pos] = target_color
            
            return False
        
        if backtrack(0):
            return new_colors
        return None

    # 1. Generate initial coloring
    # Start with a small number of colors and increase if necessary
    initial_colors = None
    for k in range(3, 10):
        initial_colors = greedy_construct(k)
        if initial_colors is not None:
            break
    
    if initial_colors is None:
        # Fallback to seed-like logic if greedy fails (shouldn't happen for k>=4 usually)
        # Use the seed logic from inspiration 1 as a robust fallback
        colors = []
        for x in range(1, 129):
            found = False
            for color in range(128):
                # Check 4-term APs ending at x (x-1 is 0-indexed x-1)
                # x is 1-indexed in seed, 0-indexed in our list.
                # Seed: x in 1..128. List index x-1.
                # Condition 1: all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                # Condition 2: all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))
                
                cond1 = True
                for d in range(1, (x - 1) // 3 + 1):
                    s = set()
                    s.add(color)
                    for j in (1, 2, 3):
                        idx = x - 1 - j * d
                        if idx < 0: break
                        s.add(colors[idx])
                    if len(s) < 3:
                        cond1 = False
                        break
                
                if not cond1:
                    continue
                
                cond2 = True
                for d in range(1, (x - 1) // 2 + 1):
                    ok = False
                    for j in (1, 2):
                        idx = x - 1 - j * d
                        if idx < 0: break
                        if colors[idx] != color:
                            ok = True
                            break
                    if not ok:
                        cond2 = False
                        break
                
                if cond2:
                    colors.append(color)
                    found = True
                    break
            
            if not found:
                colors.append(0) # Should not happen
        
        initial_colors = colors

    current_colors = initial_colors
    current_count = len(set(current_colors))
    
    # 2. Local Search to reduce color count
    improved = True
    while improved and time.time() < deadline:
        improved = False
        colors_in_use = list(set(current_colors))
        # Sort by frequency (ascending) to try removing rare colors first
        colors_in_use.sort(key=lambda c: current_colors.count(c))
        
        for c in colors_in_use:
            if time.time() > deadline:
                break
            
            # Try to remove color c
            # Use a short time limit for each attempt
            new_colors = try_remove_color(current_colors, c, time_limit=1.0)
            
            if new_colors is not None:
                current_colors = new_colors
                current_count = len(set(current_colors))
                improved = True
                break # Restart loop as colors changed
        
        # If we couldn't remove any color in this pass, try a few random restarts of the local search?
        # Or just break. The greedy construction might have found the minimum for this heuristic.
        # To improve odds, we could try different greedy seeds (randomizing color order) if we have time.
        
        if not improved and time.time() < deadline - 10:
            # Try a few random restarts of the greedy construction with different color orderings
            # to see if we can get a better starting point for color reduction.
            for attempt in range(5):
                if time.time() > deadline:
                    break
                rng = random.Random(attempt)
                # Modify greedy to shuffle color choice?
                # For now, just try increasing max_colors in greedy? No, that increases colors.
                # We want to find a coloring with FEWER colors.
                # If greedy(3) failed, greedy(4) succeeded, can we reduce 4 to 3?
                # The loop above tries to remove colors from the current solution.
                # If current solution uses 4 colors, it tries to remove one.
                # If it fails, we are stuck at 4.
                # Maybe a different 4-coloring allows reduction to 3?
                # Let's try to generate a random 4-coloring and reduce it.
                
                # Randomized greedy: pick a random valid color instead of smallest.
                def greedy_random(max_colors, rng):
                    colors = [-1] * N
                    for pos in range(N):
                        forbidden_mask = 0
                        stuck = False
                        for ti in pos_taps[pos]:
                            a, b, c, d = all_taps[ti]
                            pts = [a, b, c, d]
                            others = [p for p in pts if p != pos]
                            assigned_colors = []
                            all_assigned = True
                            for p in others:
                                if colors[p] == -1:
                                    all_assigned = False
                                    break
                                assigned_colors.append(colors[p])
                            if not all_assigned:
                                continue
                            s = set(assigned_colors)
                            if len(s) == 1:
                                stuck = True
                                break
                            elif len(s) == 2:
                                for cc in s:
                                    forbidden_mask |= (1 << cc)
                        if stuck:
                            return None
                        
                        valid_colors = []
                        for cc in range(max_colors):
                            if not (forbidden_mask & (1 << cc)):
                                valid_colors.append(cc)
                        
                        if not valid_colors:
                            return None
                        
                        colors[pos] = rng.choice(valid_colors)
                    
                    if not is_valid_coloring(colors):
                        return None
                    return colors

                # Try to find a coloring with current_count colors, then reduce
                # Or just try to find a coloring with 3 colors directly?
                # If we can find a valid 3-coloring, we are done (assuming 3 is minimum).
                # Let's check if 3 is possible.
                if current_count > 3:
                    # Try to find a 3-coloring
                    for _ in range(20): # 20 random tries
                        if time.time() > deadline:
                            break
                        c3 = greedy_random(3, rng)
                        if c3 is not None:
                            current_colors = c3
                            current_count = 3
                            improved = True
                            break
                    if improved:
                        break # Restart reduction loop (will fail immediately if 3 is minimal, or succeed if we found 3)

    return current_colors

# EVOLVE-BLOCK-END
