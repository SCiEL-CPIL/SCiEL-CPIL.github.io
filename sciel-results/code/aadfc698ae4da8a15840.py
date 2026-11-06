import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 128
    
    # Precompute all non-constant 4-term APs
    # AP is (a, a+d, a+2d, a+3d)
    # a >= 0, d >= 1, a+3d < n
    aps = []
    for a in range(n):
        max_d = (n - 1 - a) // 3
        for d in range(1, max_d + 1):
            i0 = a
            i1 = a + d
            i2 = a + 2 * d
            i3 = a + 3 * d
            # All indices are distinct because d >= 1
            aps.append((i0, i1, i2, i3))
    
    # Map position to list of AP indices containing it
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    def count_violations(colors):
        count = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                count += 1
        return count

    def delta_violation(pos, old_color, new_color, colors):
        """
        Calculate change in violation count if colors[pos] changes from old_color to new_color.
        Returns (delta, new_violation_count_for_affected_aps)
        """
        delta = 0
        for ap_idx in pos_to_aps[pos]:
            ap = aps[ap_idx]
            # Get colors of the AP with old_color at pos
            # We need to check if the AP was violated before and after
            
            # Efficiently check violation status
            # Since only one element changes, we can just evaluate the set size
            
            # Before
            c0 = colors[ap[0]]
            c1 = colors[ap[1]]
            c2 = colors[ap[2]]
            c3 = colors[ap[3]]
            
            # Create a temporary state for the AP
            # We can just compute the set size for the 'before' and 'after' states
            
            # Before state
            before_set = {c0, c1, c2, c3}
            was_violated = len(before_set) < 3
            
            # After state
            # Replace color at pos
            # Note: ap contains pos. We need to identify which index in ap is pos.
            # This is inefficient. Better to precompute the index of pos in each ap?
            # Or just construct the new tuple.
            
            # Let's optimize:
            # Store ap as tuple of indices.
            # To get 'after' colors, we iterate ap indices.
            
            # Actually, let's just re-calculate the set for the 4 positions.
            # It's only 4 elements.
            
            new_colors = [c0, c1, c2, c3]
            # Find which index in ap corresponds to pos
            # ap[0] is i0, ap[1] is i1, etc.
            # pos is one of them.
            
            # To avoid finding the index every time, we can precompute for each AP and each position in it, the index.
            # But that's overkill.
            
            # Simpler:
            # before_violated is known.
            # Construct after colors
            after_c0 = new_color if ap[0] == pos else c0
            after_c1 = new_color if ap[1] == pos else c1
            after_c2 = new_color if ap[2] == pos else c2
            after_c3 = new_color if ap[3] == pos else c3
            
            after_set = {after_c0, after_c1, after_c2, after_c3}
            now_violated = len(after_set) < 3
            
            if was_violated and not now_violated:
                delta -= 1
            elif not was_violated and now_violated:
                delta += 1
        
        return delta

    # Try increasing number of colors
    best_colors = None
    best_violations = float('inf')
    best_k = 128
    
    # We want to find the minimum k.
    # Start with small k.
    for k in range(3, 129):
        if time.time() > deadline - 15:
            break
            
        # Run multiple restarts for this k
        found_valid = False
        restarts = 50 if k <= 5 else 20
        
        for r in range(restarts):
            if time.time() > deadline - 10:
                break
                
            # Initialize randomly
            colors = [random.randint(0, k - 1) for _ in range(n)]
            current_violations = count_violations(colors)
            
            if current_violations == 0:
                best_colors = colors
                best_k = k
                found_valid = True
                break
            
            best_in_run = colors[:]
            best_violations_in_run = current_violations
            
            # Simulated Annealing
            temp = 1.0
            cooling = 0.9999
            max_steps = 20000
            steps = 0
            
            while steps < max_steps and time.time() < deadline - 5:
                steps += 1
                
                # Select position
                # Strategy: Pick random position, but with bias towards high violation count?
                # For simplicity and speed, random is okay, but "most violated" is better.
                # Let's stick to random for now, but maybe weight by number of APs?
                # Actually, let's try picking a position from a random violating AP.
                # If no violations, we are done (but we check that before loop).
                
                # To pick a position involved in a violation:
                # We don't have a list of violating APs maintained dynamically in this simple version.
                # Maintaining it is complex.
                # Let's just pick a random position.
                
                pos = random.randint(0, n - 1)
                old_color = colors[pos]
                new_color = random.randint(0, k - 1)
                
                if new_color == old_color:
                    continue
                
                delta = delta_violation(pos, old_color, new_color, colors)
                new_violations = current_violations + delta
                
                # Acceptance criterion
                if new_violations <= current_violations:
                    colors[pos] = new_color
                    current_violations = new_violations
                    if current_violations < best_violations_in_run:
                        best_violations_in_run = current_violations
                        best_in_run = colors[:]
                        if current_violations == 0:
                            break
                else:
                    # Metropolis criterion
                    # P(accept) = exp(-delta / T)
                    # Use log to avoid underflow/overflow? delta is small.
                    if random.random() < pow(2.718281828, -delta / temp):
                        colors[pos] = new_color
                        current_violations = new_violations
            
            if best_violations_in_run < best_violations:
                best_violations = best_violations_in_run
                best_colors = best_in_run
                best_k = k
            
            if best_violations == 0:
                found_valid = True
                break
        
        if found_valid:
            break
        # If k=3 failed, try k=4, etc.
        # If we run out of time, we break the loop and return best found so far.
        
    if best_colors is None:
        # Fallback: Greedy
        colors = [0] * n
        for x in range(n):
            found = False
            for c in range(128):
                valid = True
                for d in range(1, x // 3 + 1):
                    i3 = x - 3 * d
                    i2 = x - 2 * d
                    i1 = x - d
                    if i3 < 0:
                        continue
                    if len({colors[i3], colors[i2], colors[i1], c}) < 3:
                        valid = False
                        break
                if valid:
                    colors[x] = c
                    found = True
                    break
            if not found:
                colors[x] = 0
        best_colors = colors
        best_k = len(set(colors))
        
    return best_colors
