"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 90s, we stop at 85s
    deadline = start_time + 85.0
    
    n = 128
    
    # 1. Precompute all non-constant 4-term APs
    # AP is (a, a+d, a+2d, a+3d) with 0-based indices
    # a in 0..n-1, d >= 1, a+3d < n
    aps = []
    pos_to_aps = [[] for _ in range(n)]
    
    for a in range(n):
        # a + 3d < n => d < (n - a) / 3
        max_d = (n - 1 - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # e is guaranteed < n by max_d calculation
            ap_idx = len(aps)
            aps.append((a, b, c, e))
            pos_to_aps[a].append(ap_idx)
            pos_to_aps[b].append(ap_idx)
            pos_to_aps[c].append(ap_idx)
            pos_to_aps[e].append(ap_idx)
    
    num_aps = len(aps)
    
    def is_ap_valid(colors, a, b, c, e):
        """Check if the 4-term AP uses at least 3 distinct colors."""
        c1, c2, c3, c4 = colors[a], colors[b], colors[c], colors[e]
        # Count distinct colors
        # If all 4 same: 1
        # If 3 same, 1 diff: 2
        # If 2 same, 2 same (pairs): 2
        # If 2 same, 2 diff: 3
        # If 1 same, 3 diff: 4
        # If all diff: 4
        
        # Efficient check:
        # Set size < 3 is invalid.
        # We can just use a set for simplicity as n is small and we do incremental checks.
        # But for speed in inner loop, manual check is better.
        
        # Invalid if distinct colors <= 2.
        # This happens if:
        # 1. All 4 are same.
        # 2. 3 are same, 1 is different.
        # 3. 2 are same, 2 are different (but the two different ones are the same as each other? No, that's 2 pairs).
        #    Pattern A A B B, A B A B, A B B A, etc.
        
        # Simpler:
        vals = [c1, c2, c3, c4]
        # If any pair is equal, we have duplicates.
        # If the set of values has size 1 or 2, it's invalid.
        
        # Manual distinct count logic:
        # c1 == c2 == c3 == c4 -> 1
        # c1 == c2 == c3 != c4 -> 2
        # c1 == c2 != c3 == c4 -> 2
        # c1 == c3 == c4 != c2 -> 2
        # c1 == c3 != c2 == c4 -> 2
        # c1 == c4 == c2 != c3 -> 2
        # c1 == c4 != c2 == c3 -> 2
        # c2 == c3 == c4 != c1 -> 2
        
        # If no 3 are equal, we check if it's 2 pairs or 1 pair + 2 singles.
        # 1 pair + 2 singles -> 3 distinct (Valid)
        # 2 pairs -> 2 distinct (Invalid)
        # 4 singles -> 4 distinct (Valid)
        
        # Let's just use a small set creation, it's fast enough in Py if called moderately.
        # But we can optimize:
        if c1 == c2:
            if c3 == c4:
                return False # AA BB
            if c3 == c1:
                return False # AAA B
            if c4 == c1:
                return False # AA B A
            # Else A A B C (B!=C, B!=A, C!=A) -> 3 distinct? 
            # Wait, if c3!=c1 and c4!=c1 and c3!=c4, then {A, B, C} size 3. Valid.
            # If c3==c4 is handled.
            # So if c1==c2, and not (c3==c4), and c3!=c1, and c4!=c1, then valid?
            # Yes, unless c3==c4 which is handled.
            # What if c3==c1? Handled.
            # So if c1==c2, valid iff c3!=c1 AND c4!=c1 AND c3!=c4.
            # Wait, if c3==c4, invalid.
            # If c3!=c4:
            #   If c3==c1 or c4==c1, invalid (AAA B or AA B A).
            #   Else valid (AA B C).
            
        # This logic is getting complex. Let's use a helper that is fast.
        # The number of APs is ~ (128*128)/6 ~ 2000.
        # Local search steps ~ 10000.
        # 20 million checks. Python set() might be slow.
        
        # Fast check:
        # distinct = 0
        # ...
        
        # Actually, let's just use the set. It's C-optimized.
        return len({c1, c2, c3, c4}) >= 3

    def get_violations(colors):
        """Return list of indices of violated APs."""
        viol = []
        for i, (a, b, c, e) in enumerate(aps):
            if not is_ap_valid(colors, a, b, c, e):
                viol.append(i)
        return viol

    def local_search(k, max_time):
        """
        Try to find a valid coloring with k colors.
        Returns (colors, valid)
        """
        end_time = time.time() + max_time
        best_colors = None
        best_violations = float('inf')
        
        # Number of restarts depends on time left
        # We'll do a loop of restarts
        while time.time() < end_time:
            # Random init
            colors = [random.randint(0, k - 1) for _ in range(n)]
            
            # Initial violation count
            violations = get_violations(colors)
            current_violations = len(violations)
            
            if current_violations == 0:
                return colors, True
            
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = list(colors)
            
            # Local search loop
            stuck_count = 0
            while time.time() < end_time:
                if current_violations == 0:
                    return colors, True
                
                # Pick a violated AP
                # Randomly pick one from the list of violations
                # We need to maintain the list of violations dynamically
                # For simplicity, we can re-scan or pick a random AP and check if it's violated.
                # Better: Maintain a list. But updating it is O(num_aps) if we scan.
                # Let's just pick a random AP index and check if it's violated.
                # If not, try again. This is inefficient if violations are few.
                
                # Alternative: Keep a set of violated indices.
                # When we change a color, we update the set for affected APs.
                
                # Let's implement the incremental update version.
                break # Switch to incremental version below
            
            # Incremental Search
            violated_set = set(violations)
            current_viol = len(violated_set)
            
            while time.time() < end_time:
                if current_viol == 0:
                    return colors, True
                
                if current_viol < best_violations:
                    best_violations = current_viol
                    best_colors = list(colors)
                
                # Pick a random violated AP
                # Convert to list for random choice? Expensive.
                # Just pick a random AP index and see if it's in violated_set.
                # If violations are sparse, this is bad.
                # Let's maintain a list of violated APs.
                viol_list = list(violated_set)
                ap_idx = random.choice(viol_list)
                a, b, c, e = aps[ap_idx]
                
                # Try to fix this AP
                best_delta = 0
                best_move = None
                
                # Try changing a, b, c, e
                for pos in (a, b, c, e):
                    old_c = colors[pos]
                    for new_c in range(k):
                        if new_c == old_c:
                            continue
                        
                        # Simulate change
                        colors[pos] = new_c
                        
                        # Calculate delta in violations
                        delta = 0
                        # Check all APs containing pos
                        for idx in pos_to_aps[pos]:
                            was_violated = idx in violated_set
                            is_violated = not is_ap_valid(colors, *aps[idx])
                            
                            if was_violated and not is_violated:
                                delta -= 1
                            elif not was_violated and is_violated:
                                delta += 1
                            
                        if delta < best_delta:
                            best_delta = delta
                            best_move = (pos, new_c)
                        
                        # Revert
                        colors[pos] = old_c
                
                if best_move:
                    pos, new_c = best_move
                    old_c = colors[pos]
                    colors[pos] = new_c
                    
                    # Update violated_set
                    for idx in pos_to_aps[pos]:
                        was_violated = idx in violated_set
                        is_violated = not is_ap_valid(colors, *aps[idx])
                        
                        if was_violated and not is_violated:
                            violated_set.remove(idx)
                            current_viol -= 1
                        elif not was_violated and is_violated:
                            violated_set.add(idx)
                            current_viol += 1
                    stuck_count = 0
                else:
                    # No improvement
                    stuck_count += 1
                    if stuck_count > 50:
                        # Random move
                        pos = random.randint(0, n - 1)
                        old_c = colors[pos]
                        new_c = random.randint(0, k - 1)
                        if new_c != old_c:
                            colors[pos] = new_c
                            for idx in pos_to_aps[pos]:
                                was_violated = idx in violated_set
                                is_violated = not is_ap_valid(colors, *aps[idx])
                                if was_violated and not is_violated:
                                    violated_set.remove(idx)
                                    current_viol -= 1
                                elif not was_violated and is_violated:
                                    violated_set.add(idx)
                                    current_viol += 1
                        stuck_count = 0 # Reset to allow more random moves
                    elif stuck_count > 200:
                        break # Restart

        return best_colors, False

    # Try k=3 first
    # Allocate 80% of time to k=3
    time_for_k3 = 0.8 * (deadline - time.time())
    colors_3, valid_3 = local_search(3, time_for_k3)
    if valid_3:
        return colors_3
    
    # Try k=4
    if time.time() < deadline - 2.0:
        time_for_k4 = deadline - time.time() - 1.0
        colors_4, valid_4 = local_search(4, time_for_k4)
        if valid_4:
            return colors_4
    
    # Fallback to seed if all else fails
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x - 1 - j * d] for j in (1, 2, 3)}) >= 3 
                    for d in range(1, (x - 1) // 3 + 1))
                and all(any(colors[x - 1 - j * d] != color for j in (1, 2)) 
                        for d in range(1, (x - 1) // 2 + 1))):
                colors.append(color)
                break
    return colors

# EVOLVE-BLOCK-END
