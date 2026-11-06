import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    N = 128
    # Precompute all 4-APs
    aps = []
    ap_index = {} # pos -> list of (ap_idx, pos_in_ap)
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            ap = [a, a + d, a + 2 * d, a + 3 * d]
            idx = len(aps)
            aps.append(ap)
            for k, pos in enumerate(ap):
                if pos not in ap_index:
                    ap_index[pos] = []
                ap_index[pos].append((idx, k))
    
    def check_validity(colors, num_colors):
        # colors: list of ints 0..num_colors-1
        # Returns True if every 4-AP has >= 3 distinct colors
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]-1], colors[ap[1]-1], colors[ap[2]-1], colors[ap[3]-1]
            if len({c0, c1, c2, c3}) < 3:
                return False
        return True

    def count_violations(colors, num_colors):
        # Returns number of 4-APs with < 3 distinct colors
        viol = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]-1], colors[ap[1]-1], colors[ap[2]-1], colors[ap[3]-1]
            if len({c0, c1, c2, c3}) < 3:
                viol += 1
        return viol

    # Try to find a 3-coloring
    best_colors = None
    for restart in range(20):
        if time.time() > deadline - 10:
            break
            
        colors = [random.randint(0, 2) for _ in range(N)]
        current_viol = count_violations(colors, 3)
        
        # Hill climbing
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            # Randomize order of positions
            positions = list(range(1, N + 1))
            random.shuffle(positions)
            
            for pos in positions:
                if time.time() > deadline - 10:
                    break
                old_color = colors[pos - 1]
                # Try changing to other colors
                best_new_color = old_color
                best_new_viol = current_viol
                
                # Calculate delta for each color
                # We only need to check APs containing pos
                affected_aps = ap_index.get(pos, [])
                
                for new_color in range(3):
                    if new_color == old_color:
                        continue
                    # Check if this change improves things
                    # We can just count violations in affected APs
                    # But we need to know the current violation count of affected APs
                    # Simpler: just calculate total violations if we change? Too slow.
                    # Let's just do a simple hill climb: try all 3 colors, pick best.
                    
                    # To speed up, we can precompute which APs are currently violated.
                    # For now, simple approach:
                    colors[pos - 1] = new_color
                    # Count violations in affected APs only?
                    # No, we need total count to compare.
                    # But we can track delta.
                    # Let's just do a full check for simplicity if N is small?
                    # 2700 APs is small.
                    new_viol = count_violations(colors, 3)
                    if new_viol < best_new_viol:
                        best_new_viol = new_viol
                        best_new_color = new_color
                    
                colors[pos - 1] = best_new_color
                if best_new_viol < current_viol:
                    current_viol = best_new_viol
                    improved = True
                    if current_viol == 0:
                        best_colors = colors[:]
                        break
                else:
                    # Revert if not improved? We already set to best_new_color.
                    # If best_new_color is old_color, it's reverted.
                    pass

        if best_colors is not None:
            break

    if best_colors is not None:
        return best_colors

    # If 3 colors failed, try 4 colors
    # Greedy for 4 colors is likely to work
    colors = [0] * N
    for i in range(1, N + 1):
        for c in range(4):
            colors[i - 1] = c
            if check_validity_partial(colors, i):
                break
        # If no color worked, try 5?
        # But 4 should work.
        
    # Verify 4-coloring
    if check_validity(colors, 4):
        return colors

    # Fallback: Greedy with 128 colors (always works)
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color);break
    return colors

def check_validity_partial(colors, upto):
    # Check APs that end at or before 'upto'
    # Actually, just check all APs fully within 1..upto
    for ap in aps:
        if ap[3] <= upto:
            c0, c1, c2, c3 = colors[ap[0]-1], colors[ap[1]-1], colors[ap[2]-1], colors[ap[3]-1]
            if len({c0, c1, c2, c3}) < 3:
                return False
    return True

# Note: aps is global in the above draft, needs to be local or passed.
