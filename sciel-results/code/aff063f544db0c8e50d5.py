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
    idx_to_aps_pos = [[] for _ in range(n)]
    for i, ap in enumerate(aps_list):
        for pos, idx in enumerate(ap):
            idx_to_aps_pos[idx].append((i, pos))
            
    # Convert to lists of tuples for fast iteration
    # Or keep as list of (ap_idx, pos)
    
    def count_bad_colors(colors):
        """Count APs with <= 2 distinct colors."""
        bad = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            # Check distinct count
            # Faster check:
            # If c0==c1 and c0==c2 and c0==c3: bad
            # If (c0==c1 and c2==c3) or (c0==c2 and c1==c3) or (c0==c3 and c1==c2): bad
            # Else if len({c0,c1,c2,c3}) <= 2: bad
            
            # Simple set check is probably fine for 5000 APs
            if len({c0, c1, c2, c3}) <= 2:
                bad += 1
        return bad

    def apply_move(colors, i, new_color, current_bad):
        """
        Calculate delta in bad count if we change colors[i] to new_color.
        Returns delta.
        """
        old_color = colors[i]
        if old_color == new_color:
            return 0
            
        delta = 0
        for ap_idx, pos in idx_to_aps_pos[i]:
            ap = aps[ap_idx]
            # Get colors
            c = [colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]]
            
            # Check if current AP is bad
            was_bad = 1 if len(set(c)) <= 2 else 0
            
            # Update color at pos
            c[pos] = new_color
            
            # Check if new AP is bad
            is_bad = 1 if len(set(c)) <= 2 else 0
            
            delta += is_bad - was_bad
            
        return delta

    # Try k = 3, 4, 5...
    for k in range(3, 10):
        if time.time() > deadline - 5:
            break
            
        # Simulated Annealing
        max_restarts = 100
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline - 2:
                break
                
            # Random initialization
            colors = np.random.randint(0, k, n).astype(np.int32)
            current_bad = count_bad_colors(colors)
            
            if current_bad == 0:
                return [int(c) for c in colors]
            
            T = 2.0
            T_min = 0.01
            cooling = 0.999
            no_improve = 0
            max_no_improve = 2000
            max_iters = 20000
            
            for it in range(max_iters):
                if time.time() > deadline - 2:
                    break
                    
                if current_bad == 0:
                    return [int(c) for c in colors]
                    
                # Pick random index
                i = np.random.randint(n)
                old_color = colors[i]
                new_color = np.random.randint(0, k)
                
                if new_color == old_color:
                    continue
                    
                # Calculate delta
                delta = apply_move(colors, i, new_color, current_bad)
                
                # Acceptance criterion
                if delta < 0:
                    colors[i] = new_color
                    current_bad += delta
                    no_improve = 0
                else:
                    # Accept with probability exp(-delta/T)
                    if random.random() < np.exp(-delta / T):
                        colors[i] = new_color
                        current_bad += delta
                    else:
                        no_improve += 1
                        
                # Cool down
                T *= cooling
                if T < T_min:
                    T = T_min
                    
                if no_improve > max_no_improve:
                    break

    # Fallback: Greedy construction similar to seed
    # This ensures we return a valid solution, though potentially with more colors
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color);break
    return colors
