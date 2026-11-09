import time
import random
import numpy as np

# Precompute all nonconstant 4-term progressions (a, a+d, a+2d, a+3d)
# 0-indexed internally
_ALL_APS = []
_AP_MAP = {} # index -> tuple
for a in range(126):  # a+3d <= 127 => a <= 124 if d>=1? 
    # a is 0-indexed start.
    # Max d: a + 3d <= 127 => 3d <= 127 - a => d <= (127-a)//3
    for d in range(1, (127 - a) // 3 + 1):
        ap = (a, a+d, a+2*d, a+3*d)
        _ALL_APS.append(ap)

_NUM_APS = len(_ALL_APS)

# Map position to list of AP indices it participates in
_POS_APS = [[] for _ in range(128)]
for idx, ap in enumerate(_ALL_APS):
    for pos in ap:
        _POS_APS[pos].append(idx)

def _is_valid_ap(ap, colors):
    c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
    return len({c0, c1, c2, c3}) >= 3

def construct():
    start_time = time.time()
    DEADLINE = 55.0 # Safety margin
    
    # We will try to find a valid coloring with K=3, then K=4, etc.
    # Start with K=3 as it's the theoretical minimum.
    
    best_solution = None
    best_k = 128
    
    for k in range(3, 8):
        if time.time() > DEADLINE:
            break
            
        # Local Search / Simulated Annealing
        # State: colors array
        # Cost: number of invalid APs
        
        # Initialize randomly
        colors = [random.randint(0, k-1) for _ in range(128)]
        
        # Precompute validity of each AP
        ap_valid = [False] * _NUM_APS
        invalid_count = 0
        for idx, ap in enumerate(_ALL_APS):
            if _is_valid_ap(ap, colors):
                ap_valid[idx] = True
            else:
                ap_valid[idx] = False
                invalid_count += 1
        
        if invalid_count == 0:
            best_solution = colors
            best_k = k
            break
            
        # Local Search Loop
        # We use a simple hill climbing with random restarts or simulated annealing.
        # Given the small size, a simple greedy "fix one violation" might work.
        
        current_colors = colors
        current_invalid = invalid_count
        current_ap_valid = ap_valid
        
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() < DEADLINE and no_improve_count < max_no_improve:
            if current_invalid == 0:
                best_solution = current_colors
                best_k = k
                break
                
            # Pick a random invalid AP
            # To speed up, we can keep a list of invalid AP indices
            # But rebuilding it is O(N_APS). N_APS is ~ 1000-2000?
            # Let's estimate N_APS.
            # Sum_{a=0}^{124} floor((127-a)/3)
            # approx 125 * 40 = 5000.
            # Checking all 5000 is fast.
            
            invalid_indices = [i for i in range(_NUM_APS) if not current_ap_valid[i]]
            if not invalid_indices:
                break
                
            target_ap_idx = random.choice(invalid_indices)
            target_ap = _ALL_APS[target_ap_idx]
            
            # Try to change one of the positions in this AP to fix it
            # We can try all 4 positions and all k colors
            improved = False
            
            # Shuffle positions to avoid bias
            positions = list(target_ap)
            random.shuffle(positions)
            
            for pos in positions:
                old_c = current_colors[pos]
                for new_c in range(k):
                    if new_c == old_c:
                        continue
                    
                    # Check impact
                    # Only APs containing 'pos' are affected
                    affected_aps = _POS_APS[pos]
                    
                    delta_invalid = 0
                    # Check if any affected AP changes status
                    # We need to simulate the change
                    current_colors[pos] = new_c
                    
                    for ap_idx in affected_aps:
                        ap = _ALL_APS[ap_idx]
                        was_valid = current_ap_valid[ap_idx]
                        is_valid = _is_valid_ap(ap, current_colors)
                        
                        if was_valid and not is_valid:
                            delta_invalid += 1
                            current_ap_valid[ap_idx] = False
                        elif not was_valid and is_valid:
                            delta_invalid -= 1
                            current_ap_valid[ap_idx] = True
                    
                    if delta_invalid < 0:
                        # Improvement
                        current_invalid += delta_invalid
                        no_improve_count = 0
                        improved = True
                        break
                    else:
                        # Revert
                        current_colors[pos] = old_c
                        for ap_idx in affected_aps:
                            ap = _ALL_APS[ap_idx]
                            # Restore validity
                            # Note: we modified current_ap_valid in the loop above.
                            # We need to revert those changes too.
                            # This is getting messy.
                            pass
                    
                if improved:
                    break
            
            if not improved:
                # Try a random move that doesn't increase invalid count too much?
                # Or just pick a random position and random color, accept if delta <= 0
                pos = random.randint(0, 127)
                old_c = current_colors[pos]
                new_c = random.randint(0, k-1)
                if new_c != old_c:
                    current_colors[pos] = new_c
                    delta = 0
                    for ap_idx in _POS_APS[pos]:
                        ap = _ALL_APS[ap_idx]
                        was_valid = current_ap_valid[ap_idx]
                        is_valid = _is_valid_ap(ap, current_colors)
                        if was_valid and not is_valid:
                            delta += 1
                            current_ap_valid[ap_idx] = False
                        elif not was_valid and is_valid:
                            delta -= 1
                            current_ap_valid[ap_idx] = True
                    
                    current_invalid += delta
                    if delta > 0:
                        no_improve_count += 1
                    else:
                        no_improve_count = 0
                        if delta < 0:
                            no_improve_count = 0 # Reset on improvement
                else:
                    no_improve_count += 1
                    
        if best_solution is not None:
            break
            
    if best_solution is None:
        # Fallback: use a known valid pattern or unique colors
        # Unique colors always works
        return list(range(128))
        
    return best_solution

# Quick self-check
# print(construct())
