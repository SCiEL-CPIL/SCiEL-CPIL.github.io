import time
import random

def _check_validity(colors, n=128):
    for a in range(n):
        max_d = (n - 1 - a) // 3
        for d in range(1, max_d + 1):
            c0 = colors[a]
            c1 = colors[a + d]
            c2 = colors[a + 2*d]
            c3 = colors[a + 3*d]
            if len({c0, c1, c2, c3}) < 3:
                return False
    return True

def _local_search(num_colors, n=128, time_limit=50):
    start_time = time.time()
    aps = []
    for a in range(n):
        max_d = (n - 1 - a) // 3
        for d in range(1, max_d + 1):
            aps.append((a, a+d, a+2*d, a+3*d))
            
    # Random restarts
    for restart in range(2000):
        if time.time() - start_time > time_limit:
            break
            
        colors = [random.randint(0, num_colors - 1) for _ in range(n)]
        
        # Local search loop
        for step in range(5000):
            if time.time() - start_time > time_limit:
                break
                
            # Find a violation
            bad_ap = None
            # Sample 5 random APs first for speed
            for _ in range(5):
                ap = random.choice(aps)
                idxs = ap
                c0, c1, c2, c3 = colors[idxs[0]], colors[idxs[1]], colors[idxs[2]], colors[idxs[3]]
                if len({c0, c1, c2, c3}) < 3:
                    bad_ap = ap
                    break
            
            if bad_ap is None:
                # Check all
                valid = True
                for ap in aps:
                    idxs = ap
                    c0, c1, c2, c3 = colors[idxs[0]], colors[idxs[1]], colors[idxs[2]], colors[idxs[3]]
                    if len({c0, c1, c2, c3}) < 3:
                        valid = False
                        break
                if valid:
                    return colors
                else:
                    # Find the first bad one
                    for ap in aps:
                        idxs = ap
                        c0, c1, c2, c3 = colors[idxs[0]], colors[idxs[1]], colors[idxs[2]], colors[idxs[3]]
                        if len({c0, c1, c2, c3}) < 3:
                            bad_ap = ap
                            break
            
            if bad_ap:
                idxs = bad_ap
                pos_idx = random.choice(idxs)
                old_color = colors[pos_idx]
                new_color = random.randint(0, num_colors - 1)
                if new_color == old_color:
                    new_color = (new_color + 1) % num_colors
                colors[pos_idx] = new_color
            else:
                break

    return None

def construct():
    start_time = time.time()
    
    # Try 3 colors
    remaining = 45 - (time.time() - start_time)
    if remaining > 5:
        result = _local_search(3, time_limit=remaining)
        if result is not None:
            if _check_validity(result):
                return result
                
    # Try 4 colors
    remaining = 55 - (time.time() - start_time)
    if remaining > 5:
        result = _local_search(4, time_limit=remaining)
        if result is not None:
            if _check_validity(result):
                return result
                
    # Fallback
    return [random.randint(0, 127) for _ in range(128)]
