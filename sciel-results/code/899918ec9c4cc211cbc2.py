import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80
    n = 128
    
    # Precompute all 4-term APs
    aps = []
    for d in range(1, (n-1)//3 + 1):
        for a in range(1, n - 3*d + 1):
            aps.append((a-1, a-1+d, a-1+2*d, a-1+3*d))
    
    num_aps = len(aps)
    
    # For each position, list of AP indices it belongs to
    pos_aps = [[] for _ in range(n)]
    for i, ap in enumerate(aps):
        for pos in ap:
            pos_aps[pos].append(i)
    
    def is_violation(c0, c1, c2, c3):
        return len({c0, c1, c2, c3}) < 3
    
    best_violations = float('inf')
    best_colors = None
    
    for k in range(3, 10):
        if time.time() > deadline - 5:
            break
        
        for restart in range(200):
            if time.time() > deadline - 2:
                break
            
            colors = np.random.randint(0, k, size=n)
            viol = np.array([is_violation(colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]) for ap in aps])
            v = int(viol.sum())
            
            if v < best_violations:
                best_violations = v
                best_colors = colors.copy()
            
            if v == 0:
                break
            
            # Local search
            for step in range(10000):
                if time.time() > deadline - 1:
                    break
                if v == 0:
                    break
                
                i = random.randint(0, n-1)
                old = int(colors[i])
                new = random.randint(0, k-1)
                if new == old:
                    continue
                
                affected = pos_aps[i]
                affected_viol = sum(1 for j in affected if viol[j])
                
                colors[i] = new
                new_affected_viol = 0
                for j in affected:
                    ap = aps[j]
                    is_v = is_violation(colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]])
                    viol[j] = is_v
                    if is_v:
                        new_affected_viol += 1
                
                delta = new_affected_viol - affected_viol
                if delta <= 0:
                    v += delta
                    if v < best_violations:
                        best_violations = v
                        best_colors = colors.copy()
                else:
                    colors[i] = old
                    for j in affected:
                        ap = aps[j]
                        viol[j] = is_violation(colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]])
            
            if best_violations == 0:
                break
        
        if best_violations == 0:
            break
    
    if best_colors is not None and best_violations == 0:
        return list(best_colors.astype(int))
    else:
        # Fallback to seed
        colors = []
        for x in range(1, 129):
            for color in range(128):
                if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 
                        for d in range(1, (x-1)//3+1))
                    and all(any(colors[x-1-j*d] != color for j in (1,2)) 
                            for d in range(1, (x-1)//2+1))):
                    colors.append(color)
                    break
        return colors
