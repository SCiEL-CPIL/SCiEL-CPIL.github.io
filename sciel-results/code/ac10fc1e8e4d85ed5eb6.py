import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    n = 128
    # Precompute all 4-APs
    aps = []
    for d in range(1, (n - 1) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    def is_valid(colors):
        for ap in aps:
            c = {colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}
            if len(c) < 3:
                return False
        return True

    def count_violations(colors):
        v = 0
        for ap in aps:
            c = {colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}
            if len(c) < 3:
                v += 1
        return v

    best_colors = None
    best_violations = float('inf')
    
    for K in range(3, 6): # Try 3, 4, 5 colors
        # If we already found a valid one with fewer colors, stop
        if best_colors is not None:
            break
            
        # Local search for K colors
        # Restart a few times if it gets stuck
        max_restarts = 20
        for r in range(max_restarts):
            if time.time() > deadline:
                break
                
            colors = [random.randint(0, K - 1) for _ in range(n)]
            v = count_violations(colors)
            
            if v == 0:
                best_colors = colors
                break
                
            best_v = v
            stuck_count = 0
            
            while time.time() < deadline:
                if v == 0:
                    best_colors = colors
                    break
                    
                # Pick a random violated AP
                # To be faster, we can just pick a random AP and check if it's violated
                # Or maintain a list of violated APs. Given small N, random sampling is fine.
                
                improved = False
                for _ in range(100): # Try 100 moves
                    ap = random.choice(aps)
                    c_set = {colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}
                    if len(c_set) >= 3:
                        continue # This AP is already valid, skip? 
                        # Actually, changing a color in a valid AP might break it.
                        # Better to pick from violated ones.
                        
                # Re-evaluate violations to pick a good one? 
                # Let's just pick a random AP, if it's violated, try to fix it.
                ap = random.choice(aps)
                c_set = {colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}
                if len(c_set) < 3:
                    # Pick a random index in this AP
                    idx = random.choice(ap)
                    old_c = colors[idx]
                    new_c = random.randint(0, K - 1)
                    colors[idx] = new_c
                    
                    new_v = count_violations(colors)
                    if new_v < v:
                        v = new_v
                        improved = True
                        stuck_count = 0
                    else:
                        # Revert or keep? 
                        # If not improved, revert to avoid random walk? 
                        # Or keep if equal? 
                        colors[idx] = old_c
                        stuck_count += 1
                        
                    if stuck_count > 50:
                        break # Restart this random initialization
                    
            if best_colors is not None:
                break

    if best_colors is not None:
        return best_colors
    
    # Fallback: return the seed-like greedy result or just random
    # The seed program returns a list of integers.
    # If we failed, we should return something valid if possible, or best effort.
    # But the task says "Minimize the number of labels".
    # If we couldn't find a valid one with 3-5 colors, we might need to fall back to a known valid one.
    # The seed program is valid. We can replicate its logic as a fallback if time permits, 
    # but it's slow. 
    # However, 3 or 4 colors should be findable.
    
    # If still None, return a simple valid coloring (e.g. using many colors)
    # Actually, using distinct colors for every position is always valid (128 colors).
    # But we want to minimize.
    # Let's just return the best found, or a random one if none valid found (though unlikely).
    
    # For safety, if no valid found, use the greedy from seed but optimized?
    # Or just return [0]*128? No, that's invalid.
    # Return [i % 128 for i in range(128)]? Valid (128 colors).
    
    if best_colors is None:
        # Fallback to a valid but high-color solution
        return list(range(128))
        
    return best_colors
