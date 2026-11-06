import random
import time
import math

def construct():
    deadline = time.time() + 85
    N = 128
    
    # Precompute all nonconstant 4-term APs (0-indexed)
    aps = []
    for a in range(1, N+1):
        for d in range(1, N+1):
            if a + 3*d <= N:
                aps.append((a-1, a+d-1, a+2*d-1, a+3*d-1))
    
    # For each position, list of AP indices containing it
    pos_to_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    def is_valid_partial(colors, changed_idx):
        for ap_idx in pos_to_aps[changed_idx]:
            i, j, k, l = aps[ap_idx]
            if len({colors[i], colors[j], colors[k], colors[l]}) < 3:
                return False
        return True
    
    def greedy_init(rng, max_colors=8):
        colors = [0] * N
        for x in range(N):
            found = False
            used_colors = list(set(colors[:x]))
            rng.shuffle(used_colors)
            for color in used_colors:
                colors[x] = color
                if is_valid_partial(colors, x):
                    found = True
                    break
            if not found:
                for color in range(max_colors):
                    colors[x] = color
                    if is_valid_partial(colors, x):
                        found = True
                        break
            if not found:
                for color in range(128):
                    colors[x] = color
                    if is_valid_partial(colors, x):
                        found = True
                        break
            if not found:
                return None
        return colors
    
    best_colors = None
    best_cost = 129
    
    rng = random.Random(42)
    
    # Phase 1: Multiple greedy inits
    for attempt in range(50):
        if time.time() > deadline - 60:
            break
        c = greedy_init(rng)
        if c is None:
            continue
        cost = len(set(c))
        if cost < best_cost:
            best_cost = cost
            best_colors = c[:]
    
    if best_colors is None:
        return list(range(N))
    
    # Phase 2: Simulated annealing
    colors = best_colors[:]
    current_cost = best_cost
    
    freq = {}
    for c in colors:
        freq[c] = freq.get(c, 0) + 1
    
    T = 3.0
    T_min = 0.01
    cooling = 0.9995
    
    while time.time() < deadline - 5 and T > T_min:
        idx = rng.randint(0, N-1)
        old_color = colors[idx]
        new_color = rng.randint(0, 127)
        if new_color == old_color:
            T *= cooling
            continue
        
        colors[idx] = new_color
        
        if is_valid_partial(colors, idx):
            freq[old_color] -= 1
            if freq[old_color] == 0:
                del freq[old_color]
            freq[new_color] = freq.get(new_color, 0) + 1
            
            new_cost = len(freq)
            delta = new_cost - current_cost
            
            if delta <= 0 or (T > 0 and rng.random() < math.exp(-delta / T)):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_colors = colors[:]
            else:
                colors[idx] = old_color
                freq[old_color] = freq.get(old_color, 0) + 1
                freq[new_color] -= 1
                if freq[new_color] == 0:
                    del freq[new_color]
        else:
            colors[idx] = old_color
        
        T *= cooling
    
    # Phase 3: Polish - try to eliminate colors one by one
    # For each color used, try to recolor all its positions to other colors
    for _ in range(5):  # repeat polish a few times
        if time.time() > deadline - 2:
            break
        colors = best_colors[:]
        current_cost = best_cost
        freq = {}
        for c in colors:
            freq[c] = freq.get(c, 0) + 1
        
        used = sorted(freq.keys())
        for target_color in used[:]:
            if target_color not in freq:
                continue
            # Try to eliminate target_color
            positions = [i for i in range(N) if colors[i] == target_color]
            rng.shuffle(positions)
            success = True
            for pos in positions:
                found = False
                for new_c in used:
                    if new_c == target_color:
                        continue
                    colors[pos] = new_c
                    if is_valid_partial(colors, pos):
                        found = True
                        break
                if not found:
                    # Try any color
                    for new_c in range(128):
                        if new_c == target_color:
                            continue
                        colors[pos] = new_c
                        if is_valid_partial(colors, pos):
                            found = True
                            break
                if not found:
                    success = False
                    break
                # Update freq
                freq[target_color] -= 1
                if freq[target_color] == 0:
                    del freq[target_color]
                freq[new_c] = freq.get(new_c, 0) + 1
            if success and target_color in freq:
                # This shouldn't happen if we removed all
                pass
            if success:
                new_cost = len(freq)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_colors = colors[:]
    
    return best_colors
