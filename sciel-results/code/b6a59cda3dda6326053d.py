import time
import random
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s buffer for 120s limit
    
    # Precompute all 4-APs: a, a+d, a+2d, a+3d with 1 <= a, a+3d <= n
    # Using 0-indexed: indices 0..n-1 represent values 1..n
    # a + 3d <= n-1 => a <= n-1-3d
    ap_list = []
    for d in range(1, (n - 1) // 3 + 1):
        max_a = n - 1 - 3 * d
        for a in range(max_a + 1):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(ap_list)
    
    # Precompute for each position, which APs it belongs to
    pos_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(ap_list):
        for pos in ap:
            pos_aps[pos].append(idx)
    
    def count_mono(coloring):
        """Count monochromatic 4-APs."""
        count = 0
        for ap in ap_list:
            c0 = coloring[ap[0]]
            if c0 == coloring[ap[1]] == coloring[ap[2]] == coloring[ap[3]]:
                count += 1
        return count
    
    def count_mono_fast(coloring):
        """Faster count using numpy."""
        arr = np.array(coloring)
        count = 0
        for ap in ap_list:
            a, b, c, d = ap
            if arr[a] == arr[b] == arr[c] == arr[d]:
                count += 1
        return count
    
    def delta_count(coloring, pos, new_color):
        """Compute change in mono count if we flip position pos to new_color."""
        old_color = coloring[pos]
        if old_color == new_color:
            return 0
        
        # For each AP containing pos, check if it was mono before and is mono after
        delta = 0
        for ap_idx in pos_aps[pos]:
            ap = ap_list[ap_idx]
            other_positions = [p for p in ap if p != pos]
            other_colors = [coloring[p] for p in other_positions]
            
            # Was it mono before?
            was_mono = all(c == old_color for c in other_colors)
            # Is it mono after?
            is_mono = all(c == new_color for c in other_colors)
            
            if was_mono and not is_mono:
                delta -= 1
            elif not was_mono and is_mono:
                delta += 1
        
        return delta
    
    # Thue-Morse initialization
    def thue_morse(n):
        """Generate Thue-Morse sequence of length n."""
        tm = [0]
        for _ in range(n.bit_length()):
            tm = tm + [1 - x for x in tm]
        return tm[:n]
    
    coloring = thue_morse(n)
    best_coloring = coloring[:]
    best_score = count_mono(coloring)
    
    # Local search with simulated annealing
    temp = 50.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick a random position to flip
        pos = random.randint(0, n - 1)
        new_color = 1 - coloring[pos]
        
        delta = delta_count(coloring, pos, new_color)
        
        # Accept if better, or with probability exp(-delta/temp) if worse
        if delta < 0:
            accept = True
        else:
            if temp > 0.01:
                r = random.random()
                accept = r < np.exp(-delta / temp)
            else:
                accept = False
        
        if accept:
            coloring[pos] = new_color
            current_score = best_score + delta  # Track incrementally
            if current_score < best_score:
                best_score = current_score
                best_coloring = coloring[:]
        
        # Cool down
        temp *= cooling
        
        # Occasionally restart temperature to escape local minima
        if random.random() < 0.001:
            temp = max(5.0, temp * 2)
    
    # Final local search pass (hill climbing only)
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for pos in range(n):
            new_color = 1 - coloring[pos]
            delta = delta_count(coloring, pos, new_color)
            if delta < 0:
                coloring[pos] = new_color
                best_score += delta
                best_coloring = coloring[:]
                improved = True
    
    return best_coloring
