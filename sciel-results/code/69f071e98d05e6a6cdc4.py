import time
import random
import math

def construct():
    n = 256
    deadline = time.time() + 110  # safety margin for 120s limit
    
    # Precompute all 4-term APs: a, a+d, a+2d, a+3d with 0-indexed, a+3d < n
    ap_list = []
    for d in range(1, (n - 1) // 3 + 1):
        max_a = n - 1 - 3 * d
        for a in range(max_a + 1):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(ap_list)
    
    # For each position, store list of (ap_index, position_in_ap) for fast delta
    # position_in_ap: 0,1,2,3 indicating which term
    pos_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(ap_list):
        for j, pos in enumerate(ap):
            pos_aps[pos].append((idx, j))
    
    # Thue-Morse initialization: t(i) = popcount(i) % 2
    def thue_morse(i):
        return bin(i).count('1') % 2
    
    colors = [thue_morse(i) for i in range(n)]
    
    # Count mono APs
    def count_mono(colors):
        count = 0
        for ap in ap_list:
            c = colors[ap[0]]
            if c == colors[ap[1]] == colors[ap[2]] == colors[ap[3]]:
                count += 1
        return count
    
    best = count_mono(colors)
    current = best
    best_colors = colors[:]
    
    # Simulated annealing
    T = 5.0
    T_min = 0.01
    cooling = 0.9995
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        if T < T_min:
            # Restart with a slightly perturbed version of best
            colors = best_colors[:]
            for _ in range(10):
                pos = rng.randrange(n)
                colors[pos] = 1 - colors[pos]
            current = count_mono(colors)
            T = 2.0
            continue
        
        # Pick a random position to flip
        pos = rng.randrange(n)
        old_color = colors[pos]
        new_color = 1 - old_color
        
        # Compute delta in mono count
        delta = 0
        for (ap_idx, j) in pos_aps[pos]:
            ap = ap_list[ap_idx]
            # Count how many of the other 3 positions match old_color
            others_match_old = 0
            others_match_new = 0
            for k in range(4):
                if k == j:
                    continue
                c = colors[ap[k]]
                if c == old_color:
                    others_match_old += 1
                else:
                    others_match_new += 1
            
            # Before flip: this AP is mono if all 4 match (they all matched old_color)
            # i.e., others_match_old == 3
            if others_match_old == 3:
                delta -= 1  # was mono, won't be after flip
            
            # After flip: this AP is mono if all 4 match new_color
            # i.e., others_match_new == 3
            if others_match_new == 3:
                delta += 1  # wasn't mono, will be after flip
        
        colors[pos] = new_color
        current += delta
        
        # Accept or reject
        if delta <= 0 or rng.random() < math.exp(-delta / T):
            if current < best:
                best = current
                best_colors = colors[:]
        else:
            colors[pos] = old_color
            current -= delta
        
        T *= cooling
    
    return best_colors

# EVOLVE-BLOCK-END
