import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    N = 256
    
    # Precompute all 4-APs as 0-indexed tuples
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    num_aps = len(aps)
    
    # For each position, store which AP indices contain it
    pos_to_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    # Initialize with Thue-Morse
    colors = [bin(i).count('1') % 2 for i in range(N)]
    
    def compute_score(c):
        s = 0
        for ap in aps:
            if c[ap[0]] == c[ap[1]] == c[ap[2]] == c[ap[3]]:
                s += 1
        return s
    
    current_score = compute_score(colors)
    best_score = current_score
    best_colors = colors[:]
    
    random.seed(42)
    T = 5.0
    T_min = 0.005
    cooling = 0.99995
    
    while time.time() < deadline:
        i = random.randint(0, N - 1)
        old_c = colors[i]
        delta = 0
        for ap_idx in pos_to_aps[i]:
            ap = aps[ap_idx]
            same = 0
            for p in ap:
                if colors[p] == old_c:
                    same += 1
            was_mono = 1 if (same == 4 or same == 0) else 0
            now_mono = 1 if (same == 1) else 0
            delta += now_mono - was_mono
        
        accept = False
        if delta <= 0:
            accept = True
        else:
            if delta / T < 700:
                prob = 2.718281828 ** (-delta / T)
            else:
                prob = 0.0
            if random.random() < prob:
                accept = True
        
        if accept:
            colors[i] = 1 - colors[i]
            current_score += delta
            if current_score < best_score:
                best_score = current_score
                best_colors = colors[:]
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_colors
