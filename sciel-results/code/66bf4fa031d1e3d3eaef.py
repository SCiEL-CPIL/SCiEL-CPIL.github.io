import time
import random
import math

def construct():
    n = 256
    deadline = time.time() + 140  # Safety margin
    
    # 1. Precompute APs
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            p0 = a - 1
            p1 = a + d - 1
            p2 = a + 2 * d - 1
            p3 = a + 3 * d - 1
            aps.append((p0, p1, p2, p3))
    
    num_aps = len(aps)
    
    # Map position to list of AP indices
    pos_aps = [[] for _ in range(n)]
    for idx in range(num_aps):
        p0, p1, p2, p3 = aps[idx]
        pos_aps[p0].append(idx)
        pos_aps[p1].append(idx)
        pos_aps[p2].append(idx)
        pos_aps[p3].append(idx)
        
    # Helper to compute initial mono count
    def count_mono(color):
        count = 0
        for p0, p1, p2, p3 in aps:
            if color[p0] == color[p1] == color[p2] == color[p3]:
                count += 1
        return count

    # Helper to compute delta for flipping position i
    def calc_delta(color, i):
        delta = 0
        c_i = color[i]
        new_c_i = 1 - c_i
        for ap_idx in pos_aps[i]:
            p0, p1, p2, p3 = aps[ap_idx]
            # Get colors of the other 3 positions
            # We need to check if the AP is currently mono
            # and if it becomes mono
            
            # Efficient check:
            # Count how many of the 4 positions have color c_i
            # and how many have color new_c_i
            
            # Actually, simpler:
            # Is it currently mono?
            # color[p0]==color[p1]==color[p2]==color[p3]
            # Since color[i] is c_i, the others must be c_i.
            
            # Let's just check the other 3.
            # If all other 3 are c_i, it's currently mono.
            # If all other 3 are new_c_i, it becomes mono.
            
            # Find the other 3 positions
            # p0, p1, p2, p3. One of them is i.
            # We can just iterate and skip i.
            
            others = []
            for p in (p0, p1, p2, p3):
                if p != i:
                    others.append(color[p])
            
            # Check if all others are c_i
            if all(v == c_i for v in others):
                delta -= 1 # Breaks a mono AP
            
            # Check if all others are new_c_i
            if all(v == new_c_i for v in others):
                delta += 1 # Creates a mono AP
                
        return delta

    # Local Search (Greedy)
    def local_search(color, max_iters=50):
        current_score = count_mono(color)
        improved = True
        while improved and max_iters > 0:
            improved = False
            max_iters -= 1
            # Try flipping each position
            # Optimization: Only try positions in mono APs?
            # If a position is not in any mono AP, flipping it might create one,
            # but it might also break others? No, if it's not in a mono AP,
            # flipping it cannot break a mono AP (since it wasn't part of one).
            # It can only create mono APs. So delta >= 0.
            # Thus, we only need to try positions that are part of at least one mono AP.
            
            mono_pos = set()
            for p0, p1, p2, p3 in aps:
                if color[p0] == color[p1] == color[p2] == color[p3]:
                    mono_pos.update([p0, p1, p2, p3])
                    
            if not mono_pos:
                break
                
            best_delta = 0
            best_pos = -1
            for p in mono_pos:
                d = calc_delta(color, p)
                if d < best_delta:
                    best_delta = d
                    best_pos = p
                    
            if best_pos != -1:
                color[best_pos] = 1 - color[best_pos]
                current_score += best_delta
                improved = True
                
        return color, current_score

    # Simulated Annealing
    def simulated_annealing(initial_color, T_start=10.0, T_end=0.1, steps=50000):
        color = initial_color[:]
        current_score = count_mono(color)
        best_color = color[:]
        best_score = current_score
        
        T = T_start
        cooling = (T_end / T_start) ** (1.0 / steps)
        
        for step in range(steps):
            if time.time() > deadline - 5:
                break
                
            # Pick random position
            i = random.randint(0, n - 1)
            
            # Calculate delta
            delta = calc_delta(color, i)
            
            # Acceptance
            if delta < 0:
                accept = True
            else:
                if T > 0:
                    accept = random.random() < math.exp(-delta / T)
                else:
                    accept = False
                    
            if accept:
                color[i] = 1 - color[i]
                current_score += delta
                
                if current_score < best_score:
                    best_score = current_score
                    best_color = color[:]
                    
            T *= cooling
            
        return best_color, best_score

    # Main logic
    random.seed(42)
    
    # Initialize with Thue-Morse
    def thue_morse(n):
        return [bin(i).count('1') % 2 for i in range(n)]
    
    best_overall_color = None
    best_overall_score = float('inf')
    
    # Start with Thue-Morse
    tm = thue_morse(n)
    tm, score_tm = local_search(tm, max_iters=20)
    best_overall_color = tm[:]
    best_overall_score = score_tm
    
    # Run SA from Thue-Morse
    sa_color, sa_score = simulated_annealing(tm, T_start=5.0, T_end=0.01, steps=20000)
    if sa_score < best_overall_score:
        best_overall_score = sa_score
        best_overall_color = sa_color[:]
        
    # Run SA from Random
    for _ in range(3):
        if time.time() > deadline - 10:
            break
        rand_color = [random.randint(0, 1) for _ in range(n)]
        rand_color, rand_score = local_search(rand_color, max_iters=10)
        
        sa_color, sa_score = simulated_annealing(rand_color, T_start=5.0, T_end=0.01, steps=10000)
        if sa_score < best_overall_score:
            best_overall_score = sa_score
            best_overall_color = sa_color[:]
            
    # Final polish
    best_overall_color, best_overall_score = local_search(best_overall_color, max_iters=100)
    
    return best_overall_color
