import time
import random
import math
import numpy as np

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
    
    # Convert to numpy for vectorized operations
    aps_arr = np.array(aps, dtype=np.int32)  # shape (num_aps, 4)
    
    # Precompute for each position i, the "others" in each AP containing i
    # others[i] is a list of lists: for each AP containing i, the other 3 positions
    # This allows vectorized delta calculation
    others = [[] for _ in range(n)]
    for i in range(n):
        for ap_idx in pos_aps[i]:
            p0, p1, p2, p3 = aps[ap_idx]
            others[i].append([p for p in (p0, p1, p2, p3) if p != i])
            
    # Convert others to numpy arrays for speed
    others_np = []
    for i in range(n):
        if others[i]:
            arr = np.array(others[i], dtype=np.int32) # shape (k, 3)
            others_np.append(arr)
        else:
            others_np.append(np.empty((0, 3), dtype=np.int32))

    def count_mono(color):
        c = np.array(color, dtype=np.int8)
        mono = (c[aps_arr[:,0]] == c[aps_arr[:,1]]) & \
               (c[aps_arr[:,1]] == c[aps_arr[:,2]]) & \
               (c[aps_arr[:,2]] == c[aps_arr[:,3]])
        return int(np.sum(mono))

    def calc_delta(color_arr, i):
        """
        Calculate change in mono count if we flip position i.
        color_arr: numpy array of colors
        i: position to flip
        Returns delta
        """
        c_i = color_arr[i]
        new_c_i = 1 - c_i
        delta = 0
        
        # Get precomputed others for i
        oth = others_np[i]
        if oth.size == 0:
            return 0
            
        # oth shape (k, 3)
        # colors of others: c[oth] shape (k, 3)
        c_others = color_arr[oth]
        
        # Count currently mono:
        # An AP is currently mono if c_i == all others.
        # c_others == c_i for all 3 columns
        currently_mono = np.all(c_others == c_i, axis=1)
        delta -= np.sum(currently_mono)
        
        # Count becomes mono:
        # An AP becomes mono if new_c_i == all others.
        # c_others == new_c_i for all 3 columns
        becomes_mono = np.all(c_others == new_c_i, axis=1)
        delta += np.sum(becomes_mono)
        
        return int(delta)

    def local_search(color, max_iters=10000):
        color_arr = np.array(color, dtype=np.int8)
        current_score = count_mono(color)
        improved = True
        iters = 0
        
        while improved and iters < max_iters and time.time() < deadline - 5:
            improved = False
            iters += 1
            
            # Try all positions? That's 256 * cost(delta).
            # cost(delta) is small (~10-20 ops). 256 * 20 = 5000 ops per iteration.
            # 10000 iters = 50M ops. Might be slow in Python loop.
            # Let's vectorize the delta calculation for ALL positions at once?
            # Hard because each i has different 'others'.
            
            # Instead, let's just pick random positions or positions in mono APs?
            # Greedy: find best move.
            # Since n=256 is small, we can try all 256 positions.
            
            best_delta = 0
            best_pos = -1
            
            for p in range(n):
                d = calc_delta(color_arr, p)
                if d < best_delta:
                    best_delta = d
                    best_pos = p
                    
            if best_pos != -1:
                color_arr[best_pos] = 1 - color_arr[best_pos]
                current_score += best_delta
                improved = True
            else:
                # No improving move found. 
                # If we want to escape local minima, we could accept a random non-improving move.
                # But for now, let's stop this local search and let SA handle it if needed.
                break
                
        return color_arr.tolist(), current_score

    # Simulated Annealing with single flips and fast delta
    def simulated_annealing(initial_color, T_start=5.0, T_end=0.01, steps=50000):
        color_arr = np.array(initial_color, dtype=np.int8)
        current_score = count_mono(initial_color)
        best_color = color_arr.copy()
        best_score = current_score
        
        T = T_start
        cooling = (T_end / T_start) ** (1.0 / steps)
        
        for step in range(steps):
            if time.time() > deadline - 5:
                break
                
            # Pick random position
            p = random.randint(0, n - 1)
            
            # Calculate delta
            delta = calc_delta(color_arr, p)
            
            # Acceptance
            if delta < 0:
                accept = True
            else:
                if T > 0.001:
                    accept = random.random() < math.exp(-delta / T)
                else:
                    accept = False
                    
            if accept:
                color_arr[p] = 1 - color_arr[p]
                current_score += delta
                
                if current_score < best_score:
                    best_score = current_score
                    best_color = color_arr.copy()
                    
            T *= cooling
            
        return best_color.tolist(), best_score

    # Main logic
    random.seed(42)
    
    best_overall_color = None
    best_overall_score = float('inf')
    
    # 1. Thue-Morse initialization
    def thue_morse(n):
        return [bin(i).count('1') % 2 for i in range(n)]
        
    tm = thue_morse(n)
    tm, score_tm = local_search(tm, max_iters=2000)
    best_overall_color = tm[:]
    best_overall_score = score_tm
    
    # 2. Random restarts with local search
    for trial in range(20):
        if time.time() > deadline - 20:
            break
        random.seed(42 + trial)
        rand_color = [random.randint(0, 1) for _ in range(n)]
        rand_color, rand_score = local_search(rand_color, max_iters=5000)
        
        if rand_score < best_overall_score:
            best_overall_score = rand_score
            best_overall_color = rand_color[:]
            
    # 3. SA from best solution found
    if time.time() < deadline - 10:
        sa_color, sa_score = simulated_annealing(best_overall_color, T_start=2.0, T_end=0.001, steps=50000)
        if sa_score < best_overall_score:
            best_overall_score = sa_score
            best_overall_color = sa_color[:]
            
        # Final polish
        best_overall_color, best_overall_score = local_search(best_overall_color, max_iters=2000)

    return best_overall_color
