import time
import random
import numpy as np

def construct():
    N = 256
    deadline = time.time() + 110
    
    # Precompute all 4-term APs as index tuples (0-based)
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(N - 3 * d):
            i0 = a
            i1 = a + d
            i2 = a + 2 * d
            i3 = a + 3 * d
            aps.append((i0, i1, i2, i3))
    
    num_aps = len(aps)
    aps_array = np.array(aps, dtype=np.int32)
    
    def count_mono(colors):
        """Count monochromatic 4-term APs. colors is a numpy array of 0s and 1s."""
        c0 = colors[aps_array[:, 0]]
        c1 = colors[aps_array[:, 1]]
        c2 = colors[aps_array[:, 2]]
        c3 = colors[aps_array[:, 3]]
        # Mono if all same: all 0 or all 1
        mono0 = (c0 == 0) & (c1 == 0) & (c2 == 0) & (c3 == 0)
        mono1 = (c0 == 1) & (c1 == 1) & (c2 == 1) & (c3 == 1)
        return int(np.sum(mono0) + np.sum(mono1))
    
    def compute_deltas(colors):
        """
        Compute the change in mono count if we flip each bit.
        Returns an array of deltas.
        """
        c = colors
        c_aps = c[aps_array] # Shape (M, 4)
        
        # Role 0: depends on cols 1,2,3
        o0 = c_aps[:, 1:4]
        eq0 = (o0[:, 0] == o0[:, 1]) & (o0[:, 1] == o0[:, 2])
        contrib0 = np.where(eq0, np.where(c_aps[:, 0] == o0[:, 0], -1, 1), 0)
        
        # Role 1: depends on cols 0,2,3
        o1 = np.column_stack((c_aps[:, 0], c_aps[:, 2], c_aps[:, 3]))
        eq1 = (o1[:, 0] == o1[:, 1]) & (o1[:, 1] == o1[:, 2])
        contrib1 = np.where(eq1, np.where(c_aps[:, 1] == o1[:, 0], -1, 1), 0)
        
        # Role 2: depends on cols 0,1,3
        o2 = np.column_stack((c_aps[:, 0], c_aps[:, 1], c_aps[:, 3]))
        eq2 = (o2[:, 0] == o2[:, 1]) & (o2[:, 1] == o2[:, 2])
        contrib2 = np.where(eq2, np.where(c_aps[:, 2] == o2[:, 0], -1, 1), 0)
        
        # Role 3: depends on cols 0,1,2
        o3 = c_aps[:, 0:3]
        eq3 = (o3[:, 0] == o3[:, 1]) & (o3[:, 1] == o3[:, 2])
        contrib3 = np.where(eq3, np.where(c_aps[:, 3] == o3[:, 0], -1, 1), 0)
        
        deltas = np.zeros(N, dtype=np.int32)
        np.add.at(deltas, aps_array[:, 0], contrib0)
        np.add.at(deltas, aps_array[:, 1], contrib1)
        np.add.at(deltas, aps_array[:, 2], contrib2)
        np.add.at(deltas, aps_array[:, 3], contrib3)
        
        return deltas

    def greedy_polish_vectorized(colors, max_iters=1000):
        """
        Local search using vectorized delta computation.
        """
        colors = colors.copy()
        current = count_mono(colors)
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            deltas = compute_deltas(colors)
            min_delta = np.min(deltas)
            
            if min_delta < 0:
                flip_mask = deltas < 0
                if np.any(flip_mask):
                    colors[flip_mask] = 1 - colors[flip_mask]
                    current = count_mono(colors)
                else:
                    break
            else:
                break
        
        return colors, current

    def simulated_annealing_vectorized(initial_colors, start_temp=1.0, end_temp=0.01, 
                            cooling_rate=0.999, max_steps=50000):
        """
        Simulated annealing using vectorized delta computation for single bit flips.
        """
        colors = initial_colors.copy()
        current = count_mono(colors)
        best_colors = colors.copy()
        best_count = current
        temp = start_temp
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            deltas = compute_deltas(colors)
            
            pos = random.randint(0, N - 1)
            delta = deltas[pos]
            
            if delta < 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
                colors[pos] = 1 - colors[pos]
                current += delta
                if current < best_count:
                    best_count = current
                    best_colors = colors.copy()
            
            temp *= cooling_rate
            if temp < end_temp:
                temp = end_temp
        
        return best_colors, best_count

    best_colors = None
    best_score = float('inf')
    
    colors_greedy = np.zeros(N, dtype=np.int8)
    for k in range(N):
        c0_count = np.sum(colors_greedy[:k] == 0)
        c1_count = np.sum(colors_greedy[:k] == 1)
        if c0_count <= c1_count:
            colors_greedy[k] = 0
        else:
            colors_greedy[k] = 1
            
    polished, score = greedy_polish_vectorized(colors_greedy)
    if score < best_score:
        best_score = score
        best_colors = polished.copy()
    
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        random_colors = np.random.randint(0, 2, N, dtype=np.int8)
        
        polished, score = greedy_polish_vectorized(random_colors, max_iters=2000)
        
        if score < best_score:
            best_score = score
            best_colors = polished.copy()
        
        if time.time() < deadline - 10:
            sa_colors, sa_score = simulated_annealing_vectorized(
                polished, start_temp=0.5, end_temp=0.01, 
                cooling_rate=0.9995, max_steps=20000
            )
            if sa_score < best_score:
                best_score = sa_score
                best_colors = sa_colors.copy()
    
    if time.time() < deadline - 2:
        final_colors, final_score = greedy_polish_vectorized(best_colors, max_iters=10000)
        if final_score < best_score:
            best_score = final_score
            best_colors = final_colors.copy()
    
    return [int(c) for c in best_colors]
