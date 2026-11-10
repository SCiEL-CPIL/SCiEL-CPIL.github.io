import time
import random
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Precompute all 4-term arithmetic progressions
    # a, a+d, a+2d, a+3d with 1 <= a, a+3d <= N
    # 0-indexed: a-1, a+d-1, a+2d-1, a+3d-1
    aps_list = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            aps_list.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    aps = np.array(aps_list, dtype=np.int32)
    M = len(aps)

    def count_mono(color):
        """
        Count monochromatic 4-APs.
        color: numpy array of 0s and 1s, shape (N,)
        """
        cols = color[aps]
        # Check if all 4 are equal
        mono = (cols[:, 0] == cols[:, 1]) & (cols[:, 1] == cols[:, 2]) & (cols[:, 2] == cols[:, 3])
        return np.sum(mono)

    def local_search(color, max_iter=200):
        """
        Hill climbing with 1-bit flips.
        """
        current_cost = count_mono(color)
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            for idx in range(N):
                old_val = color[idx]
                color[idx] = 1 - old_val
                new_cost = count_mono(color)
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                else:
                    color[idx] = old_val
            iter_count += 1
        return current_cost

    def sa_optimize(color, T_start=10.0, T_min=0.01, alpha=0.9999, max_steps=50000):
        """
        Simulated Annealing with 1, 2, 3-bit flips.
        """
        current_cost = count_mono(color)
        best_cost = current_cost
        best_color = color.copy()
        
        T = T_start
        rng = random.Random()
        
        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
            
            # Choose move size
            r = rng.random()
            if r < 0.5:
                k = 1
            elif r < 0.8:
                k = 2
            else:
                k = 3
            
            # Pick k distinct indices
            indices = rng.sample(range(N), k)
            
            # Flip
            for idx in indices:
                color[idx] = 1 - color[idx]
            
            new_cost = count_mono(color)
            delta = new_cost - current_cost
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_color = color.copy()
            else:
                # Revert
                for idx in indices:
                    color[idx] = 1 - color[idx]
            
            T *= alpha
            if T < T_min:
                T = T_min
                
            # Early exit if perfect (unlikely) or very low
            if best_cost <= 2:
                break
        
        return best_cost, best_color

    # Generate diverse initial candidates
    candidates = []
    
    # 1. Random
    for _ in range(10):
        candidates.append(np.array([random.randint(0, 1) for _ in range(N)], dtype=np.int8))
        
    # 2. Block patterns
    for bs in [2, 3, 4, 5, 8, 16]:
        candidates.append(np.array([(i // bs) % 2 for i in range(N)], dtype=np.int8))
        
    # 3. Thue-Morse like (xor of bits)
    tm = np.array([bin(i).count('1') % 2 for i in range(N)], dtype=np.int8)
    candidates.append(tm)
    
    # 4. Van der Corput / bit reversal
    vc = np.array([bin(i).count('1') % 2 for i in range(N)], dtype=np.int8) # Same as TM for this purpose, let's do something else
    # Just random shifts of TM
    for shift in range(4):
        candidates.append(np.roll(tm, shift))

    # Evaluate and refine candidates
    best_overall_cost = float('inf')
    best_overall_color = None
    
    for c in candidates:
        c = c.copy()
        # Quick local search
        cost = local_search(c, max_iter=50)
        
        if cost < best_overall_cost:
            best_overall_cost = cost
            best_overall_color = c.copy()
            
    # Run SA from the best local search result and a few others
    # We will run SA for a fixed time budget or step count, whichever comes first
    
    # We can run multiple SA instances in parallel? No, single thread.
    # We can run SA on the best candidate, then on the 2nd best, etc.
    
    # Let's run SA on the top 3 candidates from local search
    scored_candidates = []
    for c in candidates:
        c_copy = c.copy()
        cost = local_search(c_copy, max_iter=50)
        scored_candidates.append((cost, c_copy))
        
    scored_candidates.sort(key=lambda x: x[0])
    
    # Take top 3
    for cost, color in scored_candidates[:3]:
        if time.time() > deadline - 20.0:
            break
        sa_cost, sa_color = sa_optimize(color, T_start=5.0, T_min=0.01, alpha=0.9999, max_steps=20000)
        if sa_cost < best_overall_cost:
            best_overall_cost = sa_cost
            best_overall_color = sa_color.copy()
            
        # Final polish
        if time.time() < deadline - 5.0:
            local_search(best_overall_color, max_iter=100)
            best_overall_cost = count_mono(best_overall_color)

    return [int(x) for x in best_overall_color]
