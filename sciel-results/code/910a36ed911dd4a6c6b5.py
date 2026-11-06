import numpy as np
import time
import random

def construct():
        N = 256
        start_time = time.time()
        deadline = start_time + 115.0

        # Precompute 4-APs
        aps = []
        for d in range(1, N // 3 + 1):
            max_a = N - 3 * d
            for a in range(1, max_a + 1):
                aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
        aps_array = np.array(aps, dtype=np.int32)

        def count_mono(x):
            vals = x[aps_array]
            is_mono = (vals[:, 0] == vals[:, 1]) & (vals[:, 1] == vals[:, 2]) & (vals[:, 2] == vals[:, 3])
            return int(np.sum(is_mono))

        def thue_morse(n):
            seq = [0]
            while len(seq) < n:
                seq = seq + [1 - b for b in seq]
            return np.array(seq[:n], dtype=np.int8)

        # Initial Solution
        x = thue_morse(N)
        current_cost = count_mono(x)
        best_x = x.copy()
        best_cost = current_cost
        
        last_improvement_time = start_time

        # Helper for Greedy Polish
        def greedy_polish(x, current_cost, max_time=2.0):
            improved = True
            while improved and time.time() < max_time:
                improved = False
                for i in range(N):
                    old_val = x[i]
                    x[i] = 1 - old_val
                    new_cost = count_mono(x)
                    if new_cost < current_cost:
                        current_cost = new_cost
                        improved = True
                    else:
                        x[i] = old_val
            return x, current_cost

        # Initial Polish
        x, current_cost = greedy_polish(x, current_cost)
        if current_cost < best_cost:
            best_cost = current_cost
            best_x = x.copy()
            last_improvement_time = time.time()

        # Simulated Annealing Loop
        T = 5.0
        cooling = 0.999
        min_T = 0.1
        stagnation_limit = 5.0 # seconds

        while time.time() < deadline - 5.0:
            i = np.random.randint(N)
            old_val = x[i]
            x[i] = 1 - old_val
            new_cost = count_mono(x)
            
            delta = new_cost - current_cost
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_x = x.copy()
                    last_improvement_time = time.time()
            else:
                x[i] = old_val
            
            T *= cooling
            if T < min_T:
                T = min_T

            # Check for stagnation
            if time.time() - last_improvement_time > stagnation_limit:
                # Restart from best with noise
                x = best_x.copy()
                current_cost = best_cost
                # Add noise
                num_flips = np.random.randint(3, 8)
                flip_indices = np.random.choice(N, size=num_flips, replace=False)
                for idx in flip_indices:
                    x[idx] = 1 - x[idx]
                current_cost = count_mono(x)
                T = 5.0
                last_improvement_time = time.time()
                
                # Quick polish
                x, current_cost = greedy_polish(x, current_cost, max_time=1.0)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_x = x.copy()
                    last_improvement_time = time.time()

        # Final Polish
        if time.time() < deadline - 1.0:
            best_x, best_cost = greedy_polish(best_x, best_cost, max_time=deadline - time.time() - 0.5)

        return [int(v) for v in best_x]
