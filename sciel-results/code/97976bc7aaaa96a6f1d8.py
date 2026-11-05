import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    def solve_n(n, time_limit):
        # Precompute triples
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Precompute K4s and K5s
        k4s = []
        for quad in combinations(range(n), 4):
            a,b,c,d = quad
            idxs = (
                triple_to_idx[(a,b,c)],
                triple_to_idx[(a,b,d)],
                triple_to_idx[(a,c,d)],
                triple_to_idx[(b,c,d)]
            )
            k4s.append(idxs)
            
        k5s = []
        for quint in combinations(range(n), 5):
            idxs = []
            for triple in combinations(quint, 3):
                idxs.append(triple_to_idx[triple])
            k5s.append(tuple(idxs))
            
        num_k4 = len(k4s)
        num_k5 = len(k5s)
        
        # Build reverse index: triple_idx -> list of (clique_idx, clique_type)
        # We need to know which K4s and K5s a triple belongs to.
        # And for each, we need to know the other 3 or 9 triples?
        # Actually, for incremental update, we just need the count of reds/blues in that clique.
        
        triple_to_k4s = [[] for _ in range(num_triples)]
        triple_to_k5s = [[] for _ in range(num_triples)]
        
        for i, idxs in enumerate(k4s):
            for t in idxs:
                triple_to_k4s[t].append(i)
                
        for i, idxs in enumerate(k5s):
            for t in idxs:
                triple_to_k5s[t].append(i)

        # Initialize colors
        # Try a few different initializations
        # 1. Random
        # 2. Mod 3
        # 3. Mod 4
        
        best_cost = float('inf')
        best_colors = None
        
        # We will run SA for a fixed time, or multiple restarts
        # Given time limit, we do one long SA run with restarts if stuck.
        
        # State variables for incremental update
        # red_counts[i] for k4s[i]
        # blue_counts[j] for k5s[j]
        
        def init_state(colors):
            red_counts = np.zeros(num_k4, dtype=np.int8)
            blue_counts = np.zeros(num_k5, dtype=np.int8)
            
            for i, idxs in enumerate(k4s):
                # Count reds (0)
                # Using numpy sum for speed?
                # colors is int8 array. Red is 0.
                # red_count = sum(colors[idx] == 0 for idx in idxs)
                # Vectorized:
                red_counts[i] = np.sum(colors[list(idxs)] == 0)
                
            for j, idxs in enumerate(k5s):
                # Count blues (1)
                blue_counts[j] = np.sum(colors[list(idxs)] == 1)
                
            # Initial cost
            cost = np.sum(red_counts == 4) + np.sum(blue_counts == 10)
            return red_counts, blue_counts, cost

        # SA Loop
        # We try to find cost 0.
        
        # Strategy: 
        # 1. Try n=34.
        # 2. If not found, try n=33, etc.
        
        # For each n, run SA for a portion of the time.
        
        pass # Placeholder

    # Actually, let's structure the main loop to try n=34, then 33, etc.
    # Allocate time based on n.
    
    # Since n=34 is the target, give it most of the time.
    # If n=34 fails, try 33 with remaining time.
    
    # To make it robust, we define a helper that runs SA for a given n and time limit.
    
    def run_sa(n, time_limit):
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        k4s = []
        for quad in combinations(range(n), 4):
            a,b,c,d = quad
            idxs = (
                triple_to_idx[(a,b,c)],
                triple_to_idx[(a,b,d)],
                triple_to_idx[(a,c,d)],
                triple_to_idx[(b,c,d)]
            )
            k4s.append(idxs)
            
        k5s = []
        for quint in combinations(range(n), 5):
            idxs = []
            for triple in combinations(quint, 3):
                idxs.append(triple_to_idx[triple])
            k5s.append(tuple(idxs))
            
        num_k4 = len(k4s)
        num_k5 = len(k5s)
        
        triple_to_k4s = [[] for _ in range(num_triples)]
        triple_to_k5s = [[] for _ in range(num_triples)]
        
        for i, idxs in enumerate(k4s):
            for t in idxs:
                triple_to_k4s[t].append(i)
                
        for i, idxs in enumerate(k5s):
            for t in idxs:
                triple_to_k5s[t].append(i)

        # Initialization strategies
        inits = []
        # Random
        inits.append(np.random.randint(0, 2, num_triples, dtype=np.int8))
        # Mod 3
        inits.append(np.array([0 if (a+b+c)%3==0 else 1 for a,b,c in triples], dtype=np.int8))
        # Mod 2
        inits.append(np.array([0 if (a+b+c)%2==0 else 1 for a,b,c in triples], dtype=np.int8))
        
        best_cost = float('inf')
        best_colors = None
        
        for init_colors in inits:
            if time.time() > time_limit: break
            
            red_counts = np.zeros(num_k4, dtype=np.int8)
            blue_counts = np.zeros(num_k5, dtype=np.int8)
            
            for i, idxs in enumerate(k4s):
                red_counts[i] = np.sum(init_colors[list(idxs)] == 0)
            for j, idxs in enumerate(k5s):
                blue_counts[j] = np.sum(init_colors[list(idxs)] == 1)
                
            current_cost = np.sum(red_counts == 4) + np.sum(blue_counts == 10)
            
            if current_cost == 0:
                return init_colors, True
                
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = init_colors.copy()
                
            colors = init_colors.copy()
            temp = 1.0
            cool_rate = 0.9995
            
            while time.time() < time_limit:
                t = np.random.randint(num_triples)
                old_color = colors[t]
                new_color = 1 - old_color
                
                delta_cost = 0
                
                # Update K4s
                for k4_idx in triple_to_k4s[t]:
                    idxs = k4s[k4_idx]
                    # Check if t is in idxs (it is)
                    # Current state of this K4
                    # We track red_counts[k4_idx]
                    # If old_color was 0 (red), red_count was X.
                    # If new_color is 1 (blue), red_count becomes X-1.
                    # If old_color was 1 (blue), red_count was X.
                    # If new_color is 0 (red), red_count becomes X+1.
                    
                    rc = red_counts[k4_idx]
                    if old_color == 0:
                        # Red -> Blue
                        # rc decreases by 1
                        if rc == 4:
                            delta_cost -= 1 # Violation removed
                        # If rc was 3, becomes 2. No violation change.
                        # If rc was <3, no change.
                        red_counts[k4_idx] = rc - 1
                    else:
                        # Blue -> Red
                        # rc increases by 1
                        if rc == 3:
                            delta_cost += 1 # Violation added
                        red_counts[k4_idx] = rc + 1

                # Update K5s
                for k5_idx in triple_to_k5s[t]:
                    bc = blue_counts[k5_idx]
                    if old_color == 1:
                        # Blue -> Red
                        # bc decreases by 1
                        if bc == 10:
                            delta_cost -= 1
                        blue_counts[k5_idx] = bc - 1
                    else:
                        # Red -> Blue
                        # bc increases by 1
                        if bc == 9:
                            delta_cost += 1
                        blue_counts[k5_idx] = bc + 1

                new_cost = current_cost + delta_cost
                
                # Acceptance
                if new_cost < current_cost:
                    colors[t] = new_color
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colors = colors.copy()
                        if best_cost == 0:
                            return best_colors, True
                elif new_cost == current_cost:
                    if np.random.rand() < 0.5:
                        colors[t] = new_color
                        current_cost = new_cost
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_colors = colors.copy()
                else:
                    if np.random.rand() < np.exp(-delta_cost / temp):
                        colors[t] = new_color
                        current_cost = new_cost
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_colors = colors.copy()
                            
                temp *= cool_rate
                if temp < 0.01:
                    temp = 1.0 # Restart temperature
                    
        return best_colors, (best_cost == 0)

    # Main Logic
    # Try n=34, 33, 32...
    # Allocate time: 100s for 34, 30s for 33, 10s for 32, 5s for 31, 5s for 30.
    
    final_n = 9
    final_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]
    
    # Time allocation
    time_alloc = {
        34: 100.0,
        33: 30.0,
        32: 10.0,
        31: 5.0,
        30: 5.0
    }
    
    for n in [34, 33, 32, 31, 30]:
        if time.time() > deadline - 5: break
        t_limit = time.time() + time_alloc[n]
        if t_limit > deadline: t_limit = deadline
        
        try:
            colors, is_valid = run_sa(n, t_limit)
            if is_valid:
                final_n = n
                final_colors = colors.tolist()
                # If we found a valid one, we can stop if it's 34?
                # Or try to go higher? Max is 40.
                # But 34 is the benchmark.
                # If we found 34, we should probably return it immediately to save time for safety.
                # However, if we have time, we could try 35?
                # R(4,5;3) is likely <= 34 or 35.
                # If we find 34, it's a huge win.
                # Let's return 34 immediately if found.
                break
        except Exception as e:
            print(f"Error n={n}: {e}")
            continue
            
    return {'n': final_n, 'colors': final_colors}
