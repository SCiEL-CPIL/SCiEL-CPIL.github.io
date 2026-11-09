"""
Max-SAT based approach for R(4,5;3) coloring.
We try to find a valid coloring on n=34 (or as large as possible).
Strategy: Start with a random or heuristic initial coloring, then use local search
to eliminate all-red K4s and all-blue K5s.
"""
import random
import numpy as np
from itertools import combinations
import time

def construct():
    # Target n
    # The published construction has 34 points. Let's aim for that.
    # If we can't find a valid one for 34 quickly, we might drop to 33 or lower.
    # But the goal is to maximize n.
    
    # Time limit is 120s. Let's use most of it for search if needed.
    # However, a good heuristic might find it fast.
    
    # Let's try n=34 first.
    target_n = 34
    
    # Precompute all K4s (sets of 4 vertices) and K5s (sets of 5 vertices)
    # For a given n, a K4 is a set {a,b,c,d}. It is "bad" (all red) if all 4 triples are red.
    # A K5 is a set {a,b,c,d,e}. It is "bad" (all blue) if all 10 triples are blue.
    
    # We will use a bit-parallel local search.
    # State: A bit vector of length M = C(n,3).
    # 1 = Red, 0 = Blue (or vice versa, let's say 1=Red).
    
    # Precompute for each triple, which K4s it belongs to and which K5s it belongs to?
    # Actually, it's easier to precompute for each K4, the indices of its 4 triples.
    # And for each K5, the indices of its 10 triples.
    
    # Let's build these structures for n=34.
    # M = C(34,3) = 34*33*32/6 = 5984 triples.
    # K4s: C(34,4) = 34*33*32*31/24 = 46376 K4s.
    # K5s: C(34,5) = 34*33*32*31*30/120 = 278256 K5s.
    
    # This is manageable.
    
    # Let's try to find a valid coloring.
    # We will use a simulated annealing or simple hill climbing with restarts.
    # Objective: Minimize (number of all-red K4s + number of all-blue K5s).
    # If we reach 0, we have a valid coloring.
    
    def generate_coloring(n, strategy='random'):
        M = n*(n-1)*(n-2)//6
        if strategy == 'random':
            return np.random.randint(0, 2, M)
        elif strategy == 'mod3':
            # The seed strategy: red if sum % 3 == 0
            colors = []
            for a,b,c in combinations(range(n), 3):
                colors.append(1 if (a+b+c)%3 == 0 else 0)
            return np.array(colors)
        else:
            return np.random.randint(0, 2, M)
    
    def count_violations(n, colors):
        """
        Count all-red K4s and all-blue K5s.
        colors: array of 0/1, 1=Red.
        """
        # Map from (a,b,c) to index
        # We need to check all K4s and K5s.
        # This is O(C(n,4) + C(n,5)) which is ~ 320k operations.
        # In Python, this might be slow if done naively in a loop for every iteration.
        # We need to speed this up or use a smarter search.
        
        # Let's precompute the triple indices for each K4 and K5.
        # But we can't do that inside the loop for every evaluation if it's slow.
        # Let's precompute the lists of triple indices for each K4 and K5 once.
        
        # This function is called many times, so we should use the precomputed lists.
        pass # Placeholder
    
    # Precomputation for n=34
    n = target_n
    M = n*(n-1)*(n-2)//6
    
    # Create a mapping from (a,b,c) to index
    idx_map = {}
    k = 0
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                idx_map[(a,b,c)] = k
                k += 1
    
    # Precompute triple indices for each K4
    k4_triples = []
    for quad in combinations(range(n), 4):
        a,b,c,d = quad
        t1 = idx_map[(a,b,c)]
        t2 = idx_map[(a,b,d)]
        t3 = idx_map[(a,c,d)]
        t4 = idx_map[(b,c,d)]
        k4_triples.append((t1,t2,t3,t4))
        
    # Precompute triple indices for each K5
    k5_triples = []
    for quint in combinations(range(n), 5):
        a,b,c,d,e = quint
        # 10 triples
        t1 = idx_map[(a,b,c)]
        t2 = idx_map[(a,b,d)]
        t3 = idx_map[(a,b,e)]
        t4 = idx_map[(a,c,d)]
        t5 = idx_map[(a,c,e)]
        t6 = idx_map[(a,d,e)]
        t7 = idx_map[(b,c,d)]
        t8 = idx_map[(b,c,e)]
        t9 = idx_map[(b,d,e)]
        t10 = idx_map[(c,d,e)]
        k5_triples.append((t1,t2,t3,t4,t5,t6,t7,t8,t9,t10))
        
    # Convert to numpy arrays for faster access?
    # k4_arr: shape (num_k4, 4)
    # k5_arr: shape (num_k5, 10)
    k4_arr = np.array(k4_triples, dtype=np.int32)
    k5_arr = np.array(k5_triples, dtype=np.int32)
    
    def evaluate(colors):
        """
        Returns (num_red_k4, num_blue_k5)
        colors: numpy array of 0/1
        """
        # Red K4: all 4 triples are 1
        # Blue K5: all 10 triples are 0
        
        # Vectorized check for K4s
        # k4_arr is (N4, 4). colors[k4_arr] is (N4, 4).
        # We want to check if all are 1.
        # sum along axis 1 == 4
        red_k4_count = np.sum(np.all(colors[k4_arr] == 1, axis=1))
        
        # Vectorized check for K5s
        # k5_arr is (N5, 10). colors[k5_arr] is (N5, 10).
        # We want to check if all are 0.
        # sum along axis 1 == 0
        blue_k5_count = np.sum(np.all(colors[k5_arr] == 0, axis=1))
        
        return red_k4_count, blue_k5_count
    
    def total_violations(colors):
        r, b = evaluate(colors)
        return r + b
    
    # Search
    best_colors = None
    best_violations = float('inf')
    
    # Try a few initial states
    # 1. Random
    # 2. Mod 3
    # 3. Random with bias
    
    start_time = time.time()
    time_limit = 110  # Leave some buffer
    
    # Heuristic: Start with random, then hill climb.
    # If stuck, restart.
    
    # To speed up evaluation, we can use the fact that flipping one bit
    # only affects K4s and K5s containing that triple.
    # But implementing an incremental update is complex.
    # Given N4=46k and N5=278k, the vectorized evaluation is ~ 324k checks.
    # In numpy, this is very fast. Maybe 1-2 ms.
    # So we can do thousands of iterations per second.
    
    # Let's run a simple hill climbing / simulated annealing.
    
    for restart in range(50):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize
        if restart == 0:
            colors = generate_coloring(n, 'mod3')
        else:
            colors = generate_coloring(n, 'random')
            
        current_violations = total_violations(colors)
        
        if current_violations == 0:
            best_colors = colors.copy()
            break
            
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors.copy()
            
        # Hill Climbing
        accepted = 0
        rejected = 0
        for i in range(2000):
            if time.time() - start_time > time_limit:
                break
                
            # Pick a random triple to flip
            idx = np.random.randint(0, M)
            
            # Flip
            old_val = colors[idx]
            colors[idx] = 1 - old_val
            
            new_violations = total_violations(colors)
            
            # If better or equal, accept
            # If worse, maybe accept with small probability?
            # Let's stick to strict hill climbing for now, or simple SA.
            # Since we want to get to 0, strict is good if we have restarts.
            # But we might get stuck in local minima.
            # Let's use a simple "accept if not worse" or "accept if better".
            
            if new_violations <= current_violations:
                current_violations = new_violations
                accepted += 1
                if current_violations == 0:
                    best_colors = colors.copy()
                    break
            else:
                # Revert
                colors[idx] = old_val
                rejected += 1
                
        if best_colors is not None and total_violations(best_colors) == 0:
            break
            
    if best_colors is not None:
        # Check validity one last time
        r, b = evaluate(best_colors)
        if r == 0 and b == 0:
            # Convert to list of ints
            return {'n': n, 'colors': best_colors.tolist()}
    
    # If n=34 failed, try n=33?
    # The task says "Maximize n".
    # If we can't find 34, we should return the best we found.
    # But the seed returns n=9.
    # If we find a valid n=30, that's better than 9.
    
    # Let's try smaller n if 34 fails.
    for n_try in [33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9]:
        if time.time() - start_time > time_limit - 5:
            break
            
        # Re-run search for this n
        # We need to rebuild the precomputed arrays for this n
        M_try = n_try*(n_try-1)*(n_try-2)//6
        idx_map_try = {}
        k = 0
        for a in range(n_try):
            for b in range(a+1, n_try):
                for c in range(b+1, n_try):
                    idx_map_try[(a,b,c)] = k
                    k += 1
                    
        k4_triples_try = []
        for quad in combinations(range(n_try), 4):
            a,b,c,d = quad
            t1 = idx_map_try[(a,b,c)]
            t2 = idx_map_try[(a,b,d)]
            t3 = idx_map_try[(a,c,d)]
            t4 = idx_map_try[(b,c,d)]
            k4_triples_try.append((t1,t2,t3,t4))
            
        k5_triples_try = []
        for quint in combinations(range(n_try), 5):
            a,b,c,d,e = quint
            t1 = idx_map_try[(a,b,c)]
            t2 = idx_map_try[(a,b,d)]
            t3 = idx_map_try[(a,b,e)]
            t4 = idx_map_try[(a,c,d)]
            t5 = idx_map_try[(a,c,e)]
            t6 = idx_map_try[(a,d,e)]
            t7 = idx_map_try[(b,c,d)]
            t8 = idx_map_try[(b,c,e)]
            t9 = idx_map_try[(b,d,e)]
            t10 = idx_map_try[(c,d,e)]
            k5_triples_try.append((t1,t2,t3,t4,t5,t6,t7,t8,t9,t10))
            
        k4_arr_try = np.array(k4_triples_try, dtype=np.int32) if k4_triples_try else np.empty((0,4), dtype=np.int32)
        k5_arr_try = np.array(k5_triples_try, dtype=np.int32) if k5_triples_try else np.empty((0,10), dtype=np.int32)
        
        def evaluate_try(colors):
            if len(k4_arr_try) > 0:
                red_k4_count = np.sum(np.all(colors[k4_arr_try] == 1, axis=1))
            else:
                red_k4_count = 0
            if len(k5_arr_try) > 0:
                blue_k5_count = np.sum(np.all(colors[k5_arr_try] == 0, axis=1))
            else:
                blue_k5_count = 0
            return red_k4_count, blue_k5_count
        
        found = False
        for restart in range(20):
            if time.time() - start_time > time_limit:
                break
                
            colors_try = np.random.randint(0, 2, M_try)
            current_violations = evaluate_try(colors_try)[0] + evaluate_try(colors_try)[1]
            
            if current_violations == 0:
                found = True
                break
                
            for i in range(1000):
                if time.time() - start_time > time_limit:
                    break
                idx = np.random.randint(0, M_try)
                old_val = colors_try[idx]
                colors_try[idx] = 1 - old_val
                new_violations = evaluate_try(colors_try)[0] + evaluate_try(colors_try)[1]
                if new_violations <= current_violations:
                    current_violations = new_violations
                    if current_violations == 0:
                        found = True
                        break
                else:
                    colors_try[idx] = old_val
            if found:
                break
        
        if found:
            r, b = evaluate_try(colors_try)
            if r == 0 and b == 0:
                return {'n': n_try, 'colors': colors_try.tolist()}
                
    # Fallback: return seed solution
    n = 9
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
    return {'n': n, 'colors': colors}

# EVOLVE-BLOCK-END
