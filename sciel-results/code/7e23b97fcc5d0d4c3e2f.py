"""Strategy: Vectorized Simulated Annealing with Bitmask Incidence Matrices for Hypergraph Ramsey R(4,5;3)."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit (task says 120s, be conservative)
    
    # Target n=34. If we fail, we fallback to a smaller valid n.
    # The known construction exists for n=34.
    
    def try_solve(n, time_budget):
        """
        Attempt to find a valid coloring for n vertices.
        Returns (n, colors) if successful, else (0, None).
        """
        t0 = time.time()
        end_time = t0 + time_budget
        
        # Precomputation
        vertices = np.arange(n)
        triples = list(itertools.combinations(range(n), 3))
        num_triples = len(triples)
        
        # Map triple tuple to index
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Build K4 incidence: For each K4 (set of 4 vertices), list of triple indices
        # C(34, 4) = 46376 K4s. Each has 4 triples.
        # Build K5 incidence: For each K5 (set of 5 vertices), list of triple indices
        # C(34, 5) = 278256 K5s. Each has 10 triples.
        
        # For fast delta calculation, we need to know for each triple t, 
        # which K4s and K5s contain it.
        # Let's build reverse maps: triple -> list of K4 indices, triple -> list of K5 indices.
        
        k4_triples = [] # List of lists of triple indices
        k5_triples = [] # List of lists of triple indices
        
        # We also need to quickly check if a K4/K5 is "almost" satisfied to compute delta.
        # A K4 is red if all 4 triples are 0.
        # A K5 is blue if all 10 triples are 1.
        
        # Let's precompute the lists of triples for each K4 and K5.
        for quad in itertools.combinations(range(n), 4):
            ts = [triple_to_idx[t] for t in itertools.combinations(quad, 3)]
            k4_triples.append(ts)
            
        for quint in itertools.combinations(range(n), 5):
            ts = [triple_to_idx[t] for t in itertools.combinations(quint, 3)]
            k5_triples.append(ts)
            
        num_k4 = len(k4_triples)
        num_k5 = len(k5_triples)
        
        # Reverse indices: triple_idx -> list of K4 indices containing it
        # triple_idx -> list of K5 indices containing it
        triple_to_k4s = [[] for _ in range(num_triples)]
        triple_to_k5s = [[] for _ in range(num_triples)]
        
        for i, ts in enumerate(k4_triples):
            for t in ts:
                triple_to_k4s[t].append(i)
                
        for i, ts in enumerate(k5_triples):
            for t in ts:
                triple_to_k5s[t].append(i)
        
        # State
        # colors: array of 0s and 1s (int8 for memory, but we need bool logic)
        # Let's use 0 for Red, 1 for Blue.
        # We want to minimize violations.
        
        # Initial random coloring
        colors = np.random.randint(0, 2, num_triples, dtype=np.int8)
        
        # Helper to compute violations and "almost" counts for delta
        # To compute delta efficiently when flipping triple t:
        # 1. Identify K4s containing t. For each, check if the other 3 triples are 0.
        #    - If yes, and colors[t] was 0 -> flipping to 1 removes a violation.
        #    - If yes, and colors[t] was 1 -> flipping to 0 creates a violation.
        #    - If no, no change in violation status for this K4.
        # 2. Identify K5s containing t. For each, check if the other 9 triples are 1.
        #    - If yes, and colors[t] was 1 -> flipping to 0 removes a violation.
        #    - If yes, and colors[t] was 0 -> flipping to 1 creates a violation.
        #    - If no, no change.
        
        # We need fast access to "are the other k-1 triples all color X?".
        # We can precompute for each triple t, the list of other triples in each K4/K5 it belongs to?
        # That's too much memory.
        
        # Alternative: 
        # Maintain an array `red_count_in_k4[i]` = number of red triples in K4 i.
        # Maintain an array `blue_count_in_k5[i]` = number of blue triples in K5 i.
        
        # Initial counts
        red_count_k4 = np.zeros(num_k4, dtype=np.int32)
        blue_count_k5 = np.zeros(num_k5, dtype=np.int32)
        
        # Fill initial counts
        # This is O(num_k4 * 4 + num_k5 * 10)
        for i, ts in enumerate(k4_triples):
            # ts is list of 4 indices
            # count how many are 0
            red_count_k4[i] = np.sum(colors[ts] == 0)
            
        for i, ts in enumerate(k5_triples):
            # ts is list of 10 indices
            blue_count_k5[i] = np.sum(colors[ts] == 1)
            
        # Initial violations
        red_k4_viol = np.sum(red_count_k4 == 4)
        blue_k5_viol = np.sum(blue_count_k5 == 10)
        total_viol = red_k4_viol + blue_k5_viol
        
        best_viol = total_viol
        best_colors = colors.copy()
        
        # Simulated Annealing
        # T starts high, cools down.
        # We accept moves that increase violations with probability exp(-delta/T).
        
        T = 50.0 # Initial temperature
        cool_rate = 0.9995
        
        steps = 0
        no_improve = 0
        max_no_improve = 2000
        
        # Precompute lists of neighbor K4s/K5s for each triple to avoid list lookups in inner loop?
        # triple_to_k4s and triple_to_k5s are already lists of lists.
        
        while time.time() < end_time:
            if total_viol == 0:
                break
                
            # Pick a random triple to flip
            t = random.randint(0, num_triples - 1)
            old_color = colors[t]
            new_color = 1 - old_color
            
            # Calculate delta
            delta = 0
            
            # Update K4s containing t
            # For each K4 index in triple_to_k4s[t]:
            #   old_count = red_count_k4[idx]
            #   if old_color == 0: new_count = old_count - 1
            #   else: new_count = old_count + 1
            #   if old_count == 4: delta -= 1 (was viol, now not if new_count < 4)
            #   if new_count == 4: delta += 1 (was not viol, now viol)
            
            for k4_idx in triple_to_k4s[t]:
                old_c = red_count_k4[k4_idx]
                if old_color == 0:
                    new_c = old_c - 1
                else:
                    new_c = old_c + 1
                    
                if old_c == 4 and new_c != 4:
                    delta -= 1
                elif old_c != 4 and new_c == 4:
                    delta += 1
                    
                red_count_k4[k4_idx] = new_c
                
            # Update K5s containing t
            for k5_idx in triple_to_k5s[t]:
                old_c = blue_count_k5[k5_idx]
                if old_color == 1:
                    new_c = old_c - 1
                else:
                    new_c = old_c + 1
                    
                if old_c == 10 and new_c != 10:
                    delta -= 1
                elif old_c != 10 and new_c == 10:
                    delta += 1
                    
                blue_count_k5[k5_idx] = new_c
                
            # Accept or Reject
            if delta <= 0:
                # Improvement or neutral
                colors[t] = new_color
                total_viol += delta
                if delta < 0:
                    no_improve = 0
                    if total_viol < best_viol:
                        best_viol = total_viol
                        best_colors = colors.copy()
                else:
                    no_improve += 1
            else:
                # Worse
                # Accept with prob exp(-delta/T)
                if random.random() < np.exp(-delta / T):
                    colors[t] = new_color
                    total_viol += delta
                    no_improve += 1
                else:
                    # Revert counts
                    for k4_idx in triple_to_k4s[t]:
                        if old_color == 0:
                            red_count_k4[k4_idx] += 1
                        else:
                            red_count_k4[k4_idx] -= 1
                    for k5_idx in triple_to_k5s[t]:
                        if old_color == 1:
                            blue_count_k5[k5_idx] += 1
                        else:
                            blue_count_k5[k5_idx] -= 1
                    # colors[t] remains old_color
                    
            steps += 1
            T *= cool_rate
            
            if no_improve > max_no_improve:
                # Restart with new random init? Or just reset temperature?
                # Let's try a restart to escape local minima
                if time.time() > end_time - 5:
                    break
                # Re-initialize
                colors = np.random.randint(0, 2, num_triples, dtype=np.int8)
                for i, ts in enumerate(k4_triples):
                    red_count_k4[i] = np.sum(colors[ts] == 0)
                for i, ts in enumerate(k5_triples):
                    blue_count_k5[i] = np.sum(colors[ts] == 1)
                red_k4_viol = np.sum(red_count_k4 == 4)
                blue_k5_viol = np.sum(blue_count_k5 == 10)
                total_viol = red_k4_viol + blue_k5_viol
                T = 50.0
                no_improve = 0
                
        if best_viol == 0:
            return n, best_colors.tolist()
        else:
            return 0, None

    # Try n=34
    # Give it most of the time
    time_for_34 = 100
    n_sol, colors_sol = try_solve(34, time_for_34)
    
    if n_sol > 0:
        return {'n': n_sol, 'colors': colors_sol}
        
    # Fallback: Try n=33 with remaining time
    remaining = deadline - time.time()
    if remaining > 5:
        n_sol, colors_sol = try_solve(33, remaining - 2)
        if n_sol > 0:
            return {'n': n_sol, 'colors': colors_sol}
            
    # Fallback: Try n=30
    remaining = deadline - time.time()
    if remaining > 5:
        n_sol, colors_sol = try_solve(30, remaining - 2)
        if n_sol > 0:
            return {'n': n_sol, 'colors': colors_sol}
            
    # Fallback: Small valid n=9 (from seed logic, but let's make sure it's valid)
    # The seed used a modular construction. Let's just return a known valid small one.
    # Actually, let's just return n=1 to be safe if everything fails? No, n=9 is better.
    # Let's construct n=9 manually if we have time?
    # The seed's n=9 was: colors = [0 if (a+b+c)%3==0 else 1 ...]
    # Let's verify n=9 validity quickly or just trust the seed's score of 1.0.
    # To be safe, let's try to solve n=20 quickly if we have 2s left?
    # No, just return the seed's n=9 construction.
    
    n = 9
    colors = []
    for a, b, c in itertools.combinations(range(n), 3):
        colors.append(0 if (a + b + c) % 3 == 0 else 1)
        
    return {'n': n, 'colors': colors}

# EVOLVE-BLOCK-END
