import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    N = 128

    # Precompute all non-constant 4-term arithmetic progressions
    # A 4-AP is (a, a+d, a+2d, a+3d) with d > 0
    aps = []
    for d in range(1, N):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= N:
                aps.append((a, b, c, e))
                
    num_aps = len(aps)
    
    # For fast cost update, we need to know which APs contain a given index
    # index_to_aps[i] = list of indices in 'aps' list that contain i
    index_to_aps = [[] for _ in range(N + 1)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            index_to_aps[pos].append(idx)

    def count_distinct(colors, ap):
        c1, c2, c3, c4 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
        return len({c1, c2, c3, c4})

    def get_cost(colors):
        cost = 0
        for ap in aps:
            if count_distinct(colors, ap) < 3:
                cost += 1
        return cost

    # Simulated Annealing parameters
    # We try K from 3 up to a reasonable limit
    # If we find a valid coloring with K colors, we return it.
    # Since we want to MINIMIZE colors, we start with K=3.
    
    best_solution = None
    best_k = N # fallback

    for K in range(3, 11):
        k_start = time.time()
        # Allocate time based on K. Smaller K is harder, give more time?
        # Or just a fixed time slice.
        # Let's give 5 seconds per K, but stop if total time is up.
        k_deadline = min(deadline, k_start + 5.0)
        
        if time.time() > deadline:
            break
            
        # Initialize
        colors = [0] * (N + 1)
        for i in range(1, N + 1):
            colors[i] = random.randint(0, K - 1)
            
        current_cost = get_cost(colors)
        
        if current_cost == 0:
            return [colors[i] for i in range(1, N + 1)]

        # Simulated Annealing
        temperature = 10.0
        cooling_rate = 0.95
        min_temperature = 0.01
        
        # We will run SA until k_deadline
        while time.time() < k_deadline:
            if current_cost == 0:
                return [colors[i] for i in range(1, N + 1)]
                
            # Pick a random position
            pos = random.randint(1, N)
            old_color = colors[pos]
            
            # Pick a new color
            new_color = random.randint(0, K - 1)
            if new_color == old_color:
                continue
                
            # Calculate delta cost
            delta = 0
            for ap_idx in index_to_aps[pos]:
                ap = aps[ap_idx]
                old_d = count_distinct(colors, ap)
                
                # Temporarily change
                colors[pos] = new_color
                new_d = count_distinct(colors, ap)
                # Restore
                colors[pos] = old_color
                
                if old_d < 3 and new_d >= 3:
                    delta -= 1
                elif old_d >= 3 and new_d < 3:
                    delta += 1
            
            # Acceptance criterion
            if delta < 0:
                # Improvement
                colors[pos] = new_color
                current_cost += delta
            else:
                # Deterioration or neutral
                # Accept with probability exp(-delta / T)
                if delta == 0:
                    # Accept neutral moves to explore
                    if random.random() < 0.5:
                        colors[pos] = new_color
                else:
                    prob = pow(2.71828, -delta / temperature)
                    if random.random() < prob:
                        colors[pos] = new_color
                        current_cost += delta
            
            # Cool down
            temperature *= cooling_rate
            if temperature < min_temperature:
                # Restart temperature to keep exploring or break?
                # Restarting temperature is like a restart but keeps the current state.
                temperature = 5.0 
                # Or maybe just restart the whole search for this K?
                # Let's stick to cooling. If it gets too cold, we might get stuck.
                # A simple restart of the SA loop with a new random state if stuck too long?
                # For now, let's just let it cool.

    # If we didn't find a solution with K < 11, we might have found one with K=10?
    # The loop above returns immediately if found.
    # If we finish the loop without returning, it means we didn't find a valid coloring for K in 3..10.
    # This is unlikely if 3-coloring exists.
    
    # Fallback: Generate a valid coloring with more colors if necessary.
    # A simple valid coloring is using distinct colors for each position? No, that's 128 colors.
    # We can use a pattern.
    # If we didn't find a small K solution, we should try to find ANY valid solution.
    # The previous fallback was i % 128.
    # But we can do better.
    # Let's try a greedy construction for a larger K, say K=20, which should be easy.
    # Or just return the best we found? No, we must return a VALID solution.
    # The loop above only returns if current_cost == 0.
    # So if we exit the loop, we have no valid solution.
    
    # We need a guaranteed valid solution.
    # Let's use a simple heuristic: assign colors based on index modulo some number.
    # If we use 4 colors: 0,1,2,3.
    # Pattern 0,1,2,3, 0,1,2,3...
    # AP: 1,2,3,4 -> 0,1,2,3 (4 distinct) OK.
    # AP: 1,3,5,7 -> 0,2,0,2 (2 distinct) FAIL.
    
    # A known valid coloring with small number of colors?
    # The problem asks to minimize. If we can't find 3, we try 4, etc.
    # If we fail all, we must output a valid one.
    # The seed program used i % 128.
    # Let's try to find a valid one with a simple greedy algorithm using K=20.
    
    K_large = 20
    colors = [0] * (N + 1)
    for i in range(1, N + 1):
        colors[i] = random.randint(0, K_large - 1)
        
    # Local search to fix
    for _ in range(100000):
        if time.time() > deadline:
            break
        cost = get_cost(colors)
        if cost == 0:
            return [colors[i] for i in range(1, N + 1)]
        
        # Pick a bad AP and fix one of its elements
        # Or pick random element and try to improve
        pos = random.randint(1, N)
        old_color = colors[pos]
        best_new = old_color
        best_cost = cost
        
        for c in range(K_large):
            if c == old_color: continue
            delta = 0
            for ap_idx in index_to_aps[pos]:
                ap = aps[ap_idx]
                old_d = count_distinct(colors, ap)
                colors[pos] = c
                new_d = count_distinct(colors, ap)
                colors[pos] = old_color
                if old_d < 3 and new_d >= 3: delta -= 1
                elif old_d >= 3 and new_d < 3: delta += 1
            
            if cost + delta < best_cost:
                best_cost = cost + delta
                best_new = c
        
        if best_new != old_color:
            colors[pos] = best_new
            cost = best_cost
        else:
            # Random perturbation
            colors[pos] = random.randint(0, K_large - 1)
            cost = get_cost(colors)

    # If still not valid, use the absolute fallback
    return [i % 128 for i in range(1, N + 1)]

