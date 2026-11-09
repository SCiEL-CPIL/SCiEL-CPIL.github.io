import random
import time
import math
from itertools import combinations

def construct():
    """
    Finds a 2-coloring of K_n^(3) with no Red K_4^(3) and no Blue K_5^(3).
    Uses Simulated Annealing with incremental violation counting.
    """
    target_n = 34
    start_time = time.time()
    time_limit = 115  # Leave 5s for overhead
    
    # Precompute all 4-subsets and 5-subsets for the target n
    # We will run for n=target_n. If it fails, we might try smaller n, 
    # but the problem asks to maximize n, and 34 is known to exist.
    
    def solve_sa(n, time_budget):
        # Initialize
        num_triples = n * (n-1) * (n-2) // 6
        # Map triple to index
        triple_to_idx = {}
        idx = 0
        triples = []
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    triple_to_idx[(a,b,c)] = idx
                    triples.append((a,b,c))
                    idx += 1
        
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        
        # Precompute K4 and K5 candidates
        k4s = list(combinations(range(n), 4))
        k5s = list(combinations(range(n), 5))
        
        # For each K4, list its 4 triples
        k4_triples = {}
        for k4 in k4s:
            k4_triples[k4] = list(combinations(k4, 3))
            
        # For each K5, list its 10 triples
        k5_triples = {}
        for k5 in k5s:
            k5_triples[k5] = list(combinations(k5, 3))
            
        # Initialize counts
        red_k4_counts = {}
        blue_k5_counts = {}
        
        # We need to know for each triple, which K4s and K5s it belongs to
        # triple_to_k4s: list of K4s containing this triple
        # triple_to_k5s: list of K5s containing this triple
        triple_to_k4s = [[] for _ in range(num_triples)]
        triple_to_k5s = [[] for _ in range(num_triples)]
        
        for k4 in k4s:
            t_list = k4_triples[k4]
            for t in t_list:
                triple_to_k4s[triple_to_idx[t]].append(k4)
                
        for k5 in k5s:
            t_list = k5_triples[k5]
            for t in t_list:
                triple_to_k5s[triple_to_idx[t]].append(k5)
                
        # Calculate initial energy
        red_violations = 0
        blue_violations = 0
        
        for k4 in k4s:
            count = 0
            for t in k4_triples[k4]:
                if colors[triple_to_idx[t]] == 0:
                    count += 1
            red_k4_counts[k4] = count
            if count == 4:
                red_violations += 1
                
        for k5 in k5s:
            count = 0
            for t in k5_triples[k5]:
                if colors[triple_to_idx[t]] == 1:
                    count += 1
            blue_k5_counts[k5] = count
            if count == 10:
                blue_violations += 1
                
        energy = red_violations + 10 * blue_violations # Weight blue more? Or equal?
        # Actually, just sum them.
        energy = red_violations + blue_violations
        
        best_energy = energy
        best_colors = colors[:]
        
        # Simulated Annealing
        T = 100.0
        cooling = 0.999
        min_T = 0.01
        
        while T > min_T and time.time() - start_time < start_time + time_budget:
            # Pick a random triple to flip
            t_idx = random.randint(0, num_triples - 1)
            t = triples[t_idx]
            
            old_color = colors[t_idx]
            new_color = 1 - old_color
            
            # Calculate delta energy
            delta_e = 0
            
            # Update Red K4 counts
            if old_color == 0: # Red -> Blue
                # Red count decreases for all K4s containing t
                for k4 in triple_to_k4s[t_idx]:
                    old_count = red_k4_counts[k4]
                    new_count = old_count - 1
                    red_k4_counts[k4] = new_count
                    if old_count == 4:
                        delta_e -= 1 # Removed a violation
                    elif new_count == 4:
                        delta_e += 1 # Created a violation (impossible if old was 4? No, old was 4, new is 3. If old was 3, new is 2. Violation only at 4. So only removal possible here? Wait. If old was 4, we remove. If old was 3, we don't create. If old was 5? Max is 4. So only removal.)
                        # Wait, if old_count was 4, it was a violation. Now it's 3. Delta -1.
                        # If old_count was 3, it was not. Now 2. Delta 0.
                        # So only removals for Red->Blue?
                        # Yes, reducing red edges can only remove red K4s.
                    
            else: # Blue -> Red
                # Red count increases for all K4s containing t
                for k4 in triple_to_k4s[t_idx]:
                    old_count = red_k4_counts[k4]
                    new_count = old_count + 1
                    red_k4_counts[k4] = new_count
                    if new_count == 4:
                        delta_e += 1 # Created a violation
                    # If old was 3, new 4 -> +1
                    # If old was 4, new 5? Impossible.
                    
            # Update Blue K5 counts
            if old_color == 1: # Blue -> Red
                # Blue count decreases for all K5s containing t
                for k5 in triple_to_k5s[t_idx]:
                    old_count = blue_k5_counts[k5]
                    new_count = old_count - 1
                    blue_k5_counts[k5] = new_count
                    if old_count == 10:
                        delta_e -= 1 # Removed a violation
                    
            else: # Red -> Blue
                # Blue count increases for all K5s containing t
                for k5 in triple_to_k5s[t_idx]:
                    old_count = blue_k5_counts[k5]
                    new_count = old_count + 1
                    blue_k5_counts[k5] = new_count
                    if new_count == 10:
                        delta_e += 1 # Created a violation
            
            # Accept/Reject
            if delta_e < 0:
                # Always accept
                colors[t_idx] = new_color
                energy += delta_e
                if energy < best_energy:
                    best_energy = energy
                    best_colors = colors[:]
            else:
                # Accept with probability exp(-delta_e / T)
                if delta_e == 0:
                    colors[t_idx] = new_color
                    # energy unchanged
                else:
                    prob = math.exp(-delta_e / T)
                    if random.random() < prob:
                        colors[t_idx] = new_color
                        energy += delta_e
                        if energy < best_energy:
                            best_energy = energy
                            best_colors = colors[:]
            
            T *= cooling
            
        # Check if best is valid
        if best_energy == 0:
            return best_colors
        return None

    # Try n=34
    for n in [34, 33, 32]:
        if time.time() - start_time > time_limit:
            break
            
        # Run SA
        res = solve_sa(n, time_limit - (time.time() - start_time))
        if res is not None:
            return {'n': n, 'colors': res}
            
    # Fallback
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
