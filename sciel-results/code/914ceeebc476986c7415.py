import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # We will try to find a valid coloring for n=35, then 34, etc.
    # Since 34 is known to exist, we prioritize finding it.
    
    def solve_for_n(n, time_limit):
        """
        Attempts to find a valid 2-coloring of K_n^(3) with no red K4^(3) and no blue K5^(3).
        Returns colors list if found, None otherwise.
        """
        if n < 4:
            return [0] * 0 # Trivial
        
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_index = {t: i for i, t in enumerate(triples)}
        
        # Precompute K4s and K5s
        k4s = list(combinations(range(n), 4))
        k5s = list(combinations(range(n), 5))
        
        num_k4 = len(k4s)
        num_k5 = len(k5s)
        
        # Map each K4 to its 4 triple indices
        k4_triple_indices = []
        for k4 in k4s:
            t_indices = []
            for t in combinations(k4, 3):
                t_indices.append(triple_index[t])
            k4_triple_indices.append(t_indices)
            
        # Map each K5 to its 10 triple indices
        k5_triple_indices = []
        for k5 in k5s:
            t_indices = []
            for t in combinations(k5, 3):
                t_indices.append(triple_index[t])
            k5_triple_indices.append(t_indices)
            
        # Map each triple to the K4s and K5s it belongs to
        triple_to_k4s = [[] for _ in range(num_triples)]
        for i, t_indices in enumerate(k4_triple_indices):
            for t_idx in t_indices:
                triple_to_k4s[t_idx].append(i)
                
        triple_to_k5s = [[] for _ in range(num_triples)]
        for i, t_indices in enumerate(k5_triple_indices):
            for t_idx in t_indices:
                triple_to_k5s[t_idx].append(i)
        
        # Initialize random coloring
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        
        # Initialize counts
        # red_k4_counts[k4_idx] = number of red triples in that K4
        red_k4_counts = [0] * num_k4
        # blue_k5_counts[k5_idx] = number of blue triples in that K5
        blue_k5_counts = [0] * num_k5
        
        for i, t in enumerate(triples):
            c = colors[i]
            for k4_idx in triple_to_k4s[i]:
                if c == 0: # Red
                    red_k4_counts[k4_idx] += 1
            for k5_idx in triple_to_k5s[i]:
                if c == 1: # Blue
                    blue_k5_counts[k5_idx] += 1
        
        # Count initial violations
        red_violations = sum(1 for c in red_k4_counts if c == 4)
        blue_violations = sum(1 for c in blue_k5_counts if c == 10)
        total_violations = red_violations + blue_violations
        
        if total_violations == 0:
            return colors
            
        # Simulated Annealing
        T = 1.0
        cooling_rate = 0.999
        min_T = 0.01
        
        # We want to minimize total_violations
        # Let's also track "pressure" to help escape local minima
        # Pressure = count of K4s with 3 red edges + count of K5s with 9 blue edges
        # This is a heuristic to guide the search
        
        end_time = time.time() + time_limit
        best_colors = None
        best_violations = total_violations
        
        while time.time() < end_time:
            # Pick a random triple to flip
            t_idx = random.randint(0, num_triples - 1)
            
            # Calculate delta in violations
            # Current state of triple t_idx
            old_color = colors[t_idx]
            new_color = 1 - old_color
            
            delta_red_violations = 0
            delta_blue_violations = 0
            
            # Update K4 counts
            for k4_idx in triple_to_k4s[t_idx]:
                old_count = red_k4_counts[k4_idx]
                if old_color == 0: # Was red, now blue -> red count decreases
                    new_count = old_count - 1
                    if old_count == 4: delta_red_violations -= 1
                    if new_count == 4: delta_red_violations += 1
                    red_k4_counts[k4_idx] = new_count
                else: # Was blue, now red -> red count increases
                    new_count = old_count + 1
                    if old_count == 4: delta_red_violations -= 1
                    if new_count == 4: delta_red_violations += 1
                    red_k4_counts[k4_idx] = new_count
                    
            # Update K5 counts
            for k5_idx in triple_to_k5s[t_idx]:
                old_count = blue_k5_counts[k5_idx]
                if old_color == 1: # Was blue, now red -> blue count decreases
                    new_count = old_count - 1
                    if old_count == 10: delta_blue_violations -= 1
                    if new_count == 10: delta_blue_violations += 1
                    blue_k5_counts[k5_idx] = new_count
                else: # Was red, now blue -> blue count increases
                    new_count = old_count + 1
                    if old_count == 10: delta_blue_violations -= 1
                    if new_count == 10: delta_blue_violations += 1
                    blue_k5_counts[k5_idx] = new_count
            
            delta_violations = delta_red_violations + delta_blue_violations
            new_total_violations = total_violations + delta_violations
            
            # Accept if better, or with probability exp(-delta/T)
            if delta_violations <= 0:
                accept = True
            else:
                # Simple acceptance probability
                # We can use a heuristic: accept if it doesn't increase too much
                # Or standard SA
                accept = random.random() < (2.718281828 ** (-delta_violations / T))
                
            if accept:
                colors[t_idx] = new_color
                total_violations = new_total_violations
                
                if total_violations == 0:
                    return colors
                    
                if total_violations < best_violations:
                    best_violations = total_violations
                    best_colors = colors[:]
            else:
                # Revert changes
                for k4_idx in triple_to_k4s[t_idx]:
                    if old_color == 0:
                        red_k4_counts[k4_idx] += 1
                    else:
                        red_k4_counts[k4_idx] -= 1
                for k5_idx in triple_to_k5s[t_idx]:
                    if old_color == 1:
                        blue_k5_counts[k5_idx] += 1
                    else:
                        blue_k5_counts[k5_idx] -= 1
                
            T *= cooling_rate
            if T < min_T:
                T = min_T
        
        return best_colors if best_violations == 0 else None

    # Try n=35 first, then 34
    for n in [35, 34, 33, 32]:
        if time.time() > deadline - 5:
            break
            
        # Allocate time based on difficulty
        # 35 is harder, give it less time if 34 is the target?
        # Actually, just give each a chunk of time.
        time_left = deadline - time.time()
        if n == 35:
            t_limit = min(20.0, time_left - 10.0)
        elif n == 34:
            t_limit = min(60.0, time_left - 5.0)
        else:
            t_limit = time_left - 5.0
            
        if t_limit <= 0:
            continue
            
        result = solve_for_n(n, t_limit)
        if result is not None:
            return {'n': n, 'colors': result}
            
    # Fallback: small n
    n = 10
    triples = list(combinations(range(n), 3))
    colors = [0] * len(triples) # All red is invalid for n>=4, but let's just return something valid for small n
    # For n=3, no K4 or K5.
    # For n=4, all red is a red K4. So we need to break it.
    # Let's just return a valid one for n=3 or 4.
    # n=3: 1 triple. No K4, no K5. Valid.
    n = 3
    triples = list(combinations(range(n), 3))
    colors = [0] * len(triples)
    
    return {'n': n, 'colors': colors}
