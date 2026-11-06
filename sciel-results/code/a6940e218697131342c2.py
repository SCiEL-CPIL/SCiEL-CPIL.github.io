"""Optimized Sidon set constructor using simulated annealing."""
import random
import math
import time

def construct():
    """Construct a Sidon set containing 0 with all differences 1..30, minimizing max element."""
    
    def get_diffs(s):
        """Return set of all positive differences."""
        diffs = set()
        n = len(s)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(s[j] - s[i])
        return diffs
    
    def is_sidon(s):
        """Check if set is Sidon (all positive differences distinct)."""
        n = len(s)
        diff_count = {}
        for i in range(n):
            for j in range(i + 1, n):
                d = s[j] - s[i]
                diff_count[d] = diff_count.get(d, 0) + 1
                if diff_count[d] > 1:
                    return False
        return True
    
    def energy(s, target_diffs):
        """Energy: penalty for missing diffs + penalty for Sidon violations + small penalty for max element."""
        n = len(s)
        diffs = {}
        for i in range(n):
            for j in range(i + 1, n):
                d = s[j] - s[i]
                diffs[d] = diffs.get(d, 0) + 1
        
        # Count missing target differences
        missing = 0
        for d in target_diffs:
            if d not in diffs:
                missing += 1
        
        # Count Sidon violations (differences appearing more than once)
        violations = 0
        for d, cnt in diffs.items():
            if cnt > 1:
                violations += cnt - 1
        
        # Max element penalty (small weight to encourage compactness)
        max_el = max(s) if s else 0
        
        return missing * 1000 + violations * 100 + max_el * 0.001
    
    target_diffs = set(range(1, 31))
    
    # Try multiple random seeds and initializations
    best_solution = None
    best_energy = float('inf')
    
    start_time = time.time()
    max_time = 110  # Leave 10s buffer
    
    # Initial solution: use known small Sidon sets as seeds
    # Start with a reasonable initialization
    initial_sets = [
        [0, 1, 4, 10, 12, 17, 22, 28],  # Small Sidon set
        [0, 1, 3, 7, 12, 18, 25, 33],
        [0, 2, 3, 7, 12, 17, 23, 30],
    ]
    
    for init_idx in range(3):
        if time.time() - start_time > max_time:
            break
        
        random.seed(42 + init_idx)
        
        # Start from a base Sidon set and extend
        base = list(initial_sets[init_idx])
        
        # Ensure it's a valid Sidon set
        if not is_sidon(base):
            continue
        
        # Simulated annealing
        current = list(base)
        current_energy = energy(current, target_diffs)
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_solution = list(current)
        
        T = 50.0
        T_min = 0.01
        cooling = 0.9995
        max_iter = 50000
        
        for iteration in range(max_iter):
            if time.time() - start_time > max_time:
                break
            
            # Generate neighbor
            neighbor = list(current)
            
            # Random move: change one element
            if len(neighbor) > 1:
                idx = random.randint(1, len(neighbor) - 1)  # Don't change 0
                delta = random.choice([-2, -1, 1, 2, 3, 5, 8, 13, -5, -8, -13])
                new_val = neighbor[idx] + delta
                if new_val >= 0 and new_val != neighbor[idx]:
                    neighbor[idx] = new_val
            
            # Sometimes add or remove an element
            move_type = random.random()
            if move_type < 0.1 and len(neighbor) < 32:
                # Add a new element
                max_el = max(neighbor)
                new_val = random.randint(0, max_el + 20)
                if new_val not in neighbor:
                    neighbor.append(new_val)
            elif move_type < 0.15 and len(neighbor) > 5:
                # Remove an element (not 0)
                idx = random.randint(1, len(neighbor) - 1)
                neighbor.pop(idx)
            
            neighbor.sort()
            
            if len(neighbor) < 2 or 0 not in neighbor:
                continue
            
            neighbor_energy = energy(neighbor, target_diffs)
            
            # Accept if better or with probability based on temperature
            delta_e = neighbor_energy - current_energy
            if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                current = neighbor
                current_energy = neighbor_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_solution = list(current)
            
            T *= cooling
            if T < T_min:
                break
    
    # If we found a valid solution, verify and return
    if best_solution is not None:
        # Verify it's valid
        if is_sidon(best_solution) and all(d in get_diffs(best_solution) for d in range(1, 31)):
            return best_solution
    
    # Fallback: use the original greedy approach
    a = [0]
    def diffs_list(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    
    for d in range(1, 31):
        if d in set(diffs_list(sorted(a))):
            continue
        found = False
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(diffs_list(trial))) == len(diffs_list(trial)):
                a = trial
                found = True
                break
        if not found:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs_list(trial))) == len(diffs_list(trial)):
                    a = trial
                    break
                y += 1
    
    return a
