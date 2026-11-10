# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large square-difference-free subset of Z/205Z.
    
    Uses a combination of:
    1. Deterministic greedy initialization
    2. Simulated annealing with random restarts to escape local optima
    3. Local search (add/remove/swap moves)
    
    The forbidden differences are the nonzero quadratic residues modulo 205.
    """
    m = 205
    deadline = time.time() + 600  # 600 second deadline
    
    # Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        val = (x * x) % m
        if val != 0:
            forbidden.add(val)
    
    # Precompute conflict matrix: conflict[i][j] = True if i and j cannot both be in the set
    conflict = np.zeros((m, m), dtype=bool)
    for i in range(m):
        for j in range(i + 1, m):
            d1 = (i - j) % m
            d2 = (j - i) % m
            if d1 in forbidden or d2 in forbidden:
                conflict[i][j] = True
                conflict[j][i] = True
    
    def is_valid(selected_set):
        """Check if a set is valid (no forbidden differences)."""
        arr = np.array(sorted(selected_set))
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                if conflict[arr[i], arr[j]]:
                    return False
        return True
    
    def greedy_init():
        """Greedy initialization: add elements in order if they don't conflict."""
        selected = set()
        for candidate in range(m):
            ok = True
            for s in selected:
                if conflict[candidate, s]:
                    ok = False
                    break
            if ok:
                selected.add(candidate)
        return selected
    
    def greedy_init_random(seed):
        """Greedy initialization with random ordering."""
        rng = random.Random(seed)
        candidates = list(range(m))
        rng.shuffle(candidates)
        selected = set()
        for candidate in candidates:
            ok = True
            for s in selected:
                if conflict[candidate, s]:
                    ok = False
                    break
            if ok:
                selected.add(candidate)
        return selected
    
    def can_add(selected_set, candidate):
        """Check if candidate can be added to selected_set."""
        for s in selected_set:
            if conflict[candidate, s]:
                return False
        return True
    
    def simulated_annealing(initial_set, seed):
        """Simulated annealing to improve the set."""
        rng = random.Random(seed)
        current = set(initial_set)
        best = set(initial_set)
        
        # Temperature schedule
        T_start = 1.0
        T_end = 0.001
        cooling = 0.995
        T = T_start
        
        while time.time() < deadline - 5:  # Leave 5 seconds margin
            if T < T_end:
                break
            
            # Generate a random move
            move_type = rng.choice(['add', 'remove', 'swap'])
            
            if move_type == 'add':
                # Try to add a random element
                candidates = [i for i in range(m) if i not in current]
                if not candidates:
                    T *= cooling
                    continue
                candidate = rng.choice(candidates)
                if can_add(current, candidate):
                    new_set = set(current)
                    new_set.add(candidate)
                    # Always accept valid additions
                    current = new_set
                    if len(current) > len(best):
                        best = set(current)
                else:
                    # Metropolis acceptance for invalid moves (skip for now)
                    pass
            
            elif move_type == 'remove':
                # Try to remove a random element
                if not current:
                    T *= cooling
                    continue
                candidate = rng.choice(list(current))
                new_set = set(current)
                new_set.discard(candidate)
                # Only accept if it allows a better configuration later
                # For now, accept with some probability
                if rng.random() < 0.1:  # Low probability to remove
                    current = new_set
            
            elif move_type == 'swap':
                # Try to swap: remove one, add another
                if not current:
                    T *= cooling
                    continue
                remove_candidate = rng.choice(list(current))
                add_candidates = [i for i in range(m) if i not in current]
                if not add_candidates:
                    T *= cooling
                    continue
                add_candidate = rng.choice(add_candidates)
                
                new_set = set(current)
                new_set.discard(remove_candidate)
                new_set.add(add_candidate)
                
                # Check if new set is valid
                valid = True
                for a in new_set:
                    for b in new_set:
                        if a != b and conflict[a, b]:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    current = new_set
                    if len(current) > len(best):
                        best = set(current)
            
            T *= cooling
        
        return best
    
    def local_search(initial_set, seed):
        """Local search: try all possible single additions."""
        rng = random.Random(seed)
        current = set(initial_set)
        best = set(initial_set)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try adding each possible element
            for candidate in range(m):
                if candidate in current:
                    continue
                if can_add(current, candidate):
                    new_set = set(current)
                    new_set.add(candidate)
                    current = new_set
                    improved = True
                    break
            
            if len(current) > len(best):
                best = set(current)
        
        return best
    
    # Start with greedy initialization
    best_set = greedy_init()
    
    # Try multiple random initializations
    for seed in range(50):
        if time.time() > deadline - 10:
            break
        init_set = greedy_init_random(seed)
        if len(init_set) > len(best_set):
            best_set = init_set
    
    # Apply local search to the best set
    improved_set = local_search(best_set, seed=42)
    if len(improved_set) > len(best_set):
        best_set = improved_set
    
    # Apply simulated annealing with multiple restarts
    for seed in range(20):
        if time.time() > deadline - 30:
            break
        # Start from best_set or a random perturbation
        if seed % 2 == 0:
            start_set = set(best_set)
        else:
            start_set = greedy_init_random(seed)
        
        sa_result = simulated_annealing(start_set, seed)
        if len(sa_result) > len(best_set):
            best_set = sa_result
    
    # Final local search
    final_set = local_search(best_set, seed=123)
    if len(final_set) > len(best_set):
        best_set = final_set
    
    # Convert to sorted array
    result = np.array(sorted(best_set), dtype=np.int64)
    
    # Validate the result
    if not is_valid(set(result.tolist())):
        # Fallback to greedy
        result = np.array(sorted(greedy_init()), dtype=np.int64)
    
    return result

# EVOLVE-BLOCK-END
