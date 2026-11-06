import math
import time
import random

_START_TIME = None

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    deadline = _START_TIME + 85.0  # safety margin for 90s limit
    
    # Moduli from 2 to 128
    moduli = list(range(2, 129))
    n_mod = len(moduli)  # 127 moduli
    
    # Precompute gcd for all modulus pairs
    gcd_table = {}
    for i, m1 in enumerate(moduli):
        for j, m2 in enumerate(moduli):
            if i != j:
                gcd_table[(m1, m2)] = _gcd(m1, m2)
    
    # Precompute conflict counts: for each modulus m, how many other moduli are coprime to m
    # This helps with heuristic ordering
    coprime_count = {}
    for m in moduli:
        cnt = 0
        for m2 in moduli:
            if m != m2 and _gcd(m, m2) == 1:
                cnt += 1
        coprime_count[m] = cnt
    
    import random
    random.seed(42)
    
    def is_valid_addition(r, m, selected):
        """Check if (r, m) can be added to selected without conflicts."""
        for (sr, sm) in selected:
            g = gcd_table[(m, sm)]
            if (r - sr) % g == 0:
                return False
        return True
    
    def greedy_init(order_type='random'):
        """Greedy initialization with different ordering strategies."""
        if order_type == 'random':
            perm = moduli[:]
            random.shuffle(perm)
        elif order_type == 'coprime_first':
            # Sort by number of coprime moduli (descending) - more constrained first
            perm = sorted(moduli, key=lambda m: -coprime_count[m])
        elif order_type == 'coprime_first_random':
            # Group by coprime count, randomize within groups
            groups = {}
            for m in moduli:
                c = coprime_count[m]
                if c not in groups:
                    groups[c] = []
                groups[c].append(m)
            perm = []
            for c in sorted(groups.keys(), reverse=True):
                group = groups[c][:]
                random.shuffle(group)
                perm.extend(group)
        elif order_type == 'size_first':
            # Smaller moduli first (fewer residue choices)
            perm = sorted(moduli)
        elif order_type == 'size_first_random':
            perm = sorted(moduli)
            # Randomize slightly
            for i in range(len(perm) - 1):
                if random.random() < 0.3:
                    perm[i], perm[i+1] = perm[i+1], perm[i]
        else:
            perm = moduli[:]
        
        selected = []
        for m in perm:
            residues = list(range(m))
            if order_type in ('random', 'coprime_first_random', 'size_first_random'):
                random.shuffle(residues)
            
            for r in residues:
                if is_valid_addition(r, m, selected):
                    selected.append((r, m))
                    break
        
        return selected
    
    def local_search_improve(solution, max_iters=1000):
        """Improve solution by trying residue changes and additions."""
        if not solution:
            return solution
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            if time.time() > deadline:
                break
            improved = False
            iters += 1
            
            # Try changing one residue at a time
            for idx in range(len(solution)):
                r, m = solution[idx]
                best_new_r = None
                # Try all other residues, prefer ones that don't conflict
                for new_r in range(m):
                    if new_r == r:
                        continue
                    if is_valid_addition(new_r, m, [(sr, sm) for jdx, (sr, sm) in enumerate(solution) if jdx != idx]):
                        best_new_r = new_r
                        break
                
                if best_new_r is not None and best_new_r != r:
                    solution[idx] = (best_new_r, m)
                    improved = True
                    break
            
            # Try to add a new modulus
            if not improved:
                selected_mods = set(m for _, m in solution)
                # Try unselected moduli in order of increasing coprime count (less constrained)
                unselected = [m for m in moduli if m not in selected_mods]
                unselected.sort(key=lambda m: coprime_count[m])
                
                for m in unselected:
                    for r in range(m):
                        if is_valid_addition(r, m, solution):
                            solution.append((r, m))
                            improved = True
                            break
                    if improved:
                        break
        
        return solution
    
    def simulated_annealing(solution, max_iters=500):
        """Simulated annealing to escape local optima."""
        if not solution:
            return solution
        
        current = solution[:]
        current_size = len(current)
        best = current[:]
        best_size = current_size
        
        T = 5.0  # initial temperature
        cooling = 0.995
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Random move: change a residue, remove an element, or add an element
            move_type = random.random()
            
            if move_type < 0.5 and current:
                # Change a residue
                idx = random.randint(0, len(current) - 1)
                r, m = current[idx]
                new_r = random.randint(0, m - 1)
                if new_r == r:
                    continue
                old_val = current[idx]
                current[idx] = (new_r, m)
                new_size = len(current)
                
                # Check validity
                valid = True
                for i in range(len(current)):
                    for j in range(i + 1, len(current)):
                        r1, m1 = current[i]
                        r2, m2 = current[j]
                        g = gcd_table[(m1, m2)]
                        if (r1 - r2) % g == 0:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    delta = new_size - current_size
                    if delta >= 0 or random.random() < math.exp(delta / T):
                        current_size = new_size
                        if new_size > best_size:
                            best_size = new_size
                            best = current[:]
                    else:
                        current[idx] = old_val
                else:
                    # Invalid: accept with probability based on T
                    if random.random() < math.exp(-2.0 / T):
                        current_size = new_size
                    else:
                        current[idx] = old_val
                        
            elif move_type < 0.75 and len(current) > 1:
                # Remove an element
                idx = random.randint(0, len(current) - 1)
                removed = current.pop(idx)
                new_size = len(current)
                delta = new_size - current_size
                
                if delta >= 0 or random.random() < math.exp(delta / T):
                    current_size = new_size
                    if new_size > best_size:
                        best_size = new_size
                        best = current[:]
                else:
                    current.insert(idx, removed)
                    
            else:
                # Add an element
                selected_mods = set(m for _, m in current)
                unselected = [m for m in moduli if m not in selected_mods]
                if unselected:
                    m = random.choice(unselected)
                    r = random.randint(0, m - 1)
                    current.append((r, m))
                    new_size = len(current)
                    
                    # Check validity
                    valid = True
                    for i in range(len(current)):
                        for j in range(i + 1, len(current)):
                            r1, m1 = current[i]
                            r2, m2 = current[j]
                            g = gcd_table[(m1, m2)]
                            if (r1 - r2) % g == 0:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        delta = new_size - current_size
                        current_size = new_size
                        if new_size > best_size:
                            best_size = new_size
                            best = current[:]
                    else:
                        current.pop()
            
            T *= cooling
        
        return best
    
    best_solution = None
    best_count = 0
    
    # Phase 1: Multiple greedy initializations with different strategies
    strategies = ['random', 'coprime_first', 'coprime_first_random', 'size_first', 'size_first_random']
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        strategy = strategies[attempt % len(strategies)]
        solution = greedy_init(strategy)
        
        # Local search improvement
        solution = local_search_improve(solution, max_iters=200)
        
        count = len(solution)
        if count > best_count:
            best_count = count
            best_solution = solution[:]
        
        # If we're doing well, do simulated annealing on the best
        if best_count >= 40 and attempt % 10 == 5:
            if time.time() < deadline - 10:
                sa_solution = simulated_annealing(best_solution, max_iters=200)
                if len(sa_solution) > best_count:
                    best_count = len(sa_solution)
                    best_solution = sa_solution[:]
    
    # Phase 2: More intensive local search on the best solution
    if best_solution is not None:
        if time.time() < deadline - 5:
            improved = local_search_improve(best_solution, max_iters=500)
            if len(improved) > best_count:
                best_count = len(improved)
                best_solution = improved
        
        # Final simulated annealing
        if time.time() < deadline - 2:
            sa_solution = simulated_annealing(best_solution, max_iters=300)
            if len(sa_solution) > best_count:
                best_count = len(sa_solution)
                best_solution = sa_solution
    
    if best_solution is None:
        best_solution = [[1, 2], [2, 4], [4, 8], [8, 16], [16, 32], [32, 64], [64, 128]]
    
    return [[r, m] for r, m in best_solution]
