import time
import random

P = {'n': 128}

def construct():
    n = P['n']
    deadline = time.time() + 80.0  # Safety margin (90s limit)

    def make_set(lst):
        s = 0
        for x in lst:
            s |= (1 << (x - 1))
        return s

    def get_elems(s):
        return [i + 1 for i in range(n) if (s >> i) & 1]

    def is_valid_full(s):
        elems = get_elems(s)
        L = len(elems)
        for i in range(L):
            a = elems[i]
            for j in range(i + 1, L):
                b = elems[j]
                ab = a + b
                if ab > n:
                    continue
                if not ((s >> (ab - 1)) & 1):
                    continue
                for k in range(j + 1, L):
                    c = elems[k]
                    ac = a + c
                    bc = b + c
                    if ac > n or bc > n:
                        continue
                    if ((s >> (ac - 1)) & 1) and ((s >> (bc - 1)) & 1):
                        return False
        return True

    def find_conflict(s):
        """Returns a tuple (a, b, c, ab, ac, bc) if a conflict exists, else None."""
        elems = get_elems(s)
        L = len(elems)
        for i in range(L):
            a = elems[i]
            for j in range(i + 1, L):
                b = elems[j]
                ab = a + b
                if ab > n:
                    continue
                if not ((s >> (ab - 1)) & 1):
                    continue
                for k in range(j + 1, L):
                    c = elems[k]
                    ac = a + c
                    bc = b + c
                    if ac > n or bc > n:
                        continue
                    if ((s >> (ac - 1)) & 1) and ((s >> (bc - 1)) & 1):
                        return (a, b, c, ab, ac, bc)
        return None

    def can_add(s, x):
        """Check if x can be added to set s without creating a conflict."""
        if (s >> (x - 1)) & 1:
            return True
        elems = get_elems(s)
        L = len(elems)
        for i in range(L):
            a = elems[i]
            xa = x + a
            if xa > n:
                continue
            if not ((s >> (xa - 1)) & 1):
                continue
            for j in range(i + 1, L):
                b = elems[j]
                xb = x + b
                if xb > n:
                    continue
                if not ((s >> (xb - 1)) & 1):
                    continue
                ab = a + b
                if ab == x or (ab <= n and ((s >> (ab - 1)) & 1)):
                    return False
        return True

    def count_conflicts(s):
        """Count the number of conflicting triples in set s."""
        elems = get_elems(s)
        L = len(elems)
        count = 0
        for i in range(L):
            a = elems[i]
            for j in range(i + 1, L):
                b = elems[j]
                ab = a + b
                if ab > n:
                    continue
                if not ((s >> (ab - 1)) & 1):
                    continue
                for k in range(j + 1, L):
                    c = elems[k]
                    ac = a + c
                    bc = b + c
                    if ac > n or bc > n:
                        continue
                    if ((s >> (ac - 1)) & 1) and ((s >> (bc - 1)) & 1):
                        count += 1
        return count

    best_s = 0
    best_size = 0

    # Strategy 1: Constructive greedy addition
    # Start with a good base: upper half (65-128) which is naturally sum-free
    # Then greedily add lower elements if they don't create conflicts
    
    # Base: upper half
    s = make_set(range(65, n + 1))
    
    # Greedily add lower elements in order of increasing "safety"
    # (higher elements are safer because their sums go out of bounds)
    for x in range(64, 0, -1):
        if (s >> (x - 1)) & 1:
            continue
        if can_add(s, x):
            s |= (1 << (x - 1))
    
    size = bin(s).count('1')
    if size > best_size:
        best_s = s
        best_size = size

    # Strategy 2: Simulated annealing to escape local optima
    # Objective: maximize size while maintaining validity (0 conflicts)
    # We'll use a penalty-based approach: fitness = size - penalty * conflicts
    
    current_s = best_s
    current_size = best_size
    current_conflicts = count_conflicts(current_s)
    
    # Annealing parameters
    T0 = 1.0
    T_end = 0.01
    alpha = 0.9995  # Cooling rate
    T = T0
    max_iter = 200000
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        # Generate a random move: add, remove, or swap
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to add a random element
            x = random.randint(1, n)
            if (current_s >> (x - 1)) & 1:
                continue
            new_s = current_s | (1 << (x - 1))
            new_size = current_size + 1
            new_conflicts = count_conflicts(new_s)
        elif move_type < 0.7:
            # Try to remove a random element
            if current_size == 0:
                continue
            elems = get_elems(current_s)
            x = random.choice(elems)
            new_s = current_s & ~(1 << (x - 1))
            new_size = current_size - 1
            new_conflicts = count_conflicts(new_s)
        else:
            # Try a swap: remove one, add one
            if current_size == 0:
                continue
            elems = get_elems(current_s)
            x = random.choice(elems)
            y = random.randint(1, n)
            if y == x or (current_s >> (y - 1)) & 1:
                continue
            new_s = (current_s & ~(1 << (x - 1))) | (1 << (y - 1))
            new_size = current_size
            new_conflicts = count_conflicts(new_s)
        
        # Calculate energy difference
        # We want to minimize conflicts and maximize size
        # Energy = -size + penalty * conflicts
        penalty = 10.0
        old_energy = -current_size + penalty * current_conflicts
        new_energy = -new_size + penalty * new_conflicts
        delta_E = new_energy - old_energy
        
        # Accept if better or with probability exp(-delta_E / T)
        if delta_E < 0 or random.random() < pow(2.718281828, -delta_E / T):
            current_s = new_s
            current_size = new_size
            current_conflicts = new_conflicts
            
            if current_conflicts == 0 and current_size > best_size:
                best_s = current_s
                best_size = current_size
        
        # Cool down
        T *= alpha
        if T < T_end:
            T = T_end
    
    # Final greedy add pass
    for x in range(1, n + 1):
        if (best_s >> (x - 1)) & 1:
            continue
        if can_add(best_s, x):
            best_s |= (1 << (x - 1))
            best_size += 1
    
    # Final validation
    if not is_valid_full(best_s):
        # Fallback: upper half
        best_s = make_set(range(65, n + 1))
        best_size = 64
    
    result = [i + 1 for i in range(n) if (best_s >> i) & 1]
    return result
