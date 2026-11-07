"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 128}

def _is_valid(S):
    """Check if set S has no three distinct members a,b,c with all pair sums in S."""
    s = set(S)
    lst = list(S)
    n = len(lst)
    for i in range(n):
        a = lst[i]
        for j in range(i+1, n):
            b = lst[j]
            for k in range(j+1, n):
                c = lst[k]
                if (a+b) in s and (a+c) in s and (b+c) in s:
                    return False
    return True

def _count_violations(S):
    """Count the number of violating triples in S."""
    s = set(S)
    lst = list(S)
    n = len(lst)
    count = 0
    for i in range(n):
        a = lst[i]
        for j in range(i+1, n):
            b = lst[j]
            for k in range(j+1, n):
                c = lst[k]
                if (a+b) in s and (a+c) in s and (b+c) in s:
                    count += 1
    return count

def _violation_count_incremental(S, x):
    """Count violations in S if x were added (x not in S). Fast incremental check."""
    s = set(S)
    # A violation involving x: triple {a,b,x} where a,b in S
    # Conditions: a+x in S∪{x}, b+x in S∪{x}, a+b in S∪{x}
    # Since x not in S: a+x in S (not x itself unless a=0, impossible), etc.
    # So we need: a+x in S, b+x in S, a+b in S (or a+b==x, but x not in S yet)
    
    # Find all a in S such that a+x is also in S
    A = [a for a in S if a + x <= 128 and (a + x) in s]
    
    count = 0
    for i in range(len(A)):
        a = A[i]
        for j in range(i + 1, len(A)):
            b = A[j]
            # Check a+b in S (a+b != x since x not in S, and a+b > 0)
            if a + b in s:
                count += 1
    return count

def _greedy_with_order(order, n):
    """Build a set by trying elements in given order, skipping if it creates a violation.
    Uses bitmask for fast membership checks.
    """
    S_mask = 0
    S_set = set()
    
    for x in order:
        if (S_mask >> x) & 1:
            continue
            
        # Tentatively add x
        S_mask_new = S_mask | (1 << x)
        
        # Efficient check:
        # If we add x, the new sums involving x are a+x.
        # A violation is a triple {a,b,x} where a,b in S_old.
        # Conditions:
        # 1. a+x in S_new
        # 2. b+x in S_new
        # 3. a+b in S_new
        
        valid = True
        A = []
        for a in range(1, n + 1):
            if (S_mask >> a) & 1: # a in S_old
                if a + x <= n and (S_mask_new >> (a + x)) & 1: # a+x in S_new
                    A.append(a)
        
        for i in range(len(A)):
            a = A[i]
            for j in range(i + 1, len(A)):
                b = A[j]
                if a + b <= n and (S_mask_new >> (a + b)) & 1:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            S_mask = S_mask_new
            S_set.add(x)
            
    return S_set

def _simulated_annealing(rng, n, deadline, initial_set=None):
    """Simulated annealing to maximize set size while maintaining validity."""
    if initial_set is None:
        # Start with a random dense subset
        initial_set = set(range(63, n + 1))
    
    current_set = set(initial_set)
    current_violations = _count_violations(current_set)
    
    best_set = set(current_set)
    best_violations = current_violations
    best_size = len(current_set)
    
    temperature = 1.0
    cooling_rate = 0.995
    min_temperature = 0.01
    
    while time.time() < deadline and temperature > min_temperature:
        # Choose a random move: add, remove, or swap
        move_type = rng.choice(['add', 'remove', 'swap'])
        
        if move_type == 'add':
            candidates = [x for x in range(1, n + 1) if x not in current_set]
            if not candidates:
                continue
            x = rng.choice(candidates)
            new_set = current_set | {x}
            # Use fast incremental check for add moves
            new_violations = _violation_count_incremental(current_set, x)
        elif move_type == 'remove':
            if not current_set:
                continue
            x = rng.choice(list(current_set))
            new_set = current_set - {x}
            new_violations = _count_violations(new_set)
        else:  # swap
            if len(current_set) < 2:
                continue
            in_set = list(current_set)
            out_set = [x for x in range(1, n + 1) if x not in current_set]
            if not out_set:
                continue
            x_in = rng.choice(in_set)
            x_out = rng.choice(out_set)
            new_set = (current_set - {x_in}) | {x_out}
            # For swap, just do full count (could optimize but keep simple)
            new_violations = _count_violations(new_set)
        
        new_size = len(new_set)
        
        # Objective: minimize violations, maximize size
        # We want a valid set (0 violations) with maximum size.
        # If both are valid, prefer larger size.
        # If current is invalid, prefer fewer violations.
        
        delta_obj = 0
        if current_violations == 0 and new_violations == 0:
            # Both valid: prefer larger size
            delta_obj = -(new_size - len(current_set))
        elif current_violations == 0 and new_violations > 0:
            # Moving from valid to invalid: bad
            delta_obj = new_violations * 100
        elif current_violations > 0 and new_violations == 0:
            # Moving from invalid to valid: good
            delta_obj = -current_violations * 100
        else:
            # Both invalid: prefer fewer violations, then larger size
            delta_obj = (new_violations - current_violations) * 10
            if delta_obj == 0:
                delta_obj = -(new_size - len(current_set))
        
        if delta_obj <= 0 or rng.random() < pow(2.718, -delta_obj / temperature):
            current_set = new_set
            current_violations = new_violations
            
            if current_violations < best_violations or \
               (current_violations == best_violations and len(current_set) > best_size):
                best_set = set(current_set)
                best_violations = current_violations
                best_size = len(current_set)
        
        temperature *= cooling_rate
    
    return best_set

def _polish_set(S, n, max_time=5.0):
    """Try to add or swap elements to increase size while maintaining validity."""
    S = set(S)
    start_time = time.time()
    
    # First, try adding elements greedily in various orders
    improved = True
    while improved and time.time() - start_time < max_time:
        improved = False
        # Try adding each element not in S
        for x in range(1, n + 1):
            if x in S:
                continue
            # Check if adding x creates a violation
            new_S = S | {x}
            if _is_valid(new_S):
                S = new_S
                improved = True
    
    # Try swapping: remove one, add another (net +0 or +1 size)
    improved = True
    while improved and time.time() - start_time < max_time:
        improved = False
        for x_in in list(S):
            for x_out in range(1, n + 1):
                if x_out in S or x_out == x_in:
                    continue
                new_S = (S - {x_in}) | {x_out}
                if len(new_S) > len(S) and _is_valid(new_S):
                    S = new_S
                    improved = True
                    break
            if improved:
                break
    
    return S

def construct():
    n = P['n']
    deadline = time.time() + 55.0  # safety margin before 90s limit
    
    # Baseline: {63, ..., 128} has size 66
    best_set = set(range(63, n + 1))
    best_size = len(best_set)
    
    rng = random.Random(42)
    
    # Strategy 1: Greedy approaches (fast, good for initialization)
    # Try randomized greedy approaches
    for trial in range(200):
        if time.time() > deadline - 30.0:
            break
        
        # Random ordering biased toward upper half
        upper = list(range(63, n + 1))
        lower = list(range(1, 63))
        rng.shuffle(upper)
        rng.shuffle(lower)
        
        mix_ratio = rng.random()
        num_lower = int(len(lower) * mix_ratio)
        order = upper + lower[:num_lower]
        
        S = _greedy_with_order(order, n)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Strategy 2: Simulated Annealing
    # Use the best set from greedy as starting point, and also try random starts
    sa_deadline = deadline - 10.0
    
    for trial in range(10):
        if time.time() > sa_deadline:
            break
        
        if trial == 0:
            start_set = set(best_set)
        else:
            # Random start: mix of upper and lower
            upper = list(range(63, n + 1))
            lower = list(range(1, 63))
            rng.shuffle(upper)
            rng.shuffle(lower)
            num_upper = rng.randint(30, 60)
            num_lower = rng.randint(0, 20)
            start_set = set(upper[:num_upper] + lower[:num_lower])
        
        S = _simulated_annealing(rng, n, sa_deadline, initial_set=start_set)
        
        if _is_valid(S) and len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Strategy 3: Polish the best set found
    if time.time() < deadline - 2.0:
        polished = _polish_set(best_set, n, max_time=5.0)
        if _is_valid(polished) and len(polished) > best_size:
            best_size = len(polished)
            best_set = polished
    
    # Final verification
    if _is_valid(best_set):
        return [int(x) for x in sorted(best_set)]
    else:
        # Fallback to known valid set
        return [int(x) for x in range(63, n + 1)]
# EVOLVE-BLOCK-END
