# EVOLVE-BLOCK-START
import random
import time
import math

def construct():
    start = time.time()
    DEADLINE = 145  # safety margin before 150s evaluator limit
    N = 284
    PRESCRIBED = [1, 2, 5, 11]
    PRESCRIBED_SET = set(PRESCRIBED)
    MAX_SIZE = 20

    def compute_collisions(S):
        """Returns the number of duplicate sums. 0 means valid Sidon set."""
        sums = {}
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                if s in sums:
                    sums[s] += 1
                else:
                    sums[s] = 1
        collisions = 0
        for count in sums.values():
            if count > 1:
                collisions += count - 1
        return collisions

    def is_valid(S):
        return compute_collisions(S) == 0

    def greedy_construct(rng, target_size=None):
        """
        Greedy construction. If target_size is None, adds as many as possible.
        Returns a valid set.
        """
        S = list(PRESCRIBED)
        S.sort()
        sums = set()
        for i, a in enumerate(S):
            for b in S[i:]:
                sums.add(a + b)
        
        candidates = [x for x in range(1, N + 1) if x not in PRESCRIBED_SET]
        rng.shuffle(candidates)
        
        for x in candidates:
            if target_size and len(S) >= target_size:
                break
            # Check if x can be added
            # New sums: x+y for y in S, and 2x
            ok = True
            for y in S:
                if x + y in sums:
                    ok = False
                    break
            if ok and 2 * x in sums:
                ok = False
            
            if ok:
                S.append(x)
                S.sort()
                for y in S[:-1]:
                    sums.add(x + y)
                sums.add(2 * x)
                
        return S

    def simulated_annealing(rng, target_size, initial_S=None, time_limit=10.0):
        """
        Attempts to find a valid Sidon set of exactly target_size.
        Returns the set if found, else None.
        """
        if initial_S and len(initial_S) == target_size:
            S = list(initial_S)
        else:
            # Initialize with prescribed elements and random others
            S = list(PRESCRIBED)
            available = [x for x in range(1, N + 1) if x not in PRESCRIBED_SET]
            rng.shuffle(available)
            S.extend(available[:target_size - len(PRESCRIBED)])
        
        S.sort()
        current_cost = compute_collisions(S)
        
        if current_cost == 0:
            return S

        best_S = list(S)
        best_cost = current_cost
        
        T = 10.0
        T_min = 0.01
        cooling = 0.995
        
        sa_start = time.time()
        
        while T > T_min and (time.time() - sa_start) < time_limit:
            # Generate neighbor
            S_new = list(S)
            # Pick a non-prescribed index to replace
            non_prescribed_indices = [i for i, v in enumerate(S_new) if v not in PRESCRIBED_SET]
            if not non_prescribed_indices:
                break
            idx = rng.choice(non_prescribed_indices)
            
            # Pick a new value
            new_val = rng.randint(1, N)
            # Ensure distinctness
            while new_val in S_new:
                new_val = rng.randint(1, N)
            
            S_new[idx] = new_val
            S_new.sort()
            
            new_cost = compute_collisions(S_new)
            
            if new_cost < current_cost or (T > 0 and rng.random() < math.exp(-(new_cost - current_cost) / T)):
                S = S_new
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_S = list(S)
                    
                    if best_cost == 0:
                        return best_S

            T *= cooling
            
        return best_S if best_cost == 0 else None

    def csp_backtracking(target_size, time_limit=30.0, rng=None):
        """
        CSP-style backtracking to find a Sidon set of target_size.
        Uses forward checking to prune.
        """
        if rng is None:
            rng = random.Random(42)
        
        # Variables: positions 0..target_size-1
        # Fixed variables: indices corresponding to prescribed elements
        # We don't know which indices are prescribed yet, so let's assign prescribed first?
        # Actually, it's easier to just have a list of variables and their domains.
        # Prescribed elements must be in the set.
        
        # Strategy:
        # 1. Assign prescribed elements to specific "slots" (e.g., first 4 slots).
        # 2. Assign remaining target_size - 4 elements to remaining slots.
        # 3. Use backtracking with forward checking.
        
        # However, the order of assignment matters.
        # Let's fix the first 4 variables to be the prescribed elements (sorted).
        # This is valid because the set is unordered, but for backtracking,
        # fixing them reduces symmetry.
        
        fixed_vars = sorted(PRESCRIBED)
        # Domains for fixed vars are singletons.
        # Domains for other vars are [1, N] excluding fixed_vars and previously assigned.
        
        # State:
        # assigned: list of values
        # sums: set of sums
        # domains: list of sets for each variable index
        
        # We will assign variables in order 0, 1, ..., target_size-1.
        # Vars 0-3 are fixed.
        
        assigned = [None] * target_size
        for i in range(len(fixed_vars)):
            assigned[i] = fixed_vars[i]
        
        sums = set()
        # Initialize sums with fixed vars
        for i in range(len(fixed_vars)):
            for j in range(i, len(fixed_vars)):
                sums.add(assigned[i] + assigned[j])
        
        # Domains for unfixed vars
        used_values = set(fixed_vars)
        domains = [None] * target_size
        for i in range(len(fixed_vars), target_size):
            domains[i] = set(range(1, N + 1)) - used_values
        
        # Forward checking: when assigning var i, update domains of future vars
        # This is expensive to do fully, so we'll do a simplified version:
        # When we pick a value for var i, we check if it conflicts with existing sums.
        # If it does, we backtrack.
        # We don't explicitly prune domains of future vars in this simple implementation,
        # but the conflict check at assignment time is strong.
        
        # To improve, we can order variables by domain size (MRV).
        # But since first 4 are fixed, and others are symmetric, MRV is less critical initially.
        # However, as we assign, domains shrink.
        
        # Let's implement a recursive backtracker with a time limit.
        
        nodes_explored = 0
        start_time = time.time()
        
        def backtrack(idx):
            nonlocal nodes_explored
            if time.time() - start_time > time_limit:
                return False
            
            if idx == target_size:
                return True
            
            nodes_explored += 1
            
            # Get candidates for this variable
            if idx < len(fixed_vars):
                candidates = [assigned[idx]]
            else:
                candidates = list(domains[idx])
                # Shuffle for randomness to avoid getting stuck in same path
                rng.shuffle(candidates)
            
            for val in candidates:
                if val is None:
                    continue
                if val in used_values:
                    continue
                
                # Check if val can be added
                # New sums: val + assigned[i] for i < idx, and 2*val
                valid = True
                for i in range(idx):
                    if val + assigned[i] in sums:
                        valid = False
                        break
                if valid and 2 * val in sums:
                    valid = False
                
                if valid:
                    # Assign
                    assigned[idx] = val
                    used_values.add(val)
                    old_sums = set(sums)
                    for i in range(idx):
                        sums.add(val + assigned[i])
                    sums.add(2 * val)
                    
                    # Recurse
                    if backtrack(idx + 1):
                        return True
                    
                    # Unassign
                    assigned[idx] = None
                    used_values.remove(val)
                    sums = old_sums
            
            return False
        
        # To make it more effective, we can try to assign variables in a smarter order.
        # But for now, let's just run it.
        # We might need to restart with different shuffling if it fails quickly.
        
        # Let's run multiple restarts if the first one fails or times out.
        # But the time limit is global.
        
        # Actually, the above backtracking might be too slow without proper pruning.
        # Let's add a simple heuristic: try values that minimize conflicts?
        # Or just random order.
        
        # Let's try running it.
        if backtrack(0):
            return assigned
        
        return None

    # Strategy:
    # 1. Find a good starting set using greedy.
    # 2. Try to maximize size using CSP backtracking, then SA if CSP fails.
    
    rng = random.Random(42)
    
    # Phase 1: Greedy to get a baseline
    baseline_S = greedy_construct(rng)
    best_size = len(baseline_S)
    best_set = baseline_S
    
    # Phase 2: Try to find a larger set using CSP backtracking
    # We know max is likely 20. Let's try to find a valid set of size 20.
    # If that fails, try 19, etc.
    
    for target in range(MAX_SIZE, best_size, -1):
        if time.time() - start > DEADLINE - 10:
            break
        
        # Try CSP backtracking with a time limit
        # Use a shorter time limit for higher targets to allow fallback
        time_alloc = min(remaining_time(), 20.0) if 'remaining_time' in dir() else 20.0
        # Define remaining_time helper
        def remaining_time():
            return DEADLINE - time.time()
        
        time_alloc = min(remaining_time(), 20.0)
        
        result = csp_backtracking(target, time_limit=time_alloc, rng=rng)
        if result:
            best_size = target
            best_set = result
            break
    
    # Phase 3: If CSP didn't find a solution, try SA extension from baseline
    if best_size < MAX_SIZE:
        current_size = best_size
        current_set = best_set
        
        while current_size < MAX_SIZE:
            next_size = current_size + 1
            if time.time() - start > DEADLINE - 5:
                break
            
            found = False
            attempts = 2
            for attempt in range(attempts):
                if time.time() - start > DEADLINE - 5:
                    break
                
                init_set = list(current_set)
                available = [x for x in range(1, N + 1) if x not in set(init_set)]
                if not available:
                    break
                init_set.append(rng.choice(available))
                init_set.sort()
                
                res = simulated_annealing(rng, next_size, initial_S=init_set, time_limit=10.0)
                if res:
                    found = True
                    current_size = next_size
                    current_set = res
                    break
            
            if not found:
                break
        
        if current_size > best_size:
            best_size = current_size
            best_set = current_set
    
    return best_set

# EVOLVE-BLOCK-END
