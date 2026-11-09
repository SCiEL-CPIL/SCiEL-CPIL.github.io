import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Generate primes up to 1000
    def get_primes(limit):
        sieve = bytearray(b'\x01') * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = 0
        return [i for i in range(2, limit + 1) if sieve[i]]

    primes = get_primes(1000)
    prime_set = set(primes)

    # Initial set: primes >= 13
    initial_primes = [p for p in primes if p >= 13]
    
    def sum_recip_frac(prime_list):
        return sum(Fraction(1, p) for p in prime_list)

    # Fast greedy check if 1..L can be covered by given primes
    def greedy_can_cover(L, prime_list):
        used_primes = set()
        residues = {}
        covered = [False] * (L + 1)
        
        for x in range(1, L + 1):
            if covered[x]:
                continue
            
            best_p = None
            best_gain = -1
            best_r = None
            
            for p in prime_list:
                if p in used_primes:
                    continue
                
                r = x % p
                gain = 0
                for k in range(0, (L - x) // p + 1):
                    y = x + k * p
                    if not covered[y]:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
                    best_r = r
            
            if best_p is None or best_gain <= 0:
                return None
            
            used_primes.add(best_p)
            residues[best_p] = best_r
            
            for k in range(0, (L - x) // best_p + 1):
                y = x + k * best_p
                covered[y] = True
                
        return residues

    # CSP-style backtracking with forward checking for residue assignment
    def csp_solve(L, prime_list, time_budget=5.0):
        """
        Solve the covering problem using backtracking with forward checking.
        Variables: primes in prime_list
        Domains: residues 0..p-1
        Constraints: every x in 1..L is covered by at least one prime
        """
        t0 = time.time()
        n_primes = len(prime_list)
        
        # Precompute for each prime p, the set of x in 1..L that p covers for each residue
        # coverage[p_idx][r] = set of x covered
        # This is expensive for large L, so we use a different approach
        
        # Alternative: work with the uncovered set
        # State: (uncovered_set, assigned_primes, residues)
        # Use bitset for uncovered for efficiency
        
        # For L up to ~1000, we can use Python integers as bitsets
        # bit x-1 is 1 if x is uncovered
        
        # Initial uncovered: all of 1..L
        initial_uncovered = (1 << L) - 1  # bits 0 to L-1 set, representing x=1 to L
        
        # For each prime p, precompute the bitmask for each residue r
        # mask[p_idx][r] has bits set for x in 1..L where x % p == r
        prime_masks = []
        for p in prime_list:
            masks = []
            for r in range(p):
                m = 0
                # x ranges from 1 to L, x % p == r
                # First x >= 1 with x % p == r: 
                # if r == 0, first x is p (if p <= L), else none
                # if r > 0, first x is r (if r <= L), else none
                first_x = r if r > 0 else p
                if first_x > L:
                    masks.append(0)
                    continue
                for x in range(first_x, L + 1, p):
                    m |= (1 << (x - 1))
                masks.append(m)
            prime_masks.append(masks)
        
        # Order primes: smaller primes first (they cover more, so assign them first for better pruning)
        # Actually, for CSP, we should assign variables that constrain the most first
        # Smaller primes cover more numbers, so they constrain more
        order = list(range(n_primes))
        order.sort(key=lambda i: prime_list[i])  # ascending by prime value
        
        # Backtracking state
        best_solution = None
        solution_count = 0
        max_backtracks = 50000  # limit to avoid exponential blowup
        
        def backtrack(idx, uncovered, assigned):
            nonlocal best_solution, solution_count, max_backtracks
            
            if time.time() - t0 > time_budget:
                return False
            if solution_count > max_backtracks:
                return False
            
            # Check if all covered
            if uncovered == 0:
                # Found a solution
                if best_solution is None:
                    best_solution = assigned.copy()
                solution_count += 1
                return True
            
            # If no more primes to assign but still uncovered
            if idx == len(order):
                return False
            
            p_idx = order[idx]
            p = prime_list[p_idx]
            
            # Forward checking: for each possible residue, check if it helps
            # Try residues in order of how much they reduce uncovered
            candidates = []
            for r in range(p):
                mask = prime_masks[p_idx][r]
                new_uncovered = uncovered & ~mask
                # Count newly covered
                newly_covered = bin(uncovered & mask).count('1')
                if newly_covered > 0:
                    candidates.append((newly_covered, r, new_uncovered))
            
            # Sort by most newly covered first (MRV-like: most constraining first)
            candidates.sort(key=lambda x: -x[0])
            
            for _, r, new_uncovered in candidates:
                assigned[p_idx] = r
                if backtrack(idx + 1, new_uncovered, assigned):
                    return True
                # If we found a solution, we can stop (we just need one)
                if best_solution is not None:
                    return True
            
            return False
        
        assigned = [None] * n_primes
        if backtrack(0, initial_uncovered, assigned):
            # Convert back to dict
            result = {}
            for i, p_idx in enumerate(order):
                if assigned[p_idx] is not None:
                    result[prime_list[p_idx]] = assigned[p_idx]
            # Ensure all primes have an assignment (for those not used, assign 0)
            for p in prime_list:
                if p not in result:
                    result[p] = 0
            return result
        return None

    # Start with seed configuration
    current_primes = initial_primes[:]
    current_L = 163
    
    # Verify seed with greedy
    seed_assign = greedy_can_cover(current_L, current_primes)
    if seed_assign is None:
        # Try CSP
        seed_assign = csp_solve(current_L, current_primes, time_budget=10.0)
    if seed_assign is None:
        # Fallback to seed
        return [[p, (i + 1) % p] for i, p in enumerate(initial_primes)]

    # Try to extend L
    candidates_to_add = [p for p in primes if p < 13]
    candidates_to_add.sort(reverse=True) # Try 11, 7, 5, 3, 2
    
    # Track best known solution
    best_primes = current_primes[:]
    best_L = current_L
    best_assignment = seed_assign.copy()
    
    while time.time() < deadline:
        next_L = current_L + 1
        
        # Try current set with greedy first (fast)
        assign = greedy_can_cover(next_L, current_primes)
        if assign is not None:
            current_L = next_L
            best_primes = current_primes[:]
            best_L = next_L
            best_assignment = assign.copy()
            continue
        
        # Try CSP (slower but more thorough)
        assign = csp_solve(next_L, current_primes, time_budget=3.0)
        if assign is not None:
            current_L = next_L
            best_primes = current_primes[:]
            best_L = next_L
            best_assignment = assign.copy()
            continue
        
        # Try to modify the set
        success = False
        for p_add in candidates_to_add:
            if p_add in current_primes:
                continue
            
            # Calculate new set
            temp_primes = current_primes[:]
            temp_primes.append(p_add)
            # Sort descending to pick largest for removal
            temp_primes.sort(reverse=True)
            
            s_temp = sum_recip_frac(temp_primes)
            # Remove largest while sum > 1
            while s_temp > Fraction(1, 1) and len(temp_primes) > 0:
                p_remove = temp_primes.pop(0) # Largest
                s_temp -= Fraction(1, p_remove)
            
            if s_temp > Fraction(1, 1):
                continue
                
            # Check if this new set covers next_L - try greedy first
            assign = greedy_can_cover(next_L, temp_primes)
            if assign is not None:
                current_primes = temp_primes
                current_L = next_L
                best_primes = current_primes[:]
                best_L = next_L
                best_assignment = assign.copy()
                success = True
                break
            
            # Try CSP
            assign = csp_solve(next_L, temp_primes, time_budget=2.0)
            if assign is not None:
                current_primes = temp_primes
                current_L = next_L
                best_primes = current_primes[:]
                best_L = next_L
                best_assignment = assign.copy()
                success = True
                break
                
        if not success:
            # Try random swaps
            if len(current_primes) > 5:
                # Try removing a random large prime and see if we can still cover
                for _ in range(3):
                    if time.time() > deadline - 20:
                        break
                    # Remove a random prime (prefer large ones)
                    idx = random.randint(0, len(current_primes) - 1)
                    p_remove = current_primes[idx]
                    if p_remove < 13:
                        continue  # Don't remove small primes
                    temp_primes = current_primes[:]
                    temp_primes.remove(p_remove)
                    
                    # Check if we can still cover current_L
                    assign = greedy_can_cover(current_L, temp_primes)
                    if assign is not None:
                        # We can remove this prime, try to extend
                        current_primes = temp_primes
                        # Now try to extend
                        next_L2 = current_L + 1
                        assign2 = greedy_can_cover(next_L2, current_primes)
                        if assign2 is not None:
                            current_L = next_L2
                            best_primes = current_primes[:]
                            best_L = next_L2
                            best_assignment = assign2.copy()
                            success = True
                            break
            
            if not success:
                break

    # Use the best assignment found
    final_assignment = best_assignment
    final_primes = best_primes
    
    # Verify the final assignment covers 1..best_L
    covered = [False] * (best_L + 1)
    for p, r in final_assignment.items():
        for x in range(1, best_L + 1):
            if x % p == r:
                covered[x] = True
    
    # Find actual covered length
    actual_L = 0
    for x in range(1, best_L + 1):
        if covered[x]:
            actual_L = x
        else:
            break
    
    # Format output
    result = []
    for p in final_primes:
        if p in final_assignment:
            result.append([p, final_assignment[p]])
    
    return result
