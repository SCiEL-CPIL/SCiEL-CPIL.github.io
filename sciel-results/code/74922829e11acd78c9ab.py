import math
import random
import time
from fractions import Fraction

# ---- primes up to 1000 (fixed) ----
def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            step = i
            start = i * i
            sieve[start:n + 1:step] = [False] * len(sieve[start:n + 1:step])
    return [i for i, v in enumerate(sieve) if v]

_PRIMES = _primes_upto(1000)
NUM_PRIMES = len(_PRIMES)


def _coverage_length(assign):
    """assign: list of residues (0..p-1) per prime; 0 means 'not used'.
    Return largest L such that every 1..L is covered by some class."""
    MAX_CHECK = 10000
    covered = bytearray(MAX_CHECK + 1)
    for idx, r in enumerate(assign):
        if r == 0:
            continue
        p = _PRIMES[idx]
        for n in range(r, MAX_CHECK + 1, p):
            covered[n] = 1
    L = 0
    for n in range(1, MAX_CHECK + 1):
        if covered[n]:
            L = n
        else:
            break
    return L


def _is_valid(assign):
    c = Fraction(0)
    for idx, r in enumerate(assign):
        if r != 0:
            c += Fraction(1, _PRIMES[idx])
    return c <= Fraction(1, 1)


def _csp_solve(target_L, primes_to_use, time_limit):
    """
    CSP solver to find residues for given primes that cover 1..target_L.
    Uses backtracking with forward checking and MRV.
    Returns dict {prime_index: residue} or None if failed.
    """
    start = time.time()
    n_primes = len(primes_to_use)
    if n_primes == 0:
        return {}
    
    # Track which numbers are covered
    covered = bytearray(target_L + 1)  # 0 = not covered, 1 = covered
    
    # For each prime, track which residues are still possible
    # residue_domains[i] = set of possible residues for prime i
    residue_domains = [set(range(1, primes_to_use[i] + 1)) for i in range(n_primes)]
    
    # Assignment: dict prime_index -> residue
    assignment = {}
    
    # Precompute for each prime and residue, which numbers it covers
    # cover_mask[i][r] = list of numbers covered by prime i with residue r
    cover_masks = []
    for i, p in enumerate(primes_to_use):
        mask_for_prime = {}
        for r in range(1, p + 1):
            nums = []
            for n in range(r, target_L + 1, p):
                nums.append(n)
            mask_for_prime[r] = nums
        cover_masks.append(mask_for_prime)
    
    def is_complete():
        return len(assignment) == n_primes
    
    def is_consistent():
        # Check if all numbers 1..target_L are covered
        for n in range(1, target_L + 1):
            if not covered[n]:
                return False
        return True
    
    def get_uncovered():
        """Return list of uncovered numbers."""
        return [n for n in range(1, target_L + 1) if not covered[n]]
    
    def select_variable():
        """MRV: pick unassigned prime with fewest possible residues."""
        min_count = float('inf')
        best_var = -1
        for i in range(n_primes):
            if i not in assignment:
                count = len(residue_domains[i])
                if count < min_count:
                    min_count = count
                    best_var = i
                if count == 0:
                    return i  # Fail early
        return best_var
    
    def forward_check(var, residue):
        """
        After assigning var=residue, update domains of other unassigned vars.
        Returns True if consistent, False if domain wipeout.
        """
        # Save state for backtracking
        saved_domains = [set(domain) for domain in residue_domains]
        
        # Remove residue from var's domain
        residue_domains[var].discard(residue)
        
        # For each unassigned var, check if its domain is still non-empty
        for i in range(n_primes):
            if i in assignment or i == var:
                continue
            # Check if any residue in domain[i] is still possible
            # A residue is possible if it doesn't create a conflict
            # For simplicity, we just check if domain is non-empty
            if not residue_domains[i]:
                # Restore and return False
                for j in range(n_primes):
                    residue_domains[j] = saved_domains[j]
                return False
        return True
    
    def backtrack():
        if time.time() - start > time_limit:
            return None
        
        if is_complete():
            if is_consistent():
                return dict(assignment)
            return None
        
        var = select_variable()
        if var == -1:
            return None
        
        # Order values: try residues that cover more uncovered numbers first
        uncovered = get_uncovered()
        values = list(residue_domains[var])
        values.sort(key=lambda r: -sum(1 for n in cover_masks[var][r] if n in set(uncovered)))
        
        for residue in values:
            if time.time() - start > time_limit:
                return None
            
            # Assign
            assignment[var] = residue
            residue_domains[var].discard(residue)
            
            # Update covered array
            p = primes_to_use[var]
            for n in cover_masks[var][residue]:
                covered[n] = 1
            
            # Forward check
            if forward_check(var, residue):
                result = backtrack()
                if result is not None:
                    return result
            
            # Unassign
            del assignment[var]
            residue_domains[var].add(residue)
            for n in cover_masks[var][residue]:
                covered[n] = 0
        
        return None
    
    return backtrack()


def _greedy_build(target_L, time_limit):
    """
    Greedily build a set of primes and residues to cover 1..target_L.
    Returns (assign_list, success).
    """
    start = time.time()
    assign = [0] * NUM_PRIMES
    current_cost = Fraction(0)
    covered = bytearray(target_L + 1)
    
    # Track which primes are used
    used_primes = set()
    
    # Precompute coverage for each prime and residue
    # For speed, we'll compute on the fly
    
    # Greedy: at each step, pick the prime/residue that covers the most uncovered numbers
    # with the best cost-benefit ratio
    
    # First, try to cover small numbers with small primes
    # Sort primes by size
    prime_indices = sorted(range(NUM_PRIMES), key=lambda i: _PRIMES[i])
    
    for attempt in range(1000):  # Limit iterations
        if time.time() - start > time_limit:
            break
        
        # Find uncovered numbers
        uncovered = [n for n in range(1, target_L + 1) if not covered[n]]
        if not uncovered:
            break
        
        # For each unused prime, find the best residue
        best_score = -1
        best_prime = -1
        best_residue = -1
        
        for i in prime_indices:
            if i in used_primes:
                continue
            p = _PRIMES[i]
            cost = Fraction(1, p)
            if current_cost + cost > Fraction(1, 1):
                continue
            
            # Try all residues
            for r in range(1, p + 1):
                # Count how many uncovered numbers this covers
                count = 0
                for n in range(r, target_L + 1, p):
                    if not covered[n]:
                        count += 1
                if count == 0:
                    continue
                # Score: coverage / cost
                score = count * p  # Proportional to count / (1/p)
                if score > best_score:
                    best_score = score
                    best_prime = i
                    best_residue = r
        
        if best_prime == -1:
            break
        
        # Assign
        assign[best_prime] = best_residue
        used_primes.add(best_prime)
        current_cost += Fraction(1, _PRIMES[best_prime])
        
        # Update covered
        p = _PRIMES[best_prime]
        for n in range(best_residue, target_L + 1, p):
            covered[n] = 1
    
    # Check if we covered everything
    for n in range(1, target_L + 1):
        if not covered[n]:
            return assign, False
    return assign, True


def construct():
    start = time.time()
    deadline = start + 115  # safety margin before 120s kill
    
    # Start with a known good solution
    # Use the seed approach: primes from 13
    primes_from_13 = [p for p in _PRIMES if p >= 13]
    init_assign = [0] * NUM_PRIMES
    for i, p in enumerate(primes_from_13):
        idx = _PRIMES.index(p)
        r = (i + 1) % p
        if r == 0:
            r = 1
        init_assign[idx] = r
    
    if _is_valid(init_assign):
        best_assign = init_assign[:]
        best_L = _coverage_length(init_assign)
    else:
        best_assign = [0] * NUM_PRIMES
        best_L = 0
    
    # Try to improve by targeting higher L values
    # Use CSP for targeted improvements
    target_L = max(best_L + 100, 1000)
    
    while target_L <= 10000 and time.time() < deadline - 10:
        # Get primes to use: try to find a subset that can cover target_L
        # Start with primes up to some limit
        max_prime = min(1000, target_L * 2)
        primes_to_use = [p for p in _PRIMES if p <= max_prime]
        
        # Try CSP
        time_left = deadline - time.time() - 5
        if time_left > 1:
            result = _csp_solve(target_L, primes_to_use, time_left)
            if result is not None:
                # Build assign from result
                new_assign = [0] * NUM_PRIMES
                for p_idx, r in result.items():
                    p = primes_to_use[p_idx]
                    global_idx = _PRIMES.index(p)
                    new_assign[global_idx] = r
                
                if _is_valid(new_assign):
                    new_L = _coverage_length(new_assign)
                    if new_L > best_L:
                        best_L = new_L
                        best_assign = new_assign[:]
                        target_L = new_L + 100
                        continue
        
        # Try greedy
        time_left = deadline - time.time() - 5
        if time_left > 1:
            greedy_assign, success = _greedy_build(target_L, time_left)
            if success and _is_valid(greedy_assign):
                new_L = _coverage_length(greedy_assign)
                if new_L > best_L:
                    best_L = new_L
                    best_assign = greedy_assign[:]
                    target_L = new_L + 100
                    continue
        
        # Increase target
        target_L += 500
    
    # Final validation
    if best_assign is not None and _is_valid(best_assign):
        return [[_PRIMES[i], r] for i, r in enumerate(best_assign) if r != 0]
    else:
        # Fallback to seed
        return [[p, (i + 1) % p] for i, p in enumerate(primes_from_13)]
