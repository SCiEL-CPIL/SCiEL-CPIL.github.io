import time
import math
import random

MAX_MOD = 128
DEADLINE_MARGIN = 5.0  # seconds before the 90s kill

def _disjoint(r1, m1, r2, m2):
    """Check if residue classes (r1 mod m1) and (r2 mod m2) are disjoint."""
    g = math.gcd(m1, m2)
    # Disjoint if (r2 - r1) is NOT divisible by g
    return (r2 - r1) % g != 0

def _greedy_with_backtrack(moduli, deadline, max_backtrack_depth=5):
    """
    Greedy assignment with limited backtracking.
    Returns list of [residue, modulus] if successful, None otherwise.
    """
    n = len(moduli)
    
    # Precompute gcds for speed
    gcds = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            g = math.gcd(moduli[i], moduli[j])
            gcds[i][j] = g
            gcds[j][i] = g
    
    # Sort by number of conflicts (most constrained first)
    # A modulus with many other moduli sharing a gcd > 1 is more constrained
    conflict_counts = [0]*n
    for i in range(n):
        for j in range(n):
            if i != j and gcds[i][j] > 1:
                conflict_counts[i] += 1
    
    # Order: most constrained first
    order = list(range(n))
    random.shuffle(order)  # Add randomness to break ties
    order.sort(key=lambda i: conflict_counts[i], reverse=True)
    
    assigned = {}  # index -> residue
    
    def backtrack(idx, depth=0):
        if time.time() > deadline:
            return False
        if idx == n:
            return True
        
        i = order[idx]
        m = moduli[i]
        
        # Try residues in random order
        residues = list(range(m))
        random.shuffle(residues)
        
        for r in residues:
            ok = True
            for prev_i, prev_r in assigned.items():
                g = gcds[i][prev_i]
                if (r - prev_r) % g == 0:
                    ok = False
                    break
            
            if ok:
                assigned[i] = r
                if backtrack(idx + 1, depth + 1):
                    return True
                del assigned[i]
                
                # If we've exceeded backtracking depth, stop trying more residues
                if depth >= max_backtrack_depth:
                    return False
        
        return False
    
    if backtrack(0):
        result = []
        for i in range(n):
            result.append([assigned[i], moduli[i]])
        return result
    
    return None

def _greedy_simple(moduli, deadline):
    """Simple greedy without backtracking. Fast but less likely to succeed."""
    n = len(moduli)
    
    # Precompute gcds
    gcds = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            g = math.gcd(moduli[i], moduli[j])
            gcds[i][j] = g
            gcds[j][i] = g
    
    # Most constrained first
    conflict_counts = [0]*n
    for i in range(n):
        for j in range(n):
            if i != j and gcds[i][j] > 1:
                conflict_counts[i] += 1
    
    order = list(range(n))
    random.shuffle(order)
    order.sort(key=lambda i: conflict_counts[i], reverse=True)
    
    assigned = {}
    
    for i in order:
        m = moduli[i]
        residues = list(range(m))
        random.shuffle(residues)
        
        for r in residues:
            ok = True
            for prev_i, prev_r in assigned.items():
                g = gcds[i][prev_i]
                if (r - prev_r) % g == 0:
                    ok = False
                    break
            
            if ok:
                assigned[i] = r
                break
        else:
            return None
    
    result = []
    for i in range(n):
        result.append([assigned[i], moduli[i]])
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 90.0 - DEADLINE_MARGIN
    
    # Strategy: Try different subsets of moduli, starting with the most promising.
    # The best subsets likely include many coprime moduli (primes) and carefully chosen composites.
    
    # Candidate sets to try, in order of preference:
    # 1. All primes from 2 to 127 (there are 31 primes)
    # 2. All primes + some small composites that are coprime to many primes
    # 3. All numbers that are powers of 2 (2, 4, 8, ..., 128) - 7 numbers
    # 4. All even numbers (2, 4, ..., 128) - 64 numbers
    # 5. All numbers from 2 to 128 - 127 numbers (unlikely to work)
    
    # Let's think about this more carefully.
    # Two residue classes (r1 mod m1) and (r2 mod m2) are disjoint iff (r1 - r2) % gcd(m1, m2) != 0.
    # This is always possible iff we can choose r1 and r2 such that they differ mod gcd(m1, m2).
    # For a set of moduli, we need to assign residues such that for every pair, the residues differ mod their gcd.
    
    # This is equivalent to: for each prime power p^k that divides some moduli, the residues must be distinct mod p^k.
    # Actually, it's more nuanced. Let me think again.
    
    # The condition (r1 - r2) % gcd(m1, m2) != 0 means that r1 and r2 must differ in at least one prime factor of gcd(m1, m2).
    
    # A good strategy: use moduli that are pairwise coprime. Then gcd = 1, and any residues work.
    # The maximum number of pairwise coprime moduli from 2 to 128 is limited by the number of distinct primes.
    # Primes from 2 to 127: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127
    # That's 31 primes. They are all pairwise coprime, so we can assign arbitrary residues.
    
    # But we can do better by including some composite moduli that are coprime to the primes we don't use.
    # For example, 4 is coprime to 3, 5, 7, ... but not to 2.
    
    # Let's try a greedy approach with a larger set of moduli, but start with primes and add composites carefully.
    
    # First, let's get all primes up to 128
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    primes = [n for n in range(2, MAX_MOD + 1) if is_prime(n)]
    
    # Try 1: Just the primes (31 moduli, always works)
    best_result = None
    best_size = 0
    
    # Attempt with primes
    result = _greedy_simple(primes, deadline)
    if result is not None:
        best_result = result
        best_size = len(result)
    
    # Attempt 2: Primes + some composites
    # Add composites that are coprime to all primes we've chosen, or carefully chosen
    # Let's try adding composites one by one, starting with small ones
    composites = [n for n in range(2, MAX_MOD + 1) if not is_prime(n)]
    random.shuffle(composites)
    
    current_moduli = primes[:]
    current_result = best_result[:] if best_result else []
    
    for c in composites:
        if time.time() > deadline:
            break
        
        # Try adding c to the current set
        test_moduli = current_moduli + [c]
        result = _greedy_simple(test_moduli, deadline)
        if result is not None:
            current_moduli = test_moduli
            current_result = result
            best_size = len(current_result)
    
    if current_result and len(current_result) > best_size:
        best_result = current_result
        best_size = len(current_result)
    
    # Attempt 3: Try all even numbers with backtracking (64 moduli)
    if time.time() < deadline - 10:
        evens = list(range(2, MAX_MOD + 1, 2))
        result = _greedy_with_backtrack(evens, deadline, max_backtrack_depth=10)
        if result is not None and len(result) > best_size:
            best_result = result
            best_size = len(result)
    
    # Attempt 4: Try a mix of primes and evens
    if time.time() < deadline - 5:
        # Use all primes and all evens that are not primes
        # Actually, let's try all numbers that are either prime or even
        mixed = [n for n in range(2, MAX_MOD + 1) if is_prime(n) or n % 2 == 0]
        random.shuffle(mixed)
        # Limit to avoid timeout
        mixed = mixed[:min(len(mixed), 80)]
        result = _greedy_with_backtrack(mixed, deadline, max_backtrack_depth=5)
        if result is not None and len(result) > best_size:
            best_result = result
            best_size = len(result)
    
    # Fallback: if nothing worked, return a known valid small set
    if best_result is None:
        best_result = [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
    
    return best_result
