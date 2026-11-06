"""Simulated annealing for prime residue class covering."""
import time
import numpy as np
from fractions import Fraction

# Deadline guard: evaluator kills after 150s; use 135s as safety margin
DEADLINE = 135.0
MAX_L = 100000  # cap for score computation

def _primes_upto(n):
    sieve = np.ones(n + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = False
    return np.where(sieve)[0]

# Precompute primes up to 1000
PRIMES = _primes_upto(1000)
NPRIMES = len(PRIMES)
# Map prime value -> index
PRIME_INDEX = {int(p): i for i, p in enumerate(PRIMES)}
# Reciprocals as Fractions for exact budget check
RECIPOCS = [Fraction(1, int(p)) for p in PRIMES]

def _coverage_length(used_primes, residues, limit=100000):
    """Return largest L such that 1..L are all covered by the classes."""
    if len(used_primes) == 0:
        return 0
    # Build coverage mask over 1..limit
    # Use a bytearray for memory efficiency
    covered = bytearray(limit + 1)  # index 0 unused
    for idx, r in zip(used_primes, residues):
        p = int(PRIMES[idx])
        # Mark numbers n in 1..limit with n % p == r
        # First number >= 1 with n % p == r:
        # if r == 0: first is p (if p <= limit)
        # else: first is r (if r >= 1) or r + p
        if r == 0:
            start = p
        else:
            start = r
            if start < 1:
                start += p
        # Mark all start, start+p, start+2p, ... <= limit
        for n in range(start, limit + 1, p):
            covered[n] = 1
    # Find first gap
    # Use numpy for speed
    arr = np.frombuffer(covered, dtype=np.uint8)
    # arr[1:] corresponds to 1..limit
    # Find first zero
    if len(arr) > 1:
        zeros = np.where(arr[1:] == 0)[0]
        if len(zeros) == 0:
            return limit
        else:
            return int(zeros[0])  # first uncovered is zeros[0]+1, so L = zeros[0]
    return 0

def _budget_ok(used_primes):
    """Check sum of 1/p <= 1 exactly."""
    s = Fraction(0, 1)
    for idx in used_primes:
        s += RECIPOCS[idx]
    return s <= 1

def construct():
    start_time = time.time()
    np.random.seed(42)
    
    # Initial state: greedy construction
    # Start with a reasonable set
    used = []  # list of prime indices
    residues = []
    
    # Greedy: add primes that cover the most new numbers
    # We'll build an initial solution
    limit = 10000  # working limit for greedy
    covered_greedy = bytearray(limit + 1)
    
    # Try adding primes in order of size (smaller primes cover more densely)
    # But we need to pick residues to maximize coverage
    # Simple greedy: for each prime, pick residue that covers most uncovered
    prime_order = sorted(range(NPRIMES), key=lambda i: int(PRIMES[i]))
    
    for pi in prime_order:
        if time.time() - start_time > DEADLINE - 5:
            break
        p = int(PRIMES[pi])
        # Try all residues, pick best
        best_r = 0
        best_gain = -1
        for r in range(p):
            # Count new coverage
            gain = 0
            if r == 0:
                start = p
            else:
                start = r
                if start < 1:
                    start += p
            for n in range(start, limit + 1, p):
                if not covered_greedy[n]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_r = r
        # Check budget
        test_used = used + [pi]
        if _budget_ok(test_used):
            # Apply
            used.append(pi)
            residues.append(best_r)
            # Update covered
            p = int(PRIMES[pi])
            r = best_r
            if r == 0:
                start = p
            else:
                start = r
                if start < 1:
                    start += p
            for n in range(start, limit + 1, p):
                covered_greedy[n] = 1
    
    # Compute initial L
    best_L = _coverage_length(used, residues, MAX_L)
    
    # Simulated annealing
    T = 50.0  # initial temperature
    T_min = 0.1
    cooling = 0.9995
    
    current_L = best_L
    current_used = list(used)
    current_residues = list(residues)
    
    # Track best
    best_used = list(used)
    best_residues = list(residues)
    best_L = current_L
    
    iteration = 0
    while time.time() - start_time < DEADLINE - 2:
        iteration += 1
        # Choose a move
        move_type = np.random.randint(0, 5)
        new_used = list(current_used)
        new_residues = list(current_residues)
        
        if move_type == 0 and len(new_used) < NPRIMES:
            # Add a new prime
            candidates = [i for i in range(NPRIMES) if i not in new_used]
            if candidates:
                pi = candidates[np.random.randint(len(candidates))]
                new_used.append(pi)
                # Pick random residue
                p = int(PRIMES[pi])
                new_residues.append(int(np.random.randint(0, p)))
                
        elif move_type == 1 and len(new_used) > 0:
            # Remove a prime
            idx = int(np.random.randint(len(new_used)))
            new_used.pop(idx)
            new_residues.pop(idx)
            
        elif move_type == 2 and len(new_used) > 0:
            # Change residue of a used prime
            idx = int(np.random.randint(len(new_used)))
            pi = new_used[idx]
            p = int(PRIMES[pi])
            new_residues[idx] = int(np.random.randint(0, p))
            
        elif move_type == 3 and len(new_used) >= 2:
            # Swap residues of two used primes
            i, j = np.random.choice(len(new_used), 2, replace=False)
            new_residues[i], new_residues[j] = new_residues[j], new_residues[i]
            
        elif move_type == 4 and len(new_used) > 0:
            # Replace one class with a different prime
            idx = int(np.random.randint(len(new_used)))
            old_pi = new_used[idx]
            candidates = [i for i in range(NPRIMES) if i != old_pi and i not in new_used]
            if candidates:
                new_pi = candidates[np.random.randint(len(candidates))]
                new_used[idx] = new_pi
                p = int(PRIMES[new_pi])
                new_residues[idx] = int(np.random.randint(0, p))
        
        # Check budget
        if not _budget_ok(new_used):
            # Reject
            T *= cooling
            continue
        
        # Evaluate
        new_L = _coverage_length(new_used, new_residues, MAX_L)
        
        # Acceptance
        if new_L >= current_L:
            current_L = new_L
            current_used = new_used
            current_residues = new_residues
            if new_L > best_L:
                best_L = new_L
                best_used = list(new_used)
                best_residues = list(new_residues)
        else:
            # Accept with probability exp((new_L - current_L) / T)
            delta = new_L - current_L
            if np.random.rand() < np.exp(delta / T):
                current_L = new_L
                current_used = new_used
                current_residues = new_residues
        
        T = max(T * cooling, T_min)
    
    # Return best
    result = [[int(PRIMES[pi]), r] for pi, r in zip(best_used, best_residues)]
    return result

if __name__ == "__main__":
    result = construct()
    print(f"Found {len(result)} classes, L = {max(r for p, r in result) if result else 0}")
