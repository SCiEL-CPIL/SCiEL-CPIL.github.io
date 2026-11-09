import time
import math
import random
from math import gcd

# EVOLVE-BLOCK-START

_N = 512
_LIMIT = 511
_DEADLINE = time.time() + 145.0  # Safety margin for 150s limit

# Precompute weights
_W = [0.0] * (_LIMIT + 1)
for a in range(1, _LIMIT + 1):
    _W[a] = 1.0 / (_N - a)

# Precompute prime factors for each number 1..511
# This allows fast conflict checking: two numbers conflict if they share a prime factor
_prime_factors = [[] for _ in range(_LIMIT + 1)]
for i in range(2, _LIMIT + 1):
    x = i
    while x > 1:
        # Find smallest prime factor
        for p in range(2, int(math.isqrt(x)) + 1):
            if x % p == 0:
                _prime_factors[i].append(p)
                x //= p
                while x % p == 0:
                    x //= p
                break
        if x > 1:
            _prime_factors[i].append(x)

# Precompute conflict masks? 
# A set of numbers is valid if no two share a prime factor.
# We can represent the "used primes" as a set or a bitmask.
# Number of primes < 512 is ~97. A 128-bit integer or Python int bitmask works well.
_prime_to_idx = {}
_primes_list = []
for i in range(2, _LIMIT + 1):
    # Sieve to get primes
    pass

# Let's just generate primes up to 511
def _sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i*i, n+1, i):
                is_prime[j] = False
    return [i for i in range(2, n+1) if is_prime[i]]

_primes = _sieve(_LIMIT)
_prime_index = {p: i for i, p in enumerate(_primes)}

# Mask for each number: bits corresponding to its prime factors
_num_mask = [0] * (_LIMIT + 1)
for i in range(2, _LIMIT + 1):
    m = 0
    for p in _prime_factors[i]:
        m |= (1 << _prime_index[p])
    _num_mask[i] = m

# 1 has no prime factors, mask is 0.

def _is_valid_addition(candidate, used_mask):
    """Check if candidate can be added to a set with used_mask."""
    return (_num_mask[candidate] & used_mask) == 0

def _greedy_initial():
    """Generate an initial valid set using greedy heuristic."""
    chosen = []
    used_mask = 0
    # Sort by weight descending
    sorted_indices = sorted(range(1, _LIMIT + 1), key=lambda x: -_W[x])
    for a in sorted_indices:
        if _is_valid_addition(a, used_mask):
            chosen.append(a)
            used_mask |= _num_mask[a]
    return chosen, used_mask

def _score_set(chosen):
    return sum(_W[a] for a in chosen)

def construct():
    # 1. Initialize with greedy
    current_set, current_mask = _greedy_initial()
    current_score = _score_set(current_set)
    
    best_set = list(current_set)
    best_score = current_score
    
    # 2. Simulated Annealing
    # Parameters
    T = 1.0  # Initial temperature (score units)
    cooling_rate = 0.999
    min_T = 1e-4
    
    # We need to move efficiently.
    # Move: Remove one element, Add one element.
    # To ensure we can add, we need to check conflicts.
    # Removing an element frees up its prime mask.
    
    # Precompute list of all valid candidates for quick access?
    # Or just iterate 1..511.
    
    # To speed up, we can maintain a list of "available" numbers?
    # But availability changes as we remove/add.
    # Since 511 is small, iterating all 511 numbers for each move is O(511).
    # Moves: ~100,000. 100,000 * 511 * (bitwise ops) is fast in Python?
    # 51M ops might take ~5-10s. Should be fine within 145s.
    
    # Optimization: Precompute weights and masks.
    
    # Random restarts if stuck?
    # For now, just run SA.
    
    # To allow "jumps", we can also try swapping two elements.
    # But simple remove-add is a good start.
    
    # We need a way to quickly find a candidate to add.
    # If we remove 'a', we can add any 'b' not in current_set such that 
    # _is_valid_addition(b, current_mask ^ _num_mask[a]).
    
    # Let's keep a set of currently chosen numbers for O(1) lookup.
    current_set_obj = set(current_set)
    
    while time.time() < _DEADLINE and T > min_T:
        # Pick a random element to remove
        if not current_set:
            break
        a = random.choice(current_set)
        
        # New mask after removing a
        new_mask = current_mask ^ _num_mask[a]
        
        # Find a random candidate b to add
        # We can pick a random number from 1..511 not in current_set
        # But we want to bias towards high weight?
        # SA usually picks random neighbor.
        
        # To make it more efficient, let's pick a random candidate from all 1..511
        # and check if it's valid and not in set.
        # Or, better: pick from a pre-shuffled list?
        
        # Simple approach:
        # Pick random b in 1..511
        b = random.randint(1, _LIMIT)
        
        if b in current_set_obj:
            continue # Can't add if already there (unless we swap, but we removed a)
            # Actually if b == a, we are just removing and re-adding same? 
            # If b == a, new_mask doesn't change, but a is removed. 
            # If we add a back, we are at same state.
        
        if not _is_valid_addition(b, new_mask):
            continue
            
        # Calculate new score
        delta = _W[b] - _W[a]
        
        # Accept if better or with probability exp(delta/T)
        if delta > 0 or random.random() < math.exp(delta / T):
            # Apply move
            current_set_obj.remove(a)
            current_set_obj.add(b)
            current_mask = new_mask | _num_mask[b]
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_set = list(current_set_obj)
        
        T *= cooling_rate
    
    # If time remains, do local search (hill climbing)
    # Try all single swaps to improve
    improved = True
    while improved and time.time() < _DEADLINE:
        improved = False
        current_set_obj = set(best_set)
        current_mask = 0
        for a in current_set_obj:
            current_mask |= _num_mask[a]
        current_score = best_score
        
        # Try removing each a
        for a in list(current_set_obj):
            mask_without_a = current_mask ^ _num_mask[a]
            # Try adding each b
            for b in range(1, _LIMIT + 1):
                if b in current_set_obj:
                    continue
                if not _is_valid_addition(b, mask_without_a):
                    continue
                delta = _W[b] - _W[a]
                if delta > 1e-12:
                    # Accept
                    current_set_obj.remove(a)
                    current_set_obj.add(b)
                    current_mask = mask_without_a | _num_mask[b]
                    current_score += delta
                    improved = True
                    # Restart loop to check new state
                    break
            if improved:
                break
        
        if improved:
            best_set = list(current_set_obj)
            best_score = current_score
    
    return best_set

# EVOLVE-BLOCK-END
