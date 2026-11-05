"""
Strategy: Simulated Annealing over the full vector of residues.

Rationale:
- The current program uses local hill-climbing which gets stuck in local minima.
- Simulated Annealing (SA) allows escaping local minima by accepting worse solutions
  with a probability that depends on temperature.
- We optimize all 168 primes simultaneously, providing more degrees of freedom.
- We use a fast vectorized evaluation for the score (largest L with count >= 2).
- We use a cooling schedule to balance exploration and exploitation.
"""
import time
import numpy as np
import random

def _get_primes(n):
    """Sieve of Eratosthenes to get primes up to n."""
    sieve = [True] * (n + 1)
    if n >= 2:
        sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = [False] * len(sieve[i * i::i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def _compute_counts(residues, primes, max_L):
    """Compute coverage counts array for given residues."""
    counts = np.zeros(max_L + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        if r == 0:
            start = p
        else:
            start = r
        if start > max_L:
            continue
        pos = np.arange(start, max_L + 1, p)
        if len(pos) > 0:
            counts[pos] += 1
    return counts

def _find_L(counts, max_L):
    """Find largest L such that counts[1..L] >= 2."""
    below_two = np.where(counts[1:] < 2)[0]
    if len(below_two) == 0:
        return max_L
    else:
        return below_two[0]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s limit

    primes = _get_primes(1000)
    n_primes = len(primes)
    max_L = 100000

    # Precompute positions for fast updates
    # positions[i][r] is the array of indices in 1..max_L covered by residue r mod primes[i]
    positions = []
    for p in primes:
        pos_list = []
        for r in range(p):
            if r == 0:
                start = p
            else:
                start = r
            if start > max_L:
                pos_list.append(np.array([], dtype=np.int32))
            else:
                pos_list.append(np.arange(start, max_L + 1, p, dtype=np.int32))
        positions.append(pos_list)

    # Initial state: Greedy construction to start with a decent baseline
    # Greedy: for each prime, pick r that maximizes L given previous choices
    residues = [0] * n_primes
    counts = np.zeros(max_L + 1, dtype=np.int32)
    
    # A simple greedy initialization is better than random for SA
    for i, p in enumerate(primes):
        best_r = 0
        best_L = -1
        # Sample a subset of residues if p is large to speed up initialization
        # But for small primes, check all.
        if p <= 100:
            candidates = range(p)
        else:
            candidates = random.sample(range(p), min(20, p))
            
        for r in candidates:
            pos = positions[i][r]
            if len(pos) > 0:
                counts[pos] += 1
            L = _find_L(counts, max_L)
            if len(pos) > 0:
                counts[pos] -= 1
            if L > best_L:
                best_L = L
                best_r = r
        residues[i] = best_r
        pos = positions[i][best_r]
        if len(pos) > 0:
            counts[pos] += 1

    current_L = _find_L(counts, max_L)
    best_L = current_L
    best_residues = residues[:]
    best_counts = counts.copy()

    # Simulated Annealing Parameters
    # Start with high temperature, cool down
    T = 1000.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    # We will run SA until deadline
    # Each step: pick a random prime, pick a new random residue for it
    # Evaluate delta in L
    
    while time.time() < deadline:
        # Pick a random prime index
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        current_r = residues[i]
        
        # Pick a new random residue
        if p == 1: continue
        new_r = random.randint(0, p - 1)
        if new_r == current_r:
            continue
            
        # Calculate delta in counts
        old_pos = positions[i][current_r]
        new_pos = positions[i][new_r]
        
        # We need to find the new L.
        # Instead of copying the whole array, we can update counts in place, check L, and revert if rejected.
        # However, finding L requires scanning. 
        # Optimization: If we only change one prime, the count changes by -1 at old_pos and +1 at new_pos.
        # The set of positions where count < 2 can only change at old_pos (count decreases) or new_pos (count increases).
        # But L is determined by the *first* position where count < 2.
        
        # Let's just do the in-place update and find L. It's O(max_L) in worst case but usually fast if we break early.
        # Actually, _find_L uses np.where which is O(max_L). 
        # 150s / (168 * 100000 ops) is tight. 
        # Let's try to be smarter.
        
        # Update counts
        if len(old_pos) > 0:
            counts[old_pos] -= 1
        if len(new_pos) > 0:
            counts[new_pos] += 1
            
        new_L = _find_L(counts, max_L)
        
        delta_L = new_L - current_L
        
        # Acceptance criterion
        accepted = False
        if delta_L >= 0:
            accepted = True
        else:
            # Probability of accepting worse move
            # We use delta_L as the "energy" difference. 
            # Since L is an integer, small changes are common.
            # exp(delta_L / T) might be too small if T is small.
            # Let's use a scaled delta.
            # If T is large, we accept many. As T drops, we accept fewer.
            if T > 0:
                prob = np.exp(delta_L / T)
                if random.random() < prob:
                    accepted = True
            else:
                accepted = False

        if accepted:
            residues[i] = new_r
            current_L = new_L
            if current_L > best_L:
                best_L = current_L
                best_residues = residues[:]
                best_counts = counts.copy()
        else:
            # Revert
            if len(old_pos) > 0:
                counts[old_pos] += 1
            if len(new_pos) > 0:
                counts[new_pos] -= 1

        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # If we hit min temp, we might want to restart or just do hill climbing
            # For simplicity, we keep going but effectively do hill climbing
            # To avoid getting stuck, we can occasionally reheat
            if random.random() < 0.01:
                T = 100.0

    return best_residues

# Helper to ensure we don't crash if called in a weird way
if __name__ == "__main__":
    res = construct()
    print(len(res))
