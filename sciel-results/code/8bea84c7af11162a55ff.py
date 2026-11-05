"""Simulated Annealing solver for the covering problem."""
import time
import numpy as np


def _get_primes(n):
    """Sieve of Eratosthenes to get primes up to n."""
    sieve = [True] * (n + 1)
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


def _get_positions(r, p, max_L):
    """Get positions covered by residue r mod p in 1..max_L."""
    if r == 0:
        start = p
    else:
        start = r
    if start > max_L:
        return np.array([], dtype=np.int32)
    return np.arange(start, max_L + 1, p, dtype=np.int32)


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit

    primes = _get_primes(1000)
    n_primes = len(primes)
    max_L = 100000

    # Precompute positions for each prime and each possible residue
    # positions[i] is a list of arrays, positions[i][r] is the array of positions
    # covered by residue r mod primes[i]
    positions = []
    for i, p in enumerate(primes):
        pos_list = []
        for r in range(p):
            pos_list.append(_get_positions(r, p, max_L))
        positions.append(pos_list)

    # Initialize with random residues
    np.random.seed(42)
    current_residues = np.array([np.random.randint(0, p) for p in primes], dtype=np.int32)
    counts = _compute_counts(current_residues, primes, max_L)
    current_L = _find_L(counts, max_L)

    best_residues = current_residues.copy()
    best_L = current_L

    # Simulated Annealing parameters
    T_start = 1000.0
    T_end = 1.0
    # Number of steps: we want to finish in ~130 seconds
    # Each step involves:
    # 1. Pick a random prime index
    # 2. Pick a new random residue
    # 3. Update counts (O(L/p) operations)
    # 4. Compute new L (O(L) worst case, but usually fast if we track min)
    # 5. Accept/reject
    
    # To speed up L computation, we can maintain a count of how many positions
    # have count < 2. But finding the *first* such position is the bottleneck.
    # Instead, let's just recompute L from scratch but use numpy efficiently.
    # Actually, _find_L uses np.where which is O(L).
    # With 168 primes and L=100000, one full evaluation is ~168 * (100000/p_avg) updates + 100000 search.
    # p_avg is around 500. So updates are ~168 * 200 = 33600 ops. Search is 100000 ops.
    # Total ~130k ops per step. In numpy, this is fast.
    # Let's aim for ~50000 steps.
    
    n_steps = 50000
    alpha = (T_end / T_start) ** (1.0 / n_steps)

    current_T = T_start
    
    for step in range(n_steps):
        if (time.time() - start_time) > deadline - 5:
            break
            
        # Pick a random prime index
        i = np.random.randint(0, n_primes)
        p = primes[i]
        
        # Pick a new random residue different from current
        old_r = current_residues[i]
        new_r = np.random.randint(0, p)
        while new_r == old_r:
            new_r = np.random.randint(0, p)
        
        # Update counts
        old_pos = positions[i][old_r]
        new_pos = positions[i][new_r]
        
        if len(old_pos) > 0:
            counts[old_pos] -= 1
        if len(new_pos) > 0:
            counts[new_pos] += 1
            
        new_L = _find_L(counts, max_L)
        
        # Acceptance criterion
        delta_L = new_L - current_L
        if delta_L > 0:
            # Accept
            current_residues[i] = new_r
            current_L = new_L
            if new_L > best_L:
                best_L = new_L
                best_residues = current_residues.copy()
        else:
            # Accept with probability exp(delta_L / T)
            # delta_L is negative, so exp(negative/T) is small
            if np.random.rand() < np.exp(delta_L / current_T):
                current_residues[i] = new_r
                current_L = new_L
            else:
                # Reject: revert counts
                if len(new_pos) > 0:
                    counts[new_pos] -= 1
                if len(old_pos) > 0:
                    counts[old_pos] += 1
                    
        # Cool down
        current_T *= alpha

    return best_residues.tolist()
