import time
import math
import random
from collections import Counter

N = 5000
DEADLINE = 305.0  # seconds, keep safety margin under 330s

# EVOLVE-BLOCK-START

def _sieve_primes(limit):
    """Return list of all primes <= limit."""
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]

def _construct_residue_state(primes, n_max=N, seed=None):
    """
    Initialize a random state of residue classes for each prime.
    state[p] = r means we remove all x such that x % p == r.
    """
    if seed is not None:
        random.seed(seed)
    
    state = {}
    for p in primes:
        state[p] = random.randint(0, p - 1)
        
    return state

def _compute_initial_counts(state, primes, n_max=N):
    """
    Compute the coverage counts C[x] for all x in [0, n_max].
    C[x] = number of primes p such that x % p == state[p].
    Also returns the size of the admissible set (count of x with C[x]==0).
    """
    C = [0] * (n_max + 1)
    for p in primes:
        r = state[p]
        for x in range(r, n_max + 1, p):
            C[x] += 1
            
    size = sum(1 for c in C if c == 0)
    return C, size

def _get_indices_for_prime(p, n_max=N):
    """
    Pre-compute lists of indices for each residue class mod p.
    indices[r] = list of x in [0, n_max] such that x % p == r.
    """
    indices = [[] for _ in range(p)]
    for x in range(n_max + 1):
        indices[x % p].append(x)
    return indices

def _local_search_residue(primes, n_max=N, deadline=None, seed=None):
    """
    Local search / Simulated Annealing on the residue class assignments.
    """
    if seed is not None:
        random.seed(seed)
        
    # Pre-compute indices for speed
    indices_cache = {}
    for p in primes:
        indices_cache[p] = _get_indices_for_prime(p, n_max)
        
    # Initialize state
    state = _construct_residue_state(primes, n_max, seed=seed)
    C, size = _compute_initial_counts(state, primes, n_max)
    
    best_state = state.copy()
    best_size = size
    best_C = C.copy()
    
    if deadline is None:
        deadline = time.time() + 10.0
        
    start_time = time.time()
    
    # Simulated Annealing parameters
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    # We will run multiple passes or just one long SA
    # Given N=5000 and ~669 primes, one move is O(N/p).
    # Average N/p is small. So we can do many moves.
    
    while time.time() < deadline - 5:
        # Pick a random prime
        p = random.choice(primes)
        old_r = state[p]
        
        # Pick a new random residue different from old
        new_r = random.randint(0, p - 1)
        if new_r == old_r:
            continue
            
        # Calculate delta
        delta = 0
        
        # Elements in old_r: C[x] decreases by 1
        # If C[x] becomes 0, we gain 1 element
        for x in indices_cache[p][old_r]:
            C[x] -= 1
            if C[x] == 0:
                delta += 1
                
        # Elements in new_r: C[x] increases by 1
        # If C[x] was 0, we lose 1 element
        for x in indices_cache[p][new_r]:
            if C[x] == 0:
                delta -= 1
            C[x] += 1
            
        # Acceptance criterion
        accept = False
        if delta > 0:
            accept = True
        else:
            # Simulated Annealing: accept worse moves with probability exp(delta/T)
            if T > min_T:
                prob = math.exp(delta / T)
                if random.random() < prob:
                    accept = True
                    
        if accept:
            state[p] = new_r
            size += delta
            if size > best_size:
                best_size = size
                best_state = state.copy()
                best_C = C.copy()
        else:
            # Revert changes
            for x in indices_cache[p][old_r]:
                C[x] += 1
            for x in indices_cache[p][new_r]:
                C[x] -= 1
                # Note: We don't need to check C[x]==0 here because we only 
                # decrement if it was incremented. But wait, if we accepted, 
                # we don't revert. If we reject, we revert.
                # The logic above:
                # 1. Decrement old, Increment new.
                # 2. If reject, Increment old, Decrement new.
                # This restores C exactly.
                
        # Cooling
        T *= cooling
        
    # Verify best_state is admissible and return it
    # Construct the set from best_state
    # S = { x in [0, N] | for all p, x % p != best_state[p] }
    # We can just use best_C to identify the set
    final_set = [x for x in range(n_max + 1) if best_C[x] == 0]
    
    return final_set

def construct():
    """
    Construct the largest admissible tuple within [0, 5000].
    """
    start_time = time.time()
    primes = _sieve_primes(N + 1)
    
    # Filter primes to those <= N (actually all primes <= N are relevant for size up to N)
    # The condition is for p <= len(tuple). If len(tuple) <= N, we only care about p <= N.
    
    best_result = []
    best_size = 0
    
    # Run multiple restarts of the SA local search
    # Each restart takes some time. We divide the deadline among them.
    # Let's say we try to run as many as possible.
    # Each SA run should be fast.
    
    deadline = start_time + DEADLINE
    
    # First, a quick greedy baseline to have a valid solution early
    # Greedy: for each p, pick random r.
    # This is effectively one SA run with T=0 (greedy) or just random init.
    
    # Let's run several independent SA instances
    # We can run them in parallel? No, single thread.
    # We will run a loop of SA instances, each with a time slice.
    
    num_restarts = 0
    time_per_restart = 5.0 # seconds
    
    while time.time() < deadline - 10:
        seed = random.randint(0, 10**6)
        slice_deadline = min(deadline - 5, time.time() + time_per_restart)
        
        result = _local_search_residue(primes, N, deadline=slice_deadline, seed=seed)
        
        if len(result) > best_size:
            best_size = len(result)
            best_result = result
            
        num_restarts += 1
        
        # If we are running out of time, do a final polish on the best solution found
        if time.time() > deadline - 15:
            break
            
    # Final Polish: Run a short SA starting from the best found state
    if best_result and time.time() < deadline - 2:
        # Reconstruct state from best_result? 
        # It's hard to infer the exact state (removed classes) from the set.
        # But we can just run another SA with the remaining time.
        # Or, we can assume the best_result is good enough.
        # Let's just run one more SA with the remaining time.
        final_deadline = deadline - 1
        final_result = _local_search_residue(primes, N, deadline=final_deadline, seed=random.randint(0, 10**6))
        if len(final_result) > best_size:
            best_result = final_result
            best_size = len(final_result)
            
    # If no result was found (unlikely), return empty list or a simple valid set
    if not best_result:
        # Fallback: remove residue 0 for all primes
        # This is valid? 
        # S = { x | x % p != 0 for all p <= N }
        # Size is small, but valid.
        best_result = []
        for x in range(N + 1):
            valid = True
            for p in primes:
                if x % p == 0:
                    valid = False
                    break
            if valid:
                best_result.append(x)
                
    return best_result

# EVOLVE-BLOCK-END
