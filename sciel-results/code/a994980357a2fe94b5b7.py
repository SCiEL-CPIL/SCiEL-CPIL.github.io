"""Rewritten program: DP with Pareto-optimal (length, last_gap) states for global optimality."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from bisect import bisect_left, bisect_right

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill

    # Sieve of Eratosthenes for primes <= 10000
    limit = 10000
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = 0

    primes = np.array([p for p in range(2, limit + 1) if is_prime[p]], dtype=np.int32)
    n = len(primes)

    if n == 0:
        return []

    # For each prime index i, maintain a list of Pareto-optimal states:
    # Each state is (length, last_gap, prev_state_ref)
    # A state (L1, G1) dominates (L2, G2) if L1 >= L2 and G1 <= G2
    # We keep only non-dominated states.
    
    # states[i] = list of (length, last_gap, prev_idx, prev_state_idx)
    # prev_state_idx is the index into states[prev_idx] of the predecessor state
    states = [[] for _ in range(n)]
    states[0].append((1, 0, -1, -1))  # prime[0] alone, no predecessor

    # For efficient lookup: for each i, sort states by length descending, gap ascending
    # This helps in pruning and backtracking

    for i in range(1, n):
        if time.time() > deadline:
            break

        pi = int(primes[i])
        new_states = []
        # Start with the trivial state: just prime[i] alone
        new_states.append((1, 0, -1, -1))

        # For each j < i, check if we can extend any state at j
        for j in range(i):
            gap = pi - int(primes[j])
            # For each state at j, check if we can extend
            for (L, G, prev_j, prev_state_idx) in states[j]:
                if gap >= G:
                    new_states.append((L + 1, gap, j, prev_state_idx))

        # Now prune new_states to keep only Pareto-optimal ones
        # Sort by length descending, then by last_gap ascending
        new_states.sort(key=lambda x: (-x[0], x[1]))
        pruned = []
        for state in new_states:
            L, G, _, _ = state
            # Check if this state is dominated by any already-kept state
            dominated = False
            for (Lk, Gk, _, _) in pruned:
                if Lk >= L and Gk <= G:
                    dominated = True
                    break
            if not dominated:
                pruned.append(state)
        states[i] = pruned

    # Find the global best state across all primes
    best_len = 0
    best_prime_idx = 0
    best_state_idx = 0
    for i in range(n):
        for (L, G, prev_i, prev_state_idx) in states[i]:
            if L > best_len:
                best_len = L
                best_prime_idx = i
                best_state_idx = 0

    # Reconstruct the sequence
    if best_len == 0:
        return []

    seq = []
    cur_idx = best_prime_idx
    cur_state_idx = 0

    # Trace back using the stored backpointers
    while cur_idx != -1:
        seq.append(int(primes[cur_idx]))
        L, G, prev_idx, prev_state_idx = states[cur_idx][cur_state_idx]
        if prev_idx == -1:
            break
        cur_idx = prev_idx
        cur_state_idx = prev_state_idx

    seq.reverse()
    return seq
# EVOLVE-BLOCK-END
