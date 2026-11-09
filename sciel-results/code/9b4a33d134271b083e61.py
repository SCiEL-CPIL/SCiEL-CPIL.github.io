"""Rewritten program: DP with Pareto-optimal (length, last_gap) states for global optimality."""
# EVOLVE-BLOCK-START
import time
import numpy as np

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

    # For each prime i, maintain a list of Pareto-optimal states:
    # Each state is (length, last_gap, prev_index)
    # A state (L1, G1) dominates (L2, G2) if L1 >= L2 and G1 <= G2
    # We keep only non-dominated states.
    # states[i] = list of (length, last_gap, prev_idx)
    states = [[] for _ in range(n)]
    states[0].append((1, 0, -1))  # prime[0] alone, no predecessor

    # For each prime i, find best predecessors j < i
    for i in range(1, n):
        if time.time() > deadline:
            # Return best found so far
            break

        pi = int(primes[i])
        new_states = []
        # Start with the trivial state: just prime[i] alone
        new_states.append((1, 0, -1))

        for j in range(i):
            if time.time() > deadline:
                break
            gap = pi - int(primes[j])
            # For each state at j, check if we can extend
            for (L, G, prev_j) in states[j]:
                if gap >= G:
                    new_states.append((L + 1, gap, j))

        # Now prune new_states to keep only Pareto-optimal ones
        # Sort by length descending, then by last_gap ascending
        new_states.sort(key=lambda x: (-x[0], x[1]))
        pruned = []
        for state in new_states:
            L, G, _ = state
            # Check if this state is dominated by any already-kept state
            dominated = False
            for (Lk, Gk, _) in pruned:
                if Lk >= L and Gk <= G:
                    dominated = True
                    break
            if not dominated:
                pruned.append(state)
        states[i] = pruned

    # Find the global best state across all primes
    best_state = None
    best_prime_idx = 0
    for i in range(n):
        for (L, G, prev_i) in states[i]:
            if best_state is None or L > best_state[0]:
                best_state = (L, G, prev_i)
                best_prime_idx = i

    # Reconstruct the sequence
    if best_state is None:
        return []

    seq = []
    cur_idx = best_prime_idx
    # We need to trace back. But states[i] stores prev_idx which is the index of the predecessor prime.
    # However, the state at the predecessor also needs to be identified.
    # The issue: states[j] may have multiple states, and we need the specific one that was used.
    # We stored prev_j as the index j, but not which specific state at j.
    # This is a problem. Let me fix the state representation to include a reference to the specific state.

    # Actually, let me redo this more carefully.
    # Instead of storing just prev_idx, store a reference to the exact state.
    # But that's complex. Alternative: during backtracking, for each step, find the matching state.

    # Let me use a different approach: store for each state a unique ID and a backpointer.
    # This is getting complex. Let me use a simpler approach for backtracking.

    # Actually, the current approach has a flaw: when we store (L+1, gap, j), we don't know which specific state at j was used.
    # For backtracking, we need to know the exact state.

    # Let me restructure: each state is (length, last_gap, prev_state_id)
    # where prev_state_id is an index into a global list of all states.

    # Given time constraints, let me use a simpler heuristic: 
    # For backtracking, at each step, find the state at the predecessor that matches the required (length-1, some_gap <= current_gap).

    # This is getting too complex. Let me fall back to the original DP but with a correctness note.
    # The original DP may not be globally optimal, but it's fast and gives a good answer.

    # Actually, let me just return the best from the original approach since the Pareto approach is getting complex.
    # The score is already 1.3243, which is good.

    # Let me instead try to verify if the current DP is actually correct by checking if the Pareto approach gives a better answer.
    # But that's too slow to implement correctly in the time available.

    # I'll stick with the original DP approach but optimize it slightly.

    # Actually, I realize I should just return the best state found. Let me trace back properly.
    # For the Pareto states, I need to store backpointers correctly.

    # Given the complexity, let me just return the best length found and construct a valid sequence.
    # For backtracking, I'll use a greedy approach: at each step, find the predecessor that can lead to this state.

    if best_state is None:
        return [int(primes[0])]

    max_len = best_state[0]
    # Trace back
    seq = [int(primes[best_prime_idx])]
    current_gap = best_state[1]
    current_len = best_state[0]
    current_idx = best_prime_idx

    while current_len > 1:
        # Find predecessor j such that there exists a state (current_len-1, G) in states[j]
        # where G <= current_gap and primes[current_idx] - primes[j] == current_gap
        found = False
        for j in range(current_idx - 1, -1, -1):
            gap = int(primes[current_idx]) - int(primes[j])
            if gap == current_gap:
                for (L, G, _) in states[j]:
                    if L == current_len - 1 and G <= current_gap:
                        seq.append(int(primes[j]))
                        current_gap = G
                        current_len = L
                        current_idx = j
                        found = True
                        break
            if found:
                break
        if not found:
            # Should not happen, but break to avoid infinite loop
            break

    seq.reverse()
    return seq
# EVOLVE-BLOCK-END
