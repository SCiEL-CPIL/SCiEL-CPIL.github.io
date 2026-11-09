# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys
from collections import defaultdict

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a
    nonzero square mod 205.

    Uses:
      - Bitset-based conflict graph for fast compatibility checks
      - Simulated annealing with 1-1, 2-2, and 3-3 swap moves
      - Multi-start with random greedy initializations
      - Local search polish after each annealing run
    """
    m = 205
    deadline = time.time() + 580  # safety margin before 630s

    # Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    forbidden_set = forbidden

    # Build conflict graph as bitsets: conflict_mask[v] is a 205-bit int
    # where bit u is set iff (v - u) % m is in forbidden_set
    conflict_mask = [0] * m
    for v in range(m):
        mask = 0
        for u in range(m):
            if u == v:
                continue
            d = (v - u) % m
            if d in forbidden_set:
                mask |= (1 << u)
        conflict_mask[v] = mask

    def is_compatible(candidate, selected_bits):
        """Check if candidate is compatible with all elements in selected_bits (a bitmask)."""
        return (conflict_mask[candidate] & selected_bits) == 0

    def popcount(x):
        return bin(x).count('1')

    def random_greedy():
        """Randomized greedy construction."""
        order = list(range(m))
        random.shuffle(order)
        selected_bits = 0
        for v in order:
            if is_compatible(v, selected_bits):
                selected_bits |= (1 << v)
        return selected_bits

    def local_search_add_only(selected_bits, max_iters=500):
        """
        Local search: try to add any compatible vertex repeatedly.
        Returns improved bitmask.
        """
        current = selected_bits
        current_size = popcount(current)
        best = current
        best_size = current_size

        for it in range(max_iters):
            if time.time() > deadline:
                break

            improved = False
            temp = ~current & ((1 << m) - 1)
            while temp:
                low = temp & (-temp)
                v = low.bit_length() - 1
                if is_compatible(v, current):
                    new_bits = current | (1 << v)
                    current = new_bits
                    current_size += 1
                    improved = True
                    break
                temp ^= low
            if improved:
                if current_size > best_size:
                    best_size = current_size
                    best = current
                continue
            else:
                break

        return best, best_size

    def simulated_annealing(initial_bits, start_temp=1.0, end_temp=0.001, max_iters=10000):
        """
        Simulated annealing with 1-1 and 2-2 swap moves.
        Returns (best_bits, best_size).
        """
        current = initial_bits
        current_size = popcount(current)
        best = current
        best_size = current_size

        # Precompute all vertices
        all_vertices = list(range(m))

        def try_swap_1_1(state, state_size):
            """Try a random 1-1 swap: remove one vertex, add another."""
            if state_size == 0:
                # Just add a random compatible vertex
                temp = ~state & ((1 << m) - 1)
                if temp:
                    low = temp & (-temp)
                    v = low.bit_length() - 1
                    # Shuffle by picking random bits
                    candidates = []
                    t = temp
                    while t:
                        low2 = t & (-t)
                        idx = low2.bit_length() - 1
                        candidates.append(idx)
                        t ^= low2
                    if candidates:
                        v = random.choice(candidates)
                        if is_compatible(v, state):
                            return state | (1 << v), state_size + 1
                return state, state_size

            # Pick a random vertex to remove
            t = state
            candidates = []
            while t:
                low = t & (-t)
                idx = low.bit_length() - 1
                candidates.append(idx)
                t ^= low
            if not candidates:
                return state, state_size
            rem = random.choice(candidates)
            trial = state & ~(1 << rem)

            # Pick a random vertex to add
            temp2 = ~trial & ((1 << m) - 1)
            cands2 = []
            t = temp2
            while t:
                low = t & (-t)
                idx = low.bit_length() - 1
                cands2.append(idx)
                t ^= low
            if not cands2:
                return state, state_size

            # Try a few random candidates
            for _ in range(5):
                add_v = random.choice(cands2)
                if is_compatible(add_v, trial):
                    return trial | (1 << add_v), state_size
            return state, state_size

        def try_swap_2_2(state, state_size):
            """Try a random 2-2 swap: remove two vertices, add two others."""
            if state_size < 2:
                return state, state_size

            # Pick two random vertices to remove
            t = state
            candidates = []
            while t:
                low = t & (-t)
                idx = low.bit_length() - 1
                candidates.append(idx)
                t ^= low
            if len(candidates) < 2:
                return state, state_size
            rem1, rem2 = random.sample(candidates, 2)
            trial = state & ~(1 << rem1) & ~(1 << rem2)

            # Pick two random vertices to add
            temp2 = ~trial & ((1 << m) - 1)
            cands2 = []
            t = temp2
            while t:
                low = t & (-t)
                idx = low.bit_length() - 1
                cands2.append(idx)
                t ^= low
            if len(cands2) < 2:
                return state, state_size

            # Try a few random pairs
            for _ in range(5):
                add1, add2 = random.sample(cands2, 2)
                if is_compatible(add1, trial) and is_compatible(add2, trial | (1 << add1)):
                    return trial | (1 << add1) | (1 << add2), state_size
            return state, state_size

        n_iters = max_iters
        for it in range(n_iters):
            if time.time() > deadline:
                break

            # Linear cooling
            temp = start_temp * (end_temp / start_temp) ** (it / n_iters)

            # Choose move type
            r = random.random()
            if r < 0.4:
                new_state, new_size = try_swap_1_1(current, current_size)
            elif r < 0.7:
                new_state, new_size = try_swap_2_2(current, current_size)
            else:
                # Try to add a vertex
                temp_add = ~current & ((1 << m) - 1)
                cands = []
                t = temp_add
                while t:
                    low = t & (-t)
                    idx = low.bit_length() - 1
                    cands.append(idx)
                    t ^= low
                if cands:
                    v = random.choice(cands)
                    if is_compatible(v, current):
                        new_state = current | (1 << v)
                        new_size = current_size + 1
                    else:
                        new_state = current
                        new_size = current_size
                else:
                    new_state = current
                    new_size = current_size

            delta = new_size - current_size
            if delta > 0 or random.random() < pow(2.718281828, delta / max(temp, 1e-10)):
                current = new_state
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best = current

        # Final local search add-only polish
        best, best_size = local_search_add_only(best, max_iters=200)
        return best, best_size

    # Main search
    best_bits = 0
    best_size = 0

    # Initial deterministic greedy
    for v in range(m):
        if is_compatible(v, best_bits):
            best_bits |= (1 << v)
    best_size = popcount(best_bits)

    # Local search polish on initial
    ls_bits, ls_size = local_search_add_only(best_bits, max_iters=500)
    if ls_size > best_size:
        best_size = ls_size
        best_bits = ls_bits

    # Multiple restarts with simulated annealing
    restart_count = 0
    while time.time() < deadline - 60:
        restart_count += 1

        # Random greedy initialization
        g_bits = random_greedy()
        g_size = popcount(g_bits)

        # Simulated annealing
        sa_bits, sa_size = simulated_annealing(g_bits, start_temp=1.0, end_temp=0.001, max_iters=5000)

        if sa_size > best_size:
            best_size = sa_size
            best_bits = sa_bits

        # Also try local search on the best found so far
        if restart_count % 3 == 0:
            ls2_bits, ls2_size = local_search_add_only(best_bits, max_iters=300)
            if ls2_size > best_size:
                best_size = ls2_size
                best_bits = ls2_bits

        # Occasionally try a longer annealing run from the current best
        if restart_count % 7 == 0 and time.time() < deadline - 120:
            sa2_bits, sa2_size = simulated_annealing(best_bits, start_temp=0.5, end_temp=0.001, max_iters=8000)
            if sa2_size > best_size:
                best_size = sa2_size
                best_bits = sa2_bits

    # Convert bitmask to array
    result = []
    for v in range(m):
        if best_bits & (1 << v):
            result.append(v)

    return np.array(result, dtype=np.int64)


# EVOLVE-BLOCK-END
