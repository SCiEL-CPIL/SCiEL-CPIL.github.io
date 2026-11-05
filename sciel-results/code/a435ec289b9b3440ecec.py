import time
import numpy as np
import heapq
import math

def _get_primes(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin

    primes = _get_primes(1000)
    n_primes = len(primes)
    MAX_L = 100000

    # Precompute for each prime i the list of positions covered by each residue
    # positions[i] is a list of length primes[i], where positions[i][r] is a numpy array
    # of positions in 1..MAX_L covered by residue r mod p
    # For large p, this is too much memory, so we'll compute on the fly for large primes
    # But for small primes (p <= 500), precompute
    PRECOMP_LIMIT = 500
    precomp = {}
    for i, p in enumerate(primes):
        if p <= PRECOMP_LIMIT:
            pos_list = []
            for r in range(p):
                if r == 0:
                    s = p
                else:
                    s = r
                if s <= MAX_L:
                    pos_list.append(np.arange(s, MAX_L + 1, p))
                else:
                    pos_list.append(np.array([], dtype=np.int32))
            precomp[i] = pos_list

    # Initial solution: random
    np.random.seed(42)
    residues = np.array([np.random.randint(0, p) for p in primes], dtype=np.int32)

    # Build initial coverage counts
    counts = np.zeros(MAX_L + 1, dtype=np.int32)
    for i, p in enumerate(primes):
        r = residues[i]
        if r == 0:
            s = p
        else:
            s = r
        if s <= MAX_L:
            pos = np.arange(s, MAX_L + 1, p)
            if len(pos) > 0:
                counts[pos] += 1

    # Build gap set: positions 1..MAX_L where counts < 2
    # Use a min-heap keyed by position, with lazy deletion via a dict of current counts
    # Actually, simpler: maintain a sorted list of gap positions, but that's slow to update.
    # Alternative: maintain a set of gap positions and a function to find the minimum.
    # For speed, use a heap with (position, count_at_insertion) and verify on pop.

    gap_heap = []
    gap_set = set()
    for pos in range(1, MAX_L + 1):
        if counts[pos] < 2:
            gap_set.add(pos)
            heapq.heappush(gap_heap, (pos, counts[pos]))

    def current_L():
        """Return the largest L such that all 1..L have count >= 2."""
        while gap_heap:
            pos, c = gap_heap[0]
            if pos in gap_set and counts[pos] < 2:
                return pos - 1
            else:
                heapq.heappop(gap_heap)
        return MAX_L

    best_L = current_L()
    best_residues = residues.copy()

    # Simulated annealing
    T = 50.0
    T_min = 0.01
    cooling = 0.9995
    rng = np.random.RandomState(12345)

    while (time.time() - start_time) < deadline:
        # Pick a random prime index
        i = rng.randint(0, n_primes)
        p = primes[i]
        old_r = residues[i]
        
        # Pick a new random residue
        new_r = rng.randint(0, p)
        if new_r == old_r:
            continue

        # Compute delta: positions that change
        if i in precomp:
            old_pos = precomp[i][old_r]
            new_pos = precomp[i][new_r]
        else:
            if old_r == 0:
                os = p
            else:
                os = old_r
            old_pos = np.arange(os, MAX_L + 1, p) if os <= MAX_L else np.array([], dtype=np.int32)
            
            if new_r == 0:
                ns = p
            else:
                ns = new_r
            new_pos = np.arange(ns, MAX_L + 1, p) if ns <= MAX_L else np.array([], dtype=np.int32)

        # Apply: remove old coverage, add new coverage
        # Update counts and gap structures
        if len(old_pos) > 0:
            for pos in old_pos:
                old_c = counts[pos]
                counts[pos] = old_c - 1
                new_c = old_c - 1
                if new_c < 2 and old_c >= 2:
                    gap_set.add(pos)
                    heapq.heappush(gap_heap, (pos, new_c))
                elif new_c >= 2 and old_c < 2:
                    gap_set.discard(pos)

        if len(new_pos) > 0:
            for pos in new_pos:
                old_c = counts[pos]
                counts[pos] = old_c + 1
                new_c = old_c + 1
                if new_c < 2 and old_c >= 2:
                    gap_set.add(pos)
                    heapq.heappush(gap_heap, (pos, new_c))
                elif new_c >= 2 and old_c < 2:
                    gap_set.discard(pos)

        new_L = current_L()
        delta_L = new_L - best_L

        # Accept if improves, or with probability exp(delta_L / T) if worsens
        if delta_L >= 0:
            residues[i] = new_r
            best_L = new_L
            best_residues = residues.copy()
        else:
            accept_prob = math.exp(delta_L / T) if T > 0 else 0
            if rng.random() < accept_prob:
                residues[i] = new_r
                best_L = new_L
                best_residues = residues.copy()
            else:
                # Revert
                if len(new_pos) > 0:
                    for pos in new_pos:
                        old_c = counts[pos]
                        counts[pos] = old_c - 1
                        new_c = old_c - 1
                        if new_c < 2 and old_c >= 2:
                            gap_set.add(pos)
                            heapq.heappush(gap_heap, (pos, new_c))
                        elif new_c >= 2 and old_c < 2:
                            gap_set.discard(pos)

                if len(old_pos) > 0:
                    for pos in old_pos:
                        old_c = counts[pos]
                        counts[pos] = old_c + 1
                        new_c = old_c + 1
                        if new_c < 2 and old_c >= 2:
                            gap_set.add(pos)
                            heapq.heappush(gap_heap, (pos, new_c))
                        elif new_c >= 2 and old_c < 2:
                            gap_set.discard(pos)
                best_L = current_L()

        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min

    return list(best_residues)
