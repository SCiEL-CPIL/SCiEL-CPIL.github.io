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
      - Precomputed neighbor lists for faster move generation
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

    # Precompute non-conflict neighbors for each vertex
    # non_conflict[v] = list of vertices u != v that are NOT in conflict with v
    non_conflict = [[] for _ in range(m)]
    for v in range(m):
        for u in range(m):
            if u == v:
                continue
            if not (conflict_mask[v] & (1 << u)):
                non_conflict[v].append(u)

    FULL_MASK = (1 << m) - 1

    def is_compatible(candidate, selected_bits):
        """Check if candidate is compatible with all elements in selected_bits (a bitmask)."""
        return (conflict_mask[candidate] & selected_bits) == 0

    def popcount(x):
        return bin(x).count('1')

    def get_selected_vertices(bits):
        """Get list of vertices from bitmask."""
        vertices = []
        t = bits
        while t:
            low = t & (-t)
            idx = low.bit_length() - 1
            vertices.append(idx)
            t ^= low
        return vertices

    def get_unselected_vertices(bits):
        """Get list of vertices NOT in bitmask."""
        vertices = []
        t = FULL_MASK ^ bits
        while t:
            low = t & (-t)
            idx = low.bit_length() - 1
            vertices.append(idx)
            t ^= low
        return vertices

    def random_unselected_vertex(bits):
        """Pick a random vertex not in bits, or None if all selected."""
        t = FULL_MASK ^ bits
        if t == 0:
            return None
        n = popcount(t)
        r = random.randrange(n)
        while r > 0:
            low = t & (-t)
            t ^= low
            r -= 1
        return (t & (-t)).bit_length() - 1

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
            temp = FULL_MASK ^ current
            while temp:
                low = temp & (-temp)
                v = low.bit_length() - 1
                if is_compatible(v, current):
                    current |= (1 << v)
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

    def local_search_remove_to_add(selected_bits, max_iters=300):
        """
        Try removing one element then adding one or two compatible elements.
        This can escape local optima where no single addition works.
        Returns (best_bits, best_size).
        """
        current = selected_bits
        current_size = popcount(current)
        best = current
        best_size = current_size

        for it in range(max_iters):
            if time.time() > deadline:
                break

            if current_size == 0:
                break

            # Try each selected vertex for removal
            improved = False
            selected = get_selected_vertices(current)
            random.shuffle(selected)

            for rem in selected:
                trial = current & ~(1 << rem)
                trial_size = current_size - 1

                # Try to add back compatible vertices
                temp = FULL_MASK ^ trial
                added = 0
                while temp and added < 2:
                    low = temp & (-temp)
                    v = low.bit_length() - 1
                    temp ^= low
                    if is_compatible(v, trial):
                        trial |= (1 << v)
                        trial_size += 1
                        added += 1
                        if trial_size > best_size:
                            best_size = trial_size
                            best = trial
                        if trial_size > current_size:
                            current = trial
                            current_size = trial_size
                            improved = True
                            break

                if improved:
                    break

            if not improved:
                break

        return best, best_size

    def simulated_annealing(initial_bits, start_temp=1.0, end_temp=0.001, max_iters=10000):
        """
        Simulated annealing with 1-1, 2-2, 3-3 swap moves, remove-only, and add moves.
        Returns (best_bits, best_size).
        """
        current = initial_bits
        current_size = popcount(current)
        best = current
        best_size = current_size

        n_iters = max_iters
        for it in range(n_iters):
            if time.time() > deadline:
                break

            # Geometric cooling
            temp = start_temp * (end_temp / start_temp) ** (it / max(1, n_iters - 1))

            # Choose move type
            r = random.random()
            if r < 0.30:
                # 1-1 swap: remove one, add one
                if current_size == 0:
                    v = random_unselected_vertex(current)
                    if v is not None and is_compatible(v, current):
                        new_state = current | (1 << v)
                        new_size = current_size + 1
                    else:
                        new_state = current
                        new_size = current_size
                else:
                    selected = get_selected_vertices(current)
                    rem = random.choice(selected)
                    trial = current & ~(1 << rem)
                    # Find candidates compatible with trial using bitmask
                    temp_mask = FULL_MASK ^ trial
                    cands = []
                    t = temp_mask
                    while t:
                        low = t & (-t)
                        u = low.bit_length() - 1
                        t ^= low
                        if is_compatible(u, trial):
                            cands.append(u)
                    if cands:
                        add_v = random.choice(cands)
                        new_state = trial | (1 << add_v)
                        new_size = current_size
                    else:
                        new_state = current
                        new_size = current_size

            elif r < 0.55:
                # 2-2 swap: remove two, add two
                if current_size < 2:
                    new_state = current
                    new_size = current_size
                else:
                    selected = get_selected_vertices(current)
                    rem1, rem2 = random.sample(selected, 2)
                    trial = current & ~(1 << rem1) & ~(1 << rem2)
                    # Find candidates compatible with trial
                    cands = []
                    t = FULL_MASK ^ trial
                    while t:
                        low = t & (-t)
                        u = low.bit_length() - 1
                        t ^= low
                        if is_compatible(u, trial):
                            cands.append(u)
                    if len(cands) >= 2:
                        add1, add2 = random.sample(cands, 2)
                        if is_compatible(add2, trial | (1 << add1)):
                            new_state = trial | (1 << add1) | (1 << add2)
                            new_size = current_size
                        else:
                            new_state = current
                            new_size = current_size
                    else:
                        new_state = current
                        new_size = current_size

            elif r < 0.75:
                # 3-3 swap: remove three, add three
                if current_size < 3:
                    new_state = current
                    new_size = current_size
                else:
                    selected = get_selected_vertices(current)
                    rem1, rem2, rem3 = random.sample(selected, 3)
                    trial = current & ~(1 << rem1) & ~(1 << rem2) & ~(1 << rem3)
                    # Find candidates compatible with trial
                    cands = []
                    t = FULL_MASK ^ trial
                    while t:
                        low = t & (-t)
                        u = low.bit_length() - 1
                        t ^= low
                        if is_compatible(u, trial):
                            cands.append(u)
                    if len(cands) >= 3:
                        add1, add2, add3 = random.sample(cands, 3)
                        if is_compatible(add2, trial | (1 << add1)) and is_compatible(add3, trial | (1 << add1) | (1 << add2)):
                            new_state = trial | (1 << add1) | (1 << add2) | (1 << add3)
                            new_size = current_size
                        else:
                            new_state = current
                            new_size = current_size
                    else:
                        new_state = current
                        new_size = current_size

            elif r < 0.85:
                # Remove-only move: remove one element (to escape local optima)
                if current_size > 0:
                    selected = get_selected_vertices(current)
                    rem = random.choice(selected)
                    new_state = current & ~(1 << rem)
                    new_size = current_size - 1
                else:
                    new_state = current
                    new_size = current_size

            else:
                # Try to add a vertex
                v = random_unselected_vertex(current)
                if v is not None and is_compatible(v, current):
                    new_state = current | (1 << v)
                    new_size = current_size + 1
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

        # Final local search polish: add-only then remove-to-add
        best, best_size = local_search_add_only(best, max_iters=300)
        best2, best_size2 = local_search_remove_to_add(best, max_iters=200)
        if best_size2 > best_size:
            best, best_size = best2, best_size2
        # One more add-only pass after removals
        best3, best_size3 = local_search_add_only(best, max_iters=200)
        if best_size3 > best_size:
            best, best_size = best3, best_size3

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

    # Also try remove-to-add on initial
    rm_bits, rm_size = local_search_remove_to_add(best_bits, max_iters=200)
    if rm_size > best_size:
        best_size = rm_size
        best_bits = rm_bits
    # Follow up with add-only
    ls2_bits, ls2_size = local_search_add_only(best_bits, max_iters=200)
    if ls2_size > best_size:
        best_size = ls2_size
        best_bits = ls2_bits

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
            ls3_bits, ls3_size = local_search_add_only(best_bits, max_iters=200)
            if ls3_size > best_size:
                best_size = ls3_size
                best_bits = ls3_bits
            rm2_bits, rm2_size = local_search_remove_to_add(best_bits, max_iters=150)
            if rm2_size > best_size:
                best_size = rm2_size
                best_bits = rm2_bits
            ls4_bits, ls4_size = local_search_add_only(best_bits, max_iters=150)
            if ls4_size > best_size:
                best_size = ls4_size
                best_bits = ls4_bits

        # Occasionally try a longer annealing run from the current best
        if restart_count % 7 == 0 and time.time() < deadline - 120:
            sa2_bits, sa2_size = simulated_annealing(best_bits, start_temp=0.5, end_temp=0.001, max_iters=8000)
            if sa2_size > best_size:
                best_size = sa2_size
                best_bits = sa2_bits

    # Final polish: alternate add-only and remove-to-add a few times
    for _ in range(3):
        if time.time() > deadline - 10:
            break
        f1, f1s = local_search_add_only(best_bits, max_iters=200)
        if f1s > best_size:
            best_size = f1s
            best_bits = f1
        f2, f2s = local_search_remove_to_add(best_bits, max_iters=150)
        if f2s > best_size:
            best_size = f2s
            best_bits = f2

    # Convert bitmask to array
    result = []
    for v in range(m):
        if best_bits & (1 << v):
            result.append(v)

    return np.array(result, dtype=np.int64)


# EVOLVE-BLOCK-END
