"""Rewritten program with stronger algorithm exploiting arithmetic structure."""
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 82  # 82s deadline, 8s safety margin

    N = 256

    # Build set of perfect squares up to 2*N
    squares = set()
    s = 1
    while s * s <= 2 * N:
        squares.add(s * s)
        s += 1

    # conflict[a] = set of b in [1,N] such that a+b is a perfect square
    conflict = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for sq in squares:
            b = sq - a
            if 1 <= b <= N:
                conflict[a].add(b)

    # Numbers where 2a is a perfect square are forbidden (diagonal constraint)
    forbidden = set()
    for a in range(1, N + 1):
        if 2 * a in squares:
            forbidden.add(a)

    valid_numbers = [x for x in range(1, N + 1) if x not in forbidden]
    M = len(valid_numbers)

    # Map valid numbers to indices
    num_to_idx = {x: i for i, x in enumerate(valid_numbers)}
    idx_to_num = valid_numbers

    # Build adjacency as bitmasks
    adj = [0] * M
    for i, a in enumerate(valid_numbers):
        for b in conflict[a]:
            if b in num_to_idx:
                j = num_to_idx[b]
                if j != i:
                    adj[i] |= (1 << j)

    # Also add self-loop check: if 2*a is square, it's already forbidden
    # But let's double-check adjacency includes proper constraints

    # Greedy initialization with multiple strategies
    def greedy_min_degree():
        """Greedy: pick element with minimum degree among remaining, if no conflict."""
        selected = 0
        remaining = list(range(M))
        while remaining:
            # Find element with minimum conflicts with selected
            best_i = None
            best_conflicts = float('inf')
            for i in remaining:
                c = bin(adj[i] & selected).count('1')
                if c == 0:
                    best_i = i
                    best_conflicts = 0
                    break
                if c < best_conflicts:
                    best_conflicts = c
                    best_i = i
            if best_i is None or best_conflicts > 0:
                break
            selected |= (1 << best_i)
            remaining.remove(best_i)
        return selected

    def greedy_degree_order():
        """Greedy: order by degree ascending, add if no conflict."""
        degrees = [bin(adj[i]).count('1') for i in range(M)]
        order = sorted(range(M), key=lambda i: degrees[i])
        selected = 0
        for i in order:
            if (adj[i] & selected) == 0:
                selected |= (1 << i)
        return selected

    def greedy_random_order():
        """Greedy with random order."""
        order = list(range(M))
        random.shuffle(order)
        selected = 0
        for i in order:
            if (adj[i] & selected) == 0:
                selected |= (1 << i)
        return selected

    def greedy_weighted(order_weights=None):
        """Greedy with weighted ordering."""
        if order_weights is None:
            order_weights = [bin(adj[i]).count('1') for i in range(M)]
        order = sorted(range(M), key=lambda i: order_weights[i])
        selected = 0
        for i in order:
            if (adj[i] & selected) == 0:
                selected |= (1 << i)
        return selected

    # Get initial solutions from different strategies
    best_mask = greedy_degree_order()
    best_size = bin(best_mask).count('1')

    # Try min-degree greedy
    mask2 = greedy_min_degree()
    sz2 = bin(mask2).count('1')
    if sz2 > best_size:
        best_mask = mask2
        best_size = sz2

    # Try several random greedy initializations
    for _ in range(20):
        if time.time() > deadline - 10:
            break
        mask3 = greedy_random_order()
        sz3 = bin(mask3).count('1')
        if sz3 > best_size:
            best_mask = mask3
            best_size = sz3

    def fill_greedy(mask, size):
        """Add all non-conflicting unselected elements greedily."""
        changed = True
        while changed:
            changed = False
            for i in range(M):
                if not (mask >> i) & 1 and (adj[i] & mask) == 0:
                    mask |= (1 << i)
                    size += 1
                    changed = True
        return mask, size

    def local_search(start_mask, start_size, max_iters=50000):
        """Local search with 1-swap and 2-swap moves, plus greedy fill."""
        current_mask, current_size = fill_greedy(start_mask, start_size)
        no_improve = 0
        max_no_improve = 80

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            if no_improve > max_no_improve:
                break

            improved = False
            sel_indices = [i for i in range(M) if (current_mask >> i) & 1]
            unselected = [i for i in range(M) if not (current_mask >> i) & 1]

            if not sel_indices:
                break

            # Try 1-swap: remove one, add as many as possible
            if random.random() < 0.65:
                remove_idx = random.choice(sel_indices)
                trial_mask = current_mask & ~(1 << remove_idx)
                trial_size = current_size - 1

                # Try to add elements that don't conflict
                candidates = [i for i in unselected if (adj[i] & trial_mask) == 0]
                random.shuffle(candidates)
                for i in candidates:
                    if (adj[i] & trial_mask) == 0:
                        trial_mask |= (1 << i)
                        trial_size += 1

                if trial_size > current_size:
                    current_mask, current_size = fill_greedy(trial_mask, trial_size)
                    improved = True
                    no_improve = 0
                    continue
                elif trial_size == current_size and len(candidates) >= 1:
                    if random.random() < 0.2:
                        current_mask = trial_mask
                        no_improve += 1
                        continue

            # Try 2-swap: remove two, add as many as possible
            else:
                if len(sel_indices) < 2:
                    continue

                # Pick two to remove (prefer high-degree nodes)
                if len(sel_indices) > 20:
                    # Sample from high-degree nodes
                    sampled = random.sample(sel_indices, min(20, len(sel_indices)))
                    pair = random.sample(sampled, 2)
                else:
                    pair = random.sample(sel_indices, 2)
                trial_mask = current_mask & ~(1 << pair[0]) & ~(1 << pair[1])
                trial_size = current_size - 2

                # Try to add elements
                candidates = [i for i in unselected if (adj[i] & trial_mask) == 0]
                random.shuffle(candidates)
                for i in candidates:
                    if (adj[i] & trial_mask) == 0:
                        trial_mask |= (1 << i)
                        trial_size += 1

                if trial_size > current_size:
                    current_mask, current_size = fill_greedy(trial_mask, trial_size)
                    improved = True
                    no_improve = 0
                    continue
                elif trial_size == current_size and len(candidates) >= 2:
                    if random.random() < 0.12:
                        current_mask = trial_mask
                        no_improve += 1
                        continue

            if not improved:
                no_improve += 1

        # Final fill
        current_mask, current_size = fill_greedy(current_mask, current_size)
        return current_mask, current_size

    # Run local search from best
    best_mask, best_size = local_search(best_mask, best_size, max_iters=60000)

    # Additional restarts with local search
    for restart in range(40):
        if time.time() > deadline - 8:
            break

        # New random greedy
        mask = greedy_random_order()
        sz = bin(mask).count('1')

        # Local search from this
        mask, sz = local_search(mask, sz, max_iters=12000)

        if sz > best_size:
            best_mask = mask
            best_size = sz

    # Final polish from best with more iterations
    best_mask, best_size = local_search(best_mask, best_size, max_iters=50000)

    # Extra final fill
    best_mask, best_size = fill_greedy(best_mask, best_size)

    # Convert back to actual numbers
    result = [idx_to_num[i] for i in range(M) if (best_mask >> i) & 1]

    if not result:
        result = [x for x in range(1, N + 1, 3) if x not in forbidden]

    # Verify validity and fix if needed
    def is_valid(sol):
        for i, a in enumerate(sol):
            if 2 * a in squares:
                return False
            for b in sol[i:]:
                if a + b in squares:
                    return False
        return True

    if not is_valid(result):
        # Rebuild from scratch with simpler approach
        result = []
        for x in valid_numbers:
            ok = True
            for y in result:
                if x + y in squares:
                    ok = False
                    break
            if ok:
                result.append(x)

    return sorted(result)
