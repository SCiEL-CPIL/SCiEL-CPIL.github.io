"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    N = 256
    # Precompute square-sum conflicts: conflict[i][j] = True if i+j is a perfect square
    # We use 0-indexed internally, values are 1..256
    # Perfect squares up to 512: 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361, 400, 441, 484
    squares = set()
    for s in range(2, 24):  # 2^2=4 to 23^2=529
        sq = s * s
        if sq <= 2 * N:
            squares.add(sq)

    # Build adjacency: adj[i] is a set of j where i+j is a perfect square (including i=j)
    # Using 0-indexed: value = i+1
    adj = [set() for _ in range(N)]
    for i in range(N):
        vi = i + 1
        for j in range(i, N):  # only j >= i since a<=b
            vj = j + 1
            if vi + vj in squares:
                adj[i].add(j)
                if j != i:
                    adj[j].add(i)

    # Bitset representation for fast conflict checking
    # adj_bits[i] is a Python int bitmask of all j such that i+j is a perfect square
    adj_bits = [0] * N
    for i in range(N):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        adj_bits[i] = mask

    def count_conflicts(selected_mask):
        """Count number of conflicting pairs (a<=b) in selected_mask."""
        count = 0
        for i in range(N):
            if selected_mask & (1 << i):
                # Count conflicts with j >= i
                conflicts = adj_bits[i] & selected_mask & (0xFFFFFFFF << i)  # j >= i
                count += bin(conflicts).count('1')
        return count

    def is_valid(selected_mask):
        """Check if selected_mask has no conflicts (including diagonal)."""
        for i in range(N):
            if selected_mask & (1 << i):
                if adj_bits[i] & selected_mask:
                    # There's a conflict involving i
                    # Need to check if it's with j >= i (since we only count each pair once)
                    # Actually, if i is in the set and adj_bits[i] & selected_mask != 0,
                    # that means there exists j in the set where i+j is a square.
                    # But we need to be careful: if j < i, it would have been counted when processing j.
                    # For validity, we just need NO conflicts at all.
                    return False
        return True

    # Greedy construction: add vertices in order of lowest degree (fewest conflicts)
    def greedy_construct():
        remaining = set(range(N))
        selected = set()
        # Order by degree (number of conflicts)
        order = sorted(range(N), key=lambda i: len(adj[i]))
        for i in order:
            if i not in remaining:
                continue
            # Check if i conflicts with any already selected
            if adj_bits[i] & _mask_from_set(selected):
                continue
            selected.add(i)
            remaining.discard(i)
        return selected

    def _mask_from_set(s):
        m = 0
        for i in s:
            m |= (1 << i)
        return m

    # Better greedy: iterative with restarts
    def greedy_with_order(order):
        selected = set()
        selected_mask = 0
        for i in order:
            if adj_bits[i] & selected_mask:
                continue
            selected.add(i)
            selected_mask |= (1 << i)
        return selected, selected_mask

    # Local search: swap vertices to improve
    def local_search(selected_mask, selected_set, time_limit):
        best_mask = selected_mask
        best_set = set(selected_set)
        best_size = len(best_set)
        no_improve_count = 0
        max_no_improve = 500

        while time.time() < time_limit:
            improved = False
            # Try adding a vertex
            for i in range(N):
                if best_mask & (1 << i):
                    continue
                if adj_bits[i] & best_mask:
                    continue
                # Can add i
                new_mask = best_mask | (1 << i)
                best_mask = new_mask
                best_set.add(i)
                best_size += 1
                improved = True
                no_improve_count = 0
                break

            if improved:
                continue

            # Try removing a vertex and adding two others
            for i in range(N):
                if not (best_mask & (1 << i)):
                    continue
                # Remove i
                temp_mask = best_mask & ~(1 << i)
                # Try to add two vertices that were conflicting with i
                candidates = [j for j in range(N) if not (best_mask & (1 << j)) and (adj_bits[i] & (1 << j))]
                random.shuffle(candidates)
                found = False
                for a in candidates:
                    if adj_bits[a] & temp_mask:
                        continue
                    for b in candidates:
                        if b == a:
                            continue
                        if adj_bits[b] & temp_mask:
                            continue
                        if adj_bits[a] & (1 << b):
                            continue
                        new_mask = temp_mask | (1 << a) | (1 << b)
                        if len(bin(new_mask).count('1')) > best_size:
                            best_mask = new_mask
                            best_set = set(j for j in range(N) if new_mask & (1 << j))
                            best_size += 1
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    no_improve_count = 0
                    break

            if not improved:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break

        return best_mask, best_set, best_size

    # Strategy 1: Greedy with multiple random orders
    best_overall_mask = 0
    best_overall_size = 0

    # First, try degree-based greedy
    order_by_degree = sorted(range(N), key=lambda i: len(adj[i]))
    sel, mask = greedy_with_order(order_by_degree)
    if len(sel) > best_overall_size:
        best_overall_size = len(sel)
        best_overall_mask = mask

    # Try multiple random restarts
    random.seed(42)
    num_restarts = 200
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break

        # Random order biased toward low-degree
        order = list(range(N))
        random.shuffle(order)
        sel, mask = greedy_with_order(order)

        # Local search
        mask, sel, size = local_search(mask, sel, deadline - 3.0)

        if size > best_overall_size:
            best_overall_size = size
            best_overall_mask = mask

    # Final local search on best solution
    best_set = set(i for i in range(N) if best_overall_mask & (1 << i))
    best_mask, best_set, best_size = local_search(best_overall_mask, best_set, deadline - 1.0)

    # Convert to 1-indexed values
    result = sorted(i + 1 for i in best_set)

    # Verify
    for idx, a in enumerate(result):
        for b in result[idx:]:
            s = a + b
            r = int(math.isqrt(s))
            if r * r == s:
                # Invalid - should not happen
                pass

    return result if result else [1]
# EVOLVE-BLOCK-END
