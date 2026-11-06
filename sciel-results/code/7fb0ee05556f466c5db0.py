import math
import time
from fractions import Fraction

def construct():
    deadline = time.time() + 130.0
    N = 511
    C = 512

    # Precompute weights
    w = [Fraction(0)] * (N + 1)
    for a in range(1, N + 1):
        w[a] = Fraction(1, C - a)

    # Precompute conflict masks as bitsets
    # conflict_mask[a] has bit b set if gcd(a,b) > 1
    conflict_mask = [0] * (N + 1)
    for a in range(1, N + 1):
        mask = 0
        for b in range(1, N + 1):
            if a != b and math.gcd(a, b) > 1:
                mask |= (1 << b)
        conflict_mask[a] = mask

    # Greedy initial solution: pick from high a to low a (high weight first)
    S_mask = 0
    for a in range(N, 0, -1):
        if (S_mask & conflict_mask[a]) == 0:
            S_mask |= (1 << a)

    best_S_mask = S_mask
    best_score = sum(w[a] for a in range(1, N + 1) if best_S_mask & (1 << a))

    def mask_to_list(mask):
        result = []
        for a in range(1, N + 1):
            if mask & (1 << a):
                result.append(a)
        return result

    def get_score(mask):
        return sum(w[a] for a in range(1, N + 1) if mask & (1 << a))

    def can_add(mask, a):
        return (mask & conflict_mask[a]) == 0

    # Local search: 1-add, 1-swap, and 2-remove-1-add moves
    def local_search(S_mask):
        nonlocal best_S_mask, best_score, deadline
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_score = get_score(S_mask)

            # Try adding each non-selected element
            for a in range(N, 0, -1):
                if time.time() > deadline:
                    break
                if S_mask & (1 << a):
                    continue
                if can_add(S_mask, a):
                    new_mask = S_mask | (1 << a)
                    new_score = current_score + w[a]
                    if new_score > best_score:
                        S_mask = new_mask
                        current_score = new_score
                        improved = True
                        if new_score > best_score:
                            best_score = new_score
                            best_S_mask = S_mask

            # Try swapping: remove a, add b
            selected = [a for a in range(1, N + 1) if S_mask & (1 << a)]
            for a in selected:
                if time.time() > deadline:
                    break
                mask_without_a = S_mask & ~(1 << a)
                for b in range(N, 0, -1):
                    if S_mask & (1 << b):
                        continue
                    if not can_add(mask_without_a, b):
                        continue
                    delta = w[b] - w[a]
                    if delta <= 0:
                        continue
                    new_mask = mask_without_a | (1 << b)
                    new_score = current_score + delta
                    if new_score > best_score:
                        S_mask = new_mask
                        current_score = new_score
                        improved = True
                        best_score = new_score
                        best_S_mask = S_mask

            # Try 2-remove-1-add: remove two selected elements, add one non-selected
            # This allows replacing two low-weight elements with one high-weight composite
            if len(selected) >= 2:
                for i in range(len(selected)):
                    if time.time() > deadline:
                        break
                    a1 = selected[i]
                    for j in range(i + 1, len(selected)):
                        if time.time() > deadline:
                            break
                        a2 = selected[j]
                        mask_without_both = S_mask & ~(1 << a1) & ~(1 << a2)
                        # Try adding each non-selected element
                        for b in range(N, 0, -1):
                            if S_mask & (1 << b):
                                continue
                            if not can_add(mask_without_both, b):
                                continue
                            delta = w[b] - w[a1] - w[a2]
                            if delta <= 0:
                                continue
                            new_mask = mask_without_both | (1 << b)
                            new_score = current_score + delta
                            if new_score > best_score:
                                S_mask = new_mask
                                current_score = new_score
                                improved = True
                                best_score = new_score
                                best_S_mask = S_mask

        return S_mask

    # Run local search from the greedy start
    S_mask = local_search(S_mask)

    # Try multiple random restarts
    import random
    random.seed(42)
    for restart in range(200):
        if time.time() > deadline:
            break
        # Random initial solution: pick elements in random order
        S_mask = 0
        order = list(range(1, N + 1))
        random.shuffle(order)
        for a in order:
            if can_add(S_mask, a):
                S_mask |= (1 << a)
        S_mask = local_search(S_mask)

    # Try a structured approach: prioritize large numbers
    # Build set greedily by weight, but also try to include composites that have high weight
    for restart in range(50):
        if time.time() > deadline:
            break
        S_mask = 0
        # Greedy by weight with random tie-breaking
        order = sorted(range(1, N + 1), key=lambda x: (-w[x], random.random()))
        for a in order:
            if can_add(S_mask, a):
                S_mask |= (1 << a)
        S_mask = local_search(S_mask)

    # Final polish: run local search again on the best solution found
    if time.time() < deadline - 5:
        best_S_mask = local_search(best_S_mask)

    return mask_to_list(best_S_mask)
