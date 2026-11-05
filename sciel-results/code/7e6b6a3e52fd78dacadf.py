import math
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s

    # First 40 primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
              31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
              73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
              127, 131, 137, 139, 149, 151, 157, 163, 167, 173]
    num_primes = len(primes)

    # Precompute cumulative product of smallest remaining primes for pruning
    # cum_prod[i] = product of primes[i:] (the smallest primes from i onward)
    # Actually we want: if we need k more primes from index i, the minimum product
    # is primes[i] * primes[i+1] * ... * primes[i+k-1]
    # Precompute: min_prod[i][k] = product of k smallest primes starting at index i
    # For efficiency, precompute prefix products of primes
    # min_prod_from(i, k) = primes[i]*primes[i+1]*...*primes[i+k-1]
    # We can compute this on the fly with a helper, or precompute a 2D table.
    # Given num_primes=40 and max k ~ 15, a 2D table is fine.

    # Precompute: prod_from[i][k] = product of primes[i..i+k-1], or inf if not enough primes
    INF = 10**30
    max_k_needed = 16
    prod_from = [[INF] * (max_k_needed + 1) for _ in range(num_primes + 1)]
    for i in range(num_primes, -1, -1):
        prod_from[i][0] = 1
        for k in range(1, max_k_needed + 1):
            if i + k <= num_primes:
                prod_from[i][k] = prod_from[i + 1][k - 1] * primes[i]
            else:
                prod_from[i][k] = INF

    # Score computation for a square-free n given as a bitmask of selected primes
    # Divisors are all subsets of the selected primes.
    # Two divisors are coprime iff their prime-index bitmasks are disjoint.
    # We need to enumerate all 2^k divisors, sort them, and count adjacent coprime pairs.
    # For k up to ~15, 2^15 = 32768 divisors, which is manageable.

    def calc_score_squarefree(selected_indices):
        """
        selected_indices: list of indices into primes[]
        Returns the tau_perp score for n = product of primes[i] for i in selected_indices.
        """
        k = len(selected_indices)
        if k == 0:
            return 0
        # Build all divisors as (value, bitmask)
        # bitmask has bits set for which selected primes are included
        divs = [(1, 0)]
        for idx in selected_indices:
            p = primes[idx]
            new_divs = []
            for (val, mask) in divs:
                new_divs.append((val * p, mask | (1 << idx)))
            divs.extend(new_divs)
        # Sort by value
        divs.sort(key=lambda x: x[0])
        score = 0
        for i in range(len(divs) - 1):
            if divs[i][1] & divs[i + 1][1] == 0:
                score += 1
        return score

    # Global best
    best_n = 2
    best_score = 0
    best_selected = []  # list of prime indices

    def update_best(selected_indices, n):
        nonlocal best_n, best_score, best_selected
        if n < 2 or n > 10**18:
            return
        score = calc_score_squarefree(selected_indices)
        if score > best_score:
            best_score = score
            best_n = n
            best_selected = selected_indices[:]

    # DFS to find best square-free subset of size k
    def dfs_subset(idx, current_prod, selected, k):
        nonlocal best_score
        if time.time() > deadline:
            return

        # If we've selected k primes, evaluate
        if len(selected) == k:
            if current_prod <= 10**18:
                update_best(selected, current_prod)
            return

        # Pruning: not enough primes left
        remaining = num_primes - idx
        needed = k - len(selected)
        if remaining < needed:
            return

        # Pruning: even the smallest possible product of `needed` more primes
        # from index idx exceeds the remaining budget
        if current_prod * prod_from[idx][needed] > 10**18:
            return

        # Option 1: include primes[idx]
        new_prod = current_prod * primes[idx]
        if new_prod <= 10**18:
            selected.append(idx)
            dfs_subset(idx + 1, new_prod, selected, k)
            selected.pop()

        # Option 2: skip primes[idx]
        # Only if we can still fill `needed` primes from idx+1 onward
        if num_primes - (idx + 1) >= needed:
            # Prune: can we still reach the needed count with product constraint?
            if current_prod * prod_from[idx + 1][needed] <= 10**18:
                dfs_subset(idx + 1, current_prod, selected, k)

    # Search for subsets of size 15, 14, 13 (descending, as larger sets likely score higher)
    for k in [15, 14, 13]:
        if time.time() > deadline - 20:
            break
        selected = []
        dfs_subset(0, 1, selected, k)

    # Local refinement: try swapping one selected prime for a larger unselected one
    # to potentially allow a different combination, or try adding/removing primes
    def local_refine(selected_indices, max_iterations=200):
        nonlocal best_n, best_score, best_selected
        current = selected_indices[:]
        if not current:
            return current
        current_n = 1
        for idx in current:
            current_n *= primes[idx]
        current_score = calc_score_squarefree(current)

        improved = True
        iterations = 0
        while improved and iterations < max_iterations and time.time() < deadline:
            improved = False
            iterations += 1

            # Try swapping each selected prime with a larger unselected prime
            current_set = set(current)
            for i, idx in enumerate(current):
                # Try replacing primes[idx] with primes[j] for j > idx not in current
                for j in range(idx + 1, num_primes):
                    if j in current_set:
                        continue
                    new_prod = current_n // primes[idx] * primes[j]
                    if new_prod > 10**18:
                        break
                    new_selected = current[:]
                    new_selected[i] = j
                    new_score = calc_score_squarefree(new_selected)
                    if new_score > current_score:
                        current = new_selected
                        current_n = new_prod
                        current_score = new_score
                        current_set = set(current)
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_n = current_n
                            best_selected = current[:]
                        break  # restart scan after a swap

            # Try adding a prime (if product allows)
            for j in range(num_primes):
                if j in current_set:
                    continue
                new_prod = current_n * primes[j]
                if new_prod > 10**18:
                    continue
                new_selected = current + [j]
                new_score = calc_score_squarefree(new_selected)
                if new_score > current_score:
                    current = new_selected
                    current_n = new_prod
                    current_score = new_score
                    current_set = set(current)
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_n = current_n
                        best_selected = current[:]
                    break  # restart scan after an add

            # Try removing a prime (if it improves score)
            for i in range(len(current)):
                if len(current) <= 1:
                    break
                new_selected = current[:]
                new_selected.pop(i)
                new_prod = current_n // primes[new_selected[i] if i < len(new_selected) else 0] if False else 1
                # Recompute product
                new_prod = 1
                for idx in new_selected:
                    new_prod *= primes[idx]
                new_score = calc_score_squarefree(new_selected)
                if new_score > current_score:
                    current = new_selected
                    current_n = new_prod
                    current_score = new_score
                    current_set = set(current)
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_n = current_n
                        best_selected = current[:]
                    break  # restart scan after a remove

        return current

    # Refine the best found so far
    if best_score > 0 and best_selected:
        refined = local_refine(best_selected, max_iterations=300)

    # Also try a few heuristic candidates and refine them
    # Candidate: primorial of first k primes
    for k in range(10, 16):
        if time.time() > deadline - 5:
            break
        prod = 1
        valid = True
        for i in range(k):
            prod *= primes[i]
            if prod > 10**18:
                valid = False
                break
        if valid:
            sel = list(range(k))
            update_best(sel, prod)
            if time.time() < deadline - 10:
                local_refine(sel, max_iterations=50)

    # Final refinement
    if best_score > 0 and best_selected and time.time() < deadline - 5:
        refined = local_refine(best_selected, max_iterations=500)

    return best_n

if __name__ == "__main__":
    n = construct()
    print(f"Constructed: {n}")
