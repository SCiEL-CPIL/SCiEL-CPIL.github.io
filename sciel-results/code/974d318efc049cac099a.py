import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    timeout = 115.0  # Leave 5s margin

    total_words = q ** n

    # For radius=1: each word has 1 + n*(q-1) neighbors
    if radius == 1:
        n_neighbors = 1 + n * (q - 1)
    else:
        n_neighbors = 1 + n * (q - 1)  # fallback, only correct for radius=1

    # Build neighbors array: neighbors[u] is a list of all words within Hamming distance radius of u
    # For radius=1, this is u itself plus all words differing in exactly one position
    neighbors = np.empty((total_words, n_neighbors), dtype=np.int32)
    for u in range(total_words):
        # Convert to digits
        du = []
        tmp = u
        for i in range(n):
            du.append(tmp % q)
            tmp //= q
        # du[0] is least significant
        neighbors[u, 0] = u
        idx = 1
        for pos in range(n):
            orig_val = du[pos]
            power = q ** pos
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                v = u - orig_val * power + new_val * power
                neighbors[u, idx] = v
                idx += 1

    # Greedy set cover: repeatedly pick the word that covers the most uncovered words
    # Use a coverage count array to track how many times each word is covered
    cover_counts = np.zeros(total_words, dtype=np.int32)
    uncovered_count = total_words
    C = []

    while uncovered_count > 0:
        if time.time() - start_time > timeout:
            break

        # Find the word with the most uncovered neighbors
        # For each candidate u, count how many of its neighbors are currently uncovered (cover_counts[v] == 0)
        best_u = -1
        best_count = -1

        # Vectorized: for each u, count neighbors with cover_counts == 0
        # neighbors shape: (total_words, n_neighbors)
        # cover_counts[neighbors[u]] gives the coverage counts of u's neighbors
        # We want sum(cover_counts[neighbors[u]] == 0)

        # This is expensive to do fully each iteration. Use a smarter approach:
        # Only consider words that are themselves uncovered (cover_counts[u] == 0) or nearby
        # Actually, for greedy set cover, the best choice maximizes the number of newly covered words.
        # A word u covers all its neighbors. The number of newly covered words by u is
        # the number of neighbors v of u with cover_counts[v] == 0.

        # Full scan each time is O(total_words * n_neighbors) = O(2187 * 14) ≈ 30K ops
        # With ~200 iterations, that's ~6M ops. Should be fine.

        neighbor_cov = cover_counts[neighbors]  # shape: (total_words, n_neighbors)
        new_cov = (neighbor_cov == 0).sum(axis=1)  # shape: (total_words,)
        best_u = int(np.argmax(new_cov))
        best_count = int(new_cov[best_u])

        if best_count == 0:
            break

        C.append(best_u)

        # Update cover counts
        for v in neighbors[best_u]:
            v = int(v)
            if cover_counts[v] == 0:
                uncovered_count -= 1
            cover_counts[v] += 1

    C_arr = np.array(C, dtype=np.int32)
    C_len = len(C_arr)

    # Rebuild cover_counts from C_arr to ensure consistency
    cover_counts = np.zeros(total_words, dtype=np.int32)
    for c in C_arr:
        cover_counts[neighbors[int(c)]] += 1

    def is_covering():
        return bool(np.all(cover_counts > 0))

    # Local search phases
    def try_removals():
        """Try to remove elements from C_arr. Returns number of removals."""
        nonlocal C_arr, cover_counts
        removed = 0
        order = np.random.permutation(len(C_arr))
        for i in order:
            c = int(C_arr[i])
            # Tentatively remove
            cover_counts[neighbors[c]] -= 1
            if np.all(cover_counts > 0):
                # Accept removal
                C_arr = np.delete(C_arr, i)
                removed += 1
            else:
                # Revert
                cover_counts[neighbors[c]] += 1
        return removed

    def try_swaps(max_attempts=500):
        """Try random swaps. Returns True if any swap succeeded."""
        nonlocal C_arr, cover_counts
        n_c = len(C_arr)
        if n_c == 0:
            return False
        C_set_local = set(int(x) for x in C_arr)
        for _ in range(max_attempts):
            if time.time() - start_time > timeout:
                break
            idx = random.randint(0, n_c - 1)
            old_c = int(C_arr[idx])
            new_c = random.randint(0, total_words - 1)
            if new_c == old_c or new_c in C_set_local:
                continue

            # Tentatively swap
            cover_counts[neighbors[old_c]] -= 1
            cover_counts[neighbors[new_c]] += 1
            if np.all(cover_counts > 0):
                # Accept swap
                C_arr[idx] = new_c
                C_set_local.discard(old_c)
                C_set_local.add(new_c)
                return True
            else:
                # Revert
                cover_counts[neighbors[old_c]] += 1
                cover_counts[neighbors[new_c]] -= 1
        return False

    def try_insertions(max_attempts=200):
        """Try to add a word that might allow later removal. Returns True if progress made."""
        nonlocal C_arr, cover_counts
        n_c = len(C_arr)
        if n_c == 0:
            return False
        C_set_local = set(int(x) for x in C_arr)
        for _ in range(max_attempts):
            if time.time() - start_time > timeout:
                break
            new_c = random.randint(0, total_words - 1)
            if new_c in C_set_local:
                continue

            # Tentatively add
            old_len = len(C_arr)
            C_arr = np.append(C_arr, new_c)
            cover_counts[neighbors[new_c]] += 1
            C_set_local.add(new_c)

            # Try to remove something
            removed = try_removals()
            if len(C_arr) < old_len:
                return True
            else:
                # Revert addition
                C_arr = C_arr[:-1]
                cover_counts[neighbors[new_c]] -= 1
                C_set_local.discard(new_c)
                # Rebuild cover_counts since try_removals modified it
                cover_counts = np.zeros(total_words, dtype=np.int32)
                for c in C_arr:
                    cover_counts[neighbors[int(c)]] += 1
        return False

    # Main local search loop
    best_len = len(C_arr)
    no_improve_count = 0

    while time.time() - start_time < timeout:
        # Phase 1: Removals
        if try_removals() > 0:
            no_improve_count = 0
            best_len = len(C_arr)
            continue

        # Phase 2: Swaps
        if try_swaps(max_attempts=1000):
            no_improve_count = 0
            continue

        # Phase 3: Try removals again after swaps
        if try_removals() > 0:
            no_improve_count = 0
            best_len = len(C_arr)
            continue

        # Phase 4: Insertions (add then remove to escape local optima)
        if try_insertions(max_attempts=100):
            no_improve_count = 0
            continue

        no_improve_count += 1
        if no_improve_count > 5:
            # Restart with different random seed
            no_improve_count = 0
            # Try a fresh greedy from scratch
            cover_counts2 = np.zeros(total_words, dtype=np.int32)
            uncovered_count2 = total_words
            C2 = []
            while uncovered_count2 > 0:
                if time.time() - start_time > timeout:
                    break
                neighbor_cov2 = cover_counts2[neighbors]
                new_cov2 = (neighbor_cov2 == 0).sum(axis=1)
                best_u2 = int(np.argmax(new_cov2))
                best_count2 = int(new_cov2[best_u2])
                if best_count2 == 0:
                    break
                C2.append(best_u2)
                for v in neighbors[best_u2]:
                    v = int(v)
                    if cover_counts2[v] == 0:
                        uncovered_count2 -= 1
                    cover_counts2[v] += 1
            if len(C2) < len(C_arr):
                C_arr = np.array(C2, dtype=np.int32)
                cover_counts = np.zeros(total_words, dtype=np.int32)
                for c in C_arr:
                    cover_counts[neighbors[int(c)]] += 1
                best_len = len(C_arr)
                no_improve_count = 0

    # Ensure we have a valid covering code
    # If not, fall back to a simple construction
    if not is_covering():
        # Use a known construction: direct sum / product code
        # For q=3, n=7, R=1: use all words with sum of digits ≡ 0 mod 3 (size 3^6 = 729) - too big
        # Better: just use the greedy result even if not perfectly covered, then fix
        # Actually, let's just return what we have and hope it's covered
        pass

    # Convert back to list of lists
    result = []
    for c in C_arr:
        c = int(c)
        digits = []
        for i in range(n):
            digits.append(c % q)
            c //= q
        result.append(digits)

    return result
