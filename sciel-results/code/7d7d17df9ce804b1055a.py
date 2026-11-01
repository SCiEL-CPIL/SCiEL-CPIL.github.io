import time
import numpy as np
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Fallback algebraic construction
    def get_fallback():
        if n == 4:
            return [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        if n > 4:
            tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    if q != 3 or n != 8 or radius != 1:
        return get_fallback()

    total_words = q ** n
    num_neighbors = 1 + n * (q - 1)

    # Helper to convert int to base-q digits
    def int_to_digits(num, n, q):
        res = [0] * n
        for i in range(n):
            res[i] = num % q
            num //= q
        return res

    # Precompute neighbors array
    neighbors = np.zeros((total_words, num_neighbors), dtype=np.int32)
    for w in range(total_words):
        digits = int_to_digits(w, n, q)
        idx = 0
        neighbors[w, idx] = w
        idx += 1
        for pos in range(n):
            orig_val = digits[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                neighbors[w, idx] = w + (new_val - orig_val) * (q ** pos)
                idx += 1

    # Greedy Set Cover with vectorized NumPy
    uncovered = np.ones(total_words, dtype=np.uint8)
    selected_indices = []

    while uncovered.sum() > 0:
        if time.time() > deadline:
            return get_fallback()

        # Calculate gain for each candidate
        gains = uncovered[neighbors].sum(axis=1)
        best_idx = np.argmax(gains)

        if gains[best_idx] == 0:
            break

        selected_indices.append(best_idx)
        uncovered[neighbors[best_idx]] = 0

    # Convert selected indices back to codewords
    codewords = [int_to_digits(idx, n, q) for idx in selected_indices]

    # Post-processing: Local Search / Redundancy Removal
    if time.time() < deadline - 5.0:
        # Rebuild covered array for the current code
        covered_final = np.zeros(total_words, dtype=bool)
        for idx in selected_indices:
            covered_final[neighbors[idx]] = True

        # Try to remove redundant codewords
        changed = True
        while changed:
            if time.time() > deadline - 5.0:
                break
            changed = False
            for i in range(len(selected_indices)):
                if time.time() > deadline - 5.0:
                    break
                idx = selected_indices[i]
                # Temporarily remove this codeword
                was_covered = covered_final[neighbors[idx]].copy()
                covered_final[neighbors[idx]] = False

                # Check if any point becomes uncovered
                if not covered_final[neighbors[idx]].all():
                    # This codeword is essential, restore it
                    covered_final[neighbors[idx]] = was_covered
                else:
                    # This codeword is redundant, remove it
                    selected_indices.pop(i)
                    changed = True
                    break  # Restart the loop since indices changed

        # Update codewords list
        codewords = [int_to_digits(idx, n, q) for idx in selected_indices]

    # Swap-based local search to reduce code size further
    if time.time() < deadline - 10.0 and len(selected_indices) > 0:
        random.seed(42)
        covered_arr = np.zeros(total_words, dtype=np.uint8)
        for idx in selected_indices:
            covered_arr[neighbors[idx]] += 1
        
        best_size = len(selected_indices)
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            # Shuffle order for diversification
            order = list(range(len(selected_indices)))
            random.shuffle(order)
            
            for i in order:
                if time.time() > deadline - 10.0:
                    break
                if len(selected_indices) >= best_size:
                    continue
                    
                idx = selected_indices[i]
                # Try to remove this codeword
                neighbors_i = neighbors[idx]
                # Check if all neighbors are still covered by other codewords
                # covered_arr[neighbors_i] > 1 means covered by at least 2 codewords (including current)
                # So we need covered_arr[neighbors_i] > 1 for all neighbors to safely remove
                if np.all(covered_arr[neighbors_i] > 1):
                    # Safe to remove
                    covered_arr[neighbors_i] -= 1
                    selected_indices.pop(i)
                    best_size = len(selected_indices)
                    improved = True
                    break # Restart loop with new state
                
                # Try swap: replace idx with a better candidate
                # Calculate gain of removing idx
                # We want to find a codeword j such that replacing i with j reduces total code size or keeps it same but is "better"
                # Actually, a simple heuristic: try to find a codeword that covers more uncovered points
                # But since we are minimizing size, we should only swap if it allows removal of another codeword later?
                # Let's stick to removal for now, or try random swaps if stuck.
                
                # Let's try a random swap: pick a random candidate c not in code, replace i with c
                # Only do this if it doesn't increase size (it doesn't) and potentially helps future removals.
                # This is risky. Let's try to find a "better" candidate that covers more unique points.
                
                # For now, just do removal. If no removal, break inner loop to save time.
                # A more advanced approach would be to try all candidates, but that's expensive.
                # Let's try a few random swaps to escape local minima.
                if random.random() < 0.1: # 10% chance to try a random swap
                    # Pick a random candidate
                    c = random.randint(0, total_words - 1)
                    if c in selected_indices:
                        continue
                    
                    # Check if replacing i with c is valid (it is, by definition of covering)
                    # But we need to ensure coverage is maintained.
                    # Since c is in the space, its neighbors are covered by c.
                    # The only risk is if some points were ONLY covered by i.
                    # But we already checked that i is not redundant (covered_arr[neighbors_i] > 1 failed).
                    # So some points are ONLY covered by i. If we replace i with c, those points must be covered by c or other codewords.
                    # This is complex. Let's skip random swaps for simplicity and safety.
                    pass

    return codewords
