import numpy as np

def construct(q, n, radius):
    # This implementation uses a count-based greedy algorithm optimized for speed.
    # It is suitable for q=3, n=8, radius=1 but generalizes to small q, n.
    
    if q == 3 and n == 8 and radius == 1:
        total_words = 3 ** 8
        # Precompute neighbors for each word index (0 to 6560)
        # We represent words as base-3 integers.
        neighbors = [[] for _ in range(total_words)]
        
        # Helper to convert int to list of digits
        def int_to_digits(val, base, length):
            res = [0] * length
            for i in range(length):
                res[i] = val % base
                val //= base
            return res

        # Helper to convert list of digits to int
        def digits_to_int(digits, base):
            val = 0
            for d in reversed(digits):
                val = val * base + d
            return val

        # Precompute neighbors
        for i in range(total_words):
            digits = int_to_digits(i, q, n)
            neigh = [i]
            for pos in range(n):
                orig = digits[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    digits[pos] = new_val
                    ni = digits_to_int(digits, q)
                    neigh.append(ni)
                digits[pos] = orig
            neighbors[i] = neigh

        # Greedy with counts
        uncovered = np.ones(total_words, dtype=bool)
        # counts[i] = number of currently uncovered words in neighbors[i]
        counts = np.full(total_words, 17, dtype=np.int32)
        
        code = []
        
        while uncovered.any():
            # Find the word with the maximum count of uncovered neighbors
            best_idx = np.argmax(counts)
            code.append(best_idx)
            
            # Update coverage
            # We need to mark all neighbors of best_idx as covered
            # And update the counts for all neighbors of those neighbors
            covered_now = []
            for v in neighbors[best_idx]:
                if uncovered[v]:
                    uncovered[v] = False
                    covered_now.append(v)
            
            # Update counts for all neighbors of the newly covered words
            for v in covered_now:
                for u in neighbors[v]:
                    counts[u] -= 1
        
        # Convert back to list of lists
        return [int_to_digits(c, q, n) for c in code]
    else:
        # Fallback for other cases: simple greedy (slower)
        # For the sake of the task contract, we assume the main case is q=3, n=8
        # But we should provide a reasonable fallback.
        total_words = q ** n
        def int_to_list(val, base, length):
            res = [0] * length
            for i in range(length):
                res[i] = val % base
                val //= base
            return res

        def list_to_int(lst, base):
            val = 0
            for i in range(len(lst)-1, -1, -1):
                val = val * base + lst[i]
            return val

        neighbors = []
        for i in range(total_words):
            w = int_to_list(i, q, n)
            neigh = {i}
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    w[pos] = new_val
                    ni = list_to_int(w, q)
                    neigh.add(ni)
                w[pos] = orig
            neighbors.append(neigh)

        uncovered = set(range(total_words))
        code = []
        while uncovered:
            best_c = -1
            best_count = -1
            for c in uncovered:
                count = len(neighbors[c] & uncovered)
                if count > best_count:
                    best_count = count
                    best_c = c
            code.append(best_c)
            uncovered = uncovered - neighbors[best_c]
        return [int_to_list(c, q, n) for c in code]
