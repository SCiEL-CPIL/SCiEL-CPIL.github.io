<<<<<<< SEARCH
def _local_search(values, products, deadline):
    """Local search using the prime-exponent-vector encoding as a fast, exact
    distinctness surrogate. Tries remove-and-reinsert and insert-new moves."""
    values = list(values)
    best = len(values)
    best_values = list(values)
    prefix = [()]
    for v in values:
        prefix.append(prefix[-1] + _EXP[v])
    n = len(values)
    m = n + 1
    diff = [None] * (m * m)

    def _distinct():
        for i in range(n):
            pe = prefix[i]
            base = i * m
            for j in range(i, n):
                s = pe
                for k in range(j + 1, n):
                    s = s + _EXP[k]
                diff[base + k] = s
        for a in range(m * m):
            d = diff[a]
            if d is not None:
                for b in range(a + 1, m * m):
                    if diff[b] == d:
                        return False
        return True

    for _ in range(4):  # Multiple passes
        improved = False

        # Remove-and-reinsert: drop one element, re-greedy the whole range.
        for i in range(n):
            if time.time() > deadline:
                break
            vals_without = values[:i] + values[i + 1:]
            prods_without = _all_products(vals_without)
            new_vals, new_prods = _greedy_pass(vals_without, prods_without, start_from=2)
            if len(new_vals) > best:
                best = len(new_vals)
                best_values = list(new_vals)
                values = list(new_vals)
                prefix = [()]
                for v in values:
                    prefix.append(prefix[-1] + _EXP[v])
                n = len(values)
                m = n + 1
                diff = [None] * (m * m)
                improved = True

        # Insert-new: try inserting a fresh value at each gap.
        used_set = set(values)
        for i in range(len(values) + 1):
            if time.time() > deadline:
                break
            lo = (values[i - 1] + 1) if i > 0 else 2
            hi = (values[i] - 1) if i < len(values) else 128
            for v in range(lo, hi + 1):
                if v in used_set:
                    continue
                test_vals = values[:i] + [v] + values[i:]
                test_prods = _all_products(test_vals)
                if len(test_prods) == len(test_vals) * (len(test_vals) + 1) // 2:
                    new_vals, new_prods = _greedy_pass(test_vals, test_prods, start_from=128)
                    if len(new_vals) > best:
                        best = len(new_vals)
                        best_values = list(new_vals)
                        values = list(new_vals)
                        prefix = [()]
                        for v in values:
                            prefix.append(prefix[-1] + _EXP[v])
                        n = len(values)
                        m = n + 1
                        diff = [None] * (m * m)
                        used_set = set(values)
                        improved = True
        if not improved:
            break

    return best_values
=======
def _is_valid_vectors(vecs):
    """Check if all nonempty consecutive block sums of exponent vectors are distinct."""
    n = len(vecs)
    if n == 0:
        return True
    seen = set()
    # Prefix sums of vectors
    # vecs[i] is a tuple. We need sums of vecs[i..j].
    # We can compute prefix sums P[0]=0, P[k] = sum(vecs[0..k-1]).
    # Sum(i..j) = P[j+1] - P[i].
    # Since vectors have non-negative components, we can just store tuples.
    # However, P[j+1] - P[i] requires subtraction.
    # Alternatively, just compute sums directly. O(N^2) tuples.
    for i in range(n):
        s = [0] * len(_PRIMES)
        for j in range(i, n):
            for k in range(len(_PRIMES)):
                s[k] += vecs[j][k]
            if tuple(s) in seen:
                return False
            seen.add(tuple(s))
    return True

def _local_search(values, deadline):
    """Local search using vector sums for fast validity checks."""
    values = list(values)
    best = len(values)
    best_values = list(values)
    vecs = [_EXP[v] for v in values]
    
    # Helper to update vecs
    def update_vecs(vals):
        return [_EXP[v] for v in vals]

    for _ in range(20):  # More passes if fast
        improved = False
        n = len(values)
        
        # 1. Try Insertions
        used_set = set(values)
        for i in range(n + 1):
            if time.time() > deadline:
                return best_values
            lo = (values[i - 1] + 1) if i > 0 else 2
            hi = (values[i] - 1) if i < n else 128
            for v in range(lo, hi + 1):
                if v in used_set:
                    continue
                test_vals = values[:i] + [v] + values[i:]
                test_vecs = update_vecs(test_vals)
                if _is_valid_vectors(test_vecs):
                    values = test_vals
                    vecs = test_vecs
                    used_set = set(values)
                    if len(values) > best:
                        best = len(values)
                        best_values = list(values)
                    improved = True
                    break # Restart loop for this i? Or just continue? 
                    # Breaking the inner loop and restarting the outer is safer for consistency
                    # But we can just continue to try more insertions? 
                    # Usually, if we insert, the set of valid insertions changes.
                    # Let's break out of the v loop and restart the i loop to be safe.
                    # Actually, just `break` the v loop. The i loop continues to i+1.
                    # This is fine.
        
        # 2. Try Replacements (Remove one, Insert one)
        # This is expensive. Maybe just try "Remove" then "Greedy Fill"?
        # Or "Swap" if order doesn't matter? Order DOES matter (strictly increasing).
        # So we can only replace $a_i$ with $x$ such that $a_{i-1} < x < a_{i+1}$.
        for i in range(n):
            if time.time() > deadline:
                return best_values
            lo = (values[i - 1] + 1) if i > 0 else 2
            hi = (values[i + 1] - 1) if i < n - 1 else 128
            # We must keep strictly increasing.
            # If we replace values[i] with x, we need x > values[i-1] and x < values[i+1].
            # And x != values[i].
            for v in range(lo, hi + 1):
                if v == values[i]:
                    continue
                test_vals = values[:i] + [v] + values[i+1:]
                # Check strict increasing
                # Since we picked v in range, it is strictly increasing.
                test_vecs = update_vecs(test_vals)
                if _is_valid_vectors(test_vecs):
                    # Accept if it allows for more insertions later?
                    # For now, accept any valid replacement.
                    # To ensure we don't get stuck in a local minimum of "validity"
                    # that prevents future insertions, we might want to prefer replacements
                    # that open up space?
                    # Simple heuristic: accept if valid.
                    values = test_vals
                    vecs = test_vecs
                    improved = True
                    break # Restart i loop?
        
        if not improved:
            break
            
    return best_values
>>>>>>> REPLACE

I also need to update the call to `_local_search` in `construct` to match the new signature (removing `products`).

