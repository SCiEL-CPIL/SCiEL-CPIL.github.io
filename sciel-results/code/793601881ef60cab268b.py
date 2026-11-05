"""Dense increasing sequences with unique block products."""
import time
import itertools

# EVOLVE-BLOCK-START

def _check_validity(seq):
    """Verify strictly increasing in [2,128] and all nonempty consecutive block products distinct."""
    if not seq:
        return False
    n = len(seq)
    if any(x < 2 or x > 128 for x in seq):
        return False
    for i in range(n - 1):
        if seq[i] >= seq[i + 1]:
            return False
    products = set()
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            if prod in products:
                return False
            products.add(prod)
    return True


def _greedy_baseline():
    """Original greedy: try each x from 2..128, append if no conflict."""
    values, products = [], set()
    for x in range(2, 129):
        tails = [x]
        prod = x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    return values


def _local_search(initial_seq, time_limit=45.0):
    """
    Local search: try to insert missing numbers at various positions,
    or swap adjacent elements, to increase length.
    """
    start_time = time.time()
    best_seq = list(initial_seq)
    best_len = len(best_seq)
    
    # Set of numbers not in best_seq
    all_nums = set(range(2, 129))
    in_seq = set(best_seq)
    missing = sorted(all_nums - in_seq)
    
    def _products_of(seq):
        """Compute all nonempty consecutive block products."""
        prods = set()
        n = len(seq)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                prods.add(p)
        return prods
    
    def _try_insert(seq, pos, val):
        """Try inserting val at position pos; return new seq or None if invalid."""
        new_seq = seq[:pos] + [val] + seq[pos:]
        # Check strictly increasing
        for i in range(len(new_seq) - 1):
            if new_seq[i] >= new_seq[i + 1]:
                return None
        # Check all block products distinct
        n = len(new_seq)
        prods = set()
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= new_seq[j]
                if p in prods:
                    return None
                prods.add(p)
        return new_seq
    
    def _try_swap(seq, i, j):
        """Try swapping positions i and j; return new seq or None if invalid."""
        new_seq = seq[:]
        new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
        # Check strictly increasing
        for k in range(len(new_seq) - 1):
            if new_seq[k] >= new_seq[k + 1]:
                return None
        # Check all block products distinct
        n = len(new_seq)
        prods = set()
        for k in range(n):
            p = 1
            for m in range(k, n):
                p *= new_seq[m]
                if p in prods:
                    return None
                prods.add(p)
        return new_seq
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Try inserting each missing number at each valid position
        for val in missing:
            if time.time() - start_time > time_limit:
                break
            # Find valid positions: between best_seq[pos-1] and best_seq[pos]
            for pos in range(len(best_seq) + 1):
                if pos > 0 and val <= best_seq[pos - 1]:
                    continue
                if pos < len(best_seq) and val >= best_seq[pos]:
                    continue
                new_seq = _try_insert(best_seq, pos, val)
                if new_seq is not None:
                    best_seq = new_seq
                    best_len = len(best_seq)
                    in_seq.add(val)
                    missing.remove(val)
                    improved = True
                    break
            if improved:
                break
        # Try swapping adjacent elements to potentially allow further insertions
        for i in range(len(best_seq) - 1):
            if time.time() - start_time > time_limit:
                break
            new_seq = _try_swap(best_seq, i, i + 1)
            if new_seq is not None:
                # Check if this swap allows inserting something
                in_seq2 = set(new_seq)
                missing2 = sorted(all_nums - in_seq2)
                for val in missing2:
                    for pos in range(len(new_seq) + 1):
                        if pos > 0 and val <= new_seq[pos - 1]:
                            continue
                        if pos < len(new_seq) and val >= new_seq[pos]:
                            continue
                        ns2 = _try_insert(new_seq, pos, val)
                        if ns2 is not None:
                            best_seq = ns2
                            best_len = len(best_seq)
                            improved = True
                            break
                    if improved:
                        break
            if improved:
                break
    
    return best_seq


def _multi_start_greedy(time_limit=30.0):
    """
    Try multiple greedy strategies with different orderings to find longer sequences.
    """
    start_time = time.time()
    best = _greedy_baseline()
    best_len = len(best)
    
    # Try different starting points and orderings
    strategies = []
    
    # Strategy 1: standard ascending (already done)
    # Strategy 2: try skipping some small numbers to allow more later
    for skip_set in [set(), {2}, {3}, {4}, {2,3}, {2,4}, {3,4}, {5}, {2,5}, {3,5}]:
        if time.time() - start_time > time_limit:
            break
        values, products = [], set()
        for x in range(2, 129):
            if x in skip_set:
                continue
            tails = [x]
            prod = x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        if len(values) > best_len:
            best = values
            best_len = len(values)
    
    # Strategy 3: greedy but try to fill gaps after initial pass
    for skip_set in [set(), {2}, {3}]:
        if time.time() - start_time > time_limit:
            break
        values, products = [], set()
        for x in range(2, 129):
            if x in skip_set:
                continue
            tails = [x]
            prod = x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        # Try to insert skipped numbers
        for x in sorted(skip_set):
            for pos in range(len(values) + 1):
                if pos > 0 and x <= values[pos - 1]:
                    continue
                if pos < len(values) and x >= values[pos]:
                    continue
                new_vals = values[:pos] + [x] + values[pos:]
                # Check products
                n = len(new_vals)
                prods = set()
                ok = True
                for i in range(n):
                    p = 1
                    for j in range(i, n):
                        p *= new_vals[j]
                        if p in prods:
                            ok = False
                            break
                        prods.add(p)
                    if not ok:
                        break
                if ok:
                    values = new_vals
                    break
        if len(values) > best_len:
            best = values
            best_len = len(values)
    
    return best


def construct():
    total_deadline = time.time() + 80.0  # 80s safety margin from 90s limit
    
    # Phase 1: baseline greedy
    best = _greedy_baseline()
    
    # Phase 2: multi-start greedy variations
    remaining = total_deadline - time.time()
    if remaining > 5.0:
        candidate = _multi_start_greedy(time_limit=min(remaining * 0.4, 30.0))
        if len(candidate) > len(best):
            best = candidate
    
    # Phase 3: local search from best
    remaining = total_deadline - time.time()
    if remaining > 5.0:
        candidate = _local_search(best, time_limit=min(remaining * 0.7, 45.0))
        if len(candidate) > len(best):
            best = candidate
    
    # Final validation
    if _check_validity(best):
        return best
    
    # Fallback: return baseline (which should be valid)
    return _greedy_baseline()

# EVOLVE-BLOCK-END
