"""Dense increasing sequences with unique block products."""
import time
import itertools
import random

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


def _get_products(seq):
    """Compute all nonempty consecutive block products as a set."""
    prods = set()
    n = len(seq)
    for i in range(n):
        p = 1
        for j in range(i, n):
            p *= seq[j]
            prods.add(p)
    return prods


def _is_valid_insertion(seq, pos, val):
    """Check if inserting val at pos keeps seq strictly increasing and all products distinct."""
    # Check strictly increasing
    if pos > 0 and val <= seq[pos - 1]:
        return False
    if pos < len(seq) and val >= seq[pos]:
        return False
    
    # Check all block products distinct
    n = len(seq) + 1
    prods = set()
    for i in range(n):
        p = 1
        for j in range(i, n):
            if j == pos:
                p *= val
            else:
                p *= seq[j - 1 if j > pos else j]
            if p in prods:
                return False
            prods.add(p)
    return True


def _try_insert(seq, pos, val):
    """Try inserting val at position pos; return new seq or None if invalid."""
    if not _is_valid_insertion(seq, pos, val):
        return None
    return seq[:pos] + [val] + seq[pos:]


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


def _try_replace(seq, idx, new_val):
    """Try replacing element at idx with new_val; return new seq or None if invalid."""
    # Check strictly increasing
    if idx > 0 and new_val <= seq[idx - 1]:
        return None
    if idx < len(seq) - 1 and new_val >= seq[idx + 1]:
        return None
    
    new_seq = seq[:]
    new_seq[idx] = new_val
    
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


def _local_search(initial_seq, time_limit=45.0, rng=None):
    """
    Local search: try to insert missing numbers, swap, or replace elements
    to increase length.
    """
    if rng is None:
        rng = random.Random()
    
    start_time = time.time()
    best_seq = list(initial_seq)
    best_len = len(best_seq)
    
    # Set of numbers not in best_seq
    all_nums = set(range(2, 129))
    in_seq = set(best_seq)
    missing = sorted(all_nums - in_seq)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # Try inserting each missing number at each valid position
        for val in missing:
            if time.time() - start_time > time_limit:
                break
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
        
        # Try swapping adjacent elements
        if not improved:
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
        
        # Try replacing elements with missing values
        if not improved:
            for idx in range(len(best_seq)):
                if time.time() - start_time > time_limit:
                    break
                for val in missing:
                    # val must be > seq[idx-1] if idx > 0
                    # val must be < seq[idx+1] if idx < len-1
                    if idx > 0 and val <= best_seq[idx - 1]:
                        continue
                    if idx < len(best_seq) - 1 and val >= best_seq[idx + 1]:
                        continue
                    new_seq = _try_replace(best_seq, idx, val)
                    if new_seq is not None:
                        # Check if this replacement allows inserting something
                        in_seq3 = set(new_seq)
                        missing3 = sorted(all_nums - in_seq3)
                        for v2 in missing3:
                            for pos in range(len(new_seq) + 1):
                                if pos > 0 and v2 <= new_seq[pos - 1]:
                                    continue
                                if pos < len(new_seq) and v2 >= new_seq[pos]:
                                    continue
                                ns3 = _try_insert(new_seq, pos, v2)
                                if ns3 is not None:
                                    best_seq = ns3
                                    best_len = len(best_seq)
                                    improved = True
                                    break
                            if improved:
                                break
                    if improved:
                        break
                if improved:
                    break
        
        # Random perturbation: randomly remove a few elements and try to reinsert
        if not improved and (time.time() - start_time) < time_limit - 5.0:
            if len(best_seq) > 5:
                perturbed = list(best_seq)
                # Remove 1-3 random elements
                remove_count = min(rng.randint(1, 3), len(perturbed) - 3)
                remove_indices = set(rng.sample(range(len(perturbed)), remove_count))
                for i in sorted(remove_indices, reverse=True):
                    del perturbed[i]
                
                # Now try to greedily add back missing elements
                in_seq_p = set(perturbed)
                missing_p = sorted(all_nums - in_seq_p)
                
                for val in missing_p:
                    if time.time() - start_time > time_limit:
                        break
                    for pos in range(len(perturbed) + 1):
                        if pos > 0 and val <= perturbed[pos - 1]:
                            continue
                        if pos < len(perturbed) and val >= perturbed[pos]:
                            continue
                        new_seq = _try_insert(perturbed, pos, val)
                        if new_seq is not None:
                            perturbed = new_seq
                            break
                
                if len(perturbed) > best_len:
                    best_seq = perturbed
                    best_len = len(best_seq)
                    improved = True
    
    return best_seq


def _multi_start_greedy(time_limit=30.0, rng=None):
    """
    Try multiple greedy strategies with different orderings to find longer sequences.
    """
    if rng is None:
        rng = random.Random()
    
    start_time = time.time()
    best = _greedy_baseline()
    best_len = len(best)
    
    # Try different starting points and orderings
    strategies = []
    
    # Strategy 1: standard ascending (already done)
    # Strategy 2: try skipping some small numbers to allow more later
    skip_sets = [set(), {2}, {3}, {4}, {2,3}, {2,4}, {3,4}, {5}, {2,5}, {3,5}, {6}, {7}, {8}, {2,6}, {3,6}, {4,6}]
    for skip_set in skip_sets:
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
    for skip_set in [set(), {2}, {3}, {4}]:
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
    
    # Strategy 4: Randomized greedy - shuffle order of consideration
    for trial in range(20):
        if time.time() - start_time > time_limit:
            break
        candidates = list(range(2, 129))
        rng.shuffle(candidates)
        values, products = [], set()
        for x in candidates:
            tails = [x]
            prod = x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        values.sort()
        if len(values) > best_len:
            best = values
            best_len = len(values)
    
    return best


def construct():
    total_deadline = time.time() + 85.0  # 85s safety margin from 90s limit
    rng = random.Random(42)
    
    # Phase 1: baseline greedy
    best = _greedy_baseline()
    
    # Phase 2: multi-start greedy variations
    remaining = total_deadline - time.time()
    if remaining > 5.0:
        candidate = _multi_start_greedy(time_limit=min(remaining * 0.3, 25.0), rng=rng)
        if len(candidate) > len(best):
            best = candidate
    
    # Phase 3: local search from best
    remaining = total_deadline - time.time()
    if remaining > 5.0:
        candidate = _local_search(best, time_limit=min(remaining * 0.5, 30.0), rng=rng)
        if len(candidate) > len(best):
            best = candidate
    
    # Phase 4: More local search iterations with different random seeds
    remaining = total_deadline - time.time()
    if remaining > 5.0:
        for seed in [123, 456, 789, 321]:
            remaining = total_deadline - time.time()
            if remaining < 3.0:
                break
            rng2 = random.Random(seed)
            candidate = _local_search(best, time_limit=min(remaining * 0.25, 15.0), rng=rng2)
            if len(candidate) > len(best):
                best = candidate
    
    # Phase 5: Final local search polish
    remaining = total_deadline - time.time()
    if remaining > 3.0:
        candidate = _local_search(best, time_limit=min(remaining * 0.8, 10.0), rng=rng)
        if len(candidate) > len(best):
            best = candidate
    
    # Final validation
    if _check_validity(best):
        return best
    
    # Fallback: return baseline (which should be valid)
    return _greedy_baseline()

# EVOLVE-BLOCK-END
