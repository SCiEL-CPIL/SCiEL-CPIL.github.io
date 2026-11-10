import time
import random
from itertools import product as iproduct

def _compute_block_products(seq):
    """Return set of all nonempty consecutive block products."""
    n = len(seq)
    prods = set()
    for i in range(n):
        p = 1
        for j in range(i, n):
            p *= seq[j]
            prods.add(p)
    return prods

def _valid_insertion(seq, pos, val, existing_prods):
    """Check if inserting val at pos in seq keeps all block products distinct."""
    # We need to check all blocks that include the new element
    # and verify they don't collide with existing products or each other
    n = len(seq)
    new_prods = set()
    # Blocks ending at pos-1 (before insertion) are unchanged
    # We only need to check blocks that include the new element at position pos
    for i in range(max(0, pos - 1), min(n + 1, pos + 1)):
        # Actually, we need all blocks that include the new element
        # New element is at position pos in the new sequence
        # Blocks starting at i and ending at j where i <= pos <= j
        for i in range(0, n + 1):
            if i > pos:
                break
            p = 1
            for j in range(i, n + 1):
                if j == pos:
                    p *= val
                else:
                    p *= seq[j]
                if j >= pos:
                    # This block includes the new element
                    if p in existing_prods:
                        return False
                    if p in new_prods:
                        return False
                    new_prods.add(p)
                if p > 128 ** (n + 1):  # overflow guard, unlikely to hit
                    break
    return True

def _try_backtrack(candidates, start_idx, current_seq, current_prods, best_seq, max_len, time_limit):
    """Backtracking search to find a long valid sequence."""
    start_time = time.time()
    if time.time() - start_time > time_limit:
        return
    
    if len(current_seq) > len(best_seq):
        best_seq[:] = current_seq
    
    if start_idx >= len(candidates):
        return
    
    # Try adding candidates in various orders
    for i in range(start_idx, len(candidates)):
        val = candidates[i]
        # Try inserting val at various positions
        n = len(current_seq)
        # Only try a few positions for efficiency: end, and a few near the end
        positions = [n]
        if n >= 2:
            positions = [n, n - 1, n - 2]
        
        for pos in positions:
            # Check if inserting val at pos is valid
            valid = True
            new_prods = set()
            # Compute all new block products that include val
            for i2 in range(0, n + 1):
                if i2 > pos:
                    break
                p = 1
                for j2 in range(i2, n + 1):
                    if j2 == pos:
                        p *= val
                    else:
                        p *= current_seq[j2]
                    if j2 >= pos:
                        if p in current_prods:
                            valid = False
                            break
                        if p in new_prods:
                            valid = False
                            break
                        new_prods.add(p)
                if not valid:
                    break
            
            if valid:
                # Insert val
                new_seq = current_seq[:pos] + [val] + current_seq[pos:]
                new_prods_set = current_prods | new_prods
                _try_backtrack(candidates, i + 1, new_seq, new_prods_set, best_seq, max_len, time_limit)
                # Backtrack
                if time.time() - start_time > time_limit:
                    return

def _local_search(seq, time_limit):
    """Try to improve the sequence via local moves."""
    seq = list(seq)
    best_len = len(seq)
    start_time = time.time()
    improved = True
    
    while improved and time.time() - start_time < time_limit:
        improved = False
        best_prods = _compute_block_products(seq)
        
        # Try inserting missing numbers
        missing = [x for x in range(2, 129) if x not in seq]
        random.shuffle(missing)
        
        for val in missing:
            if time.time() - start_time > time_limit:
                break
            n = len(seq)
            # Try inserting at each position
            for pos in range(n + 1):
                new_seq = seq[:pos] + [val] + seq[pos:]
                new_prods = _compute_block_products(new_seq)
                if len(new_prods) == len(new_seq) * (len(new_seq) + 1) // 2:
                    # All products distinct
                    seq = new_seq
                    best_prods = new_prods
                    improved = True
                    break
            if improved:
                break
        
        # Try swapping pairs to potentially allow more insertions
        if not improved:
            n = len(seq)
            for i in range(n):
                for j in range(i + 1, n):
                    if time.time() - start_time > time_limit:
                        break
                    new_seq = seq[:]
                    new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
                    if new_seq == seq:
                        continue
                    new_prods = _compute_block_products(new_seq)
                    if len(new_prods) == len(new_seq) * (len(new_seq) + 1) // 2:
                        # Check if this swap allows more insertions
                        # Just accept if valid
                        seq = new_seq
                        improved = True
                        break
                if improved:
                    break
    
    return seq

def construct():
    start_time = time.time()
    time_limit = 55  # Leave 5s buffer
    
    # Strategy 1: Greedy with different orderings
    best_seq = []
    
    # Original greedy (increasing order)
    values, products = [], set()
    for x in range(2, 129):
        tails, prod = [x], x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    best_seq = values
    
    # Try decreasing order
    values, products = [], set()
    for x in range(128, 1, -1):
        tails, prod = [x], x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    if len(values) > len(best_seq):
        best_seq = values
    
    # Try random orderings
    random.seed(42)
    for trial in range(20):
        if time.time() - start_time > time_limit - 20:
            break
        candidates = list(range(2, 129))
        random.shuffle(candidates)
        values, products = [], set()
        for x in candidates:
            tails, prod = [x], x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        if len(values) > len(best_seq):
            best_seq = values
    
    # Strategy 2: Local search improvement
    if time.time() - start_time < time_limit - 10:
        improved_seq = _local_search(best_seq, time_limit - (time.time() - start_time) - 5)
        if len(improved_seq) > len(best_seq):
            best_seq = improved_seq
    
    # Strategy 3: Try a different greedy - prioritize numbers with fewer prime factors
    if time.time() - start_time < time_limit - 5:
        def prime_factor_count(n):
            count = 0
            d = 2
            while d * d <= n:
                while n % d == 0:
                    count += 1
                    n //= d
                d += 1
            if n > 1:
                count += 1
            return count
        
        candidates = sorted(range(2, 129), key=lambda x: (prime_factor_count(x), x))
        values, products = [], set()
        for x in candidates:
            tails, prod = [x], x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        if len(values) > len(best_seq):
            best_seq = values
    
    return best_seq
