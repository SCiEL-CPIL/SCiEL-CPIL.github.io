"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _all_block_products(seq):
    """Return set of all nonempty consecutive block products."""
    prods = set()
    n = len(seq)
    for i in range(n):
        p = 1
        for j in range(i, n):
            p *= seq[j]
            prods.add(p)
    return prods

def _is_valid(seq):
    """Check strictly increasing in [2,128] and all block products distinct."""
    if not seq:
        return False
    if len(seq) != len(set(seq)):
        return False
    if seq[0] < 2 or seq[-1] > 128:
        return False
    for i in range(len(seq) - 1):
        if seq[i] >= seq[i + 1]:
            return False
    prods = _all_block_products(seq)
    return len(prods) == sum(len(seq) - i for i in range(len(seq)))

def _greedy_construct():
    values, products = [], set()
    for x in range(2, 129):
        tails, prod = [x], x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    return values

def _is_valid_fast(seq):
    """Fast validity check with early termination."""
    n = len(seq)
    if n == 0:
        return False
    if seq[0] < 2 or seq[-1] > 128:
        return False
    for i in range(n - 1):
        if seq[i] >= seq[i + 1]:
            return False
    
    prods = set()
    expected_count = n * (n + 1) // 2
    current_count = 0
    for i in range(n):
        p = 1
        for j in range(i, n):
            p *= seq[j]
            if p in prods:
                return False
            prods.add(p)
            current_count += 1
            if current_count > expected_count:
                return False # Safety check, shouldn't happen
    return current_count == expected_count

def _try_improve(seq, deadline):
    """Local search with 1-out/1-in moves only."""
    best = list(seq)
    best_len = len(best)
    
    while time.time() < deadline:
        improved = False
        n = len(best)
        
        # 1-out/1-in: remove one element, try inserting a different one
        for i in range(n):
            if time.time() >= deadline:
                break
            candidate = list(best)
            candidate.pop(i)
            # Try inserting each number from 2..128 that's not in candidate
            for x in range(2, 129):
                if x in candidate:
                    continue
                # Insert at best position to maintain increasing order
                pos = i
                while pos < len(candidate) and candidate[pos] < x:
                    pos += 1
                new_seq = candidate[:pos] + [x] + candidate[pos:]
                
                # Quick checks before full validation
                if new_seq[0] < 2 or new_seq[-1] > 128:
                    continue
                # Check strictly increasing
                valid_order = True
                for k in range(len(new_seq) - 1):
                    if new_seq[k] >= new_seq[k+1]:
                        valid_order = False
                        break
                if not valid_order:
                    continue
                
                if _is_valid_fast(new_seq):
                    if len(new_seq) > best_len:
                        best = new_seq
                        best_len = len(best)
                        improved = True
                    # If same length, we don't swap to avoid cycles, 
                    # but we could track best sum or other metric if needed.
                    # Here we just want max length.
        
        if not improved:
            break
    
    return best

def construct():
    deadline = time.time() + 80  # Safety margin before 90s kill
    
    # Start with greedy
    best = _greedy_construct()
    best_len = len(best)
    
    # Try to improve with local search
    if time.time() < deadline:
        improved = _try_improve(best, deadline)
        if len(improved) > best_len:
            best = improved
            best_len = len(best)
    
    return best
# EVOLVE-BLOCK-END
