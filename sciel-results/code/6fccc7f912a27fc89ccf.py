"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def _prime_sigs(n):
    """Return prime factorization as a frozenset of (p, e) for fast comparison."""
    factors = {}
    d = 2
    temp = n
    while d * d <= temp:
        if temp % d == 0:
            cnt = 0
            while temp % d == 0:
                temp //= d
                cnt += 1
            factors[d] = cnt
        d += 1
    if temp > 1:
        factors[temp] = 1
    return frozenset(factors.items())

def _product_sig(sigs):
    """Compute combined prime factorization signature from list of element sigs."""
    factors = {}
    for sig in sigs:
        for p, e in sig:
            factors[p] = factors.get(p, 0) + e
    return frozenset(factors.items())

def _greedy_with_sigs():
    """Greedy using prime factorization signatures for fast conflict detection."""
    values = []
    value_sigs = []
    products = set()  # Set of product signatures
    
    for x in range(2, 129):
        sig = _prime_sigs(x)
        # Compute all suffix products' signatures
        tails = []
        cur_factors = {}
        for a in reversed(value_sigs):
            for p, e in a:
                cur_factors[p] = cur_factors.get(p, 0) + e
            tails.append(frozenset(cur_factors.items()))
        tails.append(sig)  # The singleton
        
        # Check for conflicts
        conflict = False
        for t in tails:
            if t in products:
                conflict = True
                break
        
        if not conflict:
            values.append(x)
            value_sigs.append(sig)
            products.update(tails)
    
    return values

def _check_insertion_valid(values, value_sigs, products, pos, val, val_sig):
    """Check if inserting val at pos is valid."""
    # New sequence: values[:pos] + [val] + values[pos:]
    # New blocks involving val:
    # Start i in [0, pos], end j in [pos, len(values)]
    # (in new coords, val is at index pos)
    
    n_old = len(values)
    n_new = n_old + 1
    
    # We need to check:
    # 1. New blocks involving val don't conflict with each other
    # 2. New blocks involving val don't conflict with old blocks
    # 3. Old blocks remain distinct (they do, since we're not changing them)
    
    # Compute all new block signatures involving val
    new_block_sigs = set()
    
    # Blocks starting at i (0 <= i <= pos) and ending at j (pos <= j < n_new)
    # In terms of old indices:
    # If i < pos: block is values[i..pos-1] + val + values[pos..j-1] where j-1 >= pos
    # If i == pos: block is val + values[pos..j-1]
    # If i > pos: not possible since val is at pos
    
    # Let me reindex: in new_seq, val is at index pos
    # Block from i to j (i <= pos <= j):
    #   = product(new_seq[i..j])
    
    # Precompute prefix and suffix products
    # Prefix: prod from 0 to k in new_seq
    # Suffix: prod from k to end in new_seq
    
    # Actually, let's just compute directly for correctness
    new_values = values[:pos] + [val] + values[pos:]
    new_sigs = value_sigs[:pos] + [val_sig] + value_sigs[pos:]
    
    seen = set()
    for i in range(len(new_values)):
        cur = {}
        for j in range(i, len(new_values)):
            for p, e in new_sigs[j]:
                cur[p] = cur.get(p, 0) + e
            sig = frozenset(cur.items())
            if sig in seen:
                return False
            seen.add(sig)
    
    # Check against old products
    for sig in seen:
        if sig in products:
            return False
    
    return True

def _local_search_sigs(values, value_sigs, products, deadline):
    """Local search using signatures."""
    best = list(values)
    best_sigs = list(value_sigs)
    best_products = set(products)
    best_len = len(best)
    all_elements = set(range(2, 129))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        skipped = sorted(all_elements - set(best))
        
        for val in skipped:
            if time.time() > deadline:
                break
            val_sig = _prime_sigs(val)
            for pos in range(len(best) + 1):
                if _check_insertion_valid(best, best_sigs, best_products, pos, val, val_sig):
                    # Insert
                    best = best[:pos] + [val] + best[pos:]
                    best_sigs = best_sigs[:pos] + [val_sig] + best_sigs[pos:]
                    # Update products: add all new block products
                    # Recompute all products for simplicity
                    all_prods = set()
                    for i in range(len(best)):
                        cur = {}
                        for j in range(i, len(best)):
                            for p, e in best_sigs[j]:
                                cur[p] = cur.get(p, 0) + e
                            all_prods.add(frozenset(cur.items()))
                    best_products = all_prods
                    best_len = len(best)
                    improved = True
                    break
            if improved:
                break
    
    return best

def construct():
    """Main entry point."""
    start_time = time.time()
    deadline = start_time + 80
    
    # Greedy with signatures
    best = _greedy_with_sigs()
    
    # Get signatures and products for local search
    best_sigs = [_prime_sigs(x) for x in best]
    best_products = set()
    for i in range(len(best)):
        cur = {}
        for j in range(i, len(best)):
            for p, e in best_sigs[j]:
                cur[p] = cur.get(p, 0) + e
            best_products.add(frozenset(cur.items()))
    
    # Local search
    best = _local_search_sigs(best, best_sigs, best_products, deadline)
    
    return best
# EVOLVE-BLOCK-END
