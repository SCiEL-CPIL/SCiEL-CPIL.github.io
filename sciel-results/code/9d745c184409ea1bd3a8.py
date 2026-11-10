"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _compute_all_products(seq):
    """Compute all consecutive block products for a sequence."""
    prods = set()
    n = len(seq)
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            prods.add(prod)
    return prods

def _is_valid_sequence(seq):
    """Check if sequence is strictly increasing and all block products are distinct."""
    if not seq:
        return True
    # Check strictly increasing
    for i in range(1, len(seq)):
        if seq[i] <= seq[i-1]:
            return False
    # Check products
    n = len(seq)
    expected = n * (n + 1) // 2
    prods = _compute_all_products(seq)
    return len(prods) == expected

def _local_search(seq, deadline):
    """Apply local search with 1-out/1-in moves to improve sequence length or enable further growth."""
    best = list(seq)
    improved = True
    # We allow a few passes to ensure convergence
    max_passes = 10
    passes = 0
    
    while improved and passes < max_passes and time.time() < deadline:
        improved = False
        passes += 1
        n = len(best)
        available = [x for x in range(2, 129) if x not in best]
        
        # Try 1-out/1-in moves: Remove one, add one.
        # This maintains length but might change the set to allow more insertions later?
        # Actually, for maximizing length, we primarily want to ADD.
        # But if greedy got stuck, maybe swapping allows adding?
        # Let's first try simple insertions (1-in).
        
        # 1. Try to insert any available number
        for x in available:
            if time.time() >= deadline:
                break
            # Try inserting x at all valid positions
            for pos in range(len(best) + 1):
                # Check increasing order quickly
                if pos > 0 and best[pos-1] >= x:
                    continue
                if pos < len(best) and best[pos] <= x:
                    continue
                
                new_seq = best[:pos] + [x] + best[pos:]
                if _is_valid_sequence(new_seq):
                    best = new_seq
                    available.remove(x)
                    improved = True
                    break
            if improved:
                break
        
        if time.time() >= deadline:
            break
            
        # 2. If no insertion worked, try 1-out/1-in swaps to potentially unlock insertions
        # Or just accept that we are stuck?
        # A swap doesn't increase length directly, but might allow a future insertion.
        # However, the problem asks to maximize length. 
        # If we can't insert, we can't increase length.
        # Swaps only help if they allow *subsequent* insertions.
        # Let's try a limited number of swaps to see if they enable insertions.
        
        found_swap = False
        for i in range(len(best)):
            if time.time() >= deadline:
                break
            for x in available:
                if time.time() >= deadline:
                    break
                # Remove best[i], insert x at best[i]'s position (or nearby)
                base = best[:i] + best[i+1:]
                # Try inserting x into base at various positions
                for pos in range(len(base) + 1):
                    if pos > 0 and base[pos-1] >= x:
                        continue
                    if pos < len(base) and base[pos] <= x:
                        continue
                    
                    new_seq = base[:pos] + [x] + base[pos:]
                    if _is_valid_sequence(new_seq):
                        # Check if this swap allows an immediate insertion?
                        # If yes, accept. If no, maybe accept anyway to explore?
                        # To be safe and fast, only accept if it allows immediate insertion or is "better" somehow?
                        # Since objective is length, and length is same, we should only accept if it leads to growth.
                        # Let's check if any other available number can now be inserted.
                        # This is expensive. 
                        # Alternative: Just accept the swap if it's valid, hoping it helps later?
                        # Risk: Infinite loop of swaps.
                        # Better: Only do swaps if we have time, and limit them.
                        
                        # Let's just accept the swap to change the state.
                        best = new_seq
                        available.remove(x)
                        available.append(best[i]) # The removed element becomes available
                        improved = True
                        found_swap = True
                        break
                if found_swap:
                    break
            if found_swap:
                break
                
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    # Greedy construction
    values, products = [], set()
    for x in range(2, 129):
        tails, prod = [x], x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    
    # Try to improve with local search if time permits
    if time.time() < deadline - 2.0:
        improved = _local_search(values, deadline)
        if len(improved) > len(values):
            values = improved
    
    return values
# EVOLVE-BLOCK-END
