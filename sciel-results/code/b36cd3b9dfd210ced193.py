"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _all_products(values):
    """Return set of all nonempty consecutive block products."""
    n = len(values)
    prods = set()
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= values[j]
            prods.add(prod)
    return prods

def _can_add(values, products, x):
    """Check if x can be appended to values (strictly increasing, no product collision)."""
    if values and x <= values[-1]:
        return False
    tails = [x]
    prod = x
    for a in reversed(values):
        prod *= a
        tails.append(prod)
    if products.intersection(tails):
        return False
    return True

def _greedy_pass(values, products, start_from=2):
    """Continue greedy from current state."""
    values = list(values)
    products = set(products)
    for x in range(start_from, 129):
        if _can_add(values, products, x):
            tails = [x]
            prod = x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            values.append(x)
            products.update(tails)
    return values, products

def _local_search(values, products, deadline):
    """Apply local search: remove-and-reinsert, insert-new, swap, replace operations."""
    values = list(values)
    products = set(products)
    best = len(values)
    best_values = list(values)
    
    # We run multiple passes until no improvement or deadline
    max_passes = 50
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        improved = False
        
        # 1. Remove-and-reinsert: drop one element, re-greedy the whole range.
        # This is often the most effective move for escaping local optima.
        for i in range(len(values)):
            if time.time() > deadline:
                break
            vals_without = values[:i] + values[i+1:]
            prods_without = _all_products(vals_without)
            new_vals, new_prods = _greedy_pass(vals_without, prods_without, start_from=2)
            if len(new_vals) > best:
                best = len(new_vals)
                best_values = list(new_vals)
                values = list(new_vals)
                products = set(new_prods)
                improved = True
                # Don't break immediately, try other removals? 
                # Usually one improvement resets the loop. Let's continue to find more.
        
        # 2. Swap: try swapping two adjacent elements (if still strictly increasing)
        # This is a low-cost move.
        for i in range(len(values) - 1):
            if time.time() > deadline:
                break
            # Check if swapping maintains strictly increasing order
            # values[i] < values[i+1]
            # After swap: values[i+1] < values[i] is required, which is false.
            # So simple adjacent swap breaks the "strictly increasing" constraint 
            # UNLESS we are not constrained to the original order?
            # Wait, the task says "Return strictly increasing integers".
            # So the sequence MUST be strictly increasing.
            # Therefore, we cannot simply swap two elements if it violates the order.
            # The "Swap" in the inspiration likely refers to swapping the *positions* in the 
            # search process or perhaps it was a different constraint.
            # However, if we swap values[i] and values[i+1], the new sequence is 
            # ... values[i+1], values[i] ... which is NOT strictly increasing.
            # So "Swap" is invalid for this specific task constraint unless it means 
            # something else (e.g., swapping the *decision* to include/exclude in a different context).
            # Let's ignore simple adjacent swaps as they violate the strictly increasing constraint.
            # Instead, "Replace" is the valid move: change values[i] to a different valid number.
            
        # 3. Replace: try replacing one element with a different value
        used_set = set(values)
        for i in range(len(values)):
            if time.time() > deadline:
                break
            lo = (values[i-1] + 1) if i > 0 else 2
            hi = (values[i+1] - 1) if i < len(values) - 1 else 128
            # Try a few random candidates in the range to avoid O(N^2) overhead if range is large
            # But range is at most 127, so it's small.
            candidates = []
            if lo <= hi:
                # Randomize the order to escape biases
                candidates = list(range(lo, hi + 1))
                random.shuffle(candidates)
            
            for v in candidates:
                if v == values[i]:
                    continue
                test_vals = values[:i] + [v] + values[i+1:]
                # Quick check: is it strictly increasing?
                # Yes, by construction of lo/hi.
                test_prods = _all_products(test_vals)
                if len(test_prods) == len(test_vals) * (len(test_vals) + 1) // 2:
                    # Valid sequence. Now try to extend it greedily?
                    # Or just accept it if it's longer? It has the same length.
                    # The goal is to MAXIMIZE LENGTH.
                    # Replacing an element keeps the length the same.
                    # However, a different element might allow *future* extensions 
                    # that the current element blocks.
                    # So we should re-greedy from this new state.
                    new_vals, new_prods = _greedy_pass(test_vals, test_prods, start_from=128)
                    # Note: start_from=128 means we don't re-scan 2..128, we just check if 
                    # we can add anything? No, _greedy_pass scans from start_from to 128.
                    # If we replace values[i], we might have freed up a spot for a smaller number?
                    # No, the sequence is fixed in the first i-1 and last n-i-1.
                    # We replaced values[i]. The new value v is between neighbors.
                    # The greedy pass from 128 downwards? No, _greedy_pass goes UP.
                    # If we changed an element, we might be able to add elements that were 
                    # previously blocked? 
                    # Actually, _greedy_pass(values, products, start_from=2) is the safest way 
                    # to see if we can extend the sequence further.
                    # But it's expensive. 
                    # Let's just check if the new sequence itself is valid (it is) and 
                    # if it has the same length. It does.
                    # The benefit is only if it allows *more* elements to be added later.
                    # So we should call _greedy_pass(test_vals, test_prods, start_from=2) 
                    # to see if the "landscape" has changed to allow more elements.
                    # This is expensive (O(N^2) per candidate). 
                    # Let's optimize: only do this if we are stuck.
                    
                    # For now, let's just update if we find a valid replacement that 
                    # leads to a longer sequence after re-greedy.
                    new_vals, new_prods = _greedy_pass(test_vals, test_prods, start_from=2)
                    if len(new_vals) > best:
                        best = len(new_vals)
                        best_values = list(new_vals)
                        values = list(new_vals)
                        products = set(new_prods)
                        improved = True
                        break # Restart loop with new state
        
        if not improved:
            break
            
    return best_values

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    # 1. Run Beam Search to get a good initial lower bound
    def run_beam_search():
        # State: seq_tuple -> prods_set
        # We start with the empty state
        beam = {(): set()}
        
        # Beam width: needs to be large enough to explore diverse subsets
        W = 500
        
        best_seq = []
        best_prods = set()
        best_len = 0
        
        for x in range(2, 129):
            if time.time() > deadline:
                break
                
            next_beam = {}
            
            for seq, prods in beam.items():
                # Option 1: Skip x
                if seq not in next_beam:
                    next_beam[seq] = prods
                
                # Option 2: Include x
                # Check if x can be appended
                if not seq or x > seq[-1]:
                    tails = [x]
                    prod = x
                    for a in reversed(seq):
                        prod *= a
                        tails.append(prod)
                    
                    # Check collision
                    valid = True
                    for t in tails:
                        if t in prods:
                            valid = False
                            break
                    
                    if valid:
                        new_seq = seq + (x,)
                        new_prods = set(prods)
                        new_prods.update(tails)
                        next_beam[new_seq] = new_prods
            
            # Prune beam
            # Sort by length desc, then by sum of elements asc
            sorted_seqs = sorted(next_beam.keys(), key=lambda s: (-len(s), sum(s) if s else 0))
            
            # Keep top W
            beam = {s: next_beam[s] for s in sorted_seqs[:W]}
            
            # Update best
            if sorted_seqs:
                top_seq = sorted_seqs[0]
                if len(top_seq) > best_len:
                    best_len = len(top_seq)
                    best_seq = list(top_seq)
                    best_prods = next_beam[top_seq]

        return best_seq, best_prods

    best_seq, best_prods = run_beam_search()
    
    # 2. Run Local Search to refine the best sequence
    if time.time() < deadline:
        refined_seq = _local_search(best_seq, best_prods, deadline)
        if len(refined_seq) > len(best_seq):
            best_seq = refined_seq
            # Recalculate products for the final result just in case, though _local_search returns valid seq
            # best_prods = _all_products(best_seq)

    return best_seq
# EVOLVE-BLOCK-END
