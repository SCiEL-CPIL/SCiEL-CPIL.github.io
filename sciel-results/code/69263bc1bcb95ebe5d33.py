import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    def compute_products(seq):
        """Compute all consecutive block products for a sequence."""
        products = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                if prod in products:
                    return None  # collision
                products.add(prod)
        return products
    
    def is_valid(seq):
        """Check if a strictly increasing sequence has all distinct consecutive block products."""
        if len(seq) == 0:
            return True
        if not all(seq[i] < seq[i+1] for i in range(len(seq)-1)):
            return False
        if any(x < 2 or x > 128 for x in seq):
            return False
        return compute_products(seq) is not None
    
    def greedy_build(start=2, end=129, skip_set=None):
        """Greedily build a valid sequence, optionally skipping certain values."""
        seq = []
        products = set()
        for x in range(start, end):
            if skip_set and x in skip_set:
                continue
            # Try adding x
            new_products = set(products)
            prod = x
            valid = True
            # Compute new products ending at x
            for i in range(len(seq)-1, -1, -1):
                prod *= seq[i]
                if prod in new_products:
                    valid = False
                    break
                new_products.add(prod)
            if valid:
                seq.append(x)
                products = new_products
        return seq
    
    def greedy_with_insertion(base_seq, candidates):
        """Try to improve a base sequence by inserting candidates at optimal positions."""
        seq = list(base_seq)
        products = compute_products(seq)
        if products is None:
            return seq
        
        for x in sorted(candidates, key=lambda v: (v not in seq, v)):
            if x in seq:
                continue
            # Try all insertion positions
            best_pos = None
            best_new_products = None
            for pos in range(len(seq) + 1):
                # Check if x fits at pos (strictly increasing)
                if pos > 0 and seq[pos-1] >= x:
                    continue
                if pos < len(seq) and seq[pos] <= x:
                    continue
                
                new_seq = seq[:pos] + [x] + seq[pos:]
                new_prods = set(products)
                valid = True
                
                # Check new products involving x
                # Products starting at x
                prod = x
                for j in range(pos, len(new_seq)):
                    if j > pos:
                        prod *= new_seq[j]
                    if prod in new_prods:
                        valid = False
                        break
                    new_prods.add(prod)
                
                # Products ending at x
                if valid:
                    prod = x
                    for i in range(pos-1, -1, -1):
                        prod *= new_seq[i]
                        if prod in new_prods:
                            valid = False
                            break
                        new_prods.add(prod)
                
                # Products that include x but don't start or end at x
                if valid:
                    for i in range(pos):
                        prod = new_seq[i]
                        for j in range(i+1, pos):
                            prod *= new_seq[j]
                            prod *= x
                            for k in range(pos+1, len(new_seq)):
                                prod *= new_seq[k]
                                if prod in new_prods:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                
                if valid:
                    best_pos = pos
                    best_new_products = new_prods
                    break
            
            if best_pos is not None:
                seq = seq[:best_pos] + [x] + seq[best_pos:]
                products = best_new_products
        
        return seq
    
    def beam_search(width=150, max_time=15):
        """Beam search to find a good sequence."""
        end_time = time.time() + max_time
        
        # State: (seq_tuple, products_frozen, last_element)
        beam = [((), frozenset(), 0)]
        best_seq = []
        best_len = 0
        
        for x in range(2, 129):
            if time.time() > end_time:
                break
            
            next_beam = []
            for seq, prods, last in beam:
                # Option 1: Skip x
                next_beam.append((seq, prods, last))
                
                # Option 2: Include x
                if last >= x:
                    continue
                    
                # Compute new products
                new_prods = set(prods)
                prod = x
                valid = True
                for i in range(len(seq)-1, -1, -1):
                    prod *= seq[i]
                    if prod in new_prods:
                        valid = False
                        break
                    new_prods.add(prod)
                
                if valid:
                    new_seq = seq + (x,)
                    next_beam.append((new_seq, frozenset(new_prods), x))
                    if len(new_seq) > best_len:
                        best_len = len(new_seq)
                        best_seq = list(new_seq)
            
            # Deduplicate and prune
            unique = {}
            for seq, prods, last in next_beam:
                if seq not in unique:
                    unique[seq] = (prods, last)
            
            # Sort by length desc, then by sum asc (prefer smaller values)
            sorted_seqs = sorted(unique.keys(), key=lambda s: (-len(s), sum(s) if s else 0))
            beam = [(s, unique[s][0], unique[s][1]) for s in sorted_seqs[:width]]
        
        return best_seq
    
    def local_search_insertion(seq, time_limit=8):
        """Improve sequence by trying to insert missing elements."""
        end_time = time.time() + time_limit
        best_seq = list(seq)
        best_len = len(seq)
        
        in_seq = set(best_seq)
        candidates = [x for x in range(2, 129) if x not in in_seq]
        random.shuffle(candidates)
        
        improved = True
        while improved and time.time() < end_time:
            improved = False
            for x in candidates:
                if x in in_seq:
                    continue
                new_seq = greedy_with_insertion(best_seq, [x])
                if len(new_seq) > best_len:
                    best_seq = new_seq
                    best_len = len(best_seq)
                    in_seq = set(best_seq)
                    improved = True
                    # Re-shuffle remaining candidates
                    candidates = [c for c in candidates if c not in in_seq]
                    random.shuffle(candidates)
                    break
        
        return best_seq
    
    # Strategy: Multi-start greedy + beam search + local search
    
    best_seq = []
    best_len = 0
    
    # 1. Basic greedy
    greedy_seq = greedy_build()
    best_seq = list(greedy_seq)
    best_len = len(greedy_seq)
    
    # 2. Multi-start greedy with different skip sets
    if time.time() < deadline - 30:
        for trial in range(5):
            # Randomly skip a few values to explore different paths
            skip_set = set()
            if trial > 0:
                # Skip 1-3 random values from the first half
                skip_candidates = list(range(2, 65))
                random.shuffle(skip_candidates)
                skip_set = set(skip_candidates[:min(trial, 3)])
            
            trial_seq = greedy_build(skip_set=skip_set)
            if len(trial_seq) > best_len:
                best_seq = list(trial_seq)
                best_len = len(trial_seq)
    
    # 3. Beam search with remaining time
    remaining_time = deadline - time.time() - 10  # Leave 10s for local search
    if remaining_time > 5:
        beam_seq = beam_search(width=150, max_time=min(remaining_time, 20))
        if len(beam_seq) > best_len:
            best_seq = beam_seq
            best_len = len(beam_seq)
    
    # 4. Local search: try inserting missing elements
    remaining_time = deadline - time.time() - 3
    if remaining_time > 2:
        ls_seq = local_search_insertion(best_seq, time_limit=min(remaining_time, 10))
        if len(ls_seq) > best_len:
            best_seq = ls_seq
            best_len = len(ls_seq)
    
    # 5. Final validation
    if not is_valid(best_seq):
        # Fallback to greedy
        best_seq = greedy_build()
    
    return best_seq

if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    t1 = time.time()
    print(f"Length: {len(result)}")
    print(f"Time: {t1-t0:.2f}s")
    print(result[:20], "...")
