import time
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
    
    def greedy_build():
        """Greedily build a valid sequence."""
        seq = []
        products = set()
        for x in range(2, 129):
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
    
    def local_search(seq, products, time_limit=10):
        """Improve sequence via local search (add, remove, swap operations)."""
        end_time = time.time() + time_limit
        best_seq = list(seq)
        best_len = len(seq)
        best_products = set(products)
        
        # First, try adding elements
        improved = True
        while improved and time.time() < end_time:
            improved = False
            for x in range(2, 129):
                if x in best_seq:
                    continue
                # Check if x fits (strictly increasing)
                # Find insertion point
                pos = 0
                for i, v in enumerate(best_seq):
                    if v < x:
                        pos = i + 1
                    else:
                        break
                # Try inserting at pos
                new_seq = best_seq[:pos] + [x] + best_seq[pos:]
                if is_valid(new_seq):
                    best_seq = new_seq
                    best_products = compute_products(best_seq)
                    best_len = len(best_seq)
                    improved = True
                    break
        
        # Then try swaps
        improved = True
        while improved and time.time() < end_time:
            improved = False
            for i in range(len(best_seq)):
                for x in range(2, 129):
                    if x in best_seq:
                        continue
                    if x == best_seq[i]:
                        continue
                    new_seq = list(best_seq)
                    new_seq[i] = x
                    # Check strictly increasing
                    if not all(new_seq[j] < new_seq[j+1] for j in range(len(new_seq)-1)):
                        continue
                    if is_valid(new_seq):
                        best_seq = new_seq
                        best_products = compute_products(best_seq)
                        best_len = len(best_seq)
                        improved = True
                        break
                if improved:
                    break
        
        return best_seq, best_products, best_len
    
    def beam_search(width=200, max_time=20):
        """Beam search to find a good sequence."""
        end_time = time.time() + max_time
        
        # State: (seq_tuple, products_set, last_element)
        # Start with empty sequence
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
    
    # Strategy: Combine greedy, beam search, and local search
    
    # 1. Get greedy solution
    greedy_seq = greedy_build()
    greedy_products = compute_products(greedy_seq)
    best_seq = list(greedy_seq)
    best_len = len(greedy_seq)
    
    # 2. Beam search
    beam_seq = beam_search(width=200, max_time=20)
    if len(beam_seq) > best_len:
        best_seq = beam_seq
        best_len = len(beam_seq)
        best_products = compute_products(best_seq)
    else:
        best_products = greedy_products
    
    # 3. Local search on the best found
    if time.time() < deadline - 5:
        ls_seq, ls_products, ls_len = local_search(best_seq, best_products, time_limit=15)
        if ls_len > best_len:
            best_seq = ls_seq
            best_len = ls_len
            best_products = ls_products
    
    # 4. Final validation
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
