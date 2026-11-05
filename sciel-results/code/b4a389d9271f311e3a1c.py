"""Combined program: beam search with targeted local repair."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s kill
    
    # --- Utility Functions ---
    
    def get_new_products(seq, c):
        """Get the list of new products when appending c to seq."""
        prods = []
        prod = c
        prods.append(prod)
        for a in reversed(seq):
            prod *= a
            prods.append(prod)
        return prods
    
    def has_collision(prods, product_set):
        """Check if any product in prods is in product_set."""
        for p in prods:
            if p in product_set:
                return True
        return False
    
    def compute_product_set(seq):
        """Compute all consecutive block products for a sequence."""
        products = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                products.add(prod)
        return products
    
    def is_valid(seq):
        """Check if sequence is strictly increasing and all block products are distinct."""
        if len(seq) == 0:
            return True
        for i in range(len(seq) - 1):
            if seq[i] >= seq[i+1]:
                return False
        if any(x < 2 or x > 128 for x in seq):
            return False
        n = len(seq)
        expected_count = n * (n + 1) // 2
        products = compute_product_set(seq)
        return len(products) == expected_count
    
    # --- Greedy Initial Solution ---
    
    def run_greedy():
        """Fast greedy heuristic to generate a strong initial sequence."""
        values = []
        products = set()
        for x in range(2, 129):
            if time.time() > deadline - 5:
                break
            tails = [x]
            prod = x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            has_collision = False
            for p in tails:
                if p in products:
                    has_collision = True
                    break
            if not has_collision:
                values.append(x)
                products.update(tails)
        return values, products
    
    # --- Beam Search ---
    
    def run_beam_search(initial_seq, initial_prods, beam_width=200, max_candidates=30):
        """Beam search to extend and improve the sequence."""
        best_seq = list(initial_seq)
        best_len = len(initial_seq)
        
        beams = [(list(initial_seq), set(initial_prods))]
        
        while time.time() < deadline - 5:
            new_beams = []
            
            for seq, prods in beams:
                if not seq:
                    last_val = 1
                else:
                    last_val = seq[-1]
                
                end = min(last_val + max_candidates + 1, 129)
                for c in range(last_val + 1, end):
                    new_prods_list = get_new_products(seq, c)
                    
                    if not has_collision(new_prods_list, prods):
                        new_seq = seq + [c]
                        new_prod_set = prods.union(new_prods_list)
                        new_beams.append((new_seq, new_prod_set))
                        
                        if len(new_seq) > best_len:
                            best_len = len(new_seq)
                            best_seq = new_seq
            
            if not new_beams:
                break
            
            # Sort beams: prefer longer, then smaller last value (more room)
            def beam_sort_key(b):
                seq, prods = b
                return (-len(seq), seq[-1] if seq else 0, len(prods))
            
            new_beams.sort(key=beam_sort_key)
            beams = new_beams[:beam_width]
            
            # Reduce beam width if we have a long sequence
            if best_len >= 35:
                beam_width = max(50, beam_width // 2)
            
            # Early termination if we've found a very long sequence
            if best_len >= 50:
                break
        
        return best_seq
    
    # --- Targeted Local Repair ---
    
    def local_repair(seq, prods, max_iter=100):
        """
        Try to extend the sequence by inserting values at various positions
        or replacing values to avoid collisions.
        """
        best_seq = list(seq)
        best_len = len(seq)
        
        for _ in range(max_iter):
            if time.time() > deadline - 3:
                break
            
            # Try to insert a value at each position
            improved = False
            for pos in range(len(best_seq) + 1):
                if time.time() > deadline - 3:
                    break
                
                # Determine range of candidates to try at this position
                if pos == 0:
                    min_val = 2
                else:
                    min_val = best_seq[pos-1] + 1
                
                if pos == len(best_seq):
                    max_val = 128
                else:
                    max_val = best_seq[pos] - 1
                
                if min_val > max_val:
                    continue
                
                # Try a few candidates
                candidates = list(range(min_val, min(max_val + 1, min_val + 20)))
                random.shuffle(candidates)
                
                for c in candidates:
                    # Build new sequence
                    new_seq = best_seq[:pos] + [c] + best_seq[pos:]
                    
                    # Check strictly increasing
                    valid = True
                    for i in range(len(new_seq) - 1):
                        if new_seq[i] >= new_seq[i+1]:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Check all products are distinct
                    n = len(new_seq)
                    expected_count = n * (n + 1) // 2
                    all_prods = set()
                    for i in range(n):
                        prod = 1
                        for j in range(i, n):
                            prod *= new_seq[j]
                            all_prods.add(prod)
                    
                    if len(all_prods) == expected_count:
                        best_seq = new_seq
                        best_len = len(new_seq)
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                break
        
        return best_seq
    
    # --- Main Logic ---
    
    # Get greedy initial solution
    greedy_seq, greedy_prods = run_greedy()
    
    # Run beam search to improve
    beam_seq = run_beam_search(greedy_seq, greedy_prods)
    
    # Run local repair on the beam search result
    final_seq = local_repair(beam_seq, compute_product_set(beam_seq))
    
    # Verify the final sequence
    if not is_valid(final_seq):
        # Fall back to greedy if repair broke something
        final_seq = greedy_seq
    
    return final_seq
# EVOLVE-BLOCK-END
