import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    def compute_all_products(values):
        """Compute all consecutive block products for a sequence."""
        products = set()
        n = len(values)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= values[j]
                products.add(prod)
        return products

    def is_valid_extension(seq, products, x):
        """
        Check if x can be appended to seq.
        Returns (is_valid, new_products_set)
        """
        new_products = set(products)
        prod = x
        # Check tails: x, seq[-1]*x, seq[-2]*seq[-1]*x, ...
        for a in reversed(seq):
            # Check if prod is already in the existing products
            if prod in products:
                return False, None
            new_products.add(prod)
            prod *= a
            # No need to check new products against each other, they are distinct
            
        return True, new_products

    def greedy_construct():
        """Fast greedy baseline."""
        values = []
        products = set()
        for x in range(2, 129):
            valid, new_prods = is_valid_extension(values, products, x)
            if valid:
                values.append(x)
                products = new_prods
        return values

    # 1. Get a baseline
    best_seq = greedy_construct()
    best_len = len(best_seq)
    
    # 2. Beam Search
    # Beam: (sequence, products_set)
    seeds = []
    seeds.append(([], set()))
    
    if best_seq:
        seeds.append((best_seq, compute_all_products(best_seq)))
        for k in range(1, min(5, len(best_seq))):
            var_seq = best_seq[:-k]
            if var_seq:
                seeds.append((var_seq, compute_all_products(var_seq)))
            else:
                seeds.append(([], set()))

    unique_seeds = []
    seen_seqs = set()
    for s, p in seeds:
        key = tuple(s)
        if key not in seen_seqs:
            seen_seqs.add(key)
            unique_seeds.append((s, p))
    
    beams = unique_seeds
    
    BEAM_WIDTH = 500
    MAX_CANDIDATES = 25
    
    while time.time() < deadline - 5.0:
        new_beams = []
        
        for seq, prods in beams:
            last_val = seq[-1] if seq else 1
            
            start_cand = last_val + 1
            end_cand = min(last_val + MAX_CANDIDATES, 128)
            
            for c in range(start_cand, end_cand + 1):
                valid, new_prods = is_valid_extension(seq, prods, c)
                if valid:
                    new_seq = seq + [c]
                    new_beams.append((new_seq, new_prods))
                    
                    if len(new_seq) > best_len:
                        best_len = len(new_seq)
                        best_seq = new_seq

        if not new_beams:
            break
            
        def beam_sort_key(b):
            seq, _ = b
            return (-len(seq), seq[-1] if seq else 0)
        
        new_beams.sort(key=beam_sort_key)
        
        beams = new_beams[:BEAM_WIDTH]

    # 3. Local Polish
    def try_insertion(seq, x):
        """Try to insert x into seq at any valid position. Returns new_seq if successful, else None."""
        n = len(seq)
        for pos in range(n + 1):
            if pos > 0 and seq[pos-1] >= x:
                continue
            if pos < n and x >= seq[pos]:
                continue
            
            new_seq = seq[:pos] + [x] + seq[pos:]
            
            products = set()
            m = len(new_seq)
            valid = True
            for i in range(m):
                prod = 1
                for j in range(i, m):
                    prod *= new_seq[j]
                    if prod in products:
                        valid = False
                        break
                    products.add(prod)
                if not valid:
                    break
            
            if valid:
                return new_seq
        
        return None

    current_best = list(best_seq)
    current_len = len(current_best)
    
    all_numbers = set(range(2, 129))
    current_set = set(current_best)
    missing = list(all_numbers - current_set)
    
    random.seed(42)
    random.shuffle(missing)
    
    for x in missing:
        if time.time() > deadline - 1.0:
            break
        if x in current_set:
            continue
            
        new_seq = try_insertion(current_best, x)
        if new_seq:
            current_best = new_seq
            current_set.add(x)
            current_len += 1
            
    return current_best
