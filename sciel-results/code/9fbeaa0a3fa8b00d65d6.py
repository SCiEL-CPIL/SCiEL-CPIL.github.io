import time
import random
from math import log

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin for 60s limit
    
    def compute_tails(seq, x):
        """Compute new tail products if x is appended to seq."""
        new_tails = [x]
        prod = x
        for a in reversed(seq):
            prod *= a
            new_tails.append(prod)
        return new_tails

    def build_product_set(seq):
        """Build the full set of consecutive block products for seq."""
        products_set = set()
        for i in range(len(seq)):
            prod = seq[i]
            products_set.add(prod)
            for j in range(i + 1, len(seq)):
                prod *= seq[j]
                products_set.add(prod)
        return products_set

    def is_valid_addition(seq, x, products_set):
        """Check if x can be appended to seq without collision."""
        new_tails = compute_tails(seq, x)
        if products_set.intersection(new_tails):
            return False, None
        return True, new_tails

    def greedy_construct(start_val=2, end_val=128):
        """Standard greedy: pick each valid x in increasing order."""
        seq = []
        products_set = set()
        for x in range(start_val, end_val + 1):
            if time.time() > deadline:
                break
            valid, new_tails = is_valid_addition(seq, x, products_set)
            if valid:
                seq.append(x)
                products_set.update(new_tails)
        return seq, products_set

    def local_search_perturb(seq, products_set, perturb_count=3):
        """Perturb the sequence by removing some elements and re-greedy-ing."""
        if not seq or time.time() > deadline:
            return seq, products_set
            
        new_seq = list(seq)
        for _ in range(perturb_count):
            if len(new_seq) <= 3:
                break
            idx = random.randint(1, len(new_seq) - 1)
            new_seq.pop(idx)
            
        new_set = build_product_set(new_seq)
        
        last_x = new_seq[-1] if new_seq else 1
        for x in range(last_x + 1, 129):
            if time.time() > deadline:
                break
            valid, new_tails = is_valid_addition(new_seq, x, new_set)
            if valid:
                new_seq.append(x)
                new_set.update(new_tails)
                
        return new_seq, new_set

    def insertion_polish(seq):
        """Try to insert missing numbers at every possible position."""
        if not seq or time.time() > deadline:
            return seq
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            in_seq = set(seq)
            for x in range(2, 129):
                if x in in_seq:
                    continue
                if time.time() > deadline:
                    break
                # Find insertion position
                pos = None
                for i, a in enumerate(seq):
                    if a > x:
                        pos = i
                        break
                if pos is None:
                    pos = len(seq)
                
                # Check if inserting x at pos is valid
                # Only need to check blocks that include x
                # A block including x is [i..j] where i <= pos <= j
                # Its product = (prod of seq[i..pos-1]) * x * (prod of seq[pos..j])
                # We need to check these don't collide with existing products or each other
                new_seq = seq[:pos] + [x] + seq[pos:]
                
                # Build product set for new_seq efficiently
                # Existing products that don't include x are unchanged
                # New products are those that include x
                # Products including x: for each i in [0..pos] and j in [pos..len(new_seq)-1]
                #   product = new_seq[i] * new_seq[i+1] * ... * new_seq[j]
                #   where new_seq[pos] = x
                
                # Precompute prefix products of seq up to pos-1 and suffix products from pos
                # Actually, let's just do the full check but optimized
                valid = True
                all_prods = set()
                
                # First add all products NOT involving position pos in new_seq
                # These are blocks entirely before pos or entirely after pos
                # Blocks before pos: in seq[0..pos-1]
                for i in range(pos):
                    prod = new_seq[i]
                    if prod in all_prods:
                        valid = False
                        break
                    all_prods.add(prod)
                    for j in range(i + 1, pos):
                        prod *= new_seq[j]
                        all_prods.add(prod)
                if not valid:
                    continue
                    
                # Blocks after pos: in new_seq[pos+1..end] = seq[pos..end]
                for i in range(pos + 1, len(new_seq)):
                    prod = new_seq[i]
                    if prod in all_prods:
                        valid = False
                        break
                    all_prods.add(prod)
                    for j in range(i + 1, len(new_seq)):
                        prod *= new_seq[j]
                        all_prods.add(prod)
                if not valid:
                    continue
                    
                # Now blocks that include x at position pos
                # Block [i..j] where i <= pos <= j
                # Product = (prod of new_seq[i..pos-1]) * x * (prod of new_seq[pos+1..j])
                # Precompute left products: L[i] = new_seq[i]*...*new_seq[pos-1] for i in 0..pos
                # L[pos] = 1 (empty left)
                # Precompute right products: R[j] = new_seq[pos+1]*...*new_seq[j] for j in pos+1..len-1
                # R[pos] = 1 (empty right)
                
                left_prods = [1] * (pos + 1)
                prod = 1
                for i in range(pos - 1, -1, -1):
                    prod *= new_seq[i]
                    left_prods[i] = prod
                left_prods[pos] = 1
                
                right_prods = [1] * (len(new_seq) - pos)
                prod = 1
                for j in range(pos + 1, len(new_seq)):
                    prod *= new_seq[j]
                    right_prods[j - pos] = prod
                right_prods[0] = 1
                
                for i in range(0, pos + 1):
                    lp = left_prods[i]
                    for j in range(pos, len(new_seq)):
                        rp = right_prods[j - pos]
                        block_prod = lp * x * rp
                        if block_prod in all_prods:
                            valid = False
                            break
                        all_prods.add(block_prod)
                    if not valid:
                        break
                
                if valid:
                    seq = new_seq
                    improved = True
                    break
        
        return seq

    best_seq = []
    best_products = set()
    
    # 1. Standard Greedy
    seq, products_set = greedy_construct(2, 128)
    if len(seq) > len(best_seq):
        best_seq = seq
        best_products = products_set

    # 2. Try different start values
    for sv in range(3, 15):
        if time.time() > deadline:
            break
        seq, products_set = greedy_construct(sv, 128)
        if len(seq) > len(best_seq):
            best_seq = seq
            best_products = products_set

    # 3. Local search with random restarts and perturbations
    random.seed(42)
    trials = 0
    while time.time() < deadline:
        trials += 1
        
        if trials % 3 == 0 and best_seq:
            seq = list(best_seq)
            products_set = build_product_set(seq)
        else:
            start_val = random.randint(2, 12)
            base_skip = random.uniform(0.1, 0.5)
            seq = []
            products_set = set()
            for x in range(start_val, 129):
                if time.time() > deadline:
                    break
                if x < 15:
                    p_skip = base_skip * (15 - x) / 15.0
                    if random.random() < p_skip:
                        continue
                valid, new_tails = is_valid_addition(seq, x, products_set)
                if valid:
                    seq.append(x)
                    products_set.update(new_tails)
        
        seq2, products_set2 = local_search_perturb(seq, products_set, perturb_count=random.randint(1, 4))
        
        last_x = seq2[-1] if seq2 else 1
        for x in range(last_x + 1, 129):
            if time.time() > deadline:
                break
            valid, new_tails = is_valid_addition(seq2, x, products_set2)
            if valid:
                seq2.append(x)
                products_set2.update(new_tails)
        
        if len(seq2) > len(best_seq):
            best_seq = seq2
            best_products = products_set2
            
    # 4. Final polish: try to insert missing numbers into best_seq
    if best_seq:
        best_seq = insertion_polish(best_seq)
        
    return best_seq
