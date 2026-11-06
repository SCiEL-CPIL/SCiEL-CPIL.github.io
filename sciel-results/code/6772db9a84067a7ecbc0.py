"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def _greedy_construct():
    """Deterministic greedy: iterate 2..128, add if no product conflict."""
    values, products = [], set()
    for x in range(2, 129):
        tails = []
        prod = x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        tails.append(x)  # singleton
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    return values


def _all_products(seq):
    """Compute all nonempty consecutive block products for seq."""
    products = set()
    n = len(seq)
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            products.add(prod)
    return products


def _check_insertion(seq, pos, x, existing_products):
    """
    Check if inserting x at position pos in seq creates any duplicate products.
    Only computes the new products that involve x (O(n) instead of O(n^2)).
    Returns True if valid, False otherwise.
    """
    n = len(seq)
    # New products are all blocks [i..j] where i <= pos <= j in the new sequence.
    # In the old sequence, these correspond to:
    # - [i..pos-1] + [x] + [pos..j] for various i, j
    # Actually, let's think in terms of the new sequence indices.
    # New sequence: seq[0..pos-1], x, seq[pos..n-1]
    # A block in the new seq that includes x: starts at some i in [0, pos] and ends at some j in [pos, n-1]
    # where the block in the new sequence is new_seq[i..j] and includes position pos (which is x).
    # So i can be 0..pos, j can be pos..n-1 (in new indexing).
    # But we need to compute the product of new_seq[i..j].
    
    # Let's compute products of blocks that include x:
    # For each start s in [0, pos] (new index) and end e in [pos, n-1] (new index):
    # The product is prod(new_seq[s..e])
    
    # Precompute: left products ending at pos-1 (in old seq), i.e., prod(seq[i..pos-1]) for i=0..pos-1
    # And right products starting at pos (in old seq), i.e., prod(seq[pos..j]) for j=pos..n-1
    # And x itself.
    
    # A block including x in the new sequence:
    # new_seq = [seq[0], ..., seq[pos-1], x, seq[pos], ..., seq[n-1]]
    # Block from new index s to new index e where s <= pos <= e.
    # If s == pos and e == pos: product = x
    # If s < pos and e == pos: product = prod(seq[s..pos-1]) * x
    # If s == pos and e > pos: product = x * prod(seq[pos..e])
    # If s < pos and e > pos: product = prod(seq[s..pos-1]) * x * prod(seq[pos..e])
    
    # Let's compute:
    # left_prods[i] = prod(seq[i..pos-1]) for i = 0, 1, ..., pos-1 (and left_prods[pos] = 1 for the case s=pos)
    # right_prods[j] = prod(seq[pos..j]) for j = pos, pos+1, ..., n-1 (and right_prods[pos-1] = 1 for the case e=pos-1... wait)
    
    # Actually, let me re-index. In the new sequence:
    # Index 0..pos-1: seq[0]..seq[pos-1]
    # Index pos: x
    # Index pos+1..n: seq[pos]..seq[n-1]
    
    # A block [s..e] in new indices includes x iff s <= pos <= e.
    # Product = prod(new_seq[s..e])
    
    # Let L[i] = prod(new_seq[i..pos-1]) for i = 0..pos-1, and L[pos] = 1
    # Let R[j] = prod(new_seq[pos+1..j]) for j = pos+1..n, and R[pos] = 1
    # Then for s <= pos <= e:
    #   product = L[s] * x * R[e] if s < pos and e > pos
    #   product = L[s] * x if s < pos and e == pos
    #   product = x * R[e] if s == pos and e > pos
    #   product = x if s == pos and e == pos
    
    # More uniformly: product = L[s] * x * R[e] where L[pos]=1, R[pos]=1
    # But wait, if s=pos and e=pos, L[pos]*x*R[pos] = 1*x*1 = x. Correct.
    # If s=pos-1, e=pos: L[pos-1]*x*R[pos] = prod(seq[pos-1..pos-1])*x*1 = seq[pos-1]*x. Correct.
    # If s=pos, e=pos+1: L[pos]*x*R[pos+1] = 1*x*prod(seq[pos..pos]) = x*seq[pos]. Correct.
    # If s=pos-1, e=pos+1: L[pos-1]*x*R[pos+1] = seq[pos-1]*x*seq[pos]. Correct.
    
    # So compute L[0..pos] and R[pos..n-1] (in new indexing, but let's use old indices for clarity)
    
    # L[i] for i in 0..pos: L[i] = prod(seq[i..pos-1]) if i < pos, else 1
    # R[j] for j in pos..n-1: R[j] = prod(seq[pos..j]) if j >= pos, else 1
    # Wait, R should be for the right part. Let me redefine:
    
    # For the right side: R[j] = prod(seq[pos..j]) for j = pos, pos+1, ..., n-1
    # For the left side: L[i] = prod(seq[i..pos-1]) for i = 0, 1, ..., pos-1, and L[pos] = 1
    
    # Then for new block [s..e] with s <= pos <= e:
    #   If s == pos and e == pos: prod = x
    #   If s < pos and e == pos: prod = L[s] * x
    #   If s == pos and e > pos: prod = x * R[e]
    #   If s < pos and e > pos: prod = L[s] * x * R[e]
    
    # Unified: prod = L[s] * x * R[e] where L[pos]=1, R[pos]=1
    # But R is indexed from pos to n-1, and e ranges from pos to n-1 in new indexing... 
    # Wait, in new indexing, e goes from pos to n (where n is the length of old seq, so new length is n+1).
    # New index e corresponds to:
    #   e = pos: x
    #   e = pos+1: seq[pos]
    #   e = pos+2: seq[pos+1]
    #   ...
    #   e = n: seq[n-1]
    
    # So R_new[e] for e in pos..n: R_new[pos] = 1, R_new[pos+1] = seq[pos], R_new[pos+2] = seq[pos]*seq[pos+1], etc.
    # And L_new[s] for s in 0..pos: L_new[pos] = 1, L_new[pos-1] = seq[pos-1], L_new[pos-2] = seq[pos-2]*seq[pos-1], etc.
    
    # Let me just compute them:
    left_prods = [1] * (pos + 1)  # left_prods[s] for s=0..pos
    prod = 1
    for i in range(pos - 1, -1, -1):
        prod *= seq[i]
        left_prods[i] = prod
    left_prods[pos] = 1
    
    right_prods = [1] * (n - pos + 1)  # right_prods[e] for e=pos..n (new index)
    prod = 1
    for j in range(pos, n):
        prod *= seq[j]
        right_prods[j - pos] = prod
    # right_prods[0] corresponds to e=pos, right_prods[1] to e=pos+1, etc.
    
    # Now check all new blocks:
    for s in range(pos + 1):
        for e_offset in range(n - pos + 1):
            e = pos + e_offset
            p = left_prods[s] * x * right_prods[e_offset]
            if p in existing_products:
                return False
    return True


def _try_insert_fast(seq, x, products_set):
    """
    Try to insert x into sorted seq. Returns (new_seq, new_products) or (None, None).
    Uses fast O(n) check per position.
    """
    n = len(seq)
    # Find insertion position
    pos = 0
    for i in range(n):
        if x < seq[i]:
            pos = i
            break
        else:
            pos = i + 1
    
    # Check if x is already in seq
    if pos < n and seq[pos] == x:
        return None, None
    if pos > 0 and seq[pos - 1] == x:
        return None, None
    
    # Check validity using fast method
    if _check_insertion(seq, pos, x, products_set):
        new_seq = seq[:pos] + [x] + seq[pos:]
        # Compute new products set
        new_products = set(products_set)
        # Add the new products involving x
        left_prods = [1] * (pos + 1)
        prod = 1
        for i in range(pos - 1, -1, -1):
            prod *= seq[i]
            left_prods[i] = prod
        left_prods[pos] = 1
        
        right_prods = [1] * (n - pos + 1)
        prod = 1
        for j in range(pos, n):
            prod *= seq[j]
            right_prods[j - pos] = prod
        
        for s in range(pos + 1):
            for e_offset in range(n - pos + 1):
                p = left_prods[s] * x * right_prods[e_offset]
                new_products.add(p)
        
        return new_seq, new_products
    
    return None, None


def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # Get initial solution via greedy
    best_seq = _greedy_construct()
    best_len = len(best_seq)
    best_products = _all_products(best_seq)
    
    rng = random.Random(42)
    all_numbers = set(range(2, 129))
    
    while time.time() < deadline:
        seq = best_seq[:]
        products = set(best_products)
        current_len = len(seq)
        
        # Randomly remove 1-3 elements
        if current_len > 3:
            num_removals = rng.randint(1, 3)
            for _ in range(num_removals):
                if len(seq) > 2:
                    idx = rng.randint(0, len(seq) - 1)
                    seq.pop(idx)
            # Recompute products after removals
            products = _all_products(seq)
            current_len = len(seq)
        
        # Try to add back elements
        available = all_numbers - set(seq)
        avail_list = list(available)
        rng.shuffle(avail_list)
        
        improved = False
        for x in avail_list:
            new_seq, new_products = _try_insert_fast(seq, x, products)
            if new_seq is not None:
                seq = new_seq
                products = new_products
                improved = True
                if len(seq) > best_len:
                    best_seq = seq[:]
                    best_products = products
                    best_len = len(seq)
        
        # Local search: try swapping out one element for another
        for _ in range(3):
            if time.time() >= deadline:
                break
            if len(seq) < 2:
                break
            idx = rng.randint(0, len(seq) - 1)
            removed = seq.pop(idx)
            products = _all_products(seq)
            
            available = all_numbers - set(seq)
            avail_list = list(available)
            rng.shuffle(avail_list)
            added = False
            for x in avail_list[:15]:
                new_seq, new_products = _try_insert_fast(seq, x, products)
                if new_seq is not None and len(new_seq) > len(seq):
                    seq = new_seq
                    products = new_products
                    added = True
                    if len(seq) > best_len:
                        best_seq = seq[:]
                        best_products = products
                        best_len = len(seq)
                    break
                elif new_seq is not None and rng.random() < 0.2:
                    seq = new_seq
                    products = new_products
                    added = True
                    break
            if not added:
                # Put back
                pos = 0
                for i in range(len(seq)):
                    if seq[i] > removed:
                        pos = i
                        break
                seq.insert(pos, removed)
                products = _all_products(seq)
        
        if len(seq) > best_len:
            best_seq = seq[:]
            best_products = products
            best_len = len(seq)
    
    return best_seq

# EVOLVE-BLOCK-END
