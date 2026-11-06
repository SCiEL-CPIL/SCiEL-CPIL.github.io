import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Primes up to 128
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    num_primes = len(primes)
    
    # Precompute prime factorization vectors
    vectors = {}
    for x in range(2, 129):
        vec = [0] * num_primes
        temp = x
        for i, p in enumerate(primes):
            while temp % p == 0:
                vec[i] += 1
                temp //= p
        vectors[x] = tuple(vec)
    
    def get_all_product_vectors(seq):
        n = len(seq)
        if n == 0:
            return set()
        products = set()
        vecs = vectors
        np_ = num_primes
        for i in range(n):
            current = [0] * np_
            for j in range(i, n):
                v_j = vecs[seq[j]]
                for k in range(np_):
                    current[k] += v_j[k]
                products.add(tuple(current))
        return products

    def is_valid(seq):
        n = len(seq)
        prods = get_all_product_vectors(seq)
        return len(prods) == n * (n + 1) // 2

    def check_insertion(seq, pos, x):
        """Check if inserting x at position pos keeps all products distinct.
        Only need to check products that include x, since existing products are already distinct."""
        n = len(seq)
        vecs = vectors
        np_ = num_primes
        v_x = vecs[x]
        
        # Collect all existing products
        existing_prods = set()
        for i in range(n):
            current = [0] * np_
            for j in range(i, n):
                v_j = vecs[seq[j]]
                for k in range(np_):
                    current[k] += v_j[k]
                existing_prods.add(tuple(current))
        
        # Products that start at x and extend right: x * seq[pos] * ... * seq[j]
        current_right = list(v_x)
        for j in range(pos, n):
            v_j = vecs[seq[j]]
            for k in range(np_):
                current_right[k] += v_j[k]
            prod = tuple(current_right)
            if prod in existing_prods:
                return False
            existing_prods.add(prod)
        
        # Products that end at x and extend left: seq[i] * ... * seq[pos-1] * x
        current_left = list(v_x)
        for i in range(pos - 1, -1, -1):
            v_i = vecs[seq[i]]
            for k in range(np_):
                current_left[k] += v_i[k]
            prod = tuple(current_left)
            if prod in existing_prods:
                return False
            existing_prods.add(prod)
        
        # Products that span x: from some i < pos to some j >= pos
        # Product = seq[i]*...*seq[pos-1] * x * seq[pos]*...*seq[j]
        # Compute right parts: x*seq[pos]*...*seq[j] for j from pos to n-1
        right_parts = []
        current_r = list(v_x)
        for j in range(pos, n):
            v_j = vecs[seq[j]]
            for k in range(np_):
                current_r[k] += v_j[k]
            right_parts.append(tuple(current_r))
        
        # For each left extension, check all right extensions
        current_l = list(v_x)  # starts as just x
        for i in range(pos - 1, -1, -1):
            v_i = vecs[seq[i]]
            for k in range(np_):
                current_l[k] += v_i[k]
            # Now current_l = seq[i]*...*seq[pos-1]*x
            # For each j >= pos, product = seq[i]*...*seq[pos-1]*x*seq[pos]*...*seq[j]
            # = current_l + (seq[pos]*...*seq[j])
            # right_parts[j] = x*seq[pos]*...*seq[j]
            # So product = current_l + right_parts[j] - x
            for j in range(pos, n):
                prod_vec = [current_l[k] + right_parts[j][k] - v_x[k] for k in range(np_)]
                prod = tuple(prod_vec)
                if prod in existing_prods:
                    return False
                existing_prods.add(prod)
        
        return True

    # Better greedy: try multiple orderings
    def greedy_construct(order=None):
        seq = []
        if order is None:
            order = list(range(2, 129))
        for x in order:
            # Find insertion position (maintain sorted order)
            pos = 0
            for i, val in enumerate(seq):
                if x < val:
                    pos = i
                    break
                else:
                    pos = i + 1
            if check_insertion(seq, pos, x):
                seq = seq[:pos] + [x] + seq[pos:]
        return seq

    # Try several greedy strategies
    best_seq = greedy_construct()
    best_len = len(best_seq)
    
    # Try reverse order
    seq2 = greedy_construct(list(range(128, 1, -1)))
    if len(seq2) > best_len:
        best_seq = seq2
        best_len = len(seq2)
    
    # Try primes first, then composites
    primes_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    composites = [x for x in range(2, 129) if x not in primes_list]
    seq3 = greedy_construct(primes_list + composites)
    if len(seq3) > best_len:
        best_seq = seq3
        best_len = len(seq3)

    rng = random.Random(42)
    no_improve_count = 0
    
    while time.time() < deadline:
        seq = best_seq[:]
        current_len = len(seq)
        
        # Random Move Selection
        move_type = rng.random()
        
        if move_type < 0.85:
            # Try to insert a missing element
            missing = [x for x in range(2, 129) if x not in seq]
            if not missing:
                continue
            x = rng.choice(missing)
            
            # Find insertion pos
            pos = 0
            for i, val in enumerate(seq):
                if x < val:
                    pos = i
                    break
                else:
                    pos = i + 1
            
            if check_insertion(seq, pos, x):
                seq = seq[:pos] + [x] + seq[pos:]
                if len(seq) > best_len:
                    best_seq = seq[:]
                    best_len = len(seq)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        else:
            # Try a swap: remove one, add one
            if len(seq) == 0:
                continue
            missing = [x for x in range(2, 129) if x not in seq]
            if not missing:
                continue
            
            x_new = rng.choice(missing)
            y_old_idx = rng.randint(0, len(seq) - 1)
            
            # Remove y_old
            seq_without_y = seq[:y_old_idx] + seq[y_old_idx+1:]
            
            # Insert x_new into seq_without_y
            pos = 0
            for i, val in enumerate(seq_without_y):
                if x_new < val:
                    pos = i
                    break
                else:
                    pos = i + 1
            
            if check_insertion(seq_without_y, pos, x_new):
                new_seq = seq_without_y[:pos] + [x_new] + seq_without_y[pos:]
                if len(new_seq) > best_len:
                    best_seq = new_seq[:]
                    best_len = len(new_seq)
                    no_improve_count = 0
                elif len(new_seq) == best_len:
                    if rng.random() < 0.2:
                        best_seq = new_seq[:]
                        no_improve_count = 0
                
        # If stuck, try a more aggressive restart
        if no_improve_count > 1500:
            seq = best_seq[:]
            if len(seq) >= 2:
                idx1 = rng.randint(0, len(seq)-1)
                idx2 = rng.randint(0, len(seq)-1)
                if idx1 == idx2:
                    idx2 = (idx2 + 1) % len(seq)
                if idx1 > idx2:
                    idx1, idx2 = idx2, idx1
                seq.pop(idx1)
                seq.pop(idx2)
            best_seq = seq[:]
            best_len = len(best_seq)
            no_improve_count = 0

    return best_seq
