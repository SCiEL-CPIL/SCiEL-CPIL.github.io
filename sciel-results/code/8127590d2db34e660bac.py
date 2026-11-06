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


def _get_all_products(seq):
    """Return set of all nonempty consecutive block products."""
    products = set()
    n = len(seq)
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            products.add(prod)
    return products


def _try_insert_fast(seq, x, existing_products):
    """Try to insert x into seq at the correct sorted position.
    Only checks new products involving x. Returns (new_seq, new_products, True) or (None, None, False)."""
    pos = -1
    for i in range(len(seq)):
        if seq[i] > x:
            pos = i
            break
    if pos == -1:
        pos = len(seq)
    if pos < len(seq) and seq[pos] == x:
        return None, None, False
    if pos > 0 and seq[pos - 1] == x:
        return None, None, False

    n = len(seq)
    # In new_seq: indices 0..pos-1 are seq[0..pos-1], index pos is x, indices pos+1..n are seq[pos..n-1]
    # New blocks are those spanning index pos in new_seq: new_seq[i:j+1] for i in [0,pos], j in [pos,n]
    # product = (prod of seq[i:pos]) * x * (prod of seq[pos:j])

    # Precompute left_prods[i] = product of seq[i:pos] for i in 0..pos
    left_prods = [1] * (pos + 1)
    for i in range(pos - 1, -1, -1):
        left_prods[i] = left_prods[i + 1] * seq[i]

    # Precompute right_prods[k] = product of seq[pos:pos+k] for k in 0..(n-pos)
    right_prods = [1] * (n - pos + 1)
    for k in range(1, n - pos + 1):
        right_prods[k] = right_prods[k - 1] * seq[pos + k - 1]

    new_prods = set()
    for i in range(pos + 1):
        for k in range(n - pos + 1):
            total = left_prods[i] * x * right_prods[k]
            if total in existing_products or total in new_prods:
                return None, None, False
            new_prods.add(total)

    new_seq = seq[:pos] + [x] + seq[pos:]
    new_products = existing_products | new_prods
    return new_seq, new_products, True


def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    # Get initial solution via greedy
    best_seq = _greedy_construct()
    best_len = len(best_seq)
    best_products = _get_all_products(best_seq)

    rng = random.Random(42)

    all_numbers = set(range(2, 129))

    while time.time() < deadline:
        # Start from best_seq
        seq = best_seq[:]
        seq_products = best_products.copy()
        current_len = len(seq)

        # Randomly remove 1-3 elements to allow rearrangement
        if current_len > 3:
            num_removals = rng.randint(1, 3)
            for _ in range(num_removals):
                if len(seq) > 2:
                    idx = rng.randint(0, len(seq) - 1)
                    seq.pop(idx)
            # Recompute products after removals
            seq_products = _get_all_products(seq)

        # Try to add back removed or new elements
        available = all_numbers - set(seq)
        avail_list = list(available)
        rng.shuffle(avail_list)

        for x in avail_list:
            if time.time() >= deadline:
                break
            new_seq, new_products, ok = _try_insert_fast(seq, x, seq_products)
            if ok:
                seq = new_seq
                seq_products = new_products
                if len(seq) > best_len:
                    best_seq = seq[:]
                    best_products = seq_products.copy()
                    best_len = len(seq)

        # Local search: remove one element and try to add a different one
        for _ in range(8):
            if time.time() >= deadline:
                break
            if len(seq) < 2:
                break
            idx = rng.randint(0, len(seq) - 1)
            removed = seq.pop(idx)
            # Recompute products after removal
            seq_products = _get_all_products(seq)
            available = all_numbers - set(seq)
            avail_list = list(available)
            rng.shuffle(avail_list)
            added = False
            for x in avail_list[:30]:
                if time.time() >= deadline:
                    break
                new_seq, new_products, ok = _try_insert_fast(seq, x, seq_products)
                if ok and len(new_seq) > len(seq):
                    seq = new_seq
                    seq_products = new_products
                    added = True
                    if len(seq) > best_len:
                        best_seq = seq[:]
                        best_products = seq_products.copy()
                        best_len = len(seq)
                    break
                elif ok:
                    if rng.random() < 0.3:
                        seq = new_seq
                        seq_products = new_products
                        added = True
                        break
            if not added:
                # Put back the removed element
                pos = 0
                for i in range(len(seq)):
                    if seq[i] > removed:
                        pos = i
                        break
                seq.insert(pos, removed)
                seq_products = _get_all_products(seq)

        if len(seq) > best_len:
            best_seq = seq[:]
            best_products = seq_products.copy()
            best_len = len(seq)

    return best_seq

# EVOLVE-BLOCK-END
