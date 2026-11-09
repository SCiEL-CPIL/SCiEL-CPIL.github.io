import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    # 1. Greedy
    values = []
    products = set()
    for x in range(2, 129):
        if time.time() > deadline: break
        new_prods = []
        p = 1
        for a in reversed(values):
            p *= a
            new_prods.append(x * p)
        new_prods.append(x)
        if products.intersection(new_prods):
            continue
        values.append(x)
        products.update(new_prods)
        
    best_seq = list(values)
    
    # 2. Local Search (insert-only, preserving strict increase)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        candidates = [x for x in range(2, 129) if x not in best_seq]
        for x in candidates:
            if time.time() > deadline:
                break
            # Only positions where x fits strictly between neighbors
            lo = 0
            while lo < len(best_seq) and best_seq[lo] < x:
                lo += 1
            hi = len(best_seq)
            while hi > 0 and best_seq[hi - 1] > x:
                hi -= 1
            for i in range(hi, lo + 1):
                new_seq = best_seq[:i] + [x] + best_seq[i:]
                if is_valid(new_seq):
                    best_seq = new_seq
                    improved = True
                    break
            if improved:
                break

    # Final safety: ensure strictly increasing and within [2,128]
    result = []
    for v in best_seq:
        if not result or v > result[-1]:
            result.append(v)
    if result and is_valid(result):
        return result
    return [2]

def is_valid(seq):
    products = set()
    n = len(seq)
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            if prod in products:
                return False
            products.add(prod)
    return True
