import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    def is_valid_bitset(n, adj_bits, timeout=None):
        t_start = time.time()
        for k in range(6, n + 1):
            for comb in itertools.combinations(range(n), k):
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                
                # Check if all degrees in induced subgraph are equal
                first_deg = (adj_bits[comb[0]] & mask).bit_count()
                is_reg = True
                for v in comb[1:]:
                    if (adj_bits[v] & mask).bit_count() != first_deg:
                        is_reg = False
                        break
                
                if is_reg:
                    return False
                
                if timeout and (time.time() - t_start) > timeout:
                    return None
        return True

    def find_bad_subset_sampled(n, adj_bits, num_samples=100, timeout=None):
        t_start = time.time()
        for _ in range(num_samples):
            k = random.randint(6, n)
            comb = tuple(random.sample(range(n), k))
            mask = 0
            for v in comb:
                mask |= (1 << v)
            
            first_deg = (adj_bits[comb[0]] & mask).bit_count()
            is_reg = True
            for v in comb[1:]:
                if (adj_bits[v] & mask).bit_count() != first_deg:
                    is_reg = False
                    break
            
            if is_reg:
                return comb
            
            if timeout and (time.time() - t_start) > timeout:
                return None
        return None

    def search(n, time_limit):
        t_start = time.time()
        while time.time() - t_start < time_limit:
            # Random init
            adj_bits = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj_bits[i] |= (1 << j)
                        adj_bits[j] |= (1 << i)
            
            # Local search
            for _ in range(50):
                bad = find_bad_subset_sampled(n, adj_bits, num_samples=200, timeout=0.5)
                if bad is None:
                    return adj_bits
                if bad:
                    # Flip random edge in bad subset
                    u, v = random.sample(bad, 2)
                    # Toggle
                    if adj_bits[u] & (1 << v):
                        adj_bits[u] &= ~(1 << v)
                        adj_bits[v] &= ~(1 << u)
                    else:
                        adj_bits[u] |= (1 << v)
                        adj_bits[v] |= (1 << u)
                else:
                    # Timeout, random flip
                    u, v = random.sample(range(n), 2)
                    if adj_bits[u] & (1 << v):
                        adj_bits[u] &= ~(1 << v)
                        adj_bits[v] &= ~(1 << u)
                    else:
                        adj_bits[u] |= (1 << v)
                        adj_bits[v] |= (1 << u)
        return None

    candidates = []
    
    # Try 21
    if time.time() < deadline - 15:
        cand = search(21, time_limit=15)
        if cand:
            candidates.append((21, cand))
            
    # Try 20
    if time.time() < deadline - 10:
        cand = search(20, time_limit=20)
        if cand:
            candidates.append((20, cand))
            
    # Try 19
    if time.time() < deadline - 5:
        cand = search(19, time_limit=10)
        if cand:
            candidates.append((19, cand))

    # Verify
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    for n, adj_bits in candidates:
        remaining = deadline - time.time()
        if remaining < 2:
            break
        valid = is_valid_bitset(n, adj_bits, timeout=remaining - 1)
        if valid is True:
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj_bits[i] & (1 << j):
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
        elif valid is None:
            continue
        else:
            continue
            
    # Fallback
    return {'n': 5, 'edges': []}
