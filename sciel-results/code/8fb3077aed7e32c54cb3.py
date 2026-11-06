import math
import time
import random

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    moduli = list(range(2, 129))
    n = len(moduli)
    
    # Precompute GCDs
    gcds = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            g = _gcd(moduli[i], moduli[j])
            gcds[i][j] = g
            gcds[j][i] = g
            
    # Precompute divisor counts to prioritize "complex" moduli?
    # Or just use random orders.
    
    best_count = 0
    best_result = []
    
    def is_valid(i, r, selected):
        """Check if residue r for modulus index i is disjoint from all in selected."""
        for (j, rj) in selected:
            g = gcds[i][j]
            if (r - rj) % g == 0:
                return False
        return True
        
    def greedy_solve(order, residue_strategy='linear'):
        """Greedy: process moduli in given order, pick first valid residue."""
        selected = []
        for i in order:
            m = moduli[i]
            found = False
            if residue_strategy == 'linear':
                residues = range(m)
            elif residue_strategy == 'random':
                residues = list(range(m))
                random.shuffle(residues)
            else: # 'prime_coprime' or something?
                residues = range(m)
            
            for r in residues:
                if is_valid(i, r, selected):
                    selected.append((i, r))
                    found = True
                    break
            if not found:
                continue
        return selected

    # Try different strategies
    strategies = [
        ('desc', 'linear'),
        ('asc', 'linear'),
        ('random', 'linear'),
        ('random', 'random'),
        ('divisors_desc', 'linear')
    ]
    
    # Precompute divisor counts
    div_counts = [0]*n
    for i in range(n):
        m = moduli[i]
        cnt = 0
        for k in range(2, int(math.isqrt(m)) + 1):
            if m % k == 0:
                cnt += 2
                if k*k == m:
                    cnt -= 1
        if m > 1: cnt += 2 # 1 and m
        div_counts[i] = cnt
        
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Cycle through strategies or pick random
        if attempts % 5 == 1:
            order_type = 'desc'
            res_type = 'linear'
        elif attempts % 5 == 2:
            order_type = 'asc'
            res_type = 'linear'
        elif attempts % 5 == 3:
            order_type = 'random'
            res_type = 'random'
        elif attempts % 5 == 4:
            order_type = 'divisors_desc'
            res_type = 'linear'
        else:
            order_type = 'random'
            res_type = 'linear'
            
        indices = list(range(n))
        if order_type == 'desc':
            indices.sort(key=lambda i: moduli[i], reverse=True)
        elif order_type == 'asc':
            indices.sort(key=lambda i: moduli[i])
        elif order_type == 'divisors_desc':
            indices.sort(key=lambda i: div_counts[i], reverse=True)
        else:
            random.shuffle(indices)
            
        selected = greedy_solve(indices, res_type)
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_result = [[r, moduli[i]] for i, r in selected]
            
            # If we get a very good result, try to polish?
            # For now, just keep going.
            if best_count > 50:
                # Increase attempts or break?
                # Let's just break if we hit a very high bar, as it's likely optimal or near.
                # Max is 127. 50 is already quite good.
                # But we have time. Let's keep going for a bit.
                pass

    if not best_result:
        return [[1, 2]]
        
    return best_result
