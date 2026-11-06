import random
import time
import math

def construct():
    start_time = time.time()
    timeout = 85  # Leave 5s margin for safety
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    
    best_count = 0
    best_solution = []
    
    # Precompute gcds for speed
    gcd_cache = {}
    for i in range(2, max_mod + 1):
        for j in range(i, max_mod + 1):
            gcd_cache[(i, j)] = math.gcd(i, j)
            
    def get_gcd(m1, m2):
        if m1 == m2: return m1
        if m1 > m2: m1, m2 = m2, m1
        return gcd_cache[(m1, m2)]

    def try_add(selected_moduli, residues, m):
        # Check against existing moduli
        # We need r such that for all m' in selected: r % gcd(m, m') != residues[m'] % gcd(m, m')
        # We can try residues 0..m-1
        # To speed up, we can check constraints one by one and prune?
        # Simple loop is probably fine for m <= 128
        
        # Randomize start point
        start = random.randint(0, m - 1)
        for k in range(m):
            r = (start + k) % m
            valid = True
            for m_prime in selected_moduli:
                g = get_gcd(m, m_prime)
                if r % g == residues[m_prime] % g:
                    valid = False
                    break
            if valid:
                residues[m] = r
                return True
        return False

    # Heuristic: Sort moduli by "difficulty"
    # Moduli with many divisors are harder? Or prime powers?
    # Let's just shuffle.
    
    for restart in range(1000):
        if time.time() - start_time > timeout:
            break
            
        selected = []
        residues = {}
        available = moduli[:]
        random.shuffle(available)
        
        for m in available:
            if time.time() - start_time > timeout:
                break
            if try_add(selected, residues, m):
                selected.append(m)
                if len(selected) > best_count:
                    best_count = len(selected)
                    # Store copy
                    best_solution = [[residues[m2], m2] for m2 in selected]
        
        # Optional: Local search to improve current best
        # Try to swap residues of existing to allow adding more?
        # Too complex for now.
        
    if not best_solution:
        return [[(1<<k)-1,1<<(k+1)] for k in range(7)]
        
    return best_solution
