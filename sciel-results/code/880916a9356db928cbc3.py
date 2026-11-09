import time
from math import gcd

def construct():
    """
    Finds maximum number of pairwise disjoint infinite residue classes
    with distinct moduli in [2, 128].
    
    Two classes (r1,m1) and (r2,m2) are disjoint iff (r1-r2) % gcd(m1,m2) != 0.
    
    Strategy:
    1. Process moduli in order of "difficulty" (primes first, then by number of divisors)
    2. For each modulus, pick a residue that conflicts with the fewest already-selected classes
    3. Use constraint propagation to prune invalid residues early
    4. Multiple restarts with different orderings to find maximum
    """
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    max_modulus = 128
    all_moduli = list(range(2, max_modulus + 1))
    n = len(all_moduli)
    
    # Precompute gcd matrix
    gcd_mat = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            gcd_mat[i][j] = gcd(all_moduli[i], all_moduli[j])
    
    # Determine processing order: primes first, then by number of proper divisors
    def num_proper_divisors(m):
        count = 0
        for d in range(2, int(m**0.5) + 1):
            if m % d == 0:
                count += 1
                if d != m // d:
                    count += 1
        return count
    
    def is_prime(m):
        if m < 2:
            return False
        for d in range(2, int(m**0.5) + 1):
            if m % d == 0:
                return False
        return True
    
    # Sort: primes first (ascending), then composites by number of divisors (ascending)
    primes = [m for m in all_moduli if is_prime(m)]
    composites = [m for m in all_moduli if not is_prime(m)]
    composites.sort(key=lambda m: (num_proper_divisors(m), m))
    priority_order = primes + composites
    
    # Map modulus to index in all_moduli
    mod_to_idx = {m: i for i, m in enumerate(all_moduli)}
    priority_indices = [mod_to_idx[m] for m in priority_order]
    
    best_result = []
    best_size = 0
    
    def try_greedy(order, deterministic=False):
        """Try greedy assignment with given order."""
        nonlocal best_result, best_size
        
        assignment = {}  # idx -> residue
        selected_indices = set()
        
        for idx in order:
            m = all_moduli[idx]
            
            # Find all valid residues for this modulus
            valid_rs = []
            for r in range(m):
                ok = True
                for j in selected_indices:
                    g = gcd_mat[idx][j]
                    if (r - assignment[j]) % g == 0:
                        ok = False
                        break
                if ok:
                    valid_rs.append(r)
            
            if not valid_rs:
                # Dead end - return what we have
                break
            
            if deterministic:
                # Pick the residue that minimizes future conflicts
                # Score: for each unselected modulus k, count how many residues it would lose
                best_r = valid_rs[0]
                best_score = float('inf')
                for r in valid_rs:
                    score = 0
                    for k in range(n):
                        if k in selected_indices or k == idx:
                            continue
                        g = gcd_mat[idx][k]
                        # This residue r forbids all r' where (r' - r) % g == 0
                        # Number of such r' in [0, m_k) is m_k // g
                        # But we want to minimize total forbidden residues
                        score += all_moduli[k] // g
                    if score < best_score:
                        best_score = score
                        best_r = r
                r = best_r
            else:
                r = valid_rs[0]  # Deterministic for reproducibility
            
            assignment[idx] = r
            selected_indices.add(idx)
        
        # Record result
        if len(assignment) > best_size:
            best_size = len(assignment)
            best_result = [[assignment[idx], all_moduli[idx]] for idx in assignment]
        
        return len(assignment)
    
    # Phase 1: Try greedy with priority order
    try_greedy(priority_indices, deterministic=True)
    
    # Phase 2: Try greedy with reverse priority (composites first)
    reverse_order = list(reversed(priority_indices))
    try_greedy(reverse_order, deterministic=True)
    
    # Phase 3: Try greedy with natural order
    try_greedy(list(range(n)), deterministic=True)
    
    # Phase 4: Local search - try to add more classes or swap residues
    # Start from best solution and try to improve
    def local_search(initial_assignment):
        """Try to improve solution via local search."""
        nonlocal best_result, best_size
        
        assignment = dict(initial_assignment)
        selected_indices = set(assignment.keys())
        
        # Try to add unselected moduli
        for idx in range(n):
            if idx in selected_indices:
                continue
            m = all_moduli[idx]
            valid_rs = []
            for r in range(m):
                ok = True
                for j in selected_indices:
                    g = gcd_mat[idx][j]
                    if (r - assignment[j]) % g == 0:
                        ok = False
                        break
                if ok:
                    valid_rs.append(r)
            if valid_rs:
                # Pick the one with minimum conflict score
                best_r = valid_rs[0]
                best_score = float('inf')
                for r in valid_rs:
                    score = 0
                    for k in range(n):
                        if k in selected_indices or k == idx:
                            continue
                        g = gcd_mat[idx][k]
                        score += all_moduli[k] // g
                    if score < best_score:
                        best_score = score
                        best_r = r
                assignment[idx] = best_r
                selected_indices.add(idx)
        
        if len(assignment) > best_size:
            best_size = len(assignment)
            best_result = [[assignment[idx], all_moduli[idx]] for idx in assignment]
        
        return len(assignment)
    
    if best_result:
        # Convert best_result back to assignment dict
        init_assign = {}
        for r, m in best_result:
            idx = mod_to_idx[m]
            init_assign[idx] = r
        local_search(init_assign)
    
    # Phase 5: Try different deterministic orders based on various heuristics
    # Order by modulus size (small first)
    try_greedy(sorted(range(n), key=lambda i: all_moduli[i]), deterministic=True)
    
    # Order by modulus size (large first)
    try_greedy(sorted(range(n), key=lambda i: -all_moduli[i]), deterministic=True)
    
    # Check time and do more restarts if needed
    elapsed = time.time() - start_time
    if elapsed < 30:
        # Try random restarts with shuffled orders
        import random
        for restart in range(200):
            if time.time() > deadline:
                break
            order = list(range(n))
            random.shuffle(order)
            try_greedy(order, deterministic=False)
    
    # Ensure non-empty result
    if not best_result:
        best_result = [[0, 2]]
    
    return best_result

# EVOLVE-BLOCK-END
