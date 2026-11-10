import random
import time
import math

def construct():
    start_time = time.time()
    max_time = 55.0
    
    # The set of moduli is all even numbers from 2 to 128
    moduli = list(range(2, 129, 2))
    n = len(moduli)
    
    # Precompute GCDs
    gcds = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g = math.gcd(moduli[i], moduli[j])
            gcds[i][j] = g
            gcds[j][i] = g
            
    # Initialize residues randomly
    residues = [random.randint(0, m - 1) for m in moduli]
    
    # Calculate initial violations
    def count_violations(res):
        count = 0
        for i in range(n):
            ri = res[i]
            for j in range(i + 1, n):
                g = gcds[i][j]
                if ri % g == res[j] % g:
                    count += 1
        return count
        
    current_violations = count_violations(residues)
    best_violations = current_violations
    best_residues = residues[:]
    
    # Simulated Annealing / Hill Climbing
    # Temperature starts high, cools down
    T = 10.0
    cooling = 0.99
    
    while time.time() - start_time < max_time:
        if current_violations == 0:
            return [[residues[i], moduli[i]] for i in range(n)]
            
        # Pick a random index to change
        i = random.randint(0, n - 1)
        old_r = residues[i]
        
        # Calculate delta for changing i to a new random value
        # To be efficient, we can try a few random values or the best among a small sample
        # Or just one random value per iteration
        
        new_r = random.randint(0, moduli[i] - 1)
        if new_r == old_r:
            continue
            
        # Calculate change in violations for index i
        delta = 0
        for j in range(n):
            if i == j:
                continue
            g = gcds[i][j]
            # Old state
            old_match = 1 if (old_r % g == residues[j] % g) else 0
            # New state
            new_match = 1 if (new_r % g == residues[j] % g) else 0
            delta += new_match - old_match
            
        # Accept if delta < 0 (improvement)
        # Or with probability exp(-delta / T) if delta > 0
        if delta < 0:
            residues[i] = new_r
            current_violations += delta
            if current_violations < best_violations:
                best_violations = current_violations
                best_residues = residues[:]
        else:
            # Metropolis criterion
            if random.random() < math.exp(-delta / T):
                residues[i] = new_r
                current_violations += delta
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_residues = residues[:]
                    
        T *= cooling
        
    # If we didn't find a perfect solution, we must return a valid one.
    # The best we found might have violations > 0.
    # We need to construct a valid subset.
    # Greedy: Try to add moduli one by one to an empty set, checking validity.
    # This guarantees a valid solution, though maybe smaller.
    
    # Actually, if best_violations > 0, the set of all 64 is invalid.
    # We should try to find the largest valid subset.
    # However, the problem asks to maximize the number.
    # If we can't get 64, we might get 63, etc.
    
    # Let's try a greedy construction on the moduli to ensure validity.
    # We can use the best_residues as a guide or just restart greedy.
    
    # Simple Greedy for valid subset:
    valid_set = []
    valid_residues = []
    for m in moduli:
        # Try to find a residue r for m that is compatible with all in valid_set
        found = False
        # Try random residues
        for _ in range(100):
            r = random.randint(0, m - 1)
            compatible = True
            for r2, m2 in zip(valid_residues, [x[1] for x in valid_set]):
                g = math.gcd(m, m2)
                if r % g == r2 % g:
                    compatible = False
                    break
            if compatible:
                valid_set.append([r, m])
                valid_residues.append(r)
                found = True
                break
        if found:
            continue
        else:
            # Try deterministic scan if random fails
            for r in range(m):
                compatible = True
                for r2, m2 in zip(valid_residues, [x[1] for x in valid_set]):
                    g = math.gcd(m, m2)
                    if r % g == r2 % g:
                        compatible = False
                        break
                if compatible:
                    valid_set.append([r, m])
                    valid_residues.append(r)
                    found = True
                    break
                    
    # If we still didn't find anything (unlikely), return empty or best effort
    # But the task requires non-empty.
    # The greedy above should always find at least one.
    
    # To maximize, we can try to improve this greedy set with local search again?
    # Given time, the SA on the full set is likely to find 64.
    # If it doesn't, the greedy fallback is safe.
    
    # However, if SA found a solution with 0 violations, we returned it.
    # So if we reach here, SA failed to find 0 violations for all 64.
    # We return the greedy valid set.
    
    return valid_set
