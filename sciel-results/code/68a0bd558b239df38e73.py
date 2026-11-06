import math
import time
import random
from itertools import combinations

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
    
    best_count = 0
    best_result = []
    
    def is_valid(i, r, selected):
        """Check if residue r for modulus index i is disjoint from all in selected."""
        for (j, rj) in selected:
            g = gcds[i][j]
            if (r - rj) % g == 0:
                return False
        return True
    
    def greedy_solve(order):
        """Greedy: process moduli in given order, pick first valid residue."""
        selected = []
        for i in order:
            m = moduli[i]
            for r in range(m):
                if is_valid(i, r, selected):
                    selected.append((i, r))
                    break
        return selected
    
    def local_search(selected):
        """Try to improve by swapping residues or adding/removing moduli."""
        selected = list(selected)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to add a missing modulus
            in_set = set(i for i, _ in selected)
            for i in range(n):
                if i in in_set:
                    continue
                m = moduli[i]
                for r in range(m):
                    if is_valid(i, r, selected):
                        new_sel = selected + [(i, r)]
                        # Try to remove one that's now redundant? No, just add.
                        selected = new_sel
                        improved = True
                        break
                if improved:
                    break
            
            # Try to swap a residue for a better one? 
            # Or try to replace a modulus with another?
            # Simple hill climbing: try changing residue of one element
            for idx, (i, r) in enumerate(selected):
                m = moduli[i]
                # Try a different residue for i
                for r_new in range(m):
                    if r_new == r:
                        continue
                    # Check if r_new is valid against all OTHERS
                    valid = True
                    for j, (k, rk) in enumerate(selected):
                        if j == idx:
                            continue
                        g = gcds[i][k]
                        if (r_new - rk) % g == 0:
                            valid = False
                            break
                    if valid:
                        # Does this allow adding a new modulus?
                        # Or just accept if it doesn't break anything (it doesn't)
                        # But we want to increase count. So only accept if we can add more?
                        # Actually, local search on count: 
                        # Let's just try to add more after a swap?
                        # This is getting complex. Let's stick to greedy + random restarts.
                        pass
        
        return selected
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Random order of moduli
        indices = list(range(n))
        random.shuffle(indices)
        
        selected = greedy_solve(indices)
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_result = [[r, moduli[i]] for i, r in selected]
            
            # If we've found a very large set, try to polish with local search?
            # For now, just keep going.
            
            # Early exit if we hit a high bar? 
            # Max is 127. 
            # If we get > 100, it's likely near optimal.
            if best_count > 100:
                # We can try a few more times to polish, but return soon
                # Let's do a few more quick attempts
                for _ in range(5):
                    if time.time() > deadline - 1:
                        break
                    indices = list(range(n))
                    random.shuffle(indices)
                    selected = greedy_solve(indices)
                    if len(selected) > best_count:
                        best_count = len(selected)
                        best_result = [[r, moduli[i]] for i, r in selected]
                break

    if not best_result:
        # Fallback
        return [[1, 2]]
        
    return best_result
