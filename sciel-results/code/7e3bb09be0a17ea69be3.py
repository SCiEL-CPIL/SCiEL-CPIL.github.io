import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Candidate moduli: all even numbers from 20 to 200
    moduli = list(range(20, 201, 2))
    n = len(moduli)
    
    # Precompute gcds
    # gcd[i][j] = gcd(moduli[i], moduli[j])
    # n is ~90, n^2 is 8100, trivial.
    gcds = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            gcds[i][j] = math.gcd(moduli[i], moduli[j])
            gcds[j][i] = gcds[i][j]
            
    def get_conflict_count(residues):
        # residues is a list of length n
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                g = gcds[i][j]
                if residues[i] % g == residues[j] % g:
                    count += 1
        return count

    def is_conflict(i, j, r_i, r_j):
        g = gcds[i][j]
        return (r_i % g) == (r_j % g)

    # Local Search
    best_conflicts = float('inf')
    best_residues = None
    
    while time.time() < deadline:
        # Initialize
        residues = [random.randint(0, moduli[i]-1) for i in range(n)]
        
        # Calculate initial conflicts
        # We can optimize this, but n is small.
        conflicts = 0
        for i in range(n):
            for j in range(i+1, n):
                if is_conflict(i, j, residues[i], residues[j]):
                    conflicts += 1
        
        if conflicts < best_conflicts:
            best_conflicts = conflicts
            best_residues = residues[:]
            if conflicts == 0:
                break
            
        # Local search loop
        stuck_count = 0
        while time.time() < deadline and conflicts > 0:
            # Pick random index
            i = random.randint(0, n-1)
            
            # Try to improve
            improved = False
            # Try a few random residues
            for _ in range(10):
                new_r = random.randint(0, moduli[i]-1)
                if new_r == residues[i]:
                    continue
                    
                # Calculate delta conflicts
                # Old conflicts involving i
                old_conf = 0
                for j in range(n):
                    if i == j: continue
                    if is_conflict(i, j, residues[i], residues[j]):
                        old_conf += 1
                        
                # New conflicts involving i
                new_conf = 0
                for j in range(n):
                    if i == j: continue
                    if is_conflict(i, j, new_r, residues[j]):
                        new_conf += 1
                        
                delta = new_conf - old_conf
                
                if delta < 0:
                    residues[i] = new_r
                    conflicts += delta
                    improved = True
                    break
                elif delta == 0 and random.random() < 0.1:
                    residues[i] = new_r
                    # conflicts unchanged
                    improved = True
                    break
                    
            if not improved:
                stuck_count += 1
                if stuck_count > 1000:
                    break
            else:
                stuck_count = 0
                
                if conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best_residues = residues[:]
                    if conflicts == 0:
                        break

    # If we found a valid solution (0 conflicts), use it
    if best_conflicts == 0:
        classes = [[moduli[i], best_residues[i]] for i in range(n)]
        return classes
    else:
        # Fallback: Greedy construction
        # This guarantees validity
        selected = []
        selected_moduli = []
        for m in moduli:
            # Try to find a residue for m compatible with selected
            found = False
            for r in range(m):
                ok = True
                for sm, sr in selected:
                    g = math.gcd(m, sm)
                    if r % g == sr % g:
                        ok = False
                        break
                if ok:
                    selected.append((m, r))
                    selected_moduli.append(m)
                    found = True
                    break
            # If not found, skip m
        # Convert to required format
        classes = [[m, r] for m, r in selected]
        return classes
