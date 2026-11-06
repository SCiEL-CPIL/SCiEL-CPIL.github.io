import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    all_moduli = list(range(20, 201))
    
    def local_search(moduli):
        n = len(moduli)
        residues = [random.randint(0, m-1) for m in moduli]
        
        # Precompute gcds
        gcds = [[gcd(moduli[i], moduli[j]) for j in range(n)] for i in range(n)]
        
        def count_conflicts():
            c = 0
            for i in range(n):
                for j in range(i+1, n):
                    if residues[i] % gcds[i][j] == residues[j] % gcds[i][j]:
                        c += 1
            return c
        
        current_conflicts = count_conflicts()
        
        max_iters = 5000
        for _ in range(max_iters):
            if time.time() > deadline:
                return None
            if current_conflicts == 0:
                return residues
            
            # Find a conflict
            # Instead of scanning all, we can pick a random i and find a j that conflicts with i
            i = random.randint(0, n-1)
            # Find j such that (i,j) is a conflict
            conflicting_js = [j for j in range(n) if j != i and residues[i] % gcds[i][j] == residues[j] % gcds[i][j]]
            if not conflicting_js:
                # i has no conflicts, try another i
                continue
            j = random.choice(conflicting_js)
            
            # Try to fix i
            m_i = moduli[i]
            candidates = list(range(m_i))
            random.shuffle(candidates)
            fixed = False
            for new_r in candidates:
                # Check if new_r conflicts with any k != i
                ok = True
                for k in range(n):
                    if k == i: continue
                    if new_r % gcds[i][k] == residues[k] % gcds[i][k]:
                        ok = False
                        break
                if ok:
                    # Update conflicts
                    # Remove old conflicts involving i
                    for k in range(n):
                        if k == i: continue
                        old_conflict = (residues[i] % gcds[i][k] == residues[k] % gcds[i][k])
                        new_conflict = (new_r % gcds[i][k] == residues[k] % gcds[i][k])
                        if old_conflict and not new_conflict:
                            current_conflicts -= 1
                        elif not old_conflict and new_conflict:
                            current_conflicts += 1
                    residues[i] = new_r
                    fixed = True
                    break
            
            if not fixed:
                # Try to fix j
                m_j = moduli[j]
                candidates = list(range(m_j))
                random.shuffle(candidates)
                for new_r in candidates:
                    ok = True
                    for k in range(n):
                        if k == j: continue
                        if new_r % gcds[j][k] == residues[k] % gcds[j][k]:
                            ok = False
                            break
                    if ok:
                        for k in range(n):
                            if k == j: continue
                            old_conflict = (residues[j] % gcds[j][k] == residues[k] % gcds[j][k])
                            new_conflict = (new_r % gcds[j][k] == residues[k] % gcds[j][k])
                            if old_conflict and not new_conflict:
                                current_conflicts -= 1
                            elif not old_conflict and new_conflict:
                                current_conflicts += 1
                        residues[j] = new_r
                        fixed = True
                        break
            
            if not fixed:
                return None
        
        return None

    solution = None
    for _ in range(100):
        if time.time() > deadline - 5:
            break
        random.seed()
        sol = local_search(all_moduli)
        if sol:
            solution = sol
            break
    
    if solution:
        return [[m, r] for m, r in zip(all_moduli, solution)]
    
    current_moduli = list(all_moduli)
    while current_moduli:
        if time.time() > deadline - 5:
            break
        sol = None
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            random.seed()
            sol = local_search(current_moduli)
            if sol:
                break
        if sol:
            return [[m, r] for m, r in zip(current_moduli, sol)]
        current_moduli.sort()
        current_moduli.pop()
        
    return [[20 * j, j - 1] for j in range(1, 11)]
