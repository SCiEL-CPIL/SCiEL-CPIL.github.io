import time
from math import gcd
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    all_moduli = list(range(20, 201))
    
    # Precompute gcds? No, gcd is fast.
    
    def check_validity(moduli, node_limit=10000):
        """
        Returns (True, residues) if valid, (False, None) otherwise.
        """
        n = len(moduli)
        if n == 0:
            return True, []
        
        residues = [None] * n
        nodes_visited = 0
        limit_reached = False
        
        def backtrack(idx):
            nonlocal nodes_visited, limit_reached
            if limit_reached:
                return False
            nodes_visited += 1
            if nodes_visited > node_limit:
                limit_reached = True
                return False
                
            if idx == n:
                return True
            
            m = moduli[idx]
            # Try residues
            # To find a solution faster, we can try residues in a specific order.
            # Usually 0, 1, ... is fine.
            for r in range(m):
                ok = True
                for j in range(idx):
                    m_j = moduli[j]
                    g = gcd(m, m_j)
                    if (r - residues[j]) % g == 0:
                        ok = False
                        break
                if ok:
                    residues[idx] = r
                    if backtrack(idx + 1):
                        return True
                    residues[idx] = None
            return False
        
        if backtrack(0):
            return True, residues
        else:
            return False, None

    def get_score(moduli):
        return sum(Fraction(1, m) for m in moduli)

    best_score = Fraction(0, 1)
    best_classes = []
    
    # Strategy 1: Greedy with sorted moduli (small to large)
    # This is the most likely to yield a high score.
    order = sorted(all_moduli)
    
    current_set = []
    current_residues = []
    
    for m in order:
        if time.time() > deadline - 10:
            break
        trial_set = current_set + [m]
        valid, res = check_validity(trial_set, node_limit=5000)
        if valid:
            current_set.append(m)
            current_residues = res
    
    if current_set:
        s = get_score(current_set)
        if s > best_score:
            best_score = s
            best_classes = [[current_set[i], current_residues[i]] for i in range(len(current_set))]
            
    # Strategy 2: Randomized restarts to find potentially better subsets
    for attempt in range(50):
        if time.time() > deadline - 10:
            break
        
        # Shuffle, but keep small moduli somewhat prioritized?
        # Or just shuffle.
        order = all_moduli[:]
        random.shuffle(order)
        
        current_set = []
        current_residues = []
        
        for m in order:
            if time.time() > deadline - 10:
                break
            trial_set = current_set + [m]
            valid, res = check_validity(trial_set, node_limit=5000)
            if valid:
                current_set.append(m)
                current_residues = res
        
        if current_set:
            s = get_score(current_set)
            if s > best_score:
                best_score = s
                best_classes = [[current_set[i], current_residues[i]] for i in range(len(current_set))]
                
    # Strategy 3: Local Search (Add if fits) on the best set
    if best_classes and time.time() < deadline - 5:
        current_moduli = [c[0] for c in best_classes]
        current_residues = [c[1] for c in best_classes]
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for m in all_moduli:
                if time.time() > deadline - 5:
                    break
                if m in current_moduli:
                    continue
                trial_set = current_moduli + [m]
                valid, res = check_validity(trial_set, node_limit=5000)
                if valid:
                    current_moduli.append(m)
                    current_residues = res
                    s = get_score(current_moduli)
                    if s > best_score:
                        best_score = s
                        best_classes = [[current_moduli[i], current_residues[i]] for i in range(len(current_moduli))]
                    improved = True
                    break # Restart loop to check new state
                    
    if not best_classes:
        best_classes = [[20, 0], [22, 1]]
        best_score = Fraction(1, 20) + Fraction(1, 22)

    return best_classes
