"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    moduli = list(range(20, 201))
    
    def is_compatible(m_new, r_new, selected):
        for m_old, r_old in selected:
            g = math.gcd(m_new, m_old)
            if r_new % g == r_old % g:
                return False
        return True
    
    def get_valid_residues(m_new, selected):
        valid = []
        for r in range(m_new):
            if is_compatible(m_new, r, selected):
                valid.append(r)
        return valid
    
    def score_of(selected):
        s = Fraction(0, 1)
        for m, r in selected:
            s += Fraction(1, m)
        return s
    
    def run_greedy_randomized():
        selected = []
        remaining = list(moduli)
        random.shuffle(remaining)
        
        while remaining:
            m = remaining.pop(0)
            valid_r = get_valid_residues(m, selected)
            if valid_r:
                # Choose residue that maximizes future flexibility (fewer conflicts)
                # Simple heuristic: pick random valid residue
                r = random.choice(valid_r)
                selected.append([m, r])
        
        return selected
    
    def simulated_annealing(initial, max_time=30.0):
        """Improve a solution via SA: try swapping residues or removing/adding moduli."""
        current = [list(c) for c in initial]
        current_score = score_of(current)
        best = [list(c) for c in current]
        best_score = current_score
        
        t = 1.0
        cooling = 0.999
        sa_start = time.time()
        
        while t > 0.01 and (time.time() - sa_start) < max_time:
            if time.time() > deadline - 5.0:
                break
            
            # Pick a random move
            move_type = random.random()
            
            if move_type < 0.5 and len(current) > 0:
                # Change residue of a random class
                idx = random.randint(0, len(current) - 1)
                m = current[idx][0]
                old_r = current[idx][1]
                new_r = random.randint(0, m - 1)
                if new_r == old_r:
                    continue
                # Check compatibility with all others
                compatible = True
                for i, (m2, r2) in enumerate(current):
                    if i == idx:
                        continue
                    g = math.gcd(m, m2)
                    if new_r % g == r2 % g:
                        compatible = False
                        break
                if compatible:
                    delta = 0  # Score doesn't change with residue
                    current[idx][1] = new_r
                    if random.random() < math.exp(delta / t):
                        pass  # Always accept since delta=0
                    else:
                        current[idx][1] = old_r
                else:
                    pass
            
            elif move_type < 0.7 and len(current) > 1:
                # Remove a random class
                idx = random.randint(0, len(current) - 1)
                removed = current.pop(idx)
                new_score = score_of(current)
                delta = new_score - current_score
                if delta >= 0 or random.random() < math.exp(delta / t):
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best = [list(c) for c in current]
                else:
                    current.insert(idx, removed)
            
            elif move_type < 0.9:
                # Try adding a random modulus
                m = random.choice(moduli)
                if any(c[0] == m for c in current):
                    continue
                valid_r = get_valid_residues(m, current)
                if valid_r:
                    r = random.choice(valid_r)
                    current.append([m, r])
                    new_score = score_of(current)
                    delta = new_score - current_score
                    if delta >= 0 or random.random() < math.exp(delta / t):
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best = [list(c) for c in current]
                    else:
                        current.pop()
            
            t *= cooling
        
        return best, best_score
    
    # Phase 1: Multiple greedy randomized restarts
    best_classes = []
    best_score = Fraction(0, 1)
    
    num_restarts = 0
    while time.time() < deadline - 60.0:
        candidate = run_greedy_randomized()
        s = score_of(candidate)
        if s > best_score:
            best_score = s
            best_classes = candidate
        num_restarts += 1
    
    # Phase 2: Simulated annealing on best and some random candidates
    sa_candidates = [best_classes]
    for _ in range(min(5, num_restarts)):
        c = run_greedy_randomized()
        sa_candidates.append(c)
    
    for cand in sa_candidates:
        if time.time() > deadline - 10.0:
            break
        improved, improved_score = simulated_annealing(cand, max_time=15.0)
        if improved_score > best_score:
            best_score = improved_score
            best_classes = improved
    
    return best_classes
# EVOLVE-BLOCK-END
