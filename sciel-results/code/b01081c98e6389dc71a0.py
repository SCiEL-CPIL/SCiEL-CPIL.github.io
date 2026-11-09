"""Construct a set of 20 distinct positive integers minimizing max(|A+A|, |A*A|)."""
import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START

def _objective(A):
    """
    Compute max(|A+A|, |A*A|) using set operations.
    A is a list of 20 distinct positive integers.
    """
    sums = set()
    prods = set()
    for i in range(20):
        ai = A[i]
        for j in range(i, 20):
            aj = A[j]
            sums.add(ai + aj)
            prods.add(ai * aj)
    return max(len(sums), len(prods))

def _is_valid(A):
    """Check 20 distinct positive integers <= 10000."""
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    if any(x < 1 or x > 10000 for x in A):
        return False
    return True

def _generate_structured_candidates():
    """
    Generate initial candidates from structured sets to exploit collisions.
    Uses union of arithmetic and geometric progressions, plus other structured families.
    """
    candidates = []
    seen = set()
    
    def add_candidate(lst):
        if len(lst) != 20 or len(set(lst)) != 20:
            return
        if any(x < 1 or x > 10000 for x in lst):
            return
        key = tuple(sorted(lst))
        if key not in seen:
            seen.add(key)
            candidates.append(sorted(lst))
    
    # Strategy 1: Union of arithmetic and geometric progressions
    for n1 in range(4, 17):
        n2 = 20 - n1
        for a in range(1, 15):
            for d in range(1, 8):
                arith = [a + i*d for i in range(n1)]
                if max(arith) > 10000:
                    continue
                for b in range(1, 15):
                    for r in [2, 3, 4, 5, 6, 7, 8, 10]:
                        geo = []
                        val = b
                        for j in range(n2):
                            if val > 10000:
                                break
                            geo.append(val)
                            val *= r
                        if len(geo) != n2:
                            continue
                        add_candidate(arith + geo)
    
    # Strategy 2: Pure geometric
    for a in range(1, 10):
        for r in range(2, 15):
            seq = []
            val = a
            for _ in range(20):
                if val > 10000:
                    break
                seq.append(val)
                val *= r
            if len(seq) == 20:
                add_candidate(seq)
    
    # Strategy 3: Pure arithmetic
    for a in range(1, 20):
        for d in range(1, 15):
            seq = [a + i*d for i in range(20)]
            if max(seq) <= 10000:
                add_candidate(seq)
    
    # Strategy 4: Powers of small bases (mixed)
    for base1 in [2, 3, 4, 5]:
        for base2 in [2, 3, 4, 5, 6, 7]:
            if base1 == base2:
                continue
            s1 = set()
            val = 1
            for _ in range(20):
                if val > 10000:
                    break
                s1.add(val)
                val *= base1
            s2 = set()
            val = 1
            for _ in range(20):
                if val > 10000:
                    break
                s2.add(val)
                val *= base2
            # Try to pick 20 from union
            union = sorted(s1 | s2)
            if len(union) >= 20:
                # Take first 20
                add_candidate(union[:20])
                # Also try mixing
                from itertools import combinations
                # Just take a few combinations
                for start in range(0, max(1, len(union) - 20), 3):
                    add_candidate(union[start:start+20])
    
    # Strategy 5: Quadratic sequences
    for a in range(1, 10):
        for b in range(1, 10):
            for c in range(1, 10):
                seq = [a*i*i + b*i + c for i in range(20)]
                if max(seq) <= 10000 and min(seq) >= 1:
                    add_candidate(seq)
    
    # Strategy 6: Random structured
    for _ in range(50):
        n1 = random.randint(4, 16)
        n2 = 20 - n1
        a = random.randint(1, 20)
        d = random.randint(1, 10)
        arith = [a + i*d for i in range(n1)]
        b = random.randint(1, 20)
        r = random.choice([2, 3, 4, 5, 6])
        geo = []
        val = b
        for j in range(n2):
            if val > 10000:
                break
            geo.append(val)
            val *= r
        if len(geo) == n2:
            add_candidate(arith + geo)
    
    return candidates

def _nudge_move(A):
    """Nudge one element by a small amount."""
    idx = random.randrange(20)
    delta = random.choice([-5, -3, -2, -1, 1, 2, 3, 5, 10, -10])
    new_val = A[idx] + delta
    if new_val < 1 or new_val > 10000:
        return None
    if new_val in A:
        return None
    A[idx] = new_val
    return sorted(A)

def _swap_move(A):
    """Swap one element with a random new value, biased towards small numbers."""
    idx = random.randrange(20)
    old = A[idx]
    while True:
        if random.random() < 0.7:
            new = random.randint(1, 50)
        else:
            new = random.randint(1, 10000)
        if new != old and new not in A:
            break
    A[idx] = new
    return sorted(A)

def _anneal(A, start_temp, end_temp, steps, deadline):
    """Simulated annealing with nudge and swap moves."""
    current = list(A)
    current_cost = _objective(current)
    best = list(current)
    best_cost = current_cost
    temp = start_temp
    
    for step in range(steps):
        if time.time() > deadline:
            break
        r = random.random()
        if r < 0.6:
            new_A = _nudge_move(list(current))
            if new_A is None:
                continue
        else:
            new_A = _swap_move(list(current))
        
        new_cost = _objective(new_A)
        delta = new_cost - current_cost
        
        if delta < 0:
            accept = True
        else:
            if temp > 1e-9:
                prob = math.exp(-delta / temp)
            else:
                prob = 0.0
            accept = random.random() < prob
            
        if accept:
            current = new_A
            current_cost = new_cost
            if new_cost < best_cost:
                best = list(current)
                best_cost = new_cost
        
        temp = start_temp * (end_temp / start_temp) ** (step / max(steps - 1, 1))
    
    return best, best_cost

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    random.seed(42)
    
    # Phase 1: Generate structured initial solutions
    best_set = None
    best_cost = float('inf')
    
    candidates = _generate_structured_candidates()
    for A in candidates:
        if time.time() > deadline - 5:
            break
        if not _is_valid(A):
            continue
        cost = _objective(A)
        if cost < best_cost:
            best_cost = cost
            best_set = list(A)
            
    if best_set is None:
        best_set = list(range(1, 21))
        best_cost = _objective(best_set)

    # Phase 2: Multiple simulated annealing runs
    remaining = deadline - time.time()
    if remaining > 5:
        # Run annealing with multiple restarts
        num_restarts = min(5, int(remaining / 5))
        for restart in range(num_restarts):
            if time.time() > deadline - 2:
                break
            if restart == 0:
                start_A = best_set
            else:
                # Perturb best solution
                start_A = list(best_set)
                for _ in range(random.randint(2, 5)):
                    idx = random.randrange(20)
                    old = start_A[idx]
                    while True:
                        new_val = random.randint(1, 10000)
                        if new_val != old and new_val not in start_A:
                            break
                    start_A[idx] = new_val
                start_A = sorted(start_A)
            
            start_temp = 15.0
            end_temp = 0.01
            steps = 5000
            result, cost = _anneal(start_A, start_temp, end_temp, steps, deadline)
            if cost < best_cost:
                best_cost = cost
                best_set = list(result)
        
        # Phase 3: Local refinement (Hill Climbing) with full single-element replacement
        if time.time() < deadline - 3:
            current = list(best_set)
            current_cost = best_cost
            improved = True
            iter_count = 0
            while improved and time.time() < deadline - 2 and iter_count < 200:
                improved = False
                iter_count += 1
                indices = list(range(20))
                random.shuffle(indices)
                for idx in indices:
                    if time.time() > deadline - 2:
                        break
                    old = current[idx]
                    # Try a range of replacements
                    tried = set()
                    # Small nudges
                    for delta in [-20, -10, -5, -3, -2, -1, 1, 2, 3, 5, 10, 20, 50, -50]:
                        new_val = old + delta
                        if new_val < 1 or new_val > 10000 or new_val in current or new_val in tried:
                            continue
                        tried.add(new_val)
                        current[idx] = new_val
                        new_cost = _objective(current)
                        if new_cost < current_cost:
                            current_cost = new_cost
                            improved = True
                            if new_cost < best_cost:
                                best_cost = new_cost
                                best_set = list(current)
                            break
                        else:
                            current[idx] = old
                    # Random replacement
                    if not improved:
                        for _ in range(5):
                            new_val = random.randint(1, 10000)
                            if new_val in current or new_val in tried:
                                continue
                            tried.add(new_val)
                            current[idx] = new_val
                            new_cost = _objective(current)
                            if new_cost < current_cost:
                                current_cost = new_cost
                                improved = True
                                if new_cost < best_cost:
                                    best_cost = new_cost
                                    best_set = list(current)
                                break
                            else:
                                current[idx] = old

    return best_set

# EVOLVE-BLOCK-END
