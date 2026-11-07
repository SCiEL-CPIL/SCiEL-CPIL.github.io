import time
import random
import math

def construct():
    n = 128
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Precompute all "bad triple" patterns: (a,b,c) where a<b<c and a+b,a+c,b+c all in 1..n
    # We'll use these for fast constraint checking
    # A triple is "active" if all 6 values (a,b,c,a+b,a+c,b+c) are in A
    # Actually we only need a,b,c,a+b,a+c,b+c all in A
    
    # For Lagrangian relaxation:
    # Objective: maximize sum(x_i) where x_i in {0,1}
    # Constraint: for each bad triple pattern T=(a,b,c), x_a*x_b*x_c*x_{a+b}*x_{a+c}*x_{b+c} <= 0
    # Lagrangian: L(x, lambda) = sum(x_i) - sum_T lambda_T * x_a*x_b*x_c*x_{a+b}*x_{a+c}*x_{b+c}
    # Primal heuristic: greedy add with penalty from dual variables
    # Dual update: subgradient = constraint violation
    
    # First, generate a good initial solution using the two-layer approach
    # Upper block: [64, 128] - sums of two elements >= 128, so no triple can have all sums in set
    # Lower block: odd numbers in [1, 31] - sum of two odds is even, not in set
    # This gives 64 + 16 = 80 elements, and it's valid
    
    A = set(range(64, 129))  # 64 elements
    for x in range(1, 32, 2):  # odd numbers 1,3,5,...,31
        A.add(x)  # 16 more elements
    # Total: 80 elements, valid base
    
    def count_bad_triples(A):
        """Fast count of bad triples in A"""
        count = 0
        elems = sorted(A)
        for i in range(len(elems)):
            a = elems[i]
            for j in range(i+1, len(elems)):
                b = elems[j]
                ab = a + b
                if ab > n: break
                if ab not in A: continue
                for k in range(j+1, len(elems)):
                    c = elems[k]
                    bc = b + c
                    if bc > n: break
                    ac = a + c
                    if ac in A and bc in A:
                        count += 1
        return count
    
    def get_bad_triples(A):
        """Get all bad triples in A"""
        bads = []
        elems = sorted(A)
        for i in range(len(elems)):
            a = elems[i]
            for j in range(i+1, len(elems)):
                b = elems[j]
                ab = a + b
                if ab > n: break
                if ab not in A: continue
                for k in range(j+1, len(elems)):
                    c = elems[k]
                    bc = b + c
                    if bc > n: break
                    ac = a + c
                    if ac in A and bc in A:
                        bads.append((a, b, c))
        return bads
    
    # Lagrangian relaxation with subgradient method
    # Dual variables: one per potential bad triple pattern
    # We'll use a sparse representation: lambda keyed by (a,b,c)
    
    # Initialize dual variables to zero
    lam = {}  # (a,b,c) -> lambda value
    
    # Precompute all possible bad triple patterns
    all_patterns = []
    for a in range(1, n+1):
        for b in range(a+1, n+1):
            ab = a + b
            if ab > n: break
            for c in range(b+1, n+1):
                bc = b + c
                if bc > n: break
                ac = a + c
                if ac <= n:
                    all_patterns.append((a, b, c))
    
    # Greedy primal heuristic with Lagrangian penalty
    def greedy_primal(A_current, lam_dict, max_steps=5000):
        """Greedy add elements to A_current, using Lagrangian penalties"""
        A_work = set(A_current)
        candidates = [x for x in range(1, n+1) if x not in A_work]
        
        for _ in range(max_steps):
            if time.time() > deadline - 5:
                break
            if not candidates:
                break
            
            # For each candidate, compute marginal gain minus penalty
            best_candidate = None
            best_score = -1
            
            for x in candidates:
                # Try adding x
                A_test = A_work | {x}
                
                # Count new bad triples created by adding x
                # A bad triple involves x if x is one of {a,b,c,a+b,a+c,b+c}
                new_bads = 0
                for (a,b,c) in all_patterns:
                    vals = (a, b, c, a+b, a+c, b+c)
                    # Check if x is in vals and all other vals are in A_work
                    if x in vals:
                        others = [v for v in vals if v != x]
                        if all(v in A_work for v in others):
                            new_bads += 1
                
                # Penalty from Lagrangian
                penalty = 0.0
                for (a,b,c), lval in lam_dict.items():
                    vals = (a, b, c, a+b, a+c, b+c)
                    if x in vals:
                        others = [v for v in vals if v != x]
                        if all(v in A_work for v in others):
                            penalty += lval
                
                score = 1.0 - penalty - new_bads * 0.1  # small penalty for creating new bads
                
                if score > best_score:
                    best_score = score
                    best_candidate = x
            
            if best_candidate is None or best_score <= 0:
                break
            
            A_work.add(best_candidate)
            candidates.remove(best_candidate)
        
        return A_work
    
    # Subgradient dual update
    def subgradient_update(A_current, lam_dict, step_size):
        """Update dual variables using subgradient method"""
        bads = get_bad_triples(A_current)
        if not bads:
            return lam_dict, 0.0
        
        # Subgradient: for each violated constraint, increase lambda
        new_lam = dict(lam_dict)
        total_violation = 0.0
        for (a,b,c) in bads:
            new_lam[(a,b,c)] = new_lam.get((a,b,c), 0.0) + step_size
            total_violation += 1.0
        
        return new_lam, total_violation
    
    # Main optimization loop
    best_A = set(A)
    best_size = len(A)
    
    # Phase 1: Lagrangian relaxation loop
    step_size = 0.1
    for iteration in range(100):
        if time.time() > deadline - 10:
            break
        
        # Primal: greedy add
        A_new = greedy_primal(A, lam, max_steps=1000)
        
        # If better, update
        if len(A_new) > best_size:
            best_A = set(A_new)
            best_size = len(A_new)
            A = set(A_new)
        
        # Dual: subgradient update
        lam, violation = subgradient_update(A, lam, step_size)
        
        # Decrease step size
        step_size *= 0.95
        
        # If no violation, we're done
        if violation == 0:
            break
    
    # Phase 2: Local search polish on best solution
    A_work = set(best_A)
    
    # Try adding elements
    candidates = [x for x in range(1, n+1) if x not in A_work]
    random.shuffle(candidates)
    
    for x in candidates:
        if time.time() > deadline - 3:
            break
        A_test = A_work | {x}
        if count_bad_triples(A_test) == 0:
            A_work.add(x)
    
    # Try removing one and adding two
    for _ in range(200):
        if time.time() > deadline - 2:
            break
        elems = list(A_work)
        if len(elems) < 2:
            break
        y1, y2 = random.sample(elems, 2)
        A_test = A_work - {y1, y2}
        
        # Try to add two new elements
        added = 0
        for x in range(1, n+1):
            if x not in A_test and x != y1 and x != y2:
                A_try = A_test | {x}
                if count_bad_triples(A_try) == 0:
                    A_test.add(x)
                    added += 1
                    if added == 2:
                        break
        
        if len(A_test) > len(A_work):
            A_work = A_test
    
    # Final validity check
    if count_bad_triples(A_work) == 0 and len(A_work) > len(best_A):
        best_A = A_work
    
    return list(best_A)
