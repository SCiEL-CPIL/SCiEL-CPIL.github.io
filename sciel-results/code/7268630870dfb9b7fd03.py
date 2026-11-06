import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Strategy:
    # 1. Try to construct a valid coloring for n=34 using a deterministic algebraic method.
    #    The known lower bound for R(4,5;3) is 34.
    #    We will use the structure of the finite field GF(64) which has 64 elements.
    #    However, we need exactly n vertices.
    #    A common construction for hypergraph Ramsey numbers involves taking a subset of a larger structure
    #    or using specific polynomials.
    #    
    #    Actually, the standard construction for R(4,5;3) > 34 is often derived from the unique (up to isomorphism)
    #    coloring on 34 vertices, or by extending a smaller one.
    #    Let's try a direct algebraic construction on n=34.
    #    
    #    One known method for 3-uniform hypergraphs is to use the "cubic form" over a finite field.
    #    Let V = {0, 1, ..., n-1}.
    #    If we can map vertices to a field F_q such that the color of {a,b,c} depends on f(a,b,c)
    #    where f is a polynomial, we might avoid monochromatic cliques.
    #    
    #    Specifically, for R(4,5;3), the critical number is related to the maximum size of a set
    #    with no 4-set having all red triples and no 5-set having all blue triples.
    #    
    #    Let's try a simpler, robust approach first:
    #    Use the known valid coloring for n=34 if we can hardcode it or generate it via a known formula.
    #    Since I cannot lookup the exact 34-vertex coloring in memory, I will implement a
    #    fast local search with a very good initial heuristic (e.g., based on vertex sums modulo small primes)
    #    and aggressive violation repair.
    #    
    #    Wait, the prompt mentions "inspiration 1" used GF(32) for n=32.
    #    Let's try to construct n=34 directly.
    #    
    #    Alternative Strategy:
    #    Use a randomized local search but with a much faster implementation using NumPy for the
    #    violation counting and delta calculation, allowing many more steps in the time limit.
    #    
    #    However, the "different strategy" directive suggests avoiding simple tuning.
    #    Let's try the algebraic construction on GF(64) restricted to 34 elements?
    #    Or better: Use the "quadratic form" or "cubic form" construction over GF(2^k).
    #    
    #    Let's try a specific construction for n=34:
    #    The vertices are 0..33.
    #    Color triple (a,b,c) as 0 (Red) if (a+b+c) % 2 == 0? No, that creates large monochromatic sets.
    #    
    #    Let's use the following heuristic which is structurally different from pure random:
    #    "Modular Arithmetic Construction"
    #    Color (a,b,c) based on the value of (a*b + b*c + c*a) mod 3 or similar.
    #    
    #    Let's implement a robust local search with NumPy acceleration, as pure Python local search
    #    for n=34 (5984 triples) is likely too slow to find a valid solution from scratch in 110s
    #    without a very good start.
    #    
    #    Actually, n=34 is small enough that a well-optimized local search should work.
    #    The number of triples is C(34,3) = 5984.
    #    The number of 4-sets is C(34,4) = 46376.
    #    The number of 5-sets is C(34,5) = 278256.
    #    
    #    Let's optimize the local search loop heavily.
    
    def solve_n_fast(n, time_limit):
        vertices = list(range(n))
        triples = list(itertools.combinations(vertices, 3))
        num_triples = len(triples)
        triple_idx = {t: i for i, t in enumerate(triples)}
        
        # Precompute quad and pent incidences
        quads = list(itertools.combinations(vertices, 4))
        num_quads = len(quads)
        # For each triple, list of quads it belongs to
        triples_to_quads = [[] for _ in range(num_triples)]
        for i, q in enumerate(quads):
            t0 = (q[0], q[1], q[2])
            t1 = (q[0], q[1], q[3])
            t2 = (q[0], q[2], q[3])
            t3 = (q[1], q[2], q[3])
            triples_to_quads[triple_idx[t0]].append(i)
            triples_to_quads[triple_idx[t1]].append(i)
            triples_to_quads[triple_idx[t2]].append(i)
            triples_to_quads[triple_idx[t3]].append(i)
            
        pents = list(itertools.combinations(vertices, 5))
        num_pents = len(pents)
        triples_to_pents = [[] for _ in range(num_triples)]
        for i, p in enumerate(pents):
            for t in itertools.combinations(p, 3):
                triples_to_pents[triple_idx[t]].append(i)
                
        # Initial coloring: Random
        colors = np.random.randint(0, 2, size=num_triples)
        
        # Initialize counts
        # red_in_4[q] = number of red triples in quad q
        # blue_in_5[p] = number of blue triples in pent p
        red_in_4 = np.zeros(num_quads, dtype=np.int32)
        blue_in_5 = np.zeros(num_pents, dtype=np.int32)
        
        for t in range(num_triples):
            if colors[t] == 0:
                for q_idx in triples_to_quads[t]:
                    red_in_4[q_idx] += 1
            else:
                for p_idx in triples_to_pents[t]:
                    blue_in_5[p_idx] += 1
                    
        # Count violations
        red_k4_count = np.sum(red_in_4 == 4)
        blue_k5_count = np.sum(blue_in_5 == 10)
        current_violations = red_k4_count + blue_k5_count
        
        best_violations = current_violations
        best_colors = colors.copy()
        
        # Local Search
        # To speed up, we iterate through triples in a random order and try to flip them.
        # We only flip if it reduces violations.
        
        t_to_q = triples_to_quads
        t_to_p = triples_to_pents
        r4 = red_in_4
        b5 = blue_in_5
        
        # Precompute lists as lists of ints for faster iteration in Python loop
        # NumPy array indexing in a Python loop is slow.
        # Let's convert to lists of tuples or just use the lists directly.
        
        # Actually, the bottleneck is the Python loop over 5984 triples.
        # Let's try to batch process or just make the loop tight.
        
        # Better idea: Use a priority queue of "bad" triples?
        # Or simply: Randomly pick a triple, calculate delta, accept if negative.
        
        # To make it faster, we can pre-compute the neighbor lists as arrays.
        # But the delta calculation depends on current state.
        
        # Let's stick to a tight Python loop but with local variables for speed.
        
        while time.time() < time_limit:
            # Pick random triple
            t = random.randint(0, num_triples - 1)
            old_c = colors[t]
            
            # Calculate delta
            delta_r4 = 0
            delta_b5 = 0
            
            if old_c == 0: # Red -> Blue
                # Red decreases in quads
                for q_idx in t_to_q[t]:
                    val = r4[q_idx]
                    if val == 4: 
                        delta_r4 -= 1
                    # val becomes val-1. It cannot become 4.
                # Blue increases in pents
                for p_idx in t_to_p[t]:
                    val = b5[p_idx]
                    if val == 10: 
                        delta_b5 -= 1
                    elif val == 9: 
                        delta_b5 += 1 # Becomes 10
            else: # Blue -> Red
                # Red increases in quads
                for q_idx in t_to_q[t]:
                    val = r4[q_idx]
                    if val == 4: 
                        delta_r4 -= 1 # Was 4, now 3? No, Red increases.
                        # Wait. If old_c is Blue, Red count increases.
                        # If val == 3, it becomes 4 -> violation increases.
                        # If val == 4, it becomes 5 -> violation stays (was 1, still 1? No, K4 requires exactly 4 reds? 
                        # "No 4 vertices may have all four triples red". So if it has 4 reds, it's a violation.
                        # If it has 5 reds, it's not a K4^(3) in the sense of "all four". 
                        # Actually, a 4-set has exactly 4 triples. So red_in_4 can only be 0..4.
                        # So if val == 3 and we add red, it becomes 4 -> new violation.
                        # If val == 4 and we add red? Impossible, max is 4.
                        # So for Blue -> Red:
                        #   Red count in quad increases.
                        #   If val == 3, it becomes 4 -> delta_r4 += 1
                        #   If val == 4, it stays 4? No, we are changing one triple from Blue to Red.
                        #   The quad has 4 triples. If one was Blue and becomes Red, Red count increases by 1.
                        #   Max Red count is 4. So if it was 3, it becomes 4. Violation +1.
                        #   If it was 4, it was already a violation. Now it has 4 Reds? 
                        #   Wait, if a quad has 4 Reds, all 4 triples are Red. 
                        #   If we change a Blue to Red, it implies there was a Blue. So it couldn't have been 4 Reds.
                        #   So val must be <= 3.
                        #   So: if val == 3, delta_r4 += 1.
                        
                        pass # Logic correction below
                    # Correct logic for Blue -> Red:
                    # Red count increases by 1.
                    # Violation exists if count == 4.
                    # Before: violation if val == 4.
                    # After: violation if val+1 == 4 => val == 3.
                    # So:
                    #   if val == 4: delta_r4 -= 1 (impossible as argued, but safe)
                    #   if val == 3: delta_r4 += 1
                    #   else: 0
                    # Let's re-verify.
                    # If val == 4, the quad was a violation. We change one triple to Red.
                    # But if val == 4, all 4 are Red. We can't change a Blue to Red if there is no Blue.
                    # So val cannot be 4 when old_c is Blue.
                    # So only val == 3 leads to new violation.
                    pass
                # Blue decreases in pents
                for p_idx in t_to_p[t]:
                    val = b5[p_idx]
                    # Blue count decreases by 1.
                    # Violation if count == 10.
                    # Before: violation if val == 10.
                    # After: violation if val-1 == 10 => val == 11 (impossible, max 10).
                    # So if val == 10, delta_b5 -= 1.
                    # Else 0.
                    pass
            
            # Let's rewrite the delta calculation cleanly.
            
            if old_c == 0: # Red -> Blue
                # Red count decreases.
                # Violation if count == 4.
                # Before: val. After: val-1.
                # If val == 4, violation removed. delta_r4 -= 1.
                # If val-1 == 4 (val=5, impossible).
                # So: for each quad, if r4[q] == 4, delta_r4 -= 1.
                for q_idx in t_to_q[t]:
                    if r4[q_idx] == 4:
                        delta_r4 -= 1
                        
                # Blue count increases.
                # Violation if count == 10.
                # Before: val. After: val+1.
                # If val == 10, violation removed (impossible if max is 10 and we add? 
                # Wait, if val == 10, all 10 are Blue. We change one to Blue? No, Red->Blue.
                # So if val == 10, it was a violation. After change, val becomes 11? 
                # No, a pent has 10 triples. If one was Red and becomes Blue, Blue count increases.
                # If val == 10, all were Blue. So the triple being changed must have been Red?
                # Contradiction: if all 10 are Blue, the specific triple is Blue.
                # So val cannot be 10 when old_c is Red (which means the triple is Red).
                # So val <= 9.
                # If val == 9, it becomes 10 -> new violation. delta_b5 += 1.
                # If val == 10, delta_b5 -= 1 (impossible).
                for p_idx in t_to_p[t]:
                    if b5[p_idx] == 9:
                        delta_b5 += 1
                    # Note: if b5[p_idx] == 10, it's a violation. But we established val < 10.
                    # So no removal of violations for Blue->? 
                    # Wait, if old_c is Red, the triple is Red. 
                    # In the pent, this triple contributes 0 to Blue count.
                    # So Blue count is count of OTHER 9 triples being Blue.
                    # Max Blue count is 9.
                    # So val is in [0,9].
                    # If val == 9, after change it is 10. New violation.
                    
            else: # Blue -> Red
                # Red count increases.
                # Violation if count == 4.
                # Before: val. After: val+1.
                # If val == 3, it becomes 4 -> new violation. delta_r4 += 1.
                # If val == 4, it was a violation. But if val == 4, all 4 are Red.
                # The triple being changed is Blue. Contradiction.
                # So val <= 3.
                for q_idx in t_to_q[t]:
                    if r4[q_idx] == 3:
                        delta_r4 += 1
                        
                # Blue count decreases.
                # Violation if count == 10.
                # Before: val. After: val-1.
                # If val == 10, it was a violation. After change, it is 9. Violation removed.
                # So if val == 10, delta_b5 -= 1.
                for p_idx in t_to_p[t]:
                    if b5[p_idx] == 10:
                        delta_b5 -= 1
            
            delta_total = delta_r4 + delta_b5
            
            if delta_total < 0:
                # Accept
                colors[t] = 1 - old_c
                
                if old_c == 0:
                    for q_idx in t_to_q[t]: r4[q_idx] -= 1
                    for p_idx in t_to_p[t]: b5[p_idx] += 1
                else:
                    for q_idx in t_to_q[t]: r4[q_idx] += 1
                    for p_idx in t_to_p[t]: b5[p_idx] -= 1
                    
                current_violations += delta_total
                
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors.copy()
                    if best_violations == 0:
                        return n, best_colors.tolist(), True
            # Optional: Random walk to escape local minima
            # else:
            #     if random.random() < 0.01:
            #         # Accept anyway? Risky.
            #         pass
            
        if best_violations == 0:
            return n, best_colors.tolist(), True
        else:
            return n, best_colors.tolist(), False

    # Try n=34
    # Allocate 50s for n=34
    sub_deadline = time.time() + 50
    n_sol, colors_sol, valid = solve_n_fast(34, sub_deadline)
    if valid:
        return {'n': n_sol, 'colors': colors_sol}
        
    # Try n=33
    if time.time() < deadline - 10:
        sub_deadline = time.time() + 30
        n_sol, colors_sol, valid = solve_n_fast(33, sub_deadline)
        if valid:
            return {'n': n_sol, 'colors': colors_sol}
            
    # Fallback to a known small valid coloring
    # For n=9, the coloring (a+b+c)%3==0 -> 0 else 1 is not necessarily valid for R(4,5;3).
    # Let's use a random search for a smaller n to ensure validity.
    # Or just return n=1 which is trivially valid?
    # The task says "Maximize n".
    # If we fail to find 34 or 33, we should try 32, 31...
    
    for n in [32, 31, 30]:
        if time.time() > deadline - 5:
            break
        sub_deadline = time.time() + (deadline - time.time()) * 0.8
        n_sol, colors_sol, valid = solve_n_fast(n, sub_deadline)
        if valid:
            return {'n': n_sol, 'colors': colors_sol}
            
    # Last resort: return n=1
    return {'n': 1, 'colors': [0]}

