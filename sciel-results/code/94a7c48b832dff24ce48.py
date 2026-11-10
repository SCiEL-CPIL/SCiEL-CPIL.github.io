"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill
    
    # Precompute values
    N = 512
    # We want to maximize sum of 1/(512 - a) for distinct pairwise coprime a in 1..511
    # Let x = 512 - a, so x in 1..511, and we want to maximize sum 1/x
    # with the constraint that the corresponding a's are pairwise coprime.
    # a = 512 - x, so gcd(512-x, 512-y) = 1 for all pairs.
    
    # Precompute 1/x values
    values = [1.0 / x for x in range(1, 512)]  # values[i] = 1/(i+1) corresponds to x=i+1, a=512-(i+1)=511-i
    
    # Greedy initialization: pick x's in order of largest 1/x (smallest x) 
    # such that corresponding a's are pairwise coprime
    def greedy_init():
        chosen_x = []
        chosen_a = []
        for x in range(1, 512):  # x from 1 to 511, priority by 1/x descending = x ascending
            a = 512 - x
            if all(gcd(a, b) == 1 for b in chosen_a):
                chosen_x.append(x)
                chosen_a.append(a)
        return chosen_x, chosen_a
    
    def score(chosen_x):
        return sum(values[x-1] for x in chosen_x)
    
    # Local search with 2-swap moves
    def local_search(chosen_x, chosen_a, max_iter=10000):
        best_score = score(chosen_x)
        chosen_set = set(chosen_x)
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            improved = False
            # Try swapping one chosen x with one unchosen x
            for i, xi in enumerate(chosen_x):
                ai = 512 - xi
                # Try removing xi and adding some xj not in chosen
                for xj in range(1, 512):
                    if xj in chosen_set:
                        continue
                    aj = 512 - xj
                    # Check if aj is coprime with all other chosen a's (excluding ai)
                    ok = True
                    for k, xk in enumerate(chosen_x):
                        if k == i:
                            continue
                        ak = 512 - xk
                        if gcd(aj, ak) != 1:
                            ok = False
                            break
                    if not ok:
                        continue
                    # Calculate new score
                    new_score = best_score - values[xi-1] + values[xj-1]
                    if new_score > best_score + 1e-15:
                        # Perform swap
                        chosen_x[i] = xj
                        chosen_a[i] = aj
                        chosen_set.remove(xi)
                        chosen_set.add(xj)
                        best_score = new_score
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        return chosen_x, chosen_a, best_score
    
    # Simulated annealing
    def simulated_annealing(chosen_x, chosen_a, start_temp=0.01, end_temp=0.001, steps=5000):
        best_x = list(chosen_x)
        best_a = list(chosen_a)
        best_score = score(chosen_x)
        cur_score = best_score
        chosen_set = set(chosen_x)
        
        for step in range(steps):
            if time.time() > deadline - 5:
                break
            temp = start_temp * (end_temp / start_temp) ** (step / steps)
            
            # Generate a random move: swap a random chosen x with a random unchosen x
            i = random.randint(0, len(chosen_x) - 1)
            xi = chosen_x[i]
            ai = 512 - xi
            
            # Pick a random candidate xj
            candidates = [x for x in range(1, 512) if x not in chosen_set]
            if not candidates:
                break
            xj = random.choice(candidates)
            aj = 512 - xj
            
            # Check coprime constraint with all other chosen
            ok = True
            for k, xk in enumerate(chosen_x):
                if k == i:
                    continue
                ak = 512 - xk
                if gcd(aj, ak) != 1:
                    ok = False
                    break
            if not ok:
                continue
            
            new_score = cur_score - values[xi-1] + values[xj-1]
            delta = new_score - cur_score
            
            if delta > 0 or random.random() < math.exp(delta / temp):
                # Accept
                chosen_x[i] = xj
                chosen_a[i] = aj
                chosen_set.remove(xi)
                chosen_set.add(xj)
                cur_score = new_score
                
                if cur_score > best_score:
                    best_score = cur_score
                    best_x = list(chosen_x)
                    best_a = list(chosen_a)
        
        return best_x, best_a, best_score
    
    # Multi-start: run several initializations and local searches
    best_overall_x = None
    best_overall_a = None
    best_overall_score = -1
    
    # Start with greedy
    gx, ga = greedy_init()
    gs = score(gx)
    if gs > best_overall_score:
        best_overall_score = gs
        best_overall_x = list(gx)
        best_overall_a = list(ga)
    
    # Local search from greedy
    lx, la, ls = local_search(list(gx), list(ga))
    if ls > best_overall_score:
        best_overall_score = ls
        best_overall_x = list(lx)
        best_overall_a = list(la)
    
    # Multi-start with random perturbations
    random.seed(42)
    num_starts = 20
    for s in range(num_starts):
        if time.time() > deadline - 10:
            break
        
        # Perturb the greedy solution
        px = list(gx)
        pa = list(ga)
        
        # Remove a few random elements and add random valid ones
        num_remove = random.randint(1, 3)
        for _ in range(num_remove):
            if not px:
                break
            idx = random.randint(0, len(px) - 1)
            removed_x = px.pop(idx)
            pa.pop(idx)
        
        # Add random valid elements
        for _ in range(num_remove):
            candidates = [x for x in range(1, 512) if x not in set(px)]
            if not candidates:
                break
            xj = random.choice(candidates)
            aj = 512 - xj
            if all(gcd(aj, b) == 1 for b in pa):
                px.append(xj)
                pa.append(aj)
        
        ps = score(px)
        if ps > best_overall_score:
            best_overall_score = ps
            best_overall_x = list(px)
            best_overall_a = list(pa)
        
        # Local search from perturbed
        lx2, la2, ls2 = local_search(px, pa, max_iter=5000)
        if ls2 > best_overall_score:
            best_overall_score = ls2
            best_overall_x = list(lx2)
            best_overall_a = list(la2)
    
    # Simulated annealing from best found
    if time.time() < deadline - 10:
        ax, aa, ascore = simulated_annealing(list(best_overall_x), list(best_overall_a))
        if ascore > best_overall_score:
            best_overall_score = ascore
            best_overall_x = list(ax)
            best_overall_a = list(aa)
    
    # Final local search polish
    if time.time() < deadline - 5:
        fx, fa, fscore = local_search(list(best_overall_x), list(best_overall_a), max_iter=20000)
        if fscore > best_overall_score:
            best_overall_score = fscore
            best_overall_x = list(fx)
            best_overall_a = list(fa)
    
    # Return the a values (not x)
    return best_overall_a

# EVOLVE-BLOCK-END
