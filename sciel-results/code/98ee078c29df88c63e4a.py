import time
import random
import math
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n = 20
    max_val = 10000
    
    # Fast evaluation using pure Python sets
    # For n=20, there are 20*21/2 = 210 pairs.
    # Computing sums and products for 210 pairs is very fast in Python.
    def evaluate(A):
        """
        Compute max(|A+A|, |A*A|) for set A.
        A is a list of distinct integers.
        """
        sums = set()
        prods = set()
        # We can optimize by only iterating i <= j
        for i in range(len(A)):
            ai = A[i]
            for j in range(i, len(A)):
                aj = A[j]
                sums.add(ai + aj)
                prods.add(ai * aj)
        return max(len(sums), len(prods))
    
    # Strategy:
    # 1. Generate a diverse pool of candidates.
    #    Instead of just smooth numbers, we include:
    #    - All integers 1..100 (dense small numbers for small sumsets)
    #    - 5-smooth numbers (good product structure)
    #    - Powers of 2 (very sparse products, but large sums)
    #    - Random integers to fill gaps.
    
    candidates = set()
    # 1. Small integers
    for i in range(1, 101):
        candidates.add(i)
        
    # 2. 5-smooth numbers
    for a in range(0, 14):
        v2 = 1 << a
        if v2 > max_val: break
        for b in range(0, 9):
            v3 = v2 * (3 ** b)
            if v3 > max_val: break
            for c in range(0, 6):
                v5 = v3 * (5 ** c)
                if v5 > max_val: break
                candidates.add(v5)
                
    # 3. Powers of 2 and 3
    for i in range(1, 14):
        candidates.add(2 ** i)
        candidates.add(3 ** i)
        
    # 4. Some random numbers to break structural bias
    for _ in range(200):
        candidates.add(random.randint(1, max_val))
        
    candidate_list = sorted(list(candidates))
    num_candidates = len(candidate_list)
    
    # Initial Construction: Greedy with Lookahead
    # Start with a small set, then add elements that minimize the increase in objective.
    # Objective: max(|S|, |P|)
    # To balance, we can use a weighted sum or just minimize the max.
    # Let's try to keep |S| and |P| close to each other to minimize the max.
    
    A = []
    used = set()
    
    # Start with 1. It minimizes product set size (1*x=x) and sum set (1+x).
    A.append(1)
    used.add(1)
    
    # We will add 19 more elements.
    # At each step, evaluate all candidates not in A.
    # To speed up, we can limit the candidates considered to those "close" to the current set in some metric,
    # but for the first few steps, we scan all.
    
    for step in range(n - 1):
        best_cand = None
        best_score = float('inf')
        best_sum_size = 0
        best_prod_size = 0
        
        # If the set is getting large, scanning all ~500 candidates might be slow.
        # 500 candidates * evaluate (210 pairs) ~ 100k ops per step.
        # 20 steps * 100k = 2M ops. Very fast.
        
        for c in candidate_list:
            if c in used:
                continue
            
            # Tentative addition
            # We can optimize evaluation by only computing new sums/prods involving c.
            # But full evaluation is simpler and fast enough for n=20.
            
            # To be safe and fast, let's just append and evaluate.
            A.append(c)
            score = evaluate(A)
            
            # Heuristic: We prefer sets where sum and product sizes are balanced.
            # If max is driven by one set, try to reduce that one.
            # Simple min-max is fine, but adding a small penalty for imbalance might help.
            # Let's stick to minimizing the max, but tie-break by sum of sizes.
            
            if score < best_score:
                best_score = score
                best_cand = c
                # Store sizes for tie-breaking if needed
                # Recalculating sizes is expensive, so let's just take the first best found.
                # Or, we can store the actual sizes.
                # For now, simple min-max.
            
            A.pop()
            
        if best_cand is None:
            # Fallback: pick a random unused number
            for c in candidate_list:
                if c not in used:
                    best_cand = c
                    break
        
        A.append(best_cand)
        used.add(best_cand)
        
        # Sort A to keep it canonical
        A.sort()
        
        if time.time() > deadline - 10:
            break
            
    # Local Search: Hill Climbing
    # Try swapping each element in A with a candidate from the pool.
    # To make this fast, we pre-compute "neighborhoods" or just scan the pool.
    # Scanning the whole pool (~500 items) for each of 20 slots = 10k evaluations per pass.
    # Each evaluation is ~210 pairs. 10k * 210 = 2.1M ops per pass.
    # We can do ~100 passes in 1s? Probably less due to Python overhead.
    # Let's estimate: 10k evals * 10us/eval = 0.1s per pass.
    # So ~1000 passes in 100s. Plenty of time.
    
    current_score = evaluate(A)
    best_set = A[:]
    best_score = current_score
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # Shuffle candidates to avoid bias? Or just iterate.
        # Let's iterate through A indices, and for each, try swapping with candidates.
        
        for i in range(n):
            old_val = A[i]
            
            # Try swapping with candidates
            # We can limit candidates to those that are "similar" in magnitude or structure?
            # For now, scan all candidates.
            for new_val in candidate_list:
                if new_val in used:
                    continue
                if new_val == old_val:
                    continue
                
                # Swap
                A[i] = new_val
                used.remove(old_val)
                used.add(new_val)
                
                new_score = evaluate(A)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_set = A[:]
                else:
                    # Revert
                    A[i] = old_val
                    used.remove(new_val)
                    used.add(old_val)
                    
        # If no improvement, try a random kick to escape local minima
        if not improved and time.time() < deadline - 10:
            # Pick 2 random indices and swap their values with random candidates
            for _ in range(2):
                idx = random.randint(0, n - 1)
                old_val = A[idx]
                new_val = random.choice(candidate_list)
                
                if new_val in used:
                    continue
                if new_val == old_val:
                    continue
                    
                A[idx] = new_val
                used.remove(old_val)
                used.add(new_val)
                current_score = evaluate(A)
                # Don't revert immediately, let the next hill climbing pass try to fix it
                # This is a "kick"
                break # Only one kick per loop iteration to keep control
                
    return sorted(best_set)

