import math
import time
import random
import heapq
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Precompute gcd table for 20..200
    max_m = 200
    gcd_table = [[0] * (max_m + 1) for _ in range(max_m + 1)]
    for i in range(20, max_m + 1):
        for j in range(i, max_m + 1):
            gcd_table[i][j] = math.gcd(i, j)
    
    def get_gcd(m1, m2):
        return gcd_table[m1][m2]
    
    def is_valid(selected, m, r):
        """
        Check if (m, r) is disjoint from all (m_j, r_j) in selected.
        """
        for m_j, r_j in selected:
            g = get_gcd(m, m_j)
            if r % g == r_j % g:
                return False
        return True
    
    def greedy_deterministic():
        """
        Greedy: process moduli in increasing order, pick smallest valid residue.
        """
        selected = []
        for m in range(20, max_m + 1):
            for r in range(m):
                if is_valid(selected, m, r):
                    selected.append([m, r])
                    break
        return selected
    
    def greedy_random(seed):
        """
        Randomized greedy: shuffle residue order for each modulus.
        """
        rng = random.Random(seed)
        selected = []
        for m in range(20, max_m + 1):
            rs = list(range(m))
            rng.shuffle(rs)
            for r in rs:
                if is_valid(selected, m, r):
                    selected.append([m, r])
                    break
        return selected
    
    def score_solution(selected):
        """
        Compute the exact rational score and return as float.
        """
        total = Fraction(0, 1)
        for m, r in selected:
            total += Fraction(1, m)
        return float(total)
    
    def local_search(selected, max_iters=5000):
        """
        Local search: try to improve by replacing one element or adding a new one.
        """
        selected = [list(x) for x in selected]
        current_score = score_solution(selected)
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iters:
            if time.time() > deadline - 10.0:
                break
            improved = False
            iter_count += 1
            
            # Try removing each element and re-filling
            for idx in range(len(selected)):
                if time.time() > deadline - 10.0:
                    return selected, current_score
                    
                m_to_remove, r_to_remove = selected[idx]
                others = selected[:idx] + selected[idx+1:]
                
                # Try to add back any unselected modulus with a valid residue
                m_set = {m for m, r in others}
                new_selected = list(others)
                new_score = current_score - Fraction(1, m_to_remove)
                
                # Try adding all possible moduli not in new_selected
                added = False
                for m_add in range(20, max_m + 1):
                    if m_add in m_set:
                        continue
                    for r_add in range(m_add):
                        if is_valid(new_selected, m_add, r_add):
                            new_selected.append([m_add, r_add])
                            new_score += Fraction(1, m_add)
                            added = True
                            break
                
                new_score_f = float(new_score)
                if new_score_f > current_score + 1e-12:
                    selected = new_selected
                    current_score = new_score_f
                    improved = True
                    break
        
        return selected, current_score
    
    def branch_and_bound():
        """
        Branch and bound on the conflict graph.
        Vertices: all valid (m, r) pairs.
        Edge: conflict between two pairs.
        We want maximum weight independent set.
        Weight of (m, r) is 1/m.
        
        This is expensive, so we use a heuristic-based B&B.
        """
        # Build list of all candidate (m, r) pairs
        candidates = []
        for m in range(20, max_m + 1):
            for r in range(m):
                candidates.append((m, r, Fraction(1, m)))
        
        # Build conflict matrix (sparse)
        # For memory, we use a dict of sets
        # But with ~18000 candidates, this might be large. Let's try.
        # Actually, let's first filter: for each m, keep only the best few residues?
        # No, let's just do a simpler B&B.
        
        # Simpler approach: use a priority queue based on upper bound.
        # Upper bound for a set S: current_weight + sum of weights of all candidates 
        # that are compatible with all in S.
        
        # This is complex. Let's use a different strategy: 
        # Start with the greedy solution and do a "swap" based local search with a larger neighborhood.
        pass
    
    def swap_local_search(selected):
        """
        More aggressive local search: try swapping a residue for a different one, 
        or replacing a pair with two pairs.
        """
        selected = [list(x) for x in selected]
        current_score = score_solution(selected)
        improved = True
        
        while improved:
            if time.time() > deadline - 15.0:
                break
            improved = False
            
            # Try changing the residue of each selected pair
            for idx in range(len(selected)):
                m, r = selected[idx]
                others = selected[:idx] + selected[idx+1:]
                
                # Try all residues for m
                best_r = None
                best_score = None
                for r_new in range(m):
                    if r_new == r:
                        continue
                    if is_valid(others, m, r_new):
                        # Check if this enables adding more pairs
                        # For now, just accept if it doesn't reduce score (which it doesn't)
                        # The score change is 0, but it might enable future additions.
                        # So we need to check if we can add more.
                        # This is expensive. Let's skip for now.
                        pass
                
                # Try replacing (m, r) with two pairs (m1, r1) and (m2, r2) where m1, m2 != m
                # This is also expensive.
                
                # Alternative: try adding a pair that was previously impossible
                # because of a conflict, after removing a different pair.
                # This is what the previous local_search does.
        
        return selected, current_score
    
    # Strategy:
    # 1. Run deterministic greedy.
    # 2. Run several randomized greedy passes.
    # 3. Apply local search to the best of these.
    # 4. If time permits, try a more sophisticated search.
    
    best_sol = []
    best_score = 0.0
    
    # 1. Deterministic greedy
    sol = greedy_deterministic()
    s = score_solution(sol)
    if s > best_score:
        best_score = s
        best_sol = sol
    
    # 2. Randomized greedy
    num_random = 50
    for i in range(num_random):
        if time.time() > deadline - 30.0:
            break
        sol = greedy_random(i)
        s = score_solution(sol)
        if s > best_score:
            best_score = s
            best_sol = sol
    
    # 3. Local search on best solution
    if time.time() < deadline - 20.0:
        improved_sol, improved_score = local_search(best_sol, max_iters=1000)
        if improved_score > best_score:
            best_score = improved_score
            best_sol = improved_sol
    
    # 4. Try local search on a few other top solutions
    # We didn't store them, so let's just do another LS on a fresh random greedy if time allows.
    if time.time() < deadline - 10.0:
        sol = greedy_random(999)
        s = score_solution(sol)
        if s > best_score - 0.1:  # If it's competitive
            improved_sol, improved_score = local_search(sol, max_iters=500)
            if improved_score > best_score:
                best_score = improved_score
                best_sol = improved_sol
    
    return [[m, r] for m, r in best_sol]
