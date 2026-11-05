"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Moduli in [20, 200]
    moduli = list(range(20, 201))
    
    def is_compatible(m_new, r_new, selected_list):
        """Check if (m_new, r_new) is disjoint from all in selected_list."""
        for m_old, r_old in selected_list:
            g = math.gcd(m_new, m_old)
            if (r_new - r_old) % g == 0:
                return False
        return True
    
    def score_solution(selected):
        """Calculate sum of 1/m for the selected classes."""
        return sum(1.0 / m for m, r in selected)
    
    # Strategy: Use a greedy approach with backtracking/local search to find a better solution.
    # 1. Initial greedy pass: pick smallest moduli, first valid residue.
    # 2. Local search: try to swap or add moduli to improve the score.
    #    Since adding a small modulus is valuable, we prioritize adding small ones.
    #    Removing a large modulus to add a small one is beneficial if 1/small > 1/large.
    
    def greedy_initial():
        selected = []
        for m in moduli:
            if time.time() > deadline:
                break
            for r in range(m):
                if is_compatible(m, r, selected):
                    selected.append([m, r])
                    break
        return selected
    
    def local_search(selected):
        """
        Attempt to improve the solution by:
        1. Trying to add any modulus that was skipped.
        2. Trying to swap a selected modulus with a better (smaller) one if possible.
        """
        selected = list(selected)
        current_score = score_solution(selected)
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            selected_set = set(m for m, r in selected)
            
            # 1. Try to add any missing modulus
            for m in moduli:
                if m in selected_set:
                    continue
                if time.time() > deadline:
                    break
                for r in range(m):
                    if is_compatible(m, r, selected):
                        new_selected = selected + [[m, r]]
                        new_score = score_solution(new_selected)
                        if new_score > current_score:
                            selected = new_selected
                            current_score = new_score
                            improved = True
                            selected_set = set(m2 for m2, r2 in selected)
                        break  # Only need to check first valid r for this m if adding
            
            # 2. Try to swap: remove a large m, try to add a smaller m
            # Sort selected by m descending to consider removing larger ones first
            # But we want to add smaller ones, so we iterate target additions in ascending order
            # and check if we can remove a compatible larger modulus.
            
            # A more efficient swap: 
            # For each m_add in moduli (ascending) not in selected:
            #   For each m_rem in selected (descending, to remove least valuable first? No, we want to gain value)
            #   Actually, we just need to find ANY valid configuration with higher score.
            #   Since 1/m is convex decreasing, replacing a large m with a small m is good.
            
            # Let's try a simpler swap: 
            # Remove one selected modulus, try to add any missing modulus that fits.
            # This is O(N^2 * K) which might be slow.
            # Given N=181, N^2 ~ 32k. K (residues) up to 200. 32k * 200 * 181 (compat check) ~ 1.1 billion ops. Too slow in Python.
            
            # Optimization: Precompute compatibility?
            # Or just rely on the "add" step which is O(N * K * N_sel).
            # If we can't add anything, we are stuck in a local minimum.
            
            # Let's try a random restart strategy instead of complex local search if stuck.
            if not improved:
                break
        
        return selected
    
    # Run initial greedy
    best_solution = greedy_initial()
    best_score = score_solution(best_solution)
    
    # Try local search improvements
    if time.time() < deadline - 5.0:
        improved_solution = local_search(best_solution)
        improved_score = score_solution(improved_solution)
        if improved_score > best_score:
            best_solution = improved_solution
            best_score = improved_score
    
    # If we have time, try a few random restarts with different residue choices
    # The greedy choice of "first valid residue" might be suboptimal.
    # Choosing a residue that leaves more room for future small moduli is better.
    # But for now, let's see if simple local search helps.
    
    return best_solution
# EVOLVE-BLOCK-END
