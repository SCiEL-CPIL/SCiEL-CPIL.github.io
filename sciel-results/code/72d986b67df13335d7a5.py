import math
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    def is_pairwise_disjoint(classes):
        n = len(classes)
        for i in range(n):
            m1, r1 = classes[i]
            for j in range(i + 1, n):
                m2, r2 = classes[j]
                g = math.gcd(m1, m2)
                if r1 % g == r2 % g:
                    return False
        return True

    def get_conflict_graph(classes):
        # Returns a set of pairs (i, j) that conflict
        conflicts = set()
        n = len(classes)
        for i in range(n):
            m1, r1 = classes[i]
            for j in range(i + 1, n):
                m2, r2 = classes[j]
                g = math.gcd(m1, m2)
                if r1 % g == r2 % g:
                    conflicts.add((i, j))
        return conflicts

    def find_valid_residue(m, existing_classes):
        for r in range(m):
            valid = True
            for m_i, r_i in existing_classes:
                g = math.gcd(m, m_i)
                if r % g == r_i % g:
                    valid = False
                    break
            if valid:
                return r
        return -1

    def greedy_construct(order):
        current_classes = []
        for m in order:
            r = find_valid_residue(m, current_classes)
            if r != -1:
                current_classes.append([m, r])
        return current_classes

    def local_search(classes, time_limit):
        # Local search: try to swap residues or add/remove moduli
        # Since the score depends only on moduli, the main move is adding a new modulus
        # or swapping a modulus for a smaller one if it fits.
        # However, adding a modulus requires finding a valid residue.
        # If we remove a modulus, we might be able to add two smaller ones? Unlikely due to 1/m decay.
        # So we focus on adding moduli.
        
        current_score = sum(1.0 / m for m, _ in classes)
        best_classes = classes
        best_score = current_score
        
        # Try to add any modulus not in the set
        used_moduli = {m for m, _ in classes}
        candidates = [m for m in range(20, 201) if m not in used_moduli]
        
        # Sort candidates by weight (1/m) descending
        candidates.sort(key=lambda m: 1.0/m, reverse=True)
        
        while time.time() < time_limit:
            improved = False
            # Try to add the best candidate that fits
            for m in candidates:
                r = find_valid_residue(m, best_classes)
                if r != -1:
                    new_classes = best_classes + [[m, r]]
                    new_score = best_score + 1.0 / m
                    if new_score > best_score:
                        best_classes = new_classes
                        best_score = new_score
                        candidates.remove(m)
                        improved = True
                        break
            
            if not improved:
                break
                
        return best_classes, best_score

    best_score = 0.0
    best_classes = []
    
    # Strategy 1: Randomized Greedy with many restarts
    num_restarts = 500
    for _ in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Random order, but biased towards smaller moduli?
        # Or just random shuffle.
        moduli = list(range(20, 201))
        random.shuffle(moduli)
        
        # Greedy construction
        classes = greedy_construct(moduli)
        
        # Local search improvement
        classes, score = local_search(classes, deadline - 10)
        
        if score > best_score:
            best_score = score
            best_classes = classes
    
    # Strategy 2: Structured Greedy based on prime factors
    # Group moduli by their smallest prime factor?
    # Or try to build a set of moduli that are all multiples of 2, or 3, etc.
    # The current best (2.867) is likely from a mix.
    
    # Let's try a specific heuristic:
    # Start with the smallest moduli.
    # Add them if possible.
    # This is essentially the greedy above but with a fixed order (ascending).
    classes_asc = greedy_construct(list(range(20, 201)))
    classes_asc, score_asc = local_search(classes_asc, deadline - 5)
    if score_asc > best_score:
        best_score = score_asc
        best_classes = classes_asc
    
    # Try descending order (largest first) - less likely to be good, but maybe unblocks?
    classes_desc = greedy_construct(list(range(200, 19, -1)))
    classes_desc, score_desc = local_search(classes_desc, deadline - 5)
    if score_desc > best_score:
        best_score = score_desc
        best_classes = classes_desc

    # Fallback
    if not best_classes:
        best_classes = [[20, 0]]
    
    return best_classes
