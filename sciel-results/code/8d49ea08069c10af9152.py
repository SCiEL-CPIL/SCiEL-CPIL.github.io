import time
import math
import random
from fractions import Fraction

def construct():
    deadline = time.time() + 115  # Safety margin for 120s limit
    
    moduli = list(range(20, 201))
    n_mod = len(moduli)
    
    # Precompute GCDs
    gcd_table = {}
    for m1 in moduli:
        for m2 in moduli:
            gcd_table[(m1, m2)] = math.gcd(m1, m2)
    
    # Precompute conflict check helper
    def is_compatible(m1, r1, m2, r2):
        g = gcd_table.get((m1, m2), math.gcd(m1, m2))
        return (r1 % g) != (r2 % g)
    
    # We want to maximize sum(1/m)
    # Smaller m gives higher value
    # Strategy: Greedy with backtracking / local search
    # Since the constraint is pairwise disjoint, this is like a graph coloring problem
    # where vertices are moduli, and edges exist if they share a gcd > 1
    # But the constraint is more subtle: r1 != r2 mod gcd(m1, m2)
    
    # Let's use a greedy approach sorted by m (ascending) to prioritize small moduli
    # But we need to be careful about conflicts
    
    # Better approach: Use local search / simulated annealing
    # Start with a random assignment, then iteratively improve
    
    def score_assignment(assignment):
        """assignment is a dict m -> r, or None if m is not included"""
        total = Fraction(0, 1)
        for m, r in assignment.items():
            if r is not None:
                total += Fraction(1, m)
        return total
    
    def check_validity(assignment):
        """Check if assignment is valid (all pairwise disjoint)"""
        ms = [m for m in assignment if assignment[m] is not None]
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                m1, m2 = ms[i], ms[j]
                r1, r2 = assignment[m1], assignment[m2]
                if not is_compatible(m1, r1, m2, r2):
                    return False
        return True
    
    # Greedy construction: try to include all moduli, starting from small ones
    # For each modulus, try all residues and pick one that doesn't conflict
    
    def greedy_construct(order):
        """
        Try to construct a valid assignment by processing moduli in given order.
        For each modulus, try all residues and pick the first valid one.
        """
        assignment = {}
        for m in order:
            found = False
            for r in range(m):
                valid = True
                for m2, r2 in assignment.items():
                    if not is_compatible(m, r, m2, r2):
                        valid = False
                        break
                if valid:
                    assignment[m] = r
                    found = True
                    break
            if not found:
                assignment[m] = None
        return assignment
    
    # Local search: start with greedy, then try to improve by changing residues or adding/removing moduli
    
    def local_search(initial_assignment, max_iters=5000):
        """
        Improve the assignment using local search.
        Try to change residues of existing moduli, or add new moduli.
        """
        assignment = dict(initial_assignment)
        current_score = score_assignment(assignment)
        best_score = current_score
        best_assignment = dict(assignment)
        
        ms_included = [m for m in assignment if assignment[m] is not None]
        ms_excluded = [m for m in assignment if assignment[m] is None]
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Randomly pick an action:
            # 1. Change residue of an included modulus
            # 2. Try to add an excluded modulus
            # 3. Remove an included modulus (rarely needed)
            
            action = random.random()
            
            if action < 0.6 and ms_included:
                # Change residue of an included modulus
                m = random.choice(ms_included)
                old_r = assignment[m]
                new_r = random.randint(0, m - 1)
                if new_r == old_r:
                    continue
                
                assignment[m] = new_r
                
                # Check if still valid
                if check_validity(assignment):
                    new_score = score_assignment(assignment)
                    if new_score > best_score:
                        best_score = new_score
                        best_assignment = dict(assignment)
                else:
                    assignment[m] = old_r
                    
            elif action < 0.9 and ms_excluded:
                # Try to add an excluded modulus
                m = random.choice(ms_excluded)
                # Try all residues
                for r in range(m):
                    assignment[m] = r
                    if check_validity(assignment):
                        new_score = score_assignment(assignment)
                        if new_score > best_score:
                            best_score = new_score
                            best_assignment = dict(assignment)
                            ms_included.append(m)
                            ms_excluded.remove(m)
                        break
                else:
                    assignment[m] = None
            else:
                # Remove an included modulus (to free up space for others)
                if ms_included:
                    m = random.choice(ms_included)
                    assignment[m] = None
                    ms_included.remove(m)
                    ms_excluded.append(m)
                    
        return best_assignment, best_score
    
    # Multi-start: try different orderings
    
    best_overall_score = Fraction(0, 1)
    best_overall_assignment = {}
    
    # Start with ascending order (prioritize small moduli)
    order1 = sorted(moduli)
    assignment1 = greedy_construct(order1)
    best1, score1 = local_search(assignment1, max_iters=2000)
    if score1 > best_overall_score:
        best_overall_score = score1
        best_overall_assignment = best1
    
    # Try descending order
    order2 = sorted(moduli, reverse=True)
    assignment2 = greedy_construct(order2)
    best2, score2 = local_search(assignment2, max_iters=2000)
    if score2 > best_overall_score:
        best_overall_score = score2
        best_overall_assignment = best2
    
    # Try random orderings
    for _ in range(5):
        if time.time() > deadline - 10:
            break
        order = moduli[:]
        random.shuffle(order)
        assignment = greedy_construct(order)
        best, score = local_search(assignment, max_iters=1500)
        if score > best_overall_score:
            best_overall_score = score
            best_overall_assignment = best
    
    # Convert to result format
    result = [[m, r] for m, r in best_overall_assignment.items() if r is not None]
    return result

