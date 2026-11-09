import time
import math
from math import gcd
import random

MAXIMUM = 128
DEADLINE = 90.0
SAFETY_MARGIN = 3.0

def _check_pair(r1, m1, r2, m2, gcd_table):
    """Return True if classes are disjoint (gcd(m1,m2) does not divide r1-r2)."""
    g = gcd_table[m1][m2]
    return (r1 - r2) % g != 0

def _is_valid_solution(solution, gcd_table):
    """Check if a solution is valid: distinct moduli in [2,128], canonical residues, pairwise disjoint."""
    if not solution:
        return False
    seen_moduli = set()
    for r, m in solution:
        if m < 2 or m > MAXIMUM:
            return False
        if m in seen_moduli:
            return False
        seen_moduli.add(m)
        if r < 0 or r >= m:
            return False
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            r1, m1 = solution[i]
            r2, m2 = solution[j]
            g = gcd_table[m1][m2]
            if r1 % g == r2 % g:
                return False
    return True

def _greedy_on_clique(clique_moduli, deadline, num_restarts, gcd_table):
    """
    Greedily assign residues to a set of moduli (clique).
    Returns the best solution found.
    """
    best_solution = []
    
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Shuffle the order of moduli to explore different assignments
        moduli_order = list(clique_moduli)
        random.shuffle(moduli_order)
        
        current_solution = []
        
        for m in moduli_order:
            if time.time() > deadline:
                break
                
            # Try to find a residue r for m that is compatible with all in current_solution
            # Shuffle residues to avoid bias
            residues = list(range(m))
            random.shuffle(residues)
            
            found = False
            for r in residues:
                compatible = True
                for r_prev, m_prev in current_solution:
                    g = gcd_table[m][m_prev]
                    if r % g == r_prev % g:
                        compatible = False
                        break
                if compatible:
                    current_solution.append([r, m])
                    found = True
                    break
            
            # If no residue found, we just skip this modulus for this restart
        
        if len(current_solution) > len(best_solution):
            best_solution = [list(x) for x in current_solution]
            
    return best_solution

def _backtracking_construct(deadline, gcd_table):
    """
    Backtracking search with most-constrained-first heuristic.
    This explores the search space more deeply than static greedy.
    """
    all_moduli = list(range(2, MAXIMUM + 1))
    best_solution = []
    best_size = 0
    
    # Pre-compute prime factors or just use gcd_table directly
    
    def count_valid_residues(m, current_assignment, gcd_table):
        """Count how many residues r in [0, m) are valid given current_assignment."""
        count = 0
        for r in range(m):
            valid = True
            for r_old, m_old in current_assignment:
                g = gcd_table[m][m_old]
                if r % g == r_old % g:
                    valid = False
                    break
            if valid:
                count += 1
        return count

    def find_valid_residue(m, current_assignment, gcd_table):
        """Find one valid residue for m given current_assignment."""
        for r in range(m):
            valid = True
            for r_old, m_old in current_assignment:
                g = gcd_table[m][m_old]
                if r % g == r_old % g:
                    valid = False
                    break
            if valid:
                return r
        return -1

    def backtrack(current_assignment, remaining_moduli, depth):
        nonlocal best_solution, best_size
        
        if time.time() > deadline:
            return
        
        if len(current_assignment) > best_size:
            best_size = len(current_assignment)
            best_solution = [list(x) for x in current_assignment]
        
        if not remaining_moduli:
            return
        
        # Heuristic: select the modulus with the fewest valid residues (most constrained)
        # To save time, we can sample or just pick the first one if it's fast enough
        # Given MAXIMUM=128, checking all is O(N * M * K) which might be slow.
        # Let's pick the one with the smallest domain (m) or just iterate?
        # Actually, the "most constrained" is the one with min valid residues.
        # Let's approximate by picking the smallest modulus first? No, larger moduli have more residues.
        # Small moduli have fewer residues to choose from, so they are more constrained.
        
        # Sort remaining by size (ascending) to pick most constrained first?
        # Or calculate actual constraints?
        # Let's try sorting by modulus ascending as a proxy for constraint tightness.
        # Or better: calculate valid count for a few candidates?
        
        # For speed, let's just pick the smallest modulus from remaining.
        # This is a static ordering, but combined with backtracking it might work.
        # To make it dynamic, we'd need to count valid residues for all, which is expensive.
        # Let's try a hybrid: pick the modulus with the smallest m.
        
        m = min(remaining_moduli)
        remaining_moduli.remove(m)
        
        # Find valid residues for m
        valid_residues = []
        for r in range(m):
            valid = True
            for r_old, m_old in current_assignment:
                g = gcd_table[m][m_old]
                if r % g == r_old % g:
                    valid = False
                    break
            if valid:
                valid_residues.append(r)
        
        if not valid_residues:
            # Backtrack
            return
        
        # Try each valid residue
        # Shuffle to avoid bias
        random.shuffle(valid_residues)
        
        for r in valid_residues:
            current_assignment.append([r, m])
            backtrack(current_assignment, remaining_moduli, depth + 1)
            current_assignment.pop()
            
            if time.time() > deadline:
                return

    # Start backtracking
    # We can start with a small set of moduli to guide the search
    # Or start with all moduli.
    # Starting with all might be too slow.
    # Let's start with a subset or just all and rely on the heuristic.
    
    # To make it faster, we can limit the branching factor or depth.
    # But for now, let's try it on a subset of moduli that are likely to form a clique.
    # The multiples of 2 are a clique.
    # Let's try backtracking on the multiples of 2 first.
    
    clique = [m for m in range(2, MAXIMUM + 1) if m % 2 == 0]
    random.shuffle(clique)
    
    # We need to be careful with recursion depth.
    # sys.setrecursionlimit(MAXIMUM * 10)
    
    # Since backtracking can be exponential, we need to be very careful with time.
    # Let's limit the number of nodes explored or just rely on the time check.
    
    # A better approach for backtracking in this context:
    # Since we want to maximize the number of classes, and we know a clique of size 63 exists (multiples of 2),
    # we can try to find a valid assignment for all 63.
    
    # Let's try backtracking on the multiples of 2.
    backtrack([], clique, 0)
    
    return best_solution

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_solution = []
    
    # Pre-compute gcd table
    gcd_table = [[0]*(MAXIMUM+1) for _ in range(MAXIMUM+1)]
    for m1 in range(2, MAXIMUM+1):
        for m2 in range(2, MAXIMUM+1):
            gcd_table[m1][m2] = gcd(m1, m2)
    
    # Generate cliques: multiples of small primes
    # The largest clique is multiples of 2
    cliques = []
    for p in [2, 3, 5, 7, 11, 13]:
        clique = [m for m in range(2, MAXIMUM + 1) if m % p == 0]
        if len(clique) > len(best_solution):
             cliques.append(clique)
    
    # Sort cliques by size descending to try the best ones first
    cliques.sort(key=lambda x: len(x), reverse=True)
    
    # Strategy 1: Greedy on the largest cliques
    for clique in cliques:
        if time.time() > deadline:
            break
        sol = _greedy_on_clique(clique, deadline, num_restarts=500, gcd_table=gcd_table)
        if len(sol) > len(best_solution):
            best_solution = sol
            
    # Strategy 2: Backtracking on the largest clique (multiples of 2)
    if time.time() < deadline:
        sol_bt = _backtracking_construct(deadline, gcd_table)
        if len(sol_bt) > len(best_solution):
            best_solution = sol_bt
            
    if not best_solution:
        return [[0, 2]]
        
    return best_solution
