import time
import math
import random
from math import gcd

def construct():
    deadline = time.time() + 140  # Safety margin for 150s limit
    
    min_m = 20
    max_m = 200
    moduli = list(range(min_m, max_m + 1))
    n = len(moduli)
    
    # Precompute GCDs
    gcd_table = {}
    for i, m1 in enumerate(moduli):
        for j, m2 in enumerate(moduli):
            gcd_table[(m1, m2)] = gcd(m1, m2)
    
    # Precompute 1/m
    inv_m = {m: 1.0 / m for m in moduli}
    
    # Sort moduli by 1/m descending (smallest first) for branch-and-bound
    sorted_moduli = sorted(moduli, key=lambda m: -inv_m[m])
    
    # Precompute suffix sums of 1/m for upper bound
    # suffix_sum[i] = sum of 1/m for sorted_moduli[i:]
    suffix_sum = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + inv_m[sorted_moduli[i]]
    
    def can_assign_residues(selected_moduli):
        """
        Try to assign residues to selected_moduli such that all are pairwise disjoint.
        Uses backtracking. Returns dict {m: r} if successful, None otherwise.
        """
        assignment = {}
        
        def backtrack(idx):
            if idx == len(selected_moduli):
                return True
            
            m = selected_moduli[idx]
            
            # Try each residue for m
            for r in range(m):
                # Check against all previously assigned
                valid = True
                for m2, r2 in assignment.items():
                    g = gcd_table[(m, m2)]
                    if r % g == r2 % g:
                        valid = False
                        break
                
                if valid:
                    assignment[m] = r
                    if backtrack(idx + 1):
                        return True
                    del assignment[m]
            
            return False
        
        if backtrack(0):
            return assignment
        return None
    
    def branch_and_bound():
        """
        Branch-and-bound search over which moduli to include.
        Processes moduli in order of decreasing 1/m.
        Upper bound at step i: current_score + suffix_sum[i]
        """
        best_score = 0.0
        best_set = []
        
        def bnb(idx, current_score, current_set):
            nonlocal best_score, best_set
            
            # Prune: if upper bound can't beat best, return
            if current_score + suffix_sum[idx] <= best_score:
                return
            
            # Prune on time
            if time.time() > deadline - 5:
                return
            
            if idx == n:
                # Leaf: try to assign residues
                if current_score > best_score:
                    assignment = can_assign_residues(current_set)
                    if assignment is not None:
                        best_score = current_score
                        best_set = list(current_set)
                return
            
            m = sorted_moduli[idx]
            
            # Branch 1: Include m
            # Check if m conflicts with any already selected (via gcd structure)
            # Actually, conflicts depend on residues, not just moduli.
            # But if gcd(m, m2) == 1 for all m2 in current_set, m can always be added.
            # If gcd > 1, it might still be addable with right residue.
            # For upper bound purposes, we always allow inclusion.
            bnb(idx + 1, current_score + inv_m[m], current_set + [m])
            
            # Branch 2: Skip m
            bnb(idx + 1, current_score, current_set)
        
        bnb(0, 0.0, [])
        return best_set, best_score
    
    def greedy_with_residues(order):
        """
        Greedy: process moduli in given order, add if a valid residue exists.
        Returns (assignment_dict, score).
        """
        assignment = {}
        score = 0.0
        
        for m in order:
            # Find a valid residue
            found = False
            for r in range(m):
                valid = True
                for m2, r2 in assignment.items():
                    g = gcd_table[(m, m2)]
                    if r % g == r2 % g:
                        valid = False
                        break
                if valid:
                    assignment[m] = r
                    score += inv_m[m]
                    found = True
                    break
        
        return assignment, score
    
    def local_search(assignment, max_time=20):
        """
        Local search: try to add unselected moduli or swap selected with unselected.
        """
        current = dict(assignment)
        current_score = sum(inv_m[m] for m in current)
        best = dict(current)
        best_score = current_score
        start_time = time.time()
        
        improved = True
        while improved and time.time() < start_time + max_time:
            improved = False
            
            # Try adding each unselected modulus
            unselected = [m for m in moduli if m not in current]
            # Sort by 1/m descending
            unselected.sort(key=lambda m: -inv_m[m])
            
            for m in unselected:
                if time.time() > deadline - 5:
                    return best, best_score
                
                # Find a valid residue
                for r in range(m):
                    valid = True
                    for m2, r2 in current.items():
                        g = gcd_table[(m, m2)]
                        if r % g == r2 % g:
                            valid = False
                            break
                    
                    if valid:
                        new_score = current_score + inv_m[m]
                        if new_score > best_score:
                            current[m] = r
                            current_score = new_score
                            best = dict(current)
                            best_score = new_score
                            improved = True
                        break
        
        return best, best_score
    
    # Strategy:
    # 1. Greedy initial solution (smallest moduli first)
    # 2. Local search to improve
    # 3. Branch-and-bound for global search (with time limit)
    # 4. More local search rounds
    
    # Step 1: Greedy
    assignment, score = greedy_with_residues(sorted_moduli)
    best_assignment = dict(assignment)
    best_score = score
    
    # Step 2: Local search
    best_assignment, best_score = local_search(best_assignment, max_time=30)
    
    # Step 3: Branch-and-bound (limited time)
    bnb_time_limit = time.time() + 60
    deadline_bnb = bnb_time_limit
    
    # Override deadline temporarily for bnb
    old_deadline = deadline
    deadline = deadline_bnb
    
    bnb_set, bnb_score = branch_and_bound()
    
    deadline = old_deadline
    
    if bnb_score > best_score:
        # Assign residues to bnb_set
        bnb_assignment = can_assign_residues(bnb_set)
        if bnb_assignment is not None:
            best_assignment = bnb_assignment
            best_score = bnb_score
    
    # Step 4: Final local search
    if time.time() < deadline - 10:
        best_assignment, best_score = local_search(best_assignment, max_time=20)
    
    # Convert to list of lists
    return [[m, r] for m, r in best_assignment.items()]

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    score = sum(1.0 / m for m, _ in res)
    print(f"Score: {score}")
    print(f"Time: {end - start:.2f}s")
    print(f"Classes: {len(res)}")
    # Verify validity
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            m1, r1 = res[i]
            m2, r2 = res[j]
            g = gcd(m1, m2)
            if r1 % g == r2 % g:
                valid = False
                print(f"Conflict: ({m1},{r1}) and ({m2},{r2})")
    print(f"Valid: {valid}")
