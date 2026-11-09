"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from math import gcd
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    MAX_MOD = 128
    moduli = list(range(2, MAX_MOD + 1))
    
    # We need to select a subset of moduli (each used at most once) and assign
    # a residue r in [0, m) such that for any two selected pairs (r1, m1), (r2, m2):
    # the residue classes {x : x ≡ r1 (mod m1)} and {x : x ≡ r2 (mod m2)} are disjoint.
    # Two residue classes are disjoint iff gcd(m1, m2) does not divide (r1 - r2).
    # Equivalently: (r1 - r2) % gcd(m1, m2) != 0.
    
    # This is a maximum independent set problem in a graph where:
    # - Nodes are (modulus, residue) pairs
    # - Two nodes are adjacent (conflict) if their classes intersect
    # - We want max non-adjacent subset
    
    # But the search space is huge (127 moduli * up to 128 residues each).
    # Key insight: for a given pair of moduli (m1, m2), the classes intersect
    # iff r1 ≡ r2 (mod gcd(m1,m2)). So for each pair of moduli, there are
    # exactly gcd(m1,m2) "conflicting residue pairs" out of m1*m2 total.
    
    # Strategy: Greedy with randomized restarts and local search.
    # Represent a solution as a dict: modulus -> residue (or None if not selected).
    
    def is_conflict(r1, m1, r2, m2):
        g = gcd(m1, m2)
        return (r1 - r2) % g == 0
    
    def count_conflicts(assignment):
        """Count number of conflicting pairs in the assignment."""
        count = 0
        items = [(m, r) for m, r in assignment.items() if r is not None]
        for i in range(len(items)):
            for j in range(i+1, len(items)):
                if is_conflict(items[i][1], items[i][0], items[j][1], items[j][0]):
                    count += 1
        return count
    
    def greedy_construct(order, seed=0):
        """Greedy: process moduli in given order, pick best residue for each."""
        rng = random.Random(seed)
        assignment = {}
        selected = []  # list of (residue, modulus)
        
        for m in order:
            # Try all residues, pick one that doesn't conflict with existing
            valid_residues = []
            for r in range(m):
                ok = True
                for (r2, m2) in selected:
                    if is_conflict(r, m, r2, m2):
                        ok = False
                        break
                if ok:
                    valid_residues.append(r)
            
            if valid_residues:
                r = rng.choice(valid_residues)
                selected.append((r, m))
                assignment[m] = r
        
        return selected
    
    def local_search(selected, max_iters=5000):
        """Improve by trying to add unselected moduli or swap residues."""
        selected_set = set(m for (r, m) in selected)
        selected_list = list(selected)
        
        # Try to add each unselected modulus
        for m in range(2, MAX_MOD + 1):
            if m in selected_set:
                continue
            # Find a valid residue
            for r in range(m):
                ok = True
                for (r2, m2) in selected_list:
                    if is_conflict(r, m, r2, m2):
                        ok = False
                        break
                if ok:
                    selected_list.append((r, m))
                    selected_set.add(m)
                    break
        
        # Try local moves: change residue of one modulus to free up space for another
        for _ in range(max_iters):
            improved = False
            # Try swapping residue of a selected modulus
            for i in range(len(selected_list)):
                r_i, m_i = selected_list[i]
                for new_r in range(m_i):
                    if new_r == r_i:
                        continue
                    # Check if new_r works with all others
                    ok = True
                    for j in range(len(selected_list)):
                        if j == i:
                            continue
                        r_j, m_j = selected_list[j]
                        if is_conflict(new_r, m_i, r_j, m_j):
                            ok = False
                            break
                    if not ok:
                        continue
                    # Would this help? Try to add an unselected modulus
                    new_selected = list(selected_list)
                    new_selected[i] = (new_r, m_i)
                    new_set = set(m for (r, m) in new_selected)
                    
                    for m_add in range(2, MAX_MOD + 1):
                        if m_add in new_set:
                            continue
                        for r_add in range(m_add):
                            ok_add = True
                            for (r2, m2) in new_selected:
                                if is_conflict(r_add, m_add, r2, m2):
                                    ok_add = False
                                    break
                            if ok_add:
                                selected_list = new_selected
                                selected_list.append((r_add, m_add))
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        
        return selected_list
    
    best_solution = []
    
    # Deterministic greedy with several orderings
    orderings = [
        list(range(2, MAX_MOD + 1)),  # ascending
        list(range(MAX_MOD, 1, -1)),  # descending
        # Sort by number of prime factors (more factors = more gcd conflicts)
        sorted(range(2, MAX_MOD + 1), key=lambda m: sum(1 for d in range(2, int(m**0.5)+1) if m%d==0)),
        # Random orderings
    ]
    
    # Add some random orderings
    rng_master = random.Random(42)
    for _ in range(20):
        ord_copy = list(range(2, MAX_MOD + 1))
        rng_master.shuffle(ord_copy)
        orderings.append(ord_copy)
    
    for idx, order in enumerate(orderings):
        if time.time() > deadline:
            break
        sol = greedy_construct(order, seed=idx)
        sol = local_search(sol, max_iters=2000)
        if len(sol) > len(best_solution):
            best_solution = sol
    
    # More randomized restarts if time permits
    rng = random.Random(12345)
    attempt = 0
    while time.time() < deadline - 5:
        attempt += 1
        # Random ordering
        order = list(range(2, MAX_MOD + 1))
        rng.shuffle(order)
        
        sol = greedy_construct(order, seed=attempt)
        sol = local_search(sol, max_iters=3000)
        if len(sol) > len(best_solution):
            best_solution = sol
        
        # Occasionally try a different strategy: start from a random subset
        if attempt % 5 == 0:
            # Pick random subset of moduli, then fill in residues
            subset_size = rng.randint(10, 30)
            subset = rng.sample(range(2, MAX_MOD + 1), min(subset_size, 127))
            partial = []
            for m in subset:
                valid = []
                for r in range(m):
                    ok = True
                    for (r2, m2) in partial:
                        if is_conflict(r, m, r2, m2):
                            ok = False
                            break
                    if ok:
                        valid.append(r)
                if valid:
                    partial.append((rng.choice(valid), m))
            sol = local_search(partial, max_iters=5000)
            if len(sol) > len(best_solution):
                best_solution = sol
    
    # Final: try to add more moduli to the best solution
    final_set = set(m for (r, m) in best_solution)
    final_list = list(best_solution)
    for m in range(2, MAX_MOD + 1):
        if m in final_set:
            continue
        for r in range(m):
            ok = True
            for (r2, m2) in final_list:
                if is_conflict(r, m, r2, m2):
                    ok = False
                    break
            if ok:
                final_list.append((r, m))
                final_set.add(m)
                break
    
    return [[r, m] for (r, m) in final_list]

# EVOLVE-BLOCK-END
