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
    n_moduli = len(moduli)
    
    # Precompute gcds for all pairs to speed up compatibility checks
    # gcd[i][j] = gcd(moduli[i], moduli[j])
    gcd_table = [[0] * n_moduli for _ in range(n_moduli)]
    for i in range(n_moduli):
        for j in range(i, n_moduli):
            g = math.gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Precompute 1/m for each modulus
    inv_moduli = [1.0 / m for m in moduli]
    
    # Upper bound: sum of all 1/m (if we could pick all)
    total_possible_sum = sum(inv_moduli)
    
    best_score = 0.0
    best_classes = []
    
    # We will use a randomized backtracking approach.
    # To make it effective, we should order moduli strategically.
    # Moduli with many small prime factors (high gcd with others) are more restrictive.
    # Moduli that are prime or have large prime factors are less restrictive.
    # However, smaller moduli contribute more to the score.
    # Let's try sorting by descending 1/m (i.e., ascending m) first, as standard greedy does,
    # but the backtracking allows us to "back up" if a choice leads to a dead end or suboptimal path.
    
    # Actually, a better heuristic for backtracking:
    # Process moduli in an order that maximizes constraint propagation early.
    # But since we want to maximize sum(1/m), small m are important.
    # Let's stick to ascending m (20, 21, ..., 200) for the variable ordering in the tree,
    # but use randomization in the value ordering (residues).
    
    # To avoid getting stuck in one specific local optimum due to deterministic ordering,
    # we can shuffle the order of moduli slightly or randomize residue choices.
    # Given the time limit, we can run multiple backtracking passes with different random seeds/orders.
    
    def backtrack(idx, current_classes, current_score):
        """
        Recursively assign residues to moduli[idx:].
        current_classes: list of [m, r] assigned so far.
        current_score: sum of 1/m for assigned classes.
        """
        nonlocal best_score, best_classes
        
        # Check time limit
        if time.time() > deadline:
            return False # Signal to stop
        
        # If we've assigned all moduli
        if idx == n_moduli:
            if current_score > best_score:
                best_score = current_score
                best_classes = [c[:] for c in current_classes]
            return True # Continue searching other branches
        
        m = moduli[idx]
        inv_m = inv_moduli[idx]
        
        # Optimistic bound: current score + sum of 1/m for all remaining moduli
        # We can precompute suffix sums of 1/m
        # suffix_sum[i] = sum(inv_moduli[i:])
        # If current_score + suffix_sum[idx] <= best_score, prune.
        # Let's precompute suffix sums.
        
        # For now, a simpler bound: if current_score + (total_possible_sum - prefix_sum[idx]) <= best_score
        # Let's use a precomputed suffix sum array.
        
        # We need to check compatibility with all previously assigned classes.
        # A residue r is valid if for all j < idx, (r - current_classes[j][1]) % gcd_table[idx][j] != 0.
        
        # To speed up, we can maintain a bitmask or a set of forbidden residues?
        # No, the constraints depend on the specific gcd with each previous modulus.
        # So we must check each previous modulus.
        
        # Optimization: Precompute valid residues for m given the current set?
        # This is expensive if done from scratch each time.
        # Instead, iterate through residues 0..m-1 and check compatibility.
        
        # Randomize the order of residues to explore different branches
        residues = list(range(m))
        random.shuffle(residues)
        
        for r in residues:
            # Check compatibility with all assigned classes
            compatible = True
            for j in range(idx):
                m_j = moduli[j]
                r_j = current_classes[j][1]
                g = gcd_table[idx][j]
                if (r - r_j) % g == 0:
                    compatible = False
                    break
            
            if compatible:
                # Assign
                current_classes.append([m, r])
                
                # Recurse
                # We need the suffix sum for pruning. Let's pass it or compute it.
                # To keep the function signature clean, we'll use a closure or global.
                # Let's compute suffix_sum globally.
                
                # Pruning check
                # suffix_sum[idx] is the sum of 1/m for moduli[idx:].
                # Wait, we are at idx, so we are assigning moduli[idx].
                # The remaining moduli are idx+1 to n-1.
                # So the potential max score from here is current_score + inv_m + suffix_sum[idx+1].
                # If this is <= best_score, we can prune this branch.
                
                if current_score + inv_m + suffix_sum[idx+1] > best_score:
                    backtrack(idx + 1, current_classes, current_score + inv_m)
                
                # Undo
                current_classes.pop()
        
        return True

    # Precompute suffix sums of 1/m
    suffix_sum = [0.0] * (n_moduli + 1)
    for i in range(n_moduli - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + inv_moduli[i]
    
    # Run backtracking
    # Since the search space is huge (product of m_i), we need to be careful.
    # Pure backtracking might be too slow for 181 variables.
    # However, the constraints are quite tight.
    # Let's add a "fail fast" mechanism: if we hit the deadline, we return best found.
    
    # To improve exploration, we can restart with different random seeds or variable orderings.
    # But changing variable ordering in backtracking requires re-indexing gcd_table and suffix_sum.
    # Let's stick to one ordering (ascending m) but randomize residue choices.
    # This is effectively a randomized depth-first search.
    
    # One issue: if the first branch is very deep and finds a good score, it might take too long.
    # We can limit the depth of recursion or use a time-sliced approach.
    # The time check inside backtrack handles the hard stop.
    
    # Let's run the backtracking.
    # We might want to run it multiple times if it finishes quickly, but it likely won't finish quickly.
    # So we just run it once until the deadline.
    
    backtrack(0, [], 0.0)
    
    # If we didn't find any classes (shouldn't happen, but safe), return empty
    if not best_classes:
        # Fallback to a simple greedy if backtracking failed completely
        selected = []
        score = 0.0
        for i, m in enumerate(moduli):
            # Find any valid residue
            valid = False
            for r in range(m):
                compatible = True
                for j in range(len(selected)):
                    m_j = selected[j][0]
                    r_j = selected[j][1]
                    g = math.gcd(m, m_j)
                    if (r - r_j) % g == 0:
                        compatible = False
                        break
                if compatible:
                    selected.append([m, r])
                    score += 1.0 / m
                    valid = True
                    break
            if not valid:
                continue
        if score > best_score:
            best_score = score
            best_classes = selected
            
    return best_classes

# EVOLVE-BLOCK-END
