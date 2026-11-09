"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    base = 51975
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Find all divisors of 51975 greater than 1
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    divisors.sort()
    
    # Precompute coverage patterns for each (m, r) pair
    # For each modulus m, we have r in 0..m-1, and the set of residues covered
    # We'll represent coverage as a bitset or just use numpy arrays
    
    # Strategy: Start with a good greedy solution, then try local search / simulated annealing
    # The objective is to minimize uncovered residues
    
    def compute_uncovered(classes):
        """Compute number of uncovered residues for a given set of classes."""
        covered = bytearray(base)
        for m, r in classes:
            # Mark all residues congruent to r mod m as covered
            for i in range(r, base, m):
                covered[i] = 1
        return base - sum(covered)
    
    def greedy_construct():
        """Greedy: for each modulus in some order, pick the residue class that covers most uncovered."""
        covered = bytearray(base)
        classes = []
        
        # Try different orderings of moduli
        # Smaller moduli cover more residues, so process them first
        for m in divisors:
            best_r = 0
            best_count = -1
            for r in range(m):
                count = 0
                for i in range(r, base, m):
                    if not covered[i]:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_r = r
            classes.append([m, best_r])
            # Update covered
            for i in range(best_r, base, m):
                covered[i] = 1
        return classes, compute_uncovered(classes)
    
    # Initial greedy solution
    best_classes, best_score = greedy_construct()
    
    # Local search: try swapping residue classes for individual moduli
    # Or try adding/removing moduli (but we must use distinct moduli, and the problem says
    # "with every divisor of 51975 available once" - wait, let me re-read)
    
    # Re-reading: "construct() -> classes [[m, r], ...] with distinct moduli m > 1 dividing 51975"
    # So we choose a subset of divisors, each used at most once, and assign a residue to each.
    # The goal is to minimize uncovered residues.
    
    # Actually, looking at the seed program, it uses ALL divisors. Let me check if we can use a subset.
    # The task says "distinct moduli m > 1 dividing 51975" - it doesn't say we must use all.
    # But using more moduli can only help (or stay same) in terms of coverage, since each adds coverage.
    # Wait, no - if we use a modulus, we're forced to cover exactly one residue class mod m.
    # Using a modulus doesn't hurt coverage, it only adds to it. So we should use all divisors.
    
    # Let me verify: the seed program uses all divisors. So we should too.
    
    # Local search: for each modulus, try all possible residues and see if it improves
    def local_search(classes, max_iter=1000):
        classes = [list(c) for c in classes]
        current_score = compute_uncovered(classes)
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Try improving one modulus at a time
            for idx, (m, r) in enumerate(classes):
                best_r = r
                best_local_score = current_score
                
                # Try all other residues for this modulus
                for new_r in range(m):
                    if new_r == r:
                        continue
                    # Compute score if we change this modulus's residue to new_r
                    # This is expensive, so let's be smarter
                    
                # Instead, let's just try a few random alternatives
                for _ in range(3):
                    new_r = random.randint(0, m - 1)
                    if new_r == r:
                        continue
                    old_r = r
                    classes[idx][1] = new_r
                    new_score = compute_uncovered(classes)
                    if new_score < best_local_score:
                        best_local_score = new_score
                        best_r = new_r
                    classes[idx][1] = old_r
                
                if best_r != r:
                    classes[idx][1] = best_r
                    current_score = best_local_score
                    improved = True
            
            if not improved:
                break
        
        return classes, current_score
    
    # Run local search
    best_classes, best_score = local_search(best_classes, max_iter=50)
    
    # Try multiple random restarts with different initial orderings
    for restart in range(5):
        if time.time() > deadline - 20:
            break
        
        # Random ordering of moduli for greedy
        random.shuffle(divisors)
        covered = bytearray(base)
        classes = []
        for m in divisors:
            best_r = 0
            best_count = -1
            for r in range(m):
                count = 0
                for i in range(r, base, m):
                    if not covered[i]:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_r = r
            classes.append([m, best_r])
            for i in range(best_r, base, m):
                covered[i] = 1
        
        score = compute_uncovered(classes)
        if score < best_score:
            best_classes = classes
            best_score = score
        
        # Local search on this solution
        classes_ls, score_ls = local_search(classes, max_iter=30)
        if score_ls < best_score:
            best_classes = classes_ls
            best_score = score_ls
    
    return best_classes
# EVOLVE-BLOCK-END
