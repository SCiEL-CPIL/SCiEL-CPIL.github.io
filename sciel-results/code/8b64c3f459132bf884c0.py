"""
Strategy: Constraint Programming (CP) with Forward Checking and MRV.
Instead of greedy selection, we treat the selection of residues as a CSP.
We order moduli dynamically (MRV heuristic) to minimize branching factor early.
We perform a depth-first search with backtracking to maximize the number of selected classes.
"""
# EVOLVE-BLOCK-START
import math
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    MAX_MOD = 128
    
    moduli = list(range(2, MAX_MOD + 1))
    n_mods = len(moduli)
    
    # Precompute gcd table for fast access
    # idx = m - 2
    gcd_table = [[0] * n_mods for _ in range(n_mods)]
    for i in range(n_mods):
        mi = moduli[i]
        for j in range(i, n_mods):
            mj = moduli[j]
            g = math.gcd(mi, mj)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # Precompute "conflict masks" or just rely on the gcd check.
    # A residue r_i is incompatible with r_j if (r_i - r_j) % gcd(m_i, m_j) == 0.

    best_size = 0
    best_solution = []

    # Helper to update best
    def update_best(sol):
        nonlocal best_size, best_solution
        if len(sol) > best_size:
            best_size = len(sol)
            best_solution = sol

    # Initial greedy to get a good lower bound (pruning)
    # Order: Sort by number of "conflicts" or just random? 
    # Let's use a simple greedy first to set a baseline for pruning.
    def quick_greedy():
        selected = []
        # Try descending moduli first, often works well for disjointness
        for m in sorted(moduli, reverse=True):
            found = False
            for r in range(m):
                valid = True
                for r_p, m_p in selected:
                    g = math.gcd(m, m_p)
                    if (r - r_p) % g == 0:
                        valid = False
                        break
                if valid:
                    selected.append([r, m])
                    found = True
                    break
        return selected

    initial_sol = quick_greedy()
    update_best(initial_sol)
    
    # CP Search
    # We want to maximize the count.
    # We can formulate this as: Find a subset of moduli S and assignments r_m for m in S
    # such that for all m, n in S, (r_m - r_n) % gcd(m, n) != 0.
    
    # We will search for solutions of size K, starting from current best+1 upwards?
    # No, that's inefficient. 
    # Better: DFS that tries to add more nodes.
    # State: current set of (r, m) pairs.
    # Heuristic: MRV (Minimum Remaining Values).
    # For each unused modulus m, calculate how many valid residues remain given current selection.
    # Pick the modulus with the fewest valid residues (but > 0). If 0, prune.
    # If all unused have 0, we have a maximal set. Record if larger than best.
    # Backtrack.

    # To make it efficient, we need to be careful with the search space.
    # 127 moduli is too many for naive DFS.
    # However, the constraint is strong.
    
    # Optimization: Precompute valid residues for each modulus against others?
    # No, constraints depend on specific residues chosen.
    
    # Let's implement the DFS with MRV.
    
    # We need a fast way to check validity of adding a new (r, m).
    def is_compatible(r_new, m_new, current_pairs):
        for r_old, m_old in current_pairs:
            g = gcd_table[m_new - 2][m_old - 2]
            if (r_new - r_old) % g == 0:
                return False
        return True

    # MRV: For a given current set, find the modulus m that has the fewest valid residues.
    # If min valid residues is 0, return None (dead end).
    # Return (m, list_of_valid_residues, count)
    def get_mrv(current_pairs, unused_moduli):
        best_m = None
        best_residues = None
        min_count = float('inf')
        
        # Optional: Prune unused_moduli based on time or just iterate all
        for m in unused_moduli:
            valid_residues = []
            for r in range(m):
                if is_compatible(r, m, current_pairs):
                    valid_residues.append(r)
            
            count = len(valid_residues)
            if count == 0:
                return None, None, 0 # Dead end for this branch
            
            if count < min_count:
                min_count = count
                best_m = m
                best_residues = valid_residues
                
                # Early exit if we found a modulus with only 1 choice
                if min_count == 1:
                    break
                    
        if best_m is None:
            # No unused moduli? Or all processed?
            # If unused_moduli is empty, we are done.
            return None, None, 0 
        return best_m, best_residues, min_count

    # DFS
    def dfs(current_pairs, unused_moduli_set):
        nonlocal best_size, best_solution
        
        if time.time() > deadline:
            return
        
        # Update best if current is better
        if len(current_pairs) > best_size:
            best_size = len(current_pairs)
            best_solution = [list(p) for p in current_pairs]
            
        if not unused_moduli_set:
            return
            
        # MRV Step
        m, valid_residues, count = get_mrv(current_pairs, list(unused_moduli_set))
        
        if m is None:
            # Dead end or no unused moduli left
            return
            
        # Remove m from unused for this branch
        unused_moduli_set.discard(m)
        
        # Try each valid residue for m
        # Order residues? Random or specific? 
        # Maybe try residues that are "central" or just iterate.
        # To avoid symmetry issues, we can just iterate.
        
        # Optimization: If we have already found a solution of size N, 
        # and len(current) + len(unused) + 1 <= N, we can prune?
        # Actually, len(unused) is an upper bound on how many more we can add.
        if len(current_pairs) + len(unused_moduli_set) + 1 <= best_size:
            unused_moduli_set.add(m) # restore
            return

        for r in valid_residues:
            if time.time() > deadline:
                break
            current_pairs.append((r, m))
            dfs(current_pairs, unused_moduli_set)
            current_pairs.pop()
            
        unused_moduli_set.add(m) # restore

    # To make the search more effective, we should perhaps restrict the search space 
    # or use a better initial ordering. 
    # Pure MRV DFS on 127 nodes might be too slow if the branching factor isn't reduced fast enough.
    # However, the constraints are quite tight.
    
    # Let's add a random restart mechanism with a different initial "seed" of moduli 
    # or just run the DFS from the start.
    
    # One issue: The order of trying residues in `for r in valid_residues` matters.
    # We can randomize the order of residues to explore different parts of the space.
    
    import random
    random.seed(42) # For reproducibility in this block, though evaluator might not care
    
    # Let's refine the DFS to handle the "maximize" aspect better.
    # The current DFS finds *a* maximal set. It doesn't guarantee the *maximum* set 
    # unless it exhausts the search space. With 127 moduli, exhaustive search is impossible.
    # So we rely on the heuristic (MRV) to find large sets quickly.
    # To improve, we can do multiple restarts with random shuffling of the "unused_moduli" 
    # priority or random residue ordering.
    
    # Let's run the DFS a few times with different random seeds for residue ordering.
    
    def dfs_with_random_order(current_pairs, unused_moduli_set, rand):
        nonlocal best_size, best_solution
        
        if time.time() > deadline:
            return
        
        if len(current_pairs) > best_size:
            best_size = len(current_pairs)
            best_solution = [list(p) for p in current_pairs]
            
        if not unused_moduli_set:
            return
            
        m, valid_residues, count = get_mrv(current_pairs, list(unused_moduli_set))
        
        if m is None:
            return
            
        unused_moduli_set.discard(m)
        
        if len(current_pairs) + len(unused_moduli_set) + 1 <= best_size:
            unused_moduli_set.add(m)
            return

        # Shuffle residues to explore different branches
        rand.shuffle(valid_residues)
        
        for r in valid_residues:
            if time.time() > deadline:
                break
            current_pairs.append((r, m))
            dfs_with_random_order(current_pairs, unused_moduli_set, rand)
            current_pairs.pop()
            
        unused_moduli_set.add(m)

    # Run a few iterations of the CP search
    # We start with empty set.
    # To guide it, we can pre-select a few "hard" moduli? 
    # Or just let MRV do its job.
    
    # Let's try 10 restarts with different random seeds for residue ordering.
    # The MRV modulus selection is deterministic based on current state, 
    # but the order of trying residues is randomized.
    
    for i in range(20):
        if time.time() > deadline:
            break
        rand = random.Random(i)
        dfs_with_random_order([], set(moduli), rand)

    # Fallback if CP didn't find anything (shouldn't happen)
    if not best_solution:
        best_solution = [[1, 2]]

    # Final verification
    def is_valid_solution(classes):
        n = len(classes)
        for i in range(n):
            r1, m1 = classes[i]
            idx1 = m1 - 2
            for j in range(i + 1, n):
                r2, m2 = classes[j]
                idx2 = m2 - 2
                g = gcd_table[idx1][idx2]
                if (r1 - r2) % g == 0:
                    return False
        return True

    if not is_valid_solution(best_solution):
        # Fallback to greedy if CP produced invalid (should not happen)
        best_solution = initial_sol
        if not is_valid_solution(best_solution):
            best_solution = [[1, 2]]

    return best_solution
# EVOLVE-BLOCK-END
