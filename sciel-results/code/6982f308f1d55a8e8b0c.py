import math
import time
import random

def construct():
    # Deadline guard: 120s limit, use 115s to be safe
    deadline = time.time() + 115
    
    # Precompute gcd table for m in [20, 200]
    # gcd_table[m1][m2] = gcd(m1, m2)
    # Since m is up to 200, a 201x201 table is small.
    # But we only need gcd for m in [20, 200].
    # Let's just compute gcd on the fly with a cache or precomputed list.
    # Precomputing a dictionary or list of lists is faster.
    max_m = 200
    gcd_table = [[0] * (max_m + 1) for _ in range(max_m + 1)]
    for i in range(20, max_m + 1):
        for j in range(20, max_m + 1):
            gcd_table[i][j] = math.gcd(i, j)
    
    # Helper function to check if adding (m, r) is valid given current set
    def is_valid_add(m, r, selected):
        for m2, r2 in selected:
            g = gcd_table[m][m2]
            if r % g == r2 % g:
                return False
        return True

    # Helper function to calculate score
    def calc_score(selected):
        return sum(1.0 / m for m, _ in selected)
    
    # Multi-start greedy with local search
    best_score = 0.0
    best_solution = []
    
    # Number of restarts. We don't know how much time we have, so we loop until deadline.
    # Each restart:
    # 1. Randomly shuffle the order of moduli (or use a specific heuristic order)
    # 2. Greedy fill
    # 3. Local search: try to improve by swapping or removing
    
    # To maximize sum(1/m), we strongly prefer small m.
    # A good heuristic for the initial greedy is to try moduli in increasing order, 
    # but with some randomness or multiple permutations to escape local optima.
    # The current program does a single pass in increasing order.
    # A different strategy: 
    # 1. Generate many random permutations of the candidate moduli [20..200].
    # 2. For each permutation, run a greedy fill.
    # 3. Keep the best solution found.
    # 4. Apply local search to the best solution: 
    #    - Try to remove a large m and see if we can add a smaller m that was previously blocked.
    #    - Try to swap residues? (Residue choice is deterministic in greedy: first valid r).
    #      Actually, for a given set of moduli, the choice of r matters for future additions.
    #      The greedy picks the FIRST valid r. This might block future smaller moduli.
    #      We could try different r choices in the local search, but that's complex.
    #      Simpler: just vary the order of moduli in the greedy phase.
    
    # Let's implement a randomized greedy with multiple restarts.
    # The key insight from the "inspiration" is that multi-start greedy + local search helps.
    # Since the space of valid sets is small (max 181 moduli, but disjointness restricts it heavily),
    # random restarts of the greedy heuristic should find better packings.
    
    # Also, we can try to be smarter about the residue choice.
    # Instead of always picking the first valid r, we could pick a random valid r.
    # This might allow different moduli to be added later.
    
    def greedy_fill(order, randomize_residue=False):
        selected = []
        for m in order:
            if time.time() > deadline:
                break
            valid_rs = []
            for r in range(m):
                if is_valid_add(m, r, selected):
                    if randomize_residue:
                        valid_rs.append(r)
                    else:
                        selected.append([m, r])
                        break
            if randomize_residue and valid_rs:
                selected.append([m, random.choice(valid_rs)])
        return selected

    # Local search: try to improve the solution
    def local_search(selected):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_score = calc_score(selected)
            selected_set = set((m, r) for m, r in selected)
            
            # Try to remove one element and see if we can add a better one
            # Or simply try to add any missing modulus that is valid now?
            # The greedy already did that. The issue is that the ORDER matters.
            # Local search idea: 
            # 1. Try to remove a large m (low value) from the selected set.
            # 2. Then try to add a smaller m (high value) that was previously blocked.
            
            # Sort selected by m descending (largest m first to remove)
            sorted_selected = sorted(selected, key=lambda x: -x[0])
            
            for i in range(len(sorted_selected)):
                if time.time() > deadline:
                    break
                m_remove, r_remove = sorted_selected[i]
                
                # Create a new list without this element
                new_selected = [x for x in selected if x[0] != m_remove]
                
                # Try to add any m that is smaller than m_remove and not in new_selected
                # and is valid
                added_any = False
                # We want to add the smallest possible m that is valid and not already present
                present_ms = set(m for m, _ in new_selected)
                
                # Check candidates in increasing order
                for m_add in range(20, m_remove):
                    if m_add in present_ms:
                        continue
                    # Find a valid r for m_add
                    found_r = None
                    for r in range(m_add):
                        if is_valid_add(m_add, r, new_selected):
                            found_r = r
                            break
                    if found_r is not None:
                        new_selected.append([m_add, found_r])
                        added_any = True
                        break # Only add one? Or try to add as many as possible?
                        # Adding one might block others. Let's just add the best one (smallest m).
                        # Actually, if we remove one and add one, the score change is 1/m_add - 1/m_remove.
                        # Since m_add < m_remove, the score increases.
                        # We should check if the new solution is better.
                        new_score = calc_score(new_selected)
                        if new_score > current_score + 1e-9:
                            selected = new_selected
                            improved = True
                            break # Restart the while loop? Or continue?
                            # Restarting is safer to avoid using stale sorted_selected.
                            # But breaking the inner for loop and setting improved=True 
                            # will cause the while loop to re-evaluate.
                            # We should break out of the for loop to re-sort.
                            # Let's just return the improved solution and restart local search.
                            return selected
            
            # If no improvement found in this pass, stop
            if not improved:
                break
        
        return selected

    # Main loop: Multi-start
    # We'll run multiple random permutations.
    # To ensure we cover the space, we can also include the deterministic increasing order.
    
    # 1. Deterministic increasing order (baseline)
    order = list(range(20, 201))
    sol = greedy_fill(order, randomize_residue=False)
    sol = local_search(sol)
    score = calc_score(sol)
    if score > best_score:
        best_score = score
        best_solution = sol
    
    # 2. Random restarts
    # How many? We have ~110 seconds.
    # Greedy fill is fast: O(N^2 * M) roughly, where N is #selected, M is max m.
    # N is small (maybe 20-50?). M is 200.
    # Local search is more expensive.
    # Let's run as many as we can.
    
    random.seed(42) # For reproducibility in testing, but maybe not in production?
    # Actually, random seed doesn't matter for the evaluator, but good for debugging.
    # Let's use a fixed seed for consistent behavior within the 120s window.
    
    while time.time() < deadline - 5: # Leave 5s for safety
        # Generate a random permutation
        # To bias towards small m being considered earlier (which is good for score),
        # we can sort by (random_value, m) or just shuffle.
        # Shuffling completely destroys the small-m priority.
        # Maybe a "noisy" sort: key = (m + random.uniform(0, 50))
        # This keeps small m likely to be first, but allows some flexibility.
        
        order = list(range(20, 201))
        # Add noise to the modulus to create a random-ish order that still favors small m
        noisy_order = sorted(order, key=lambda m: m + random.uniform(0, 100))
        
        sol = greedy_fill(noisy_order, randomize_residue=True)
        
        # Local search
        sol = local_search(sol)
        
        score = calc_score(sol)
        if score > best_score:
            best_score = score
            best_solution = sol
            # Optional: if we found a very good score, we could tighten the search
            # but let's just keep going.
            
    return best_solution

# Test locally
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
            g = math.gcd(m1, m2)
            if r1 % g == r2 % g:
                valid = False
                print(f"Invalid: {res[i]} and {res[j]}")
    print(f"Valid: {valid}")
