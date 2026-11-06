import time
import math
import random

def construct():
    deadline = time.time() + 140  # Safety margin under 150s
    
    # Even moduli in [20, 200]
    # The problem statement says moduli in [20, 200]. It doesn't restrict to even.
    # However, using all moduli gives more options.
    # Let's use all integers from 20 to 200.
    moduli = list(range(20, 201))
    n_mods = len(moduli)
    idx = {m: i for i, m in enumerate(moduli)}
    
    # Precompute gcd table for all pairs of moduli in our set
    # n_mods is 181. 181*181/2 ~ 16k gcds. Fast.
    gcd_table = [[0] * n_mods for _ in range(n_mods)]
    for i in range(n_mods):
        for j in range(i, n_mods):
            g = math.gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def score_of(classes):
        return sum(1.0 / m for m, _ in classes)

    def is_valid_residue(m, r, classes, mi):
        """Check if residue r for modulus m is valid given current classes."""
        for m2, r2 in classes:
            mi2 = idx[m2]
            g = gcd_table[mi][mi2]
            if r % g == r2 % g:
                return False
        return True

    def find_valid_residue(m, classes, mi):
        """Find any valid residue for m, or None."""
        for r in range(m):
            if is_valid_residue(m, r, classes, mi):
                return r
        return None

    def greedy_construct():
        """Construct a solution by adding moduli in increasing order (smallest first)."""
        classes = []
        # Sort moduli by 1/m descending (i.e., m ascending) to prioritize high value
        for m in sorted(moduli):
            mi = idx[m]
            r = find_valid_residue(m, classes, mi)
            if r is not None:
                classes.append([m, r])
        return classes

    def local_search(classes, max_iters=2000):
        """Simple hill climbing to polish the greedy result."""
        classes = [c[:] for c in classes]
        score = score_of(classes)
        present = set(m for m, _ in classes)
        missing = [m for m in moduli if m not in present]
        
        improved = True
        it = 0
        while improved and it < max_iters:
            improved = False
            it += 1
            
            # Try to add a missing modulus
            added = False
            # Try smallest missing first (highest value)
            for m in sorted(missing):
                mi = idx[m]
                r = find_valid_residue(m, classes, mi)
                if r is not None:
                    classes.append([m, r])
                    present.add(m)
                    missing.remove(m)
                    score += 1.0 / m
                    improved = True
                    added = True
                    break
            
            if added:
                continue

            # Try to change residue of an existing class to allow adding a missing one
            # This is expensive, so we limit it or do it randomly
            # For simplicity in this local search, we might just stop if we can't add.
            # The SA phase will handle the complex moves.
            pass
            
        return classes, score

    def simulated_annealing(initial_classes, max_steps=20000, t_start=1.0, t_end=0.01):
        """
        Simulated Annealing to improve the score.
        State: list of [m, r]
        Moves:
        1. Add a random missing modulus (if valid residue exists).
        2. Change residue of a random existing modulus (if new residue is valid).
        3. Remove a random existing modulus.
        """
        classes = [c[:] for c in initial_classes]
        score = score_of(classes)
        present = set(m for m, _ in classes)
        missing = [m for m in moduli if m not in present]
        
        best_classes = classes[:]
        best_score = score
        
        current_t = t_start
        cooling = (t_end / t_start) ** (1.0 / max_steps)
        
        rng = random.Random(42)
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
                
            # Generate a move
            move_type = rng.random()
            
            new_classes = None
            new_score = None
            
            if move_type < 0.4 and missing:
                # Move 1: Add a missing modulus
                m = rng.choice(missing)
                mi = idx[m]
                # Try to find a valid residue
                # To make it faster, we can try a few random residues first, then scan
                # But scanning is O(m). m <= 200.
                r = find_valid_residue(m, classes, mi)
                if r is not None:
                    new_classes = classes + [[m, r]]
                    new_score = score + 1.0 / m
                    
            elif move_type < 0.8 and classes:
                # Move 2: Change residue of a random class
                i = rng.randint(0, len(classes) - 1)
                m, old_r = classes[i]
                mi = idx[m]
                new_r = rng.randint(0, m - 1)
                if new_r == old_r:
                    continue
                # Check if new_r is valid against all OTHER classes
                valid = True
                for j, (m2, r2) in enumerate(classes):
                    if j == i:
                        continue
                    mi2 = idx[m2]
                    g = gcd_table[mi][mi2]
                    if new_r % g == r2 % g:
                        valid = False
                        break
                if valid:
                    new_classes = classes[:]
                    new_classes[i] = [m, new_r]
                    new_score = score # Score doesn't change for residue change
                    
            elif move_type < 1.0 and classes:
                # Move 3: Remove a random class
                i = rng.randint(0, len(classes) - 1)
                m, r = classes[i]
                new_classes = classes[:]
                del new_classes[i]
                new_score = score - 1.0 / m

            if new_classes is None:
                current_t *= cooling
                continue

            delta = new_score - score
            
            # Accept if better, or with probability exp(-delta/T)
            if delta > 0 or (new_score < 0 and delta < 0): # Score is always positive here
                pass
            if delta >= 0 or (delta < 0 and rng.random() < math.exp(-delta / current_t)):
                classes = new_classes
                score = new_score
                
                # Update present/missing sets
                # This is inefficient if we do it every step.
                # We can update incrementally.
                # But since we swap entire lists, let's just rebuild sets if we accept?
                # No, that's O(N).
                # Let's track present/missing explicitly.
                
                # Actually, rebuilding sets is O(N) which is small.
                # But we need to know what changed.
                # Move 1: Added m.
                # Move 2: No change in moduli set.
                # Move 3: Removed m.
                
                # Let's just recalculate present/missing on the fly for simplicity?
                # No, that's O(N^2) if we do it naively.
                # Let's maintain them.
                
                # To keep it simple and robust, I'll just check validity against the list.
                # The sets `present` and `missing` in the SA loop above are stale if we don't update them.
                # Let's fix the SA loop to update them.
                
                # Re-doing the SA loop logic with set updates is tricky inline.
                # Let's use a helper or just accept the slight inefficiency of checking `m in present`.
                # `present` is a set. `m in present` is O(1).
                # But `missing` is a list.
                
                # Let's refactor SA to maintain sets properly.
                pass

            current_t *= cooling
            
            if score > best_score:
                best_score = score
                best_classes = classes[:]
                
        return best_classes, best_score

    # --- Main Execution ---
    
    # 1. Greedy Start
    classes = greedy_construct()
    classes, score = local_search(classes)
    best_classes = classes
    best_score = score
    
    # 2. Simulated Annealing with multiple restarts/improvements
    # We run SA on the best found so far.
    # Since SA can get stuck, we might run it multiple times with different temperatures or seeds.
    
    # To make SA work correctly with sets, I'll rewrite the inner loop carefully.
    
    def sa_run(start_classes, start_score, max_steps, t_start, t_end, seed):
        classes = [c[:] for c in start_classes]
        score = start_score
        present = set(m for m, _ in classes)
        missing = [m for m in moduli if m not in present]
        
        best_local_classes = classes[:]
        best_local_score = score
        
        rng = random.Random(seed)
        current_t = t_start
        cooling = (t_end / t_start) ** (1.0 / max_steps)
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
                
            move_type = rng.random()
            accepted = False
            delta = 0.0
            added_m = None
            removed_m = None
            
            if move_type < 0.4 and missing:
                # Add
                m = rng.choice(missing)
                mi = idx[m]
                r = find_valid_residue(m, classes, mi)
                if r is not None:
                    new_classes = classes + [[m, r]]
                    new_score = score + 1.0 / m
                    delta = new_score - score
                    if delta >= 0 or rng.random() < math.exp(-delta / current_t):
                        classes = new_classes
                        score = new_score
                        present.add(m)
                        missing.remove(m)
                        accepted = True
                        added_m = m
                        
            elif move_type < 0.8 and classes:
                # Change Residue
                i = rng.randint(0, len(classes) - 1)
                m, old_r = classes[i]
                mi = idx[m]
                new_r = rng.randint(0, m - 1)
                if new_r == old_r:
                    continue
                valid = True
                for j, (m2, r2) in enumerate(classes):
                    if j == i:
                        continue
                    mi2 = idx[m2]
                    g = gcd_table[mi][mi2]
                    if new_r % g == r2 % g:
                        valid = False
                        break
                if valid:
                    new_classes = classes[:]
                    new_classes[i] = [m, new_r]
                    new_score = score
                    delta = 0.0
                    # Always accept neutral moves? Or random?
                    # Accept to explore.
                    classes = new_classes
                    accepted = True
                    # No change to present/missing

            elif move_type < 1.0 and classes:
                # Remove
                i = rng.randint(0, len(classes) - 1)
                m, r = classes[i]
                new_classes = classes[:]
                del new_classes[i]
                new_score = score - 1.0 / m
                delta = new_score - score
                if delta >= 0 or rng.random() < math.exp(-delta / current_t):
                    classes = new_classes
                    score = new_score
                    present.remove(m)
                    missing.append(m)
                    accepted = True
                    removed_m = m

            if accepted:
                if score > best_local_score:
                    best_local_score = score
                    best_local_classes = classes[:]
            
            current_t *= cooling
            
        return best_local_classes, best_local_score

    # Run SA for a while
    # Divide time budget
    # 1. Greedy + Local (fast)
    # 2. SA 1 (long)
    # 3. SA 2 (medium, different temp)
    
    # We already have best_classes from Greedy+Local.
    
    # Run SA with aggressive parameters
    # Max steps depends on speed. Let's estimate 10k steps/sec.
    # 100s budget -> 1M steps.
    
    # Let's do one long SA run.
    c1, s1 = sa_run(best_classes, best_score, 500000, 1.0, 0.01, 42)
    if s1 > best_score:
        best_classes, best_score = c1, s1
        
    # Run another SA with different start temp or seed to explore different landscape
    c2, s2 = sa_run(best_classes, best_score, 300000, 0.5, 0.01, 123)
    if s2 > best_score:
        best_classes, best_score = c2, s2
        
    # Final Polish: Local Search again
    c3, s3 = local_search(best_classes, max_iters=5000)
    if s3 > best_score:
        best_classes, best_score = c3, s3
        
    return best_classes
