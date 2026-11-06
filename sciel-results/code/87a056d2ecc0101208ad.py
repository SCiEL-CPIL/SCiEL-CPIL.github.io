"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin for evaluator's 150s limit
    
    best_score = Fraction(0, 1)
    best_classes = []
    
    def find_valid_residue(m, moduli, residues):
        """
        Find a valid residue r for m given existing moduli and residues.
        Returns r or None.
        Uses a randomized strategy to pick from valid residues if multiple exist,
        but for speed in greedy phase, we might just want the first one.
        For local search, we just need to know if ONE exists.
        """
        valid_residues = []
        for r in range(m):
            ok = True
            for i in range(len(moduli)):
                m2 = moduli[i]
                g = gcd(m, m2)
                if (r - residues[i]) % g == 0:
                    ok = False
                    break
            if ok:
                # Optimization: If we just need to know if it's possible, we can return immediately?
                # But for greedy, we might want to choose a specific one.
                # Let's collect a few or just return the first one for speed.
                return r
        return None

    def can_add(m, moduli, residues):
        """Check if m can be added to the set with existing residues. Returns valid r or None."""
        return find_valid_residue(m, moduli, residues)

    def score_of(moduli_list):
        return sum(Fraction(1, m) for m in moduli_list)

    all_moduli = list(range(20, 201))
    
    # Strategy 1: Multiple Greedy Attempts with different residue choices
    # We'll use a simple greedy: sort by 1/m desc, add if possible.
    # To get diversity, we can shuffle the order slightly or change residue selection.
    
    best_greedy_classes = []
    best_greedy_score = Fraction(0, 1)
    
    for attempt in range(20):
        if time.time() > deadline - 30:
            break
            
        # Shuffle moduli slightly? Or just use sorted order.
        # Sorted by 1/m (ascending m) is best for maximizing sum.
        # But maybe skipping a small m allows two slightly larger ones?
        # 1/20 = 0.05, 1/21 ~ 0.047, 1/22 ~ 0.045.
        # Greedy usually works well here.
        
        current_moduli = []
        current_residues = []
        
        # We can randomize the residue selection to escape local minima in the greedy phase.
        # find_valid_residue currently returns the FIRST valid one.
        # Let's modify it to return a random valid one if we want diversity.
        # But for now, let's stick to first valid, but shuffle the input order of moduli?
        # No, sorting by 1/m is crucial.
        
        # Instead, let's try different "strategies" for residue selection.
        # Strategy A: First valid.
        # Strategy B: Random valid.
        
        use_random_residue = (attempt % 2 == 1)
        
        for m in all_moduli:
            if time.time() > deadline - 30:
                break
            
            # Find a valid residue
            r = None
            if use_random_residue:
                valid_rs = []
                for rr in range(m):
                    ok = True
                    for i in range(len(current_moduli)):
                        m2 = current_moduli[i]
                        g = gcd(m, m2)
                        if (rr - current_residues[i]) % g == 0:
                            ok = False
                            break
                    if ok:
                        valid_rs.append(rr)
                if valid_rs:
                    r = random.choice(valid_rs)
            else:
                r = find_valid_residue(m, current_moduli, current_residues)
                
            if r is not None:
                current_moduli.append(m)
                current_residues.append(r)
        
        if current_moduli:
            s = score_of(current_moduli)
            if s > best_greedy_score:
                best_greedy_score = s
                best_greedy_classes = [[current_moduli[i], current_residues[i]] for i in range(len(current_moduli))]
    
    if best_greedy_classes:
        best_score = best_greedy_score
        best_classes = best_greedy_classes
    
    # Strategy 2: Local Search (Hill Climbing)
    # Start from best_greedy_classes
    if best_classes and time.time() < deadline - 5:
        current_moduli = [c[0] for c in best_classes]
        current_residues = [c[1] for c in best_classes]
        current_score = best_score
        
        # Set of available moduli
        selected_set = set(current_moduli)
        available_moduli = [m for m in all_moduli if m not in selected_set]
        
        # Randomize available moduli for diversity
        random.shuffle(available_moduli)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try to swap one in, one out
            # To be efficient, we don't restart the loop immediately, we just keep trying swaps
            # until no improvement is found in a full pass or time runs out.
            
            # We'll do a few passes
            for _ in range(10):
                if time.time() > deadline - 5:
                    break
                
                # Pick a random modulus to remove
                if not current_moduli:
                    break
                out_idx = random.randint(0, len(current_moduli) - 1)
                m_out = current_moduli[out_idx]
                
                # Pick a random modulus to add
                if not available_moduli:
                    break
                in_idx = random.randint(0, len(available_moduli) - 1)
                m_in = available_moduli[in_idx]
                
                # Check if m_in can be added to (current_moduli \ {m_out})
                # We need to check compatibility with all other moduli.
                # We keep their residues fixed.
                
                trial_moduli = []
                trial_residues = []
                for i in range(len(current_moduli)):
                    if i != out_idx:
                        trial_moduli.append(current_moduli[i])
                        trial_residues.append(current_residues[i])
                
                r_in = find_valid_residue(m_in, trial_moduli, trial_residues)
                
                if r_in is not None:
                    # Calculate new score
                    # Old score - 1/m_out + 1/m_in
                    delta = Fraction(1, m_in) - Fraction(1, m_out)
                    if delta > 0:
                        # Accept swap
                        current_moduli[out_idx] = m_in
                        current_residues[out_idx] = r_in
                        current_score += delta
                        
                        # Update sets
                        selected_set.remove(m_out)
                        selected_set.add(m_in)
                        
                        # Update available list
                        # This is O(N) to remove m_in and add m_out
                        available_moduli[in_idx] = m_out
                        # m_out is now available, m_in is now selected
                        # We don't strictly need to maintain sorted order for available, just membership
                        # But we need to ensure m_in is removed from available logic.
                        # Since we picked in_idx, we replaced it.
                        # But wait, m_out might already be in available? No, it was selected.
                        # So we just swap the values in the lists.
                        
                        improved = True
                        best_score = current_score
                        best_classes = [[current_moduli[i], current_residues[i]] for i in range(len(current_moduli))]
                        
                        if time.time() > deadline - 5:
                            break
    
    # Ensure we have at least the seed result
    if not best_classes:
        best_classes = [[20 * j, j - 1] for j in range(1, 11)]
        best_score = sum(Fraction(1, 20*j) for j in range(1, 11))
    
    return best_classes
# EVOLVE-BLOCK-END
