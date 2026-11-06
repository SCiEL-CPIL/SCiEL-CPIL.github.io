"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # All possible moduli
    all_moduli = list(range(20, 201))
    
    def is_compatible(m_new, r_new, chosen):
        """
        Check if (m_new, r_new) is compatible with a list of (m_i, r_i).
        """
        for m_i, r_i in chosen:
            g = gcd(m_new, m_i)
            if (r_new - r_i) % g == 0:
                return False
        return True

    def find_compatible_residue(m_new, chosen, randomize=False):
        """
        Find a residue r for m_new that is compatible with 'chosen'.
        Returns a random compatible residue if randomize=True, else the first one.
        Returns None if no compatible residue exists.
        """
        compatible_rs = []
        for r in range(m_new):
            if is_compatible(m_new, r, chosen):
                if randomize:
                    compatible_rs.append(r)
                else:
                    return r
        if randomize and compatible_rs:
            return random.choice(compatible_rs)
        return None

    def try_add_moduli(chosen, missing_moduli):
        """
        Greedily add any missing modulus that fits.
        Repeats until no more moduli can be added.
        Returns the updated list of chosen moduli and the list of missing moduli.
        """
        improved = True
        # Sort missing moduli by value ascending (highest priority)
        missing_moduli.sort()
        
        while improved and missing_moduli:
            improved = False
            # We iterate through missing moduli. 
            # If we add one, we might need to re-check others, so we break and restart the while loop.
            # To be efficient, we can just check them in order.
            for i in range(len(missing_moduli)):
                m = missing_moduli[i]
                r = find_compatible_residue(m, chosen, randomize=False)
                if r is not None:
                    chosen.append((m, r))
                    missing_moduli.pop(i)
                    improved = True
                    break # Restart to check if adding this one opened up others or if order matters
        return chosen, missing_moduli

    def local_search(chosen, deadline):
        """
        Improve the solution using swaps and residue re-assignment.
        """
        current_score = sum(Fraction(1, m) for m, r in chosen)
        selected_set = set(m for m, r in chosen)
        all_moduli_set = set(range(20, 201))
        missing_moduli = [m for m in all_moduli_set if m not in selected_set]
        
        # First, ensure the set is maximal (no more can be added)
        chosen, missing_moduli = try_add_moduli(chosen, missing_moduli)
        current_score = sum(Fraction(1, m) for m, r in chosen)
        selected_set = set(m for m, r in chosen)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try to add any missing modulus again (in case residue changes opened up space, 
            #    though we haven't changed residues yet in this iteration, it's a safe check)
            chosen, missing_moduli = try_add_moduli(chosen, missing_moduli)
            new_score = sum(Fraction(1, m) for m, r in chosen)
            if new_score > current_score:
                current_score = new_score
                improved = True
                selected_set = set(m for m, r in chosen)
                missing_moduli = [m for m in all_moduli_set if m not in selected_set]
                continue

            if not chosen or not missing_moduli:
                break

            # 2. Try swaps
            # We want to replace a large m_out with a small m_in
            # Sort chosen by m descending (worst first)
            # Sort missing by m ascending (best first)
            
            # To avoid O(N^2) with high constant, we limit the candidates
            # But N is small (max ~100 moduli), so we can be more thorough.
            
            # Identify candidates for removal: the largest moduli in the current set
            # Identify candidates for addition: the smallest moduli in the missing set
            
            # We'll try to swap the worst few with the best few
            # Let's take the top 10 largest from chosen and top 10 smallest from missing
            chosen_sorted = sorted(chosen, key=lambda x: -x[0])
            missing_sorted = sorted(missing_moduli)
            
            # Limit search space to speed up
            # If we have time, we can expand this
            num_candidates = 15
            
            worst_candidates = chosen_sorted[:num_candidates]
            best_candidates = missing_sorted[:num_candidates]
            
            swap_made = False
            for m_out, r_out in worst_candidates:
                if swap_made:
                    break
                if time.time() > deadline:
                    break
                    
                # Temporarily remove m_out
                idx = -1
                for i, (m, r) in enumerate(chosen):
                    if m == m_out:
                        idx = i
                        break
                if idx == -1:
                    continue
                
                chosen_temp = chosen[:idx] + chosen[idx+1:]
                
                for m_in in best_candidates:
                    if m_in >= m_out:
                        continue # We only want to improve score, so m_in must be smaller
                    
                    # Check if m_in can be added to chosen_temp
                    r_in = find_compatible_residue(m_in, chosen_temp, randomize=False)
                    
                    if r_in is None:
                        # Try to repair: randomly change residues of some other moduli 
                        # to see if m_in can fit
                        # We'll try a few random re-assignments
                        found = False
                        for _ in range(5):
                            # Create a copy of chosen_temp
                            trial_chosen = [(m, r) for m, r in chosen_temp]
                            
                            # Randomly pick 2-3 moduli to re-assign
                            num_reassign = min(3, len(trial_chosen))
                            if num_reassign == 0:
                                continue
                            
                            indices_to_change = random.sample(range(len(trial_chosen)), num_reassign)
                            
                            # Try to find valid residues for these indices
                            # This is a mini-CSP. We just try random valid residues for them.
                            valid_reassign = True
                            for idx_change in indices_to_change:
                                m_c, r_c = trial_chosen[idx_change]
                                # Find a random compatible residue for m_c given the OTHERS
                                # This is tricky because they depend on each other.
                                # Simpler: just try random residues for the whole set? No, too slow.
                                # Alternative: Just try to find ANY residue for m_in in the context of trial_chosen.
                                # If we just change residues of others, we need to ensure they remain valid.
                                
                                # Let's simplify: Just try to find a residue for m_in with the current trial_chosen.
                                # If it fails, we haven't actually changed trial_chosen yet.
                                # Let's try a different repair strategy:
                                # Remove m_out. Try to add m_in. If it fails, 
                                # pick a random other modulus m_k, remove it, 
                                # try to add m_in. If success, we effectively swapped m_out for m_in 
                                # and dropped m_k? No, we want to keep the count or improve it.
                                # Actually, if we drop m_k, we lose 1/m_k. 
                                # We gain 1/m_in. We lose 1/m_out.
                                # Net change: 1/m_in - 1/m_out - 1/m_k.
                                # This might be negative.
                                
                                # Let's stick to 1-for-1 swaps first.
                                # If 1-for-1 fails, we might try 2-for-1 or 1-for-2 later.
                                # For now, if r_in is None, we skip this m_in.
                                pass
                            
                            # Since the repair logic above was complex and potentially slow/ineffective,
                            # let's just accept that if it doesn't fit immediately, it doesn't fit.
                            # However, the "Maximalization" step (try_add_moduli) already handles
                            # cases where the set wasn't maximal.
                            # The main gain comes from finding a swap that IS compatible.
                            
                            # To improve compatibility, we can randomize the residue of m_in?
                            # find_compatible_residue already picks the first one.
                            # Let's try a random one.
                            r_in_random = find_compatible_residue(m_in, chosen_temp, randomize=True)
                            if r_in_random is not None:
                                r_in = r_in_random
                                found = True
                                break
                            
                    if r_in is not None:
                        # Accept swap
                        chosen = chosen_temp
                        chosen.append((m_in, r_in))
                        
                        # Update sets
                        selected_set.remove(m_out)
                        selected_set.add(m_in)
                        
                        # Update missing list
                        # m_in is removed from missing, m_out is added to missing
                        if m_in in missing_moduli:
                            missing_moduli.remove(m_in)
                        missing_moduli.append(m_out)
                        
                        current_score += Fraction(1, m_in) - Fraction(1, m_out)
                        improved = True
                        swap_made = True
                        break # Restart swap phase with new state

        return chosen, float(current_score)

    best_classes = []
    best_score = 0.0
    
    # Run multiple iterations with different random seeds/residue choices
    num_iterations = 20
    
    for i in range(num_iterations):
        if time.time() > deadline:
            break
        
        # Greedy Construction
        # Sort moduli by 1/m (ascending m)
        chosen = []
        # We can randomize the order slightly or just use sorted
        # Using sorted is usually best for greedy
        for m in all_moduli:
            # Use random residue selection to escape local minima
            r = find_compatible_residue(m, chosen, randomize=True)
            if r is not None:
                chosen.append((m, r))
        
        # Local Search
        chosen, score = local_search(chosen, deadline)
        
        if score > best_score:
            best_score = score
            best_classes = chosen
            
        # Early stop if score is very high
        if best_score > 2.5:
            break

    # Fallback
    if not best_classes:
        for m in all_moduli:
            r = find_compatible_residue(m, best_classes, randomize=False)
            if r is not None:
                best_classes.append((m, r))
        best_classes, _ = local_search(best_classes, deadline)

    # Format result
    result = [[m, r] for m, r in best_classes]
    return result
# EVOLVE-BLOCK-END
