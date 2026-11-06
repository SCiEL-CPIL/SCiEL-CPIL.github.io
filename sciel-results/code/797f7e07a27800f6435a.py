"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Precompute GCDs for all pairs in [20, 200]
    moduli_range = range(20, 201)
    all_moduli = list(moduli_range)
    n_moduli = len(all_moduli)
    
    # gcd_matrix[i][j] = gcd(all_moduli[i], all_moduli[j])
    # Index mapping: m -> index
    m_to_idx = {m: i for i, m in enumerate(all_moduli)}
    
    gcd_matrix = [[0] * n_moduli for _ in range(n_moduli)]
    for i in range(n_moduli):
        for j in range(i, n_moduli):
            g = gcd(all_moduli[i], all_moduli[j])
            gcd_matrix[i][j] = g
            gcd_matrix[j][i] = g

    def is_compatible(m_idx, r, selected_indices, selected_residues):
        """Check if residue r for modulus at m_idx is compatible with all selected."""
        for k in selected_indices:
            g = gcd_matrix[m_idx][k]
            if (r - selected_residues[k]) % g == 0:
                return False
        return True

    def find_valid_residues(m_idx, selected_indices, selected_residues):
        """Find all valid residues for modulus at m_idx."""
        valid = []
        m = all_moduli[m_idx]
        for r in range(m):
            if is_compatible(m_idx, r, selected_indices, selected_residues):
                valid.append(r)
        return valid

    def score_of(selected_indices):
        return sum(Fraction(1, all_moduli[i]) for i in selected_indices)

    def greedy_construct(randomize=True):
        """Construct a solution using greedy approach."""
        selected_indices = []
        selected_residues = []
        
        # Process moduli in increasing order (prioritize high weight)
        for i in range(n_moduli):
            if time.time() > deadline - 5:
                break
            valid_residues = find_valid_residues(i, selected_indices, selected_residues)
            if valid_residues:
                if randomize:
                    r = random.choice(valid_residues)
                else:
                    r = valid_residues[0]
                selected_indices.append(i)
                selected_residues.append(r)
                
        return selected_indices, selected_residues

    best_score = Fraction(0, 1)
    best_indices = []
    best_residues = []

    # Phase 1: Multiple Greedy Attempts
    for _ in range(50):
        if time.time() > deadline - 20:
            break
        indices, residues = greedy_construct(randomize=True)
        s = score_of(indices)
        if s > best_score:
            best_score = s
            best_indices = indices
            best_residues = residues

    # Phase 2: Local Search
    if best_indices and time.time() < deadline - 5:
        current_indices = best_indices[:]
        current_residues = best_residues[:]
        current_score = best_score
        
        selected_set = set(current_indices)
        # available_indices are those not in selected_set
        # We want to add small moduli (low index) if possible
        
        def try_add(idx):
            """Try to add modulus idx. Returns (new_indices, new_residues, new_score) or None."""
            if idx in selected_set:
                return None
            valid_residues = find_valid_residues(idx, current_indices, current_residues)
            if valid_residues:
                # Pick a random valid residue to avoid bias
                r = random.choice(valid_residues)
                new_indices = current_indices + [idx]
                new_residues = current_residues + [r]
                new_score = current_score + Fraction(1, all_moduli[idx])
                return (new_indices, new_residues, new_score)
            return None

        def try_swap(out_idx, in_idx):
            """Try to swap out_idx with in_idx (in_idx must have smaller modulus)."""
            if in_idx in selected_set:
                return None
            if all_moduli[in_idx] >= all_moduli[out_idx]:
                return None # Only swap for better (smaller) modulus
            
            # Remove out_idx
            trial_indices = [i for i in current_indices if i != out_idx]
            trial_residues = [r for i, r in zip(current_indices, current_residues) if i != out_idx]
            
            valid_residues = find_valid_residues(in_idx, trial_indices, trial_residues)
            if valid_residues:
                r = random.choice(valid_residues)
                new_indices = trial_indices + [in_idx]
                new_residues = trial_residues + [r]
                new_score = current_score - Fraction(1, all_moduli[out_idx]) + Fraction(1, all_moduli[in_idx])
                return (new_indices, new_residues, new_score)
            return None

        def try_flip_and_add(flip_idx):
            """Try to change residue of flip_idx to enable adding a skipped modulus."""
            # Find a random valid residue for flip_idx w.r.t others
            others_indices = [i for i in current_indices if i != flip_idx]
            others_residues = [r for i, r in zip(current_indices, current_residues) if i != flip_idx]
            
            valid_residues = find_valid_residues(flip_idx, others_indices, others_residues)
            if not valid_residues:
                return None
            
            current_r = current_residues[current_indices.index(flip_idx)]
            if len(valid_residues) == 1 and valid_residues[0] == current_r:
                return None # No change possible
            
            # Pick a random different valid residue
            options = [r for r in valid_residues if r != current_r]
            if not options:
                return None
            
            new_r = random.choice(options)
            
            # Tentatively update
            trial_indices = current_indices[:]
            trial_residues = current_residues[:]
            idx_pos = current_indices.index(flip_idx)
            trial_residues[idx_pos] = new_r
            
            # Check if any skipped modulus can be added
            added = False
            new_indices = trial_indices
            new_residues = trial_residues
            new_score = current_score
            
            # Prioritize adding smallest available moduli
            # We can just check all not in selected_set
            candidates = [i for i in range(n_moduli) if i not in selected_set]
            # Sort candidates by modulus (ascending) to try best ones first
            candidates.sort(key=lambda i: all_moduli[i])
            
            for cand_idx in candidates:
                # Check if cand_idx is valid with the new residue configuration
                # Note: cand_idx is not in trial_indices
                valid_cand = find_valid_residues(cand_idx, new_indices, new_residues)
                if valid_cand:
                    r_cand = random.choice(valid_cand)
                    new_indices.append(cand_idx)
                    new_residues.append(r_cand)
                    new_score = new_score + Fraction(1, all_moduli[cand_idx])
                    added = True
                    break
            
            if added:
                return (new_indices, new_residues, new_score)
            return None

        improved = True
        stall_count = 0
        
        while improved and time.time() < deadline - 2:
            improved = False
            
            # 1. Try to add any available modulus (greedy: smallest first)
            candidates = [i for i in range(n_moduli) if i not in selected_set]
            candidates.sort(key=lambda i: all_moduli[i])
            
            for idx in candidates:
                if time.time() > deadline - 2:
                    break
                res = try_add(idx)
                if res:
                    current_indices, current_residues, current_score = res
                    selected_set.add(idx)
                    improved = True
                    best_score = current_score
                    best_indices = current_indices[:]
                    best_residues = current_residues[:]
                    stall_count = 0
                    break # Restart search with new state

            # 2. Try swaps (replace large m with small m)
            if not improved and time.time() < deadline - 2:
                # Randomly pick a selected index to remove
                # Prioritize removing larger moduli (lower priority) to make room for smaller ones
                # Sort selected by modulus descending
                selected_sorted = sorted(current_indices, key=lambda i: all_moduli[i], reverse=True)
                
                # Try a few random swaps
                for _ in range(10):
                    if time.time() > deadline - 2:
                        break
                    if not selected_sorted:
                        break
                    out_idx = random.choice(selected_sorted)
                    
                    # Find potential in_idx (smaller modulus, not selected)
                    potential_in = [i for i in range(n_moduli) if i not in selected_set and all_moduli[i] < all_moduli[out_idx]]
                    if not potential_in:
                        continue
                    
                    # Try the smallest few potential in_idx
                    potential_in.sort(key=lambda i: all_moduli[i])
                    for in_idx in potential_in[:5]:
                        res = try_swap(out_idx, in_idx)
                        if res:
                            current_indices, current_residues, current_score = res
                            selected_set.remove(out_idx)
                            selected_set.add(in_idx)
                            improved = True
                            best_score = current_score
                            best_indices = current_indices[:]
                            best_residues = current_residues[:]
                            stall_count = 0
                            break
                    if improved:
                        break

            # 3. Try flip and add (local search to escape optima)
            if not improved and time.time() < deadline - 2 and current_indices:
                # Pick a random selected index to flip
                flip_idx = random.choice(current_indices)
                res = try_flip_and_add(flip_idx)
                if res:
                    current_indices, current_residues, current_score = res
                    # Update selected_set
                    # The flip keeps the same indices, just adds one
                    # So we just need to add the new one
                    # The new one is the last element if it was added
                    # But try_flip_and_add returns the full lists
                    old_set = set(current_indices[:-1]) # Assuming only 1 added
                    # Actually, let's just rebuild selected_set
                    selected_set = set(current_indices)
                    improved = True
                    best_score = current_score
                    best_indices = current_indices[:]
                    best_residues = current_residues[:]
                    stall_count = 0

            if not improved:
                stall_count += 1
                if stall_count > 100:
                    break

    # Final validation and formatting
    if not best_indices:
        # Fallback
        best_indices, best_residues = greedy_construct(randomize=False)

    result = []
    for i, r in zip(best_indices, best_residues):
        result.append([all_moduli[i], r])
        
    return result
# EVOLVE-BLOCK-END
