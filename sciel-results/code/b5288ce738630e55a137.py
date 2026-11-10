import math
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    all_moduli = list(range(20, 201))
    
    # Precompute gcd table for all pairs
    gcd_table = {}
    for i, m1 in enumerate(all_moduli):
        for j, m2 in enumerate(all_moduli):
            if i != j:
                gcd_table[(m1, m2)] = math.gcd(m1, m2)
    
    def score_of(selected):
        """Compute sum of 1/m for selected moduli"""
        total = 0.0
        for m, r in selected:
            total += 1.0 / m
        return total
    
    def is_valid_assignment(selected):
        """Check if the assignment is pairwise disjoint"""
        n = len(selected)
        for i in range(n):
            for j in range(i + 1, n):
                m1, r1 = selected[i]
                m2, r2 = selected[j]
                g = gcd_table[(m1, m2)]
                if r1 % g == r2 % g:
                    return False
        return True
    
    def find_valid_residues(m_new, selected):
        """Find all valid residues for m_new given selected moduli"""
        valid = []
        for r in range(m_new):
            valid_flag = True
            for m_sel, r_sel in selected:
                g = gcd_table[(m_new, m_sel)]
                if r % g == r_sel % g:
                    valid_flag = False
                    break
            if valid_flag:
                valid.append(r)
        return valid
    
    def greedy_construct():
        """Greedy: add moduli in increasing order, pick smallest valid residue"""
        selected = []
        used = set()
        for m in all_moduli:
            if time.time() > deadline:
                break
            residues = find_valid_residues(m, selected)
            if residues:
                # Pick a random residue to add diversity
                r = random.choice(residues)
                selected.append((m, r))
                used.add(m)
        return selected
    
    def local_search(selected, max_iters=5000):
        """Local search: try additions, swaps, and residue changes"""
        current = list(selected)
        current_score = score_of(current)
        used = set(m for m, r in current)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # 1. Try to add an unused modulus
            added = False
            unused = [m for m in all_moduli if m not in used]
            random.shuffle(unused)
            for m_new in unused[:20]:  # Limit checks per iteration
                residues = find_valid_residues(m_new, current)
                if residues:
                    r_new = random.choice(residues)
                    new_current = current + [(m_new, r_new)]
                    new_score = current_score + 1.0 / m_new
                    if new_score > current_score:
                        current = new_current
                        current_score = new_score
                        used.add(m_new)
                        added = True
                        break
            
            if added:
                continue
            
            # 2. Try to swap: remove one, add one smaller
            swap_done = False
            existing = list(current)
            random.shuffle(existing)
            for m_old, r_old in existing[:15]:
                if time.time() > deadline:
                    break
                unused = [m for m in all_moduli if m not in used and m < m_old]
                if not unused:
                    continue
                random.shuffle(unused)
                for m_new in unused[:10]:
                    # Remove m_old, try to add m_new
                    temp = [(m, r) for m, r in current if m != m_old]
                    residues = find_valid_residues(m_new, temp)
                    if residues:
                        r_new = random.choice(residues)
                        new_current = temp + [(m_new, r_new)]
                        new_score = current_score - 1.0 / m_old + 1.0 / m_new
                        if new_score > current_score:
                            current = new_current
                            current_score = new_score
                            used.discard(m_old)
                            used.add(m_new)
                            swap_done = True
                            break
                if swap_done:
                    break
            
            if swap_done:
                continue
            
            # 3. Try to change a residue to enable an addition
            residue_changed = False
            existing = list(current)
            random.shuffle(existing)
            for m_old, r_old in existing[:10]:
                if time.time() > deadline:
                    break
                # Try different residues for m_old
                other_residues = [r for r in range(m_old) if r != r_old]
                random.shuffle(other_residues)
                for r_new in other_residues[:5]:
                    # Check if this residue is valid with others
                    valid = True
                    for m2, r2 in current:
                        if m2 == m_old:
                            continue
                        g = gcd_table[(m_old, m2)]
                        if r_new % g == r2 % g:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Try this new residue
                    new_current = [(m, r_new if m == m_old else r) for m, r in current]
                    
                    # Now try to add an unused modulus
                    unused = [m for m in all_moduli if m not in used]
                    random.shuffle(unused)
                    for m_add in unused[:10]:
                        residues_add = find_valid_residues(m_add, new_current)
                        if residues_add:
                            r_add = random.choice(residues_add)
                            final_current = new_current + [(m_add, r_add)]
                            final_score = current_score + 1.0 / m_add
                            if final_score > current_score:
                                current = final_current
                                current_score = final_score
                                used.add(m_add)
                                residue_changed = True
                                break
                    if residue_changed:
                        break
                if residue_changed:
                    break
            
            if not added and not swap_done and not residue_changed:
                # Stuck, break out
                break
        
        return current
    
    # Main search: multiple restarts with local search
    best_solution = None
    best_score = -1.0
    
    # Initial greedy construction
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Greedy with random residue choice
        selected = greedy_construct()
        
        # Local search
        improved = local_search(selected, max_iters=3000)
        
        s = score_of(improved)
        if s > best_score:
            best_score = s
            best_solution = improved
        
        # If we have a good score, reduce local search iterations to do more restarts
        if best_score > 3.0:
            pass  # Continue with current settings
    
    # Final: try to add any remaining moduli to best_solution
    if best_solution is not None:
        used = set(m for m, r in best_solution)
        unused = [m for m in all_moduli if m not in used]
        random.shuffle(unused)
        for m in unused:
            if time.time() > deadline:
                break
            residues = find_valid_residues(m, best_solution)
            if residues:
                r = random.choice(residues)
                best_solution.append((m, r))
                used.add(m)
    
    if best_solution is None:
        # Fallback: just return all moduli with residue 0 (may be invalid but better than nothing)
        best_solution = [[m, 0] for m in all_moduli]
    
    return [[m, r] for m, r in best_solution]
