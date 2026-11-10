import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    def check_conflict(res, m, selected):
        """Check if residue res with modulus m conflicts with any selected class."""
        for m2, r2 in selected:
            g = gcd(m, m2)
            if res % g == r2 % g:
                return True
        return False

    def find_valid_residue(m, selected):
        """Find any valid residue for m given selected set. Returns None if none."""
        for r in range(m):
            if not check_conflict(r, m, selected):
                return r
        return None

    def find_best_residue(m, selected):
        """Find a valid residue that minimizes future conflicts (heuristic)."""
        valid_residues = []
        for r in range(m):
            if not check_conflict(r, m, selected):
                valid_residues.append(r)
        if not valid_residues:
            return None
        
        # Heuristic: prefer residues that are "spread out" from existing ones
        # Count how many conflicts this residue would cause with potential future moduli
        best_r = valid_residues[0]
        best_score = -1
        
        for r in valid_residues:
            # Score: number of moduli in [20,200] that would NOT conflict with (m, r)
            # This is expensive, so use a simpler heuristic
            # Prefer residues that are coprime-friendly
            score = 0
            for m2 in range(20, 201):
                if m2 == m:
                    continue
                g = gcd(m, m2)
                # If r % g is "common", it's more likely to conflict
                # We want r % g to be a value that fewer other residues would take
                # Simplified: just count how many r2 in [0, m2) would conflict
                # But that's expensive. Use a proxy.
                pass
            
            # Simpler heuristic: prefer residues that are multiples of small primes
            # or just use the first valid one
            if score > best_score:
                best_score = score
                best_r = r
        
        return best_r

    def calculate_score(selected):
        total = Fraction(0, 1)
        for m, r in selected:
            total += Fraction(1, m)
        return total

    def is_valid_set(selected):
        """Check if the entire set is valid."""
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                m1, r1 = selected[i]
                m2, r2 = selected[j]
                g = gcd(m1, m2)
                if r1 % g == r2 % g:
                    return False
        return True

    # 1. Initial Greedy Construction: Ascending m (smallest first for max weight)
    best_selected = []
    best_score = Fraction(0, 1)
    
    for m in range(20, 201):
        if time.time() > deadline:
            break
        r = find_valid_residue(m, best_selected)
        if r is not None:
            best_selected.append([m, r])
    score1 = calculate_score(best_selected)
    best_score = score1
    best_selected = [x[:] for x in best_selected]
    
    # Try a few randomized restarts to see if we can do better
    num_restarts = 3
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        selected = []
        # Slightly randomized order: mostly ascending but with some shuffling
        moduli_list = list(range(20, 201))
        # Keep small moduli mostly in order but allow some randomness
        for i in range(len(moduli_list)):
            if random.random() < 0.1:
                # Swap with a nearby element
                j = min(i + random.randint(1, 5), len(moduli_list) - 1)
                moduli_list[i], moduli_list[j] = moduli_list[j], moduli_list[i]
        
        for m in moduli_list:
            if time.time() > deadline:
                break
            r = find_valid_residue(m, selected)
            if r is not None:
                selected.append([m, r])
        
        score = calculate_score(selected)
        if score > best_score:
            best_score = score
            best_selected = [x[:] for x in selected]
            
    selected = best_selected
    current_score = best_score
    
    # Identify missing moduli
    selected_moduli = set(m for m, r in selected)
    missing_moduli = [m for m in range(20, 201) if m not in selected_moduli]
    missing_moduli.sort()

    # 2. Local Search: Try to add missing moduli
    no_improvement_count = 0
    max_no_improvement = 200
    
    while time.time() < deadline and no_improvement_count < max_no_improvement:
        if not missing_moduli:
            break
            
        # Strategy 1: Add a missing modulus by perturbing one existing residue
        # Prefer smaller missing moduli (higher weight)
        if random.random() < 0.8:
            target_m = missing_moduli[0]
        else:
            target_m = random.choice(missing_moduli[:min(10, len(missing_moduli))])
            
        if not selected:
            break
            
        found_improvement = False
        
        # Try perturbing different victims
        for _ in range(5):
            victim_idx = random.randint(0, len(selected) - 1)
            victim_m, victim_r = selected[victim_idx]
            
            # Try a few random new residues for victim
            for _ in range(15):
                new_victim_r = random.randint(0, victim_m - 1)
                if new_victim_r == victim_r:
                    continue
                    
                old_r = selected[victim_idx][1]
                selected[victim_idx][1] = new_victim_r
                
                # Check if target_m can be added now
                target_r = find_valid_residue(target_m, selected)
                if target_r is not None:
                    # We can add target_m!
                    new_score = current_score + Fraction(1, target_m)
                    selected.append([target_m, target_r])
                    current_score = new_score
                    if target_m in missing_moduli:
                        missing_moduli.remove(target_m)
                    found_improvement = True
                    no_improvement_count = 0
                    break
                
                # Revert victim if not accepted
                selected[victim_idx][1] = old_r
                
            if found_improvement:
                break
        
        if found_improvement:
            continue
            
        # Strategy 2: Swap a selected modulus for a smaller missing one
        # Pick victim from larger moduli, target from smaller missing
        better_targets = [m for m in missing_moduli if m < 100]  # Focus on high-value targets
        if better_targets:
            target_m = random.choice(better_targets)
            # Find a victim with m > target_m
            victims = [(i, m) for i, (m, r) in enumerate(selected) if m > target_m]
            if victims:
                victim_idx, victim_m = random.choice(victims)
                victim_r = selected[victim_idx][1]
                
                # Remove victim
                selected.pop(victim_idx)
                
                # Try to add target
                target_r = find_valid_residue(target_m, selected)
                if target_r is not None:
                    selected.append([target_m, target_r])
                    current_score += Fraction(1, target_m) - Fraction(1, victim_m)
                    if target_m in missing_moduli:
                        missing_moduli.remove(target_m)
                    missing_moduli.append(victim_m)
                    missing_moduli.sort()
                    no_improvement_count = 0
                else:
                    # Revert
                    selected.append([victim_m, victim_r])
        
        # Track no improvement
        no_improvement_count += 1

    # 3. Final polish: Try to add any remaining missing moduli with more aggressive search
    if missing_moduli and time.time() < deadline - 5:
        for target_m in missing_moduli[:]:
            if time.time() > deadline - 2:
                break
            
            # Try all possible victims and residues
            for victim_idx in range(len(selected)):
                if time.time() > deadline - 2:
                    break
                victim_m, victim_r = selected[victim_idx]
                old_r = victim_r
                
                # Try all residues for victim
                for new_r in range(victim_m):
                    if new_r == old_r:
                        continue
                    selected[victim_idx][1] = new_r
                    
                    target_r = find_valid_residue(target_m, selected)
                    if target_r is not None:
                        selected.append([target_m, target_r])
                        current_score += Fraction(1, target_m)
                        if target_m in missing_moduli:
                            missing_moduli.remove(target_m)
                        break
                    selected[victim_idx][1] = old_r
                
                if target_m not in missing_moduli:
                    break
            
            if not missing_moduli:
                break

    # Final validation
    if not is_valid_set(selected):
        # Fallback to greedy if something went wrong
        selected = []
        for m in range(20, 201):
            r = find_valid_residue(m, selected)
            if r is not None:
                selected.append([m, r])

    return selected
