import math
import time
import random

def construct():
    deadline = time.time() + 115  # Safety margin
    
    # Moduli range [20, 200]
    all_moduli = list(range(20, 201))
    
    # Precompute gcds
    gcd_cache = {}
    def get_gcd(m1, m2):
        key = (m1, m2) if m1 < m2 else (m2, m1)
        if key not in gcd_cache:
            gcd_cache[key] = math.gcd(key[0], key[1])
        return gcd_cache[key]

    def is_valid(m, r, selected):
        """Check if (m, r) is disjoint from all in selected."""
        for m2, r2 in selected:
            g = get_gcd(m, m2)
            if r % g == r2 % g:
                return False
        return True

    def find_valid_residues(m, selected):
        """Find all valid residues for m given selected."""
        valid = []
        for r in range(m):
            if is_valid(m, r, selected):
                valid.append(r)
        return valid

    def spread_score(m, r, selected):
        """Heuristic: How 'spread out' is r from existing residues?"""
        score = 0
        for m2, r2 in selected:
            g = get_gcd(m, m2)
            if g == 1:
                continue
            d1 = r % g
            d2 = r2 % g
            diff = abs(d1 - d2)
            dist = min(diff, g - diff)
            score += dist
        return score

    def greedy_construct(seed):
        """Greedy construction with randomized tie-breaking based on spread score."""
        rng = random.Random(seed)
        selected = []
        # Sort by modulus to prioritize small m (high 1/m)
        # But we might want to skip some to pack better? 
        # Greedy by smallest m is usually best for sum 1/m.
        for m in all_moduli:
            valid_rs = find_valid_residues(m, selected)
            if not valid_rs:
                continue
            
            # If only one choice, take it
            if len(valid_rs) == 1:
                selected.append((m, valid_rs[0]))
                continue
            
            # Calculate spread score for each candidate
            # To speed up, we can sample if list is large, but m <= 200 is small
            scored_rs = []
            for r in valid_rs:
                s = spread_score(m, r, selected)
                # Add small random noise to break symmetry
                s += rng.random() * 0.1
                scored_rs.append((s, r))
            
            # Pick best spread
            scored_rs.sort(reverse=True)
            best_r = scored_rs[0][1]
            selected.append((m, best_r))
            
        return selected

    def local_search(classes, seed):
        """Improve by adding missing moduli and swapping residues."""
        rng = random.Random(seed)
        classes = [c[:] for c in classes]
        present_moduli = set(m for m, _ in classes)
        missing_moduli = [m for m in all_moduli if m not in present_moduli]
        # Sort missing by weight (1/m) descending
        missing_moduli.sort(key=lambda x: 1.0/x)
        
        best_score = sum(1.0/m for m, _ in classes)
        best_classes = [c[:] for c in classes]
        
        no_improve_count = 0
        max_no_improve = 500
        iter_count = 0
        
        while time.time() < deadline - 5 and no_improve_count < max_no_improve:
            iter_count += 1
            
            # 1. Try to add a missing modulus
            added = False
            for m in missing_moduli[:10]: # Only try top 10 missing
                valid_rs = find_valid_residues(m, classes)
                if valid_rs:
                    # Pick best spread
                    best_r = max(valid_rs, key=lambda r: spread_score(m, r, classes))
                    classes.append([m, best_r])
                    present_moduli.add(m)
                    missing_moduli.remove(m)
                    current_score = sum(1.0/m for m, _ in classes)
                    if current_score > best_score:
                        best_score = current_score
                        best_classes = [c[:] for c in classes]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    added = True
                    break
            
            if added:
                continue
                
            # 2. No addition possible. Try to swap a residue to enable addition.
            # Pick a random existing modulus to perturb
            if classes:
                idx = rng.randint(0, len(classes) - 1)
                m_i, r_i = classes[idx]
                
                # Find valid residues for m_i excluding itself
                others = [c for j, c in enumerate(classes) if j != idx]
                valid_rs = find_valid_residues(m_i, others)
                
                if len(valid_rs) > 1:
                    # Pick a random different residue
                    new_r = rng.choice(valid_rs)
                    if new_r != r_i:
                        classes[idx] = [m_i, new_r]
                        # Check if this allows adding a missing modulus
                        # We don't want to do a full scan every swap, 
                        # but we can check the top missing ones
                        for m_miss in missing_moduli[:5]:
                            v_rs = find_valid_residues(m_miss, classes)
                            if v_rs:
                                best_r = max(v_rs, key=lambda r: spread_score(m_miss, classes))
                                classes.append([m_miss, best_r])
                                present_moduli.add(m_miss)
                                missing_moduli.remove(m_miss)
                                current_score = sum(1.0/m for m, _ in classes)
                                if current_score > best_score:
                                    best_score = current_score
                                    best_classes = [c[:] for c in classes]
                                    no_improve_count = 0
                                else:
                                    no_improve_count += 1
                                break
                        else:
                            # No addition enabled, just count as no improve
                            no_improve_count += 1
            else:
                no_improve_count += 1
                
        return best_classes, best_score

    best_overall_score = 0
    best_overall_classes = []
    
    # Run multiple greedy restarts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        # Greedy phase
        init_classes = greedy_construct(i)
        
        # Local Search phase
        improved_classes, score = local_search(init_classes, i + 1000)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_classes = improved_classes
            
        # Early termination if we hit a very high score?
        # Sum of 1/m for m=20..200 approx integral 20 to 200 of 1/x dx = ln(10) ~ 2.3
        # Max possible is sum of all 1/m = ~2.99
        # If we get close to 2.99, we can stop.
        if best_overall_score > 2.95:
            break

    # Convert to required format
    return [[m, r] for m, r in best_overall_classes]
