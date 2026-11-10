import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety
    
    def score(selected):
        return sum(1.0 / m for m, r in selected)
    
    def can_add(m, r, selected):
        """Check if (m, r) can be added to selected without conflicts."""
        for existing_m, existing_r in selected:
            g = math.gcd(m, existing_m)
            if g == 1:
                return False
            if r % g == existing_r % g:
                return False
        return True
    
    def find_valid_residue(m, selected):
        """Find a residue r for m compatible with all in selected."""
        for r in range(m):
            if can_add(m, r, selected):
                return r
        return -1
    
    def greedy_construct(order):
        """Greedy construction given an ordering of moduli."""
        selected = []
        for m in order:
            r = find_valid_residue(m, selected)
            if r != -1:
                selected.append((m, r))
        return selected
    
    def local_search(selected, time_limit):
        """Try to improve by swapping or reassigning residues."""
        best = list(selected)
        best_score = score(best)
        
        while time.time() < time_limit:
            improved = False
            # Try removing one element and re-adding it with a different residue
            for i in range(len(best)):
                m_i, r_i = best[i]
                # Remove it
                remaining = [x for j, x in enumerate(best) if j != i]
                
                # Try all possible residues for m_i
                for r_try in range(m_i):
                    if can_add(m_i, r_try, remaining):
                        candidate = remaining + [(m_i, r_try)]
                        s = score(candidate)
                        if s > best_score + 1e-12:
                            best = candidate
                            best_score = s
                            improved = True
                            break
                
                # Try replacing with a different modulus
                for m_new in range(20, 201):
                    if m_new == m_i:
                        continue
                    if any(m_new == existing_m for existing_m, _ in remaining):
                        continue
                    r_new = find_valid_residue(m_new, remaining)
                    if r_new != -1:
                        candidate = remaining + [(m_new, r_new)]
                        s = score(candidate)
                        if s > best_score + 1e-12:
                            best = candidate
                            best_score = s
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                # Try adding a new modulus
                added = False
                for m_new in range(20, 201):
                    if any(m_new == existing_m for existing_m, _ in best):
                        continue
                    r_new = find_valid_residue(m_new, best)
                    if r_new != -1:
                        best = best + [(m_new, r_new)]
                        best_score = score(best)
                        added = True
                        break
                if not added:
                    break
        
        return best
    
    # Phase 1: Greedy construction with natural order
    natural_order = list(range(20, 201))
    selected = greedy_construct(natural_order)
    
    # Phase 2: Local search to improve
    if time.time() < deadline - 10:
        selected = local_search(selected, deadline - 5)
    
    # Phase 3: Try multiple random restarts with shuffled orders
    best_selected = selected
    best_score = score(selected)
    
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        random.seed(42)
        max_restarts = int(remaining_time / 2)  # ~2s per restart estimate
        max_restarts = min(max_restarts, 50)
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            # Create a shuffled order but bias toward small moduli
            # Use weighted shuffle: smaller moduli more likely to come first
            weights = [1.0 / m for m in range(20, 201)]
            shuffled = list(range(20, 201))
            for i in range(len(shuffled) - 1, 0, -1):
                # Weighted selection
                total_w = sum(weights[j] for j in range(i + 1))
                rand_val = random.random() * total_w
                cum = 0
                chosen = 0
                for j in range(i + 1):
                    cum += weights[j]
                    if rand_val <= cum:
                        chosen = j
                        break
                shuffled[i], shuffled[chosen] = shuffled[chosen], shuffled[i]
            
            candidate = greedy_construct(shuffled)
            cand_score = score(candidate)
            
            if cand_score > best_score:
                best_score = cand_score
                best_selected = candidate
                # Quick local search on the new best
                if time.time() < deadline - 15:
                    best_selected = local_search(best_selected, deadline - 5)
                    best_score = score(best_selected)
    
    # Convert to required format [[m, r], ...]
    return [[m, r] for m, r in best_selected]
