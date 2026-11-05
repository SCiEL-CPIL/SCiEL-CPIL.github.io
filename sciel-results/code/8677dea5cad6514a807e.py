import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Moduli in [20, 200]
    moduli = list(range(20, 201))
    
    def is_compatible(m_new, r_new, selected_list):
        """Check if (m_new, r_new) is disjoint from all in selected_list."""
        for m_old, r_old in selected_list:
            g = math.gcd(m_new, m_old)
            # Check if r_new == r_old (mod g)
            if (r_new - r_old) % g == 0:
                return False
        return True

    def get_valid_residues(m_new, selected_list):
        """Return list of valid residues for m_new given selected_list."""
        valids = []
        for r in range(m_new):
            if is_compatible(m_new, r, selected_list):
                valids.append(r)
        return valids

    def run_greedy_randomized(seed=0):
        random.seed(seed)
        selected = []
        score = 0.0
        # Process in increasing order of m to prioritize high 1/m values
        for m in moduli:
            valids = get_valid_residues(m, selected)
            if valids:
                # Randomly pick a valid residue to diversify search
                r = random.choice(valids)
                selected.append([m, r])
                score += 1.0 / m
        return selected, score

    best_score = 0.0
    best_classes = []
    
    # Run multiple randomized trials
    # Each trial is fast (181 moduli, small compatibility checks)
    num_trials = 5000
    for i in range(num_trials):
        if time.time() > deadline:
            break
        classes, score = run_greedy_randomized(seed=i)
        if score > best_score:
            best_score = score
            best_classes = classes
            
    return best_classes
