import random
import time

def construct():
    start_time = time.time()
    deadline = 85.0  # 5 seconds margin
    
    best_set = []
    
    # Precompute gcds? No, just use math.gcd.
    
    def is_valid(m, r, current_set):
        for r_j, m_j in current_set:
            g = __import__('math').gcd(m, m_j)
            if (r - r_j) % g == 0:
                return False
        return True

    def find_residue(m, current_set):
        # Try to find a residue r for m
        # Heuristic: try random residues first? Or sequential?
        # Sequential is deterministic. Random might escape local minima.
        candidates = list(range(m))
        random.shuffle(candidates)
        for r in candidates:
            if is_valid(m, r, current_set):
                return r
        return None

    def greedy_attempt():
        moduli = list(range(2, 129))
        random.shuffle(moduli)
        current_set = []
        
        for m in moduli:
            r = find_residue(m, current_set)
            if r is not None:
                current_set.append([r, m])
        
        return current_set

    # Run multiple attempts
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        s = greedy_attempt()
        if len(s) > len(best_set):
            best_set = s
        
        # Early exit if we found a very large set?
        # Max possible is 127 (all moduli 2..128). 
        # But pairwise non-coprime limits this.
        # Even numbers: 64.
        # If we find 64, it's likely optimal or near optimal.
        if len(best_set) >= 60:
             break

    return best_set
