import math
import random
import time

def construct():
    """Find a large set of pairwise disjoint residue classes with distinct moduli 2..128."""
    deadline = time.time() + 85  # safety margin before 90s limit
    MAX_MOD = 128
    
    # All possible moduli
    moduli = list(range(2, MAX_MOD + 1))
    
    # Precompute gcd values between all pairs of moduli
    # For moduli m1, m2 with gcd g, two classes (r1, m1) and (r2, m2) are disjoint
    # iff r1 != r2 mod g
    # We can check: (r1 - r2) % g != 0
    
    def are_disjoint(r1, m1, r2, m2):
        g = math.gcd(m1, m2)
        return (r1 - r2) % g != 0
    
    def can_add(r, m, existing):
        for (er, em) in existing:
            g = math.gcd(m, em)
            if (r - er) % g == 0:
                return False
        return True
    
    def find_valid_residue(m, existing):
        """Try to find a valid residue for modulus m given existing classes.
        Returns the residue that minimizes damage (fewest future conflicts) if possible,
        otherwise returns -1."""
        valid_residues = []
        for r in range(m):
            if can_add(r, m, existing):
                valid_residues.append(r)
        
        if not valid_residues:
            return -1
        
        # Score each valid residue by how many other moduli it would block
        # A residue r for modulus m blocks residue r' for modulus m' if (r-r')%gcd(m,m')==0
        # We want to minimize the number of blocked residues across all unused moduli
        best_r = valid_residues[0]
        best_score = float('inf')
        
        for r in valid_residues:
            score = 0
            for m2 in moduli:
                if m2 == m:
                    continue
                if any(em == m2 for (_, em) in existing):
                    continue
                g = math.gcd(m, m2)
                # Count how many residues r2 in [0, m2) are blocked
                # r2 is blocked if (r - r2) % g == 0, i.e., r2 % g == r % g
                # There are m2/g such residues
                score += m2 // g
            if score < best_score:
                best_score = score
                best_r = r
        
        return best_r
    
    best_solution = [[1, 2]]  # Start with at least one class
    best_size = 1
    
    rng = random.Random(42)
    
    # Multiple restarts with different orderings
    num_restarts = 500
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Shuffle moduli order
        shuffled = moduli[:]
        rng.shuffle(shuffled)
        
        current = []
        used_moduli = set()
        
        for m in shuffled:
            if m in used_moduli:
                continue
            if time.time() > deadline - 5:
                break
            
            r = find_valid_residue(m, current)
            if r >= 0:
                current.append([r, m])
                used_moduli.add(m)
        
        if len(current) > best_size:
            best_size = len(current)
            best_solution = current[:]
            
            # Early termination if we've found a very good solution
            if best_size >= 25:
                break
    
    return best_solution
