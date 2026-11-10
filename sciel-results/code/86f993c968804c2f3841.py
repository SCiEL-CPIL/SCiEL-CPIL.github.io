import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Candidate moduli: 2 to 128
    # We want to maximize the count.
    # Start with all even numbers (64 items).
    # We know we can't have 3 even numbers.
    # So we must discard most even numbers.
    
    # Let's try to find a valid set via backtracking.
    # To maximize count, we should include as many as possible.
    # A good heuristic is to include numbers with many small prime factors?
    # Or just try to solve the CSP for a large subset.
    
    # Since 3 even numbers are impossible, let's pick 2 even numbers.
    # And many odd numbers.
    
    # Let's define the CSP solver.
    
    def solve(moduli):
        # moduli: list of ints
        # Returns list of [residue, modulus] or None
        n = len(moduli)
        if n == 0:
            return []
        
        # Order moduli by increasing value? Or by number of constraints?
        # Let's just use the given order.
        
        assignment = {}
        
        def backtrack(i):
            if i == n:
                return True
            
            if time.time() > deadline:
                return False
            
            m = moduli[i]
            # Try all residues
            # Randomize order to avoid bias
            residues = list(range(m))
            random.shuffle(residues)
            
            for r in residues:
                valid = True
                for j in range(i):
                    n_mod = moduli[j]
                    r_n = assignment[j]
                    g = math.gcd(m, n_mod)
                    if r % g == r_n % g:
                        valid = False
                        break
                
                if valid:
                    assignment[i] = r
                    if backtrack(i + 1):
                        return True
                    del assignment[i]
            
            return False
        
        if backtrack(0):
            return [[assignment[i], moduli[i]] for i in range(n)]
        return None

    # Strategy:
    # 1. Try to find a large valid set.
    # We know even numbers are problematic.
    # Let's try sets based on multiples of 3, 5, etc.
    
    # Candidate pools
    evens = [m for m in range(2, 129) if m % 2 == 0]
    odds = [m for m in range(3, 129) if m % 2 != 0]
    
    # We can have at most 2 evens.
    # Let's pick 2 evens that are "compatible" with many odds.
    # Actually, any even $E$ and odd $O$ must have gcd > 1.
    # So if we pick $E=6$, $O$ must be multiple of 3.
    # If we pick $E=10$, $O$ must be multiple of 5.
    
    # Let's try to build a set of multiples of 3.
    # Multiples of 3: 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 102, 105, 108, 111, 114, 117, 120, 123, 126.
    # Count: 42.
    # Evens in this set: 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126.
    # Count: 21.
    # We can only keep 2 evens.
    # So we keep 2 evens and 20 odds?
    # Total 22.
    
    # Can we do better?
    # What if we use multiples of 2 and 3?
    # We can't use many 2s.
    
    # Let's just run the solver on a promising subset.
    # Subset: All multiples of 3, but limit evens to 2.
    # Which 2 evens? Maybe the ones with largest moduli?
    # Or just pick any 2.
    
    m3s = [m for m in range(3, 129) if m % 3 == 0]
    m3_evens = [m for m in m3s if m % 2 == 0]
    m3_odds = [m for m in m3s if m % 2 != 0]
    
    # Try different pairs of evens
    best_result = None
    best_len = 0
    
    # Pick 2 evens from m3_evens
    # To maximize speed, we might not try all pairs.
    # Let's just try the first 2, or random ones.
    # Actually, the parity constraint is the main killer.
    # If we pick 2 evens, they must have different parities?
    # No, they are both even, so their residues must have different parities.
    # $r_{e1} \not\equiv r_{e2} \pmod 2$.
    # And for any odd $o$, $r_o \not\equiv r_{e1} \pmod{\gcd(o, e1)}$.
    # Since $o$ is odd and multiple of 3, and $e1$ is multiple of 6, gcd is at least 3.
    # So constraint is mod 3 or higher.
    
    # Let's try a few random pairs of evens.
    random.shuffle(m3_evens)
    candidates = []
    for i in range(min(5, len(m3_evens))):
        for j in range(i+1, min(5, len(m3_evens))):
            candidates.append([m3_evens[i], m3_evens[j]] + m3_odds)
    
    # Also try sets based on multiples of 5?
    m5s = [m for m in range(5, 129) if m % 5 == 0]
    m5_evens = [m for m in m5s if m % 2 == 0]
    m5_odds = [m for m in m5s if m % 2 != 0]
    random.shuffle(m5_evens)
    for i in range(min(5, len(m5_evens))):
        for j in range(i+1, min(5, len(m5_evens))):
            candidates.append([m5_evens[i], m5_evens[j]] + m5_odds)
            
    # And multiples of 7?
    m7s = [m for m in range(7, 129) if m % 7 == 0]
    m7_evens = [m for m in m7s if m % 2 == 0]
    m7_odds = [m for m in m7s if m % 2 != 0]
    random.shuffle(m7_evens)
    for i in range(min(5, len(m7_evens))):
        for j in range(i+1, min(5, len(m7_evens))):
            candidates.append([m7_evens[i], m7_evens[j]] + m7_odds)

    # Shuffle candidates
    random.shuffle(candidates)
    
    for cand in candidates:
        if time.time() > deadline:
            break
        # Sort by modulus to make backtracking deterministic-ish
        cand.sort()
        res = solve(cand)
        if res is not None:
            if len(res) > best_len:
                best_len = len(res)
                best_result = res
    
    if best_result is None:
        # Fallback to seed if nothing found
        return [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
        
    return best_result
