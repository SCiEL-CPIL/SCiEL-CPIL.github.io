# EVOLVE-BLOCK-START
import time
import random
import math
from math import gcd

MAX_MOD = 128
DEADLINE = time.time() + 85  # safety margin before 90s kill

def _disjoint(r1, m1, r2, m2):
    """Two residue classes are disjoint iff gcd(m1,m2) does not divide (r1-r2)."""
    g = gcd(m1, m2)
    return (r1 - r2) % g != 0

def _is_valid(pairs):
    """Check all pairs are mutually disjoint."""
    n = len(pairs)
    for i in range(n):
        for j in range(i + 1, n):
            r1, m1 = pairs[i]
            r2, m2 = pairs[j]
            g = gcd(m1, m2)
            if (r1 - r2) % g == 0:
                return False
    return True

def _num_distinct_prime_factors(n):
    """Count distinct prime factors of n."""
    count = 0
    d = 2
    while d * d <= n:
        if n % d == 0:
            count += 1
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        count += 1
    return count

def _greedy_init():
    """Greedy: try to add each modulus in order of most prime factors first (more flexible)."""
    moduli = list(range(2, MAX_MOD + 1))
    # Sort by number of distinct prime factors (descending), then randomize ties
    moduli.sort(key=lambda m: (-_num_distinct_prime_factors(m), random.random()))
    pairs = []
    for m in moduli:
        # Try all residues, pick one that's disjoint from all existing
        # Use structured ordering: try residues that differ from existing ones modulo common factors
        best_r = None
        for r in range(m):
            ok = True
            for (r2, m2) in pairs:
                g = gcd(m, m2)
                if g > 1 and (r - r2) % g == 0:
                    ok = False
                    break
            if ok:
                best_r = r
                break
        if best_r is not None:
            pairs.append([best_r, m])
    return pairs

def _try_add_one(pairs, moduli_used):
    """Try to add one more pair with an unused modulus, prioritizing flexible moduli."""
    unused = [m for m in range(2, MAX_MOD + 1) if m not in moduli_used]
    if not unused:
        return None
    # Sort by prime factor count (more factors = more residue choices)
    unused.sort(key=lambda m: -_num_distinct_prime_factors(m))
    for m in unused:
        for r in range(m):
            ok = True
            for (r2, m2) in pairs:
                g = gcd(m, m2)
                if g > 1 and (r - r2) % g == 0:
                    ok = False
                    break
            if ok:
                return pairs + [[r, m]]
    return None

def _remove_add_two(pairs, moduli_used):
    """Try to remove one pair and add two new pairs with unused moduli."""
    n = len(pairs)
    if n == 0:
        return None
    
    # Pick a random pair to remove
    idx = random.randrange(n)
    removed = pairs[idx]
    pairs_no_removed = pairs[:idx] + pairs[idx+1:]
    moduli_no_removed = set(moduli_used) - {removed[1]}
    
    # Find unused moduli
    unused = [m for m in range(2, MAX_MOD + 1) if m not in moduli_no_removed]
    if len(unused) < 2:
        return None
    
    # Sort by flexibility
    unused.sort(key=lambda m: -_num_distinct_prime_factors(m))
    
    # Try pairs of unused moduli
    for i in range(min(40, len(unused))):
        m1 = unused[i]
        for j in range(i + 1, min(40, len(unused))):
            m2 = unused[j]
            # Try structured residues
            for r1 in range(m1):
                ok1 = True
                for (r, m) in pairs_no_removed:
                    g = gcd(m1, m)
                    if g > 1 and (r1 - r) % g == 0:
                        ok1 = False
                        break
                if not ok1:
                    continue
                for r2 in range(m2):
                    ok2 = True
                    for (r, m) in pairs_no_removed:
                        g = gcd(m2, m)
                        if g > 1 and (r2 - r) % g == 0:
                            ok2 = False
                            break
                    if not ok2:
                        continue
                    g12 = gcd(m1, m2)
                    if g12 > 1 and (r1 - r2) % g12 == 0:
                        continue
                    # Success
                    new_pairs = pairs_no_removed + [[r1, m1], [r2, m2]]
                    return new_pairs
    return None

def _change_residue(pairs):
    """Try to change the residue of an existing modulus to a different valid one."""
    n = len(pairs)
    if n == 0:
        return None
    
    idx = random.randrange(n)
    r, m = pairs[idx]
    others = pairs[:idx] + pairs[idx+1:]
    
    # Try all other residues
    residues = [x for x in range(m) if x != r]
    random.shuffle(residues)
    for new_r in residues:
        ok = True
        for (r2, m2) in others:
            g = gcd(m, m2)
            if g > 1 and (new_r - r2) % g == 0:
                ok = False
                break
        if ok:
            new_pairs = pairs[:]
            new_pairs[idx] = [new_r, m]
            return new_pairs
    return None

def construct():
    start = time.time()
    best = None
    best_size = 0
    
    # Multiple random restarts
    for restart in range(300):
        if time.time() > DEADLINE:
            break
        
        # Greedy initialization
        pairs = _greedy_init()
        moduli_used = set(m for _, m in pairs)
        cur_size = len(pairs)
        
        if cur_size > best_size:
            best = [p[:] for p in pairs]
            best_size = cur_size
        
        # Local search
        for step in range(800):
            if time.time() > DEADLINE:
                break
            
            move_type = random.random()
            new_pairs = None
            
            if move_type < 0.4:
                # Try add one
                new_pairs = _try_add_one(pairs, moduli_used)
            elif move_type < 0.7:
                # Remove one, add two
                new_pairs = _remove_add_two(pairs, moduli_used)
            else:
                # Change residue of existing
                new_pairs = _change_residue(pairs)

            if new_pairs is not None:
                new_size = len(new_pairs)
                if new_size > cur_size:
                    pairs = new_pairs
                    cur_size = new_size
                    moduli_used = set(m for _, m in pairs)
                    if cur_size > best_size:
                        best = [p[:] for p in pairs]
                        best_size = cur_size
                elif new_size == cur_size:
                    # Accept same size moves to escape local optima
                    if random.random() < 0.6:
                        pairs = new_pairs
                        # moduli_used unchanged for change_residue
                        pass
                # If new_size < cur_size, reject
        
        # Check if we've found a very good solution
        if best_size >= 25:
            break
    
    if best is None:
        # Fallback: simple greedy
        best = _greedy_init()
    
    # Ensure validity
    if not _is_valid(best):
        # Rebuild valid subset
        valid = []
        for p in best:
            ok = True
            for q in valid:
                if not _disjoint(p[0], p[1], q[0], q[1]):
                    ok = False
                    break
            if ok:
                valid.append(p)
        best = valid
    
    # Ensure canonical form
    result = []
    for r, m in best:
        r = r % m
        result.append([r, m])
    
    return result
# EVOLVE-BLOCK-END
