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

def _greedy_init():
    """Greedy: try to add each modulus in a random order."""
    moduli = list(range(2, MAX_MOD + 1))
    random.shuffle(moduli)
    pairs = []
    for m in moduli:
        # Try all residues, pick one that's disjoint from all existing
        # Randomize residue order for diversity
        residues = list(range(m))
        random.shuffle(residues)
        for r in residues:
            ok = True
            for (r2, m2) in pairs:
                g = gcd(m, m2)
                if (r - r2) % g == 0:
                    ok = False
                    break
            if ok:
                pairs.append([r, m])
                break
    return pairs

def _try_add_one(pairs, moduli_used):
    """Try to add one more pair with an unused modulus."""
    unused = [m for m in range(2, MAX_MOD + 1) if m not in moduli_used]
    if not unused:
        return None
    random.shuffle(unused)
    for m in unused[:50]:
        residues = list(range(m))
        random.shuffle(residues)
        for r in residues:
            ok = True
            for (r2, m2) in pairs:
                g = gcd(m, m2)
                if (r - r2) % g == 0:
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
    
    # Try random pairs of unused moduli
    random.shuffle(unused)
    for i in range(min(30, len(unused))):
        for j in range(i + 1, min(30, len(unused))):
            m1 = unused[i]
            m2 = unused[j]
            # Try random residues
            for _ in range(40):
                r1 = random.randrange(m1)
                r2 = random.randrange(m2)
                # Check both against all existing pairs
                ok1 = True
                for (r, m) in pairs_no_removed:
                    if not _disjoint(r1, m1, r, m):
                        ok1 = False
                        break
                if not ok1:
                    continue
                ok2 = True
                for (r, m) in pairs_no_removed:
                    if not _disjoint(r2, m2, r, m):
                        ok2 = False
                        break
                if not ok2:
                    continue
                if not _disjoint(r1, m1, r2, m2):
                    continue
                # Success: new solution
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
            if not _disjoint(new_r, m, r2, m2):
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
    for restart in range(200):
        if time.time() > DEADLINE:
            break
        
        # Greedy initialization
        pairs = _greedy_init()
        moduli_used = set(m for _, m in pairs)
        cur_size = len(pairs)
        
        if cur_size > best_size:
            best = [p[:] for p in pairs]
            best_size = cur_size
        
        # Local search: simulated annealing style
        temp = 10.0
        for step in range(500):
            if time.time() > DEADLINE:
                break
            
            move_type = random.random()
            new_pairs = None
            
            if move_type < 0.3:
                # Try add one
                new_pairs = _try_add_one(pairs, moduli_used)
            elif move_type < 0.6:
                # Remove one, add two
                new_pairs = _remove_add_two(pairs, moduli_used)
            elif move_type < 0.9:
                # Change residue of existing
                new_pairs = _change_residue(pairs)
            else:
                # Random restart of current solution (partial)
                # Just re-greedy from scratch? Or keep best?
                pass

            if new_pairs is not None:
                new_size = len(new_pairs)
                # For change_residue, size is same. We accept if it allows future growth?
                # Simple heuristic: Accept if size increases, or if same size but random probability (annealing)
                if new_size > cur_size:
                    pairs = new_pairs
                    cur_size = new_size
                    moduli_used = set(m for _, m in pairs)
                    if cur_size > best_size:
                        best = [p[:] for p in pairs]
                        best_size = cur_size
                elif new_size == cur_size:
                    # Accept same size moves with some probability to escape local optima
                    if random.random() < 0.5:
                        pairs = new_pairs
                        # moduli_used unchanged for change_residue
                        # But wait, _remove_add_two and _try_add_one always increase size by 1
                        # So new_size == cur_size only happens for _change_residue
                        pass
                # If new_size < cur_size, we generally reject
                
            temp *= 0.995
        
        # Check if we've found a very good solution
        if best_size >= 20:
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
