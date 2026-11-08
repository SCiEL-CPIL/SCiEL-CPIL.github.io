import time
import math
import random

def compute_score(U):
    """Compute the score for a set U."""
    if not U or 0 not in U:
        return 0.0
    U = list(set(U))
    n = len(U)
    if n < 2:
        return 0.0
    max_u = max(U)
    if max_u <= 0:
        return 0.0
    # Compute sumset and difference set
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 1:
        return 1.0
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_score = 0.0
    best_set = [0, 1, 3]
    
    # Initialize with the seed
    best_score = compute_score(best_set)
    
    # Strategy: Build sets with geometric-like structure
    # Try various constructions and local search
    
    def try_set(s):
        nonlocal best_score, best_set
        if not s or 0 not in s:
            return
        s = list(set(s))
        s.sort()
        if len(s) < 2:
            return
        sc = compute_score(s)
        if sc > best_score:
            best_score = sc
            best_set = s
    
    # Try geometric progressions
    for r in range(2, 20):
        for n in range(2, 10):
            s = [0]
            cur = 1
            for i in range(1, n):
                cur *= r
                s.append(cur)
            if time.time() > deadline:
                return best_set
            try_set(s)
    
    # Try mixed constructions: small base + geometric tail
    for r in range(2, 15):
        for n_tail in range(2, 7):
            for n_base in range(0, 4):
                s = [0]
                # Add some small elements
                for i in range(1, n_base + 1):
                    s.append(i)
                # Add geometric tail
                cur = r ** n_base
                for i in range(n_tail):
                    cur *= r
                    s.append(cur)
                if time.time() > deadline:
                    return best_set
                try_set(s)
    
    # Local search: start from best, try adding/removing/modify elements
    def local_search(current_set, max_iter=5000):
        nonlocal best_score, best_set
        current = list(set(current_set))
        current.sort()
        current_score = compute_score(current)
        
        for it in range(max_iter):
            if time.time() > deadline:
                return
            # Try modifications
            candidates = []
            
            # Try adding a new element
            max_el = max(current) if current else 0
            # Try elements near existing ones or at strategic positions
            for _ in range(10):
                if random.random() < 0.5:
                    # Add element near max
                    new_el = random.randint(1, max_el * 2 + 10)
                else:
                    # Add element in a gap
                    new_el = random.randint(1, max_el + max_el // 2 + 5)
                if new_el not in current and new_el != 0:
                    candidates.append(current + [new_el])
            
            # Try removing an element (keep 0)
            if len(current) > 2:
                removable = [x for x in current if x != 0]
                if removable:
                    idx = random.choice(range(len(removable)))
                    new_set = [x for x in current if x != removable[idx]]
                    if new_set:
                        candidates.append(new_set)
            
            # Try modifying an element
            modifiable = [x for x in current if x != 0]
            if modifiable:
                idx = random.choice(range(len(modifiable)))
                old_val = modifiable[idx]
                # Perturb
                delta = random.choice([-1, 1, -2, 2, -5, 5, -10, 10])
                new_val = old_val + delta
                if new_val > 0 and new_val not in current:
                    new_set = list(current)
                    new_set.remove(old_val)
                    new_set.append(new_val)
                    candidates.append(new_set)
            
            # Evaluate candidates
            best_candidate = None
            best_candidate_score = current_score
            for cand in candidates[:5]:  # Limit to keep fast
                cand = list(set(cand))
                cand.sort()
                if len(cand) < 2 or 0 not in cand:
                    continue
                sc = compute_score(cand)
                if sc > best_candidate_score:
                    best_candidate_score = sc
                    best_candidate = cand
            
            if best_candidate is not None:
                current = best_candidate
                current_score = best_candidate_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(current)
            else:
                # No improvement, break or continue with different random seed
                if random.random() < 0.3:
                    break
    
    # Run local search from multiple starting points
    start_sets = [
        [0, 1, 3],
        [0, 1, 4],
        [0, 1, 5],
        [0, 2, 5],
        [0, 1, 3, 9],
        [0, 1, 4, 16],
        [0, 1, 3, 10],
        [0, 1, 4, 13],
    ]
    
    for ss in start_sets:
        if time.time() > deadline:
            return best_set
        local_search(ss, max_iter=3000)
    
    # Also try a more systematic search for small sets
    # For sets of size 4-6, try structured families
    
    # Family: {0, a, b, c} with a < b < c
    # Try to maximize |U-U|/|U+U| / log(2c+1)
    for a in range(1, 15):
        for b in range(a + 1, a + 30):
            for c in range(b + 1, b + 50):
                if time.time() > deadline:
                    return best_set
                s = [0, a, b, c]
                try_set(s)
    
    # Family: {0, a, b, c, d}
    for a in range(1, 8):
        for b in range(a + 1, a + 15):
            for c in range(b + 1, b + 25):
                for d in range(c + 1, c + 30):
                    if time.time() > deadline:
                        return best_set
                    s = [0, a, b, c, d]
                    try_set(s)
    
    return best_set
