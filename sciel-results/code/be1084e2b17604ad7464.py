"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _sumset_diffset_card(u):
    """Return (|U+U|, |U-U|) for a sorted list u of nonnegative ints."""
    n = len(u)
    if n == 0:
        return 0, 0
    # U+U
    sums = set()
    for i in range(n):
        ui = u[i]
        for j in range(i, n):
            sums.add(ui + u[j])
    # U-U
    diffs = set()
    for i in range(n):
        ui = u[i]
        for j in range(n):
            diffs.add(ui - u[j])
    return len(sums), len(diffs)


def _score(u):
    """Compute the score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)."""
    if not u:
        return 0.0
    n = len(u)
    if n < 2:
        return 0.0
    max_u = max(u)
    if max_u == 0:
        return 0.0
    ss, ds = _sumset_diffset_card(u)
    if ss == 0 or ds == 0:
        return 0.0
    val = 1.0 + math.log(ds / ss) / math.log(2 * max_u + 1)
    return val


def _gaps_to_set(gaps):
    """Convert gap list to sorted set containing 0."""
    u = [0]
    for g in gaps:
        u.append(u[-1] + g)
    return u


def _set_to_gaps(u):
    """Convert sorted set to gap list."""
    if len(u) < 2:
        return []
    return [u[i+1] - u[i] for i in range(len(u)-1)]


def construct():
    """Construct a finite set U of nonnegative integers containing 0."""
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    # Start with the seed
    best_u = [0, 1, 3]
    best_score = _score(best_u)
    
    # Also try a few simple constructions as initial candidates
    candidates = [
        [0, 1, 3],
        [0, 1, 2, 4],
        [0, 1, 2, 4, 8],
        [0, 1, 3, 9, 27],
        [0, 1, 3, 6, 10, 15],
        [0, 1, 2, 4, 7, 12, 20],
    ]
    for cand in candidates:
        if time.time() > deadline:
            break
        s = _score(cand)
        if s > best_score:
            best_score = s
            best_u = list(cand)
    
    # Local search: try adding, removing, or modifying elements
    # Strategy: hill climbing with random perturbations
    random.seed(42)
    
    current_u = list(best_u)
    current_score = best_score
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        # Generate a random perturbation
        op = random.random()
        u = list(current_u)
        
        if op < 0.3:
            # Add a new element
            if len(u) > 1:
                max_val = max(u)
                # Try adding near existing elements or at a new position
                if random.random() < 0.5:
                    # Add near an existing element
                    base = random.choice(u)
                    offset = random.randint(1, max(1, max_val // 4))
                    new_val = base + offset
                else:
                    # Add at a random position
                    new_val = random.randint(1, max_val * 2 + 10)
                
                if new_val not in u:
                    u.append(new_val)
                    u.sort()
            else:
                u.append(random.randint(1, 10))
                u.sort()
        elif op < 0.5:
            # Remove an element (keep 0)
            if len(u) > 2:
                # Remove a non-zero element
                removable = [x for x in u if x != 0]
                if removable:
                    idx = u.index(random.choice(removable))
                    u.pop(idx)
        elif op < 0.8:
            # Modify an element (keep 0)
            if len(u) > 1:
                idx = random.randint(1, len(u)-1)
                old_val = u[idx]
                # Perturb
                delta = random.randint(-max(1, old_val//3), max(1, old_val//3))
                new_val = max(1, old_val + delta)
                if new_val not in u:
                    u[idx] = new_val
                    u.sort()
        else:
            # Swap two non-zero elements
            if len(u) > 2:
                idx1 = random.randint(1, len(u)-1)
                idx2 = random.randint(1, len(u)-1)
                if idx1 != idx2:
                    u[idx1], u[idx2] = u[idx2], u[idx1]
                    u.sort()
        
        if u == current_u:
            continue
        
        s = _score(u)
        if s > current_score:
            current_u = u
            current_score = s
            no_improve_count = 0
            if s > best_score:
                best_score = s
                best_u = list(u)
        else:
            no_improve_count += 1
        
        # Occasionally try a larger jump
        if no_improve_count % 50 == 0 and no_improve_count > 0:
            # Reset to best and try a different strategy
            current_u = list(best_u)
            current_score = best_score
    
    # Final local polish: try all single-element additions/removals
    if time.time() < deadline - 5:
        u = list(best_u)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try adding each candidate
            max_val = max(u) if u else 0
            candidates_to_add = set()
            for x in u:
                for d in range(1, max(2, max_val//10 + 1)):
                    candidates_to_add.add(x + d)
                    candidates_to_add.add(max(1, x - d))
            candidates_to_add.discard(0)
            
            best_add_score = _score(u)
            best_add_val = None
            for cand in candidates_to_add:
                if cand in u:
                    continue
                if cand > max_val * 3:  # don't go too far
                    continue
                trial = u + [cand]
                trial.sort()
                s = _score(trial)
                if s > best_add_score:
                    best_add_score = s
                    best_add_val = cand
            
            if best_add_val is not None:
                u.append(best_add_val)
                u.sort()
                improved = True
            
            # Try removing each non-zero element
            for i in range(len(u)):
                if u[i] == 0:
                    continue
                trial = u[:i] + u[i+1:]
                s = _score(trial)
                if s > _score(u):
                    u = trial
                    improved = True
                    break
        
        if _score(u) > best_score:
            best_score = _score(u)
            best_u = u
    
    return best_u

# EVOLVE-BLOCK-END
