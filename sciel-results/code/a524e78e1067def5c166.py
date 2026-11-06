"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
import numpy as np

# EVOLVE-BLOCK-START

def _build_adj(n, orient):
    """
    Build a numpy boolean adjacency matrix from the orientation list.
    adj[i, j] = True if i -> j.
    orient[k] corresponds to pair (i,j) in lexicographic order.
    0 means i -> j, 1 means j -> i.
    """
    adj = np.zeros((n, n), dtype=bool)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            bit = orient[idx]
            idx += 1
            if bit == 0:
                adj[i, j] = True
            else:
                adj[j, i] = True
    return adj

def _check_four_dominator_fast(n, orient):
    """
    Fast vectorized check for the 4-dominator property.
    Returns True if every 4-set has a vertex dominating the other 3.
    """
    adj = _build_adj(n, orient)
    
    # A 4-set {a,b,c,d} is bad if no vertex in it dominates the other 3.
    # v dominates S\{v} if sum(adj[v, S\{v}]) == 3.
    # Equivalently, for each v in S, the number of out-neighbors in S is 3.
    
    # We can iterate over all 4-combinations. For n=67, C(67,4) ~ 736,281.
    # This is manageable if we are clever, but even this is ~10^6 iterations.
    # Let's try to vectorize the check for all 4-sets at once?
    # That would require a tensor of shape (n, n, n, n) which is 67^4 ~ 20M elements.
    # 20M booleans is 20MB, which is fine.
    # But constructing it efficiently is the key.
    
    # Alternative: 
    # For each vertex v, let N_out(v) be the set of vertices v beats.
    # A 4-set S is good if there exists v in S such that S \ {v} is a subset of N_out(v).
    # This means S is a subset of N_out(v) U {v}.
    # Let's stick to the combination loop but optimize the inner check.
    
    # Precompute out-neighborhoods as bitsets (integers) for fast subset checks?
    # n <= 90, so we can use Python integers as bitsets.
    out_bits = [0] * n
    for i in range(n):
        for j in range(n):
            if adj[i, j]:
                out_bits[i] |= (1 << j)
    
    # Iterate over combinations
    # To speed this up, we can use a generator.
    # For n=67, 736k iterations is very fast in Python if the inner loop is simple.
    
    for a, b, c, d in itertools.combinations(range(n), 4):
        # Check if a dominates b,c,d
        if (out_bits[a] & (1 << b)) and (out_bits[a] & (1 << c)) and (out_bits[a] & (1 << d)):
            continue
        # Check if b dominates a,c,d
        if (out_bits[b] & (1 << a)) and (out_bits[b] & (1 << c)) and (out_bits[b] & (1 << d)):
            continue
        # Check if c dominates a,b,d
        if (out_bits[c] & (1 << a)) and (out_bits[c] & (1 << b)) and (out_bits[c] & (1 << d)):
            continue
        # Check if d dominates a,b,c
        if (out_bits[d] & (1 << a)) and (out_bits[d] & (1 << b)) and (out_bits[d] & (1 << c)):
            continue
        
        # If we get here, no vertex dominates the set
        return False
        
    return True

def _get_violations(n, orient):
    """
    Return a list of violating 4-sets.
    """
    adj = _build_adj(n, orient)
    out_bits = [0] * n
    for i in range(n):
        for j in range(n):
            if adj[i, j]:
                out_bits[i] |= (1 << j)
                
    violations = []
    for a, b, c, d in itertools.combinations(range(n), 4):
        is_good = False
        if (out_bits[a] & (1 << b)) and (out_bits[a] & (1 << c)) and (out_bits[a] & (1 << d)):
            is_good = True
        elif (out_bits[b] & (1 << a)) and (out_bits[b] & (1 << c)) and (out_bits[b] & (1 << d)):
            is_good = True
        elif (out_bits[c] & (1 << a)) and (out_bits[c] & (1 << b)) and (out_bits[c] & (1 << d)):
            is_good = True
        elif (out_bits[d] & (1 << a)) and (out_bits[d] & (1 << b)) and (out_bits[d] & (1 << c)):
            is_good = True
            
        if not is_good:
            violations.append((a, b, c, d))
            
    return violations

def _paley_orientation(p):
    """Return orientation list for Paley tournament on p vertices."""
    squares = {x * x % p for x in range(1, p)}
    n = p
    orient = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % p in squares:
                orient.append(0)
            else:
                orient.append(1)
    return n, orient

def _local_search(n, orient, time_limit, start_time):
    """
    Perform local search to fix violations.
    Returns (n, orient, is_valid).
    """
    current_orient = orient[:]
    violations = _get_violations(n, current_orient)
    
    if not violations:
        return n, current_orient, True
        
    # Precompute index for each pair (i, j) with i < j
    pair_to_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            pair_to_idx[(i, j)] = idx
            idx += 1
            
    best_violation_count = len(violations)
    
    while time.time() - start_time < time_limit:
        if not violations:
            return n, current_orient, True
            
        # Pick a random violation
        a, b, c, d = random.choice(violations)
        
        # Try flipping each edge in the violation
        best_flip = None
        best_new_count = best_violation_count
        
        edges = [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
        
        for u, v in edges:
            if u > v:
                u, v = v, u
            idx = pair_to_idx[(u, v)]
            
            # Flip
            old_bit = current_orient[idx]
            current_orient[idx] = 1 - old_bit
            
            # Check new violations
            new_violations = _get_violations(n, current_orient)
            new_count = len(new_violations)
            
            if new_count < best_new_count:
                best_new_count = new_count
                best_flip = idx
                best_violations = new_violations
            else:
                # Revert
                current_orient[idx] = old_bit
        
        if best_flip is not None:
            # Apply the best flip
            current_orient[best_flip] = 1 - current_orient[best_flip]
            violations = best_violations
            best_violation_count = best_new_count
        else:
            # No improvement found for this violation, try another or restart?
            # For simplicity, just pick another random violation
            # If we get stuck, we might need a more sophisticated restart strategy
            # But given the time limit, we just keep trying random violations
            pass
            
    return n, current_orient, len(violations) == 0

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # 1. Try Paley(67) first, as it is the smallest known prime candidate
    #    and potentially satisfies the property.
    #    If it doesn't, we can try to fix it with local search.
    
    p = 67
    n, orient = _paley_orientation(p)
    
    # Check if it's already valid
    if time.time() < deadline - 10:
        if _check_four_dominator_fast(n, orient):
            return {'n': n, 'orientation': orient}
            
    # If not valid, try local search to fix it
    # This is a structural change: instead of giving up, we optimize.
    n, orient, valid = _local_search(n, orient, time_limit=130, start_time=start_time)
    
    if valid:
        return {'n': n, 'orientation': orient}
        
    # If local search failed to find a valid tournament on 67,
    # try other small primes or just return the best effort.
    # However, the task requires a VALID tournament.
    # If we can't find one, we should try a larger n that is known to work?
    # The problem states "smallest known has 67 vertices".
    # This implies Paley(67) might NOT work, or we haven't found the right one.
    # Actually, Schütte's problem f(4) is the smallest n such that there EXISTS a tournament.
    # It is NOT necessarily Paley(67).
    # But Paley tournaments are highly symmetric.
    
    # Let's try a few other primes just in case, with a quick check.
    for p in [71, 79, 83]:
        if time.time() > deadline - 5:
            break
        n, orient = _paley_orientation(p)
        if _check_four_dominator_fast(n, orient):
            return {'n': n, 'orientation': orient}
            
    # Fallback: Return the result from local search on 67, even if invalid?
    # No, the task requires a valid tournament.
    # If we can't find a valid one, we might have to return an invalid one?
    # "An invalid result... scores 0".
    # So we must return a valid one.
    # If local search on 67 failed, maybe try 68? Or just return the 67 one and hope?
    # No, we must be valid.
    
    # Let's assume that if local search on 67 didn't work, we try a random tournament on a slightly larger n?
    # Or maybe we should try to construct a valid one from scratch using a different method?
    
    # For now, if we haven't found a valid one, we return the 67-vertex Paley tournament.
    # It might be invalid, which gives 0. But it's the best we can do without more time.
    # Wait, the prompt says "smallest known has 67 vertices". This suggests a valid one EXISTS on 67.
    # So local search should be able to find it if we give it enough time.
    
    return {'n': n, 'orientation': orient}

# EVOLVE-BLOCK-END
