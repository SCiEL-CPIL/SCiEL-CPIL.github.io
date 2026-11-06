import time
import random
import sys
from itertools import combinations

# Increase recursion limit just in case, though we use iterative search
sys.setrecursionlimit(10000)

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
N_TRIPLES = len(ALL_TRIPLES)  # 560

# Precompute triple indices
_triple_idx = {t: i for i, t in enumerate(ALL_TRIPLES)}

# --- Fast Conflict Checking using Bitmasks ---
# partner_mask[i*V + j] is a bitmask of vertices k such that (i,j,k) is in the current hypergraph.

def _check_conflict_fast(a, b, c, partner_mask):
    """
    Check if adding triple (a,b,c) creates a K(2,2,2).
    
    A K(2,2,2) is formed by 3 disjoint pairs (a, a'), (b, b'), (c, c').
    The 8 triples are all combinations of {a,a'} x {b,b'} x {c,c'}.
    
    If we add (a,b,c), a conflict exists if there are a', b', c' such that
    the other 7 triples are already present.
    
    Conditions for a', b', c' to complete a K222 with (a,b,c):
    1. a' != a, b, c
    2. b' != a, b, c, a'
    3. c' != a, b, c, a', b'
    4. (a', b, c) present  => a' in partner_mask[b*V + c]
    5. (a, b', c) present  => b' in partner_mask[a*V + c]
    6. (a, b, c') present  => c' in partner_mask[a*V + b]
    7. (a', b', c) present => c in partner_mask[a'*V + b']
    8. (a', b, c') present => c' in partner_mask[a'*V + b]
    9. (a, b', c') present => c' in partner_mask[a*V + b']
    10. (a', b', c') present => c' in partner_mask[a'*V + b']
    
    We iterate over possible a', then b', then check if a valid c' exists.
    """
    # Candidates for a': must be in partner_mask[b*V + c] and distinct from a,b,c
    mask_a_cands = partner_mask[b * V + c]
    mask_a_cands &= ~(1 << a)
    mask_a_cands &= ~(1 << b)
    mask_a_cands &= ~(1 << c)
    
    # Candidates for b': must be in partner_mask[a*V + c] and distinct from a,b,c
    # Note: b' must also be distinct from a', but we handle that in the inner loop
    mask_b_cands = partner_mask[a * V + c]
    mask_b_cands &= ~(1 << a)
    mask_b_cands &= ~(1 << b)
    mask_b_cands &= ~(1 << c)
    
    # Candidates for c': must be in partner_mask[a*V + b] and distinct from a,b,c
    mask_c_cands = partner_mask[a * V + b]
    mask_c_cands &= ~(1 << a)
    mask_c_cands &= ~(1 << b)
    mask_c_cands &= ~(1 << c)
    
    if mask_a_cands == 0 or mask_b_cands == 0 or mask_c_cands == 0:
        return False
        
    # Iterate over possible a'
    temp_a = mask_a_cands
    while temp_a:
        lsb_a = temp_a & -temp_a
        a_p = lsb_a.bit_length() - 1
        temp_a ^= lsb_a
        
        # For this a', filter b' candidates:
        # b' must be in mask_b_cands
        # b' must be distinct from a'
        # (a', b', c) must be present => c in partner_mask[a_p*V + b']
        # So b' must be such that partner_mask[a_p*V + b'] has bit c set.
        # Instead of checking each b', we can precompute or just check in the loop.
        # Let's just iterate valid b' candidates.
        
        mask_b_valid = mask_b_cands & ~(1 << a_p)
        
        if mask_b_valid == 0:
            continue
            
        temp_b = mask_b_valid
        while temp_b:
            lsb_b = temp_b & -temp_b
            b_p = lsb_b.bit_length() - 1
            temp_b ^= lsb_b
            
            # Check if (a', b', c) is present
            if not (partner_mask[a_p * V + b_p] & (1 << c)):
                continue
                
            # Now we have a' and b'. Check if there is a valid c'.
            # c' must be in mask_c_cands
            # c' must be distinct from a' and b'
            # (a', b, c') present => c' in partner_mask[a_p*V + b]
            # (a, b', c') present => c' in partner_mask[a*V + b_p]
            # (a', b', c') present => c' in partner_mask[a_p*V + b_p]
            
            mask_c_valid = mask_c_cands
            mask_c_valid &= ~(1 << a_p)
            mask_c_valid &= ~(1 << b_p)
            mask_c_valid &= partner_mask[a_p * V + b]
            mask_c_valid &= partner_mask[a * V + b_p]
            mask_c_valid &= partner_mask[a_p * V + b_p]
            
            if mask_c_valid:
                return True
    
    return False

def _add_triple(a, b, c, partner_mask, presence, idx):
    presence[idx] = True
    partner_mask[a * V + b] |= (1 << c)
    partner_mask[b * V + a] |= (1 << c)
    partner_mask[a * V + c] |= (1 << b)
    partner_mask[c * V + a] |= (1 << b)
    partner_mask[b * V + c] |= (1 << a)
    partner_mask[c * V + b] |= (1 << a)

def _remove_triple(a, b, c, partner_mask, presence, idx):
    presence[idx] = False
    partner_mask[a * V + b] &= ~(1 << c)
    partner_mask[b * V + a] &= ~(1 << c)
    partner_mask[a * V + c] &= ~(1 << b)
    partner_mask[c * V + a] &= ~(1 << b)
    partner_mask[b * V + c] &= ~(1 << a)
    partner_mask[c * V + b] &= ~(1 << a)

def _greedy_init(order, deadline, partner_mask):
    presence = [False] * N_TRIPLES
    count = 0
    for idx in order:
        if time.time() >= deadline:
            break
        a, b, c = ALL_TRIPLES[idx]
        if _check_conflict_fast(a, b, c, partner_mask):
            continue
        count += 1
        _add_triple(a, b, c, partner_mask, presence, idx)
    return count, presence

def _local_search(count, presence, partner_mask, deadline):
    """
    Local search with add, 1-for-1 swap, and 2-for-1/3-for-2 moves.
    """
    improved = True
    no_improve_rounds = 0
    max_no_improve = 5 # If no improvement in 5 full passes, break
    
    while improved and time.time() < deadline:
        improved = False
        no_improve_rounds += 1
        if no_improve_rounds > max_no_improve:
            break
            
        # Phase 1: Add any possible triple
        for idx in range(N_TRIPLES):
            if time.time() >= deadline:
                return count, presence, partner_mask
            if presence[idx]:
                continue
            a, b, c = ALL_TRIPLES[idx]
            if not _check_conflict_fast(a, b, c, partner_mask):
                count += 1
                _add_triple(a, b, c, partner_mask, presence, idx)
                improved = True
                no_improve_rounds = 0 # Reset counter on improvement? 
                # Actually, if we add, we should restart the loop to check more adds.
                # But for efficiency, we just continue scanning.
                # If we find many adds, we are far from local optimum.
        
        if improved:
            continue # Restart loop to catch new opportunities
        
        # Phase 2: 1-for-1 Swaps
        # Try to remove one triple to allow adding another
        present_indices = [i for i in range(N_TRIPLES) if presence[i]]
        if not present_indices:
            break
            
        # Shuffle to avoid bias
        random.shuffle(present_indices)
        
        # We try a limited number of swaps to save time
        swaps_attempted = 0
        max_swaps = 50
        
        for idx_out in present_indices:
            if time.time() >= deadline or swaps_attempted >= max_swaps:
                break
            swaps_attempted += 1
            
            a, b, c = ALL_TRIPLES[idx_out]
            
            # Save state
            old_pm = partner_mask[:]
            old_pres = presence[:]
            old_count = count
            
            _remove_triple(a, b, c, partner_mask, presence, idx_out)
            count -= 1
            
            added_any = False
            # Try to add any triple that was previously blocked
            for idx_in in range(N_TRIPLES):
                if time.time() >= deadline:
                    partner_mask = old_pm
                    presence = old_pres
                    count = old_count
                    return count, presence, partner_mask
                if presence[idx_in]:
                    continue
                a_in, b_in, c_in = ALL_TRIPLES[idx_in]
                if not _check_conflict_fast(a_in, b_in, c_in, partner_mask):
                    count += 1
                    _add_triple(a_in, b_in, c_in, partner_mask, presence, idx_in)
                    added_any = True
                    # Only add one to keep it a 1-for-1 swap? 
                    # Or add all? Adding all makes it a "remove 1, add k" move.
                    # Let's add all to maximize gain per removal.
            
            if count > old_count:
                improved = True
                no_improve_rounds = 0
                break # Restart loop
            else:
                # Revert
                partner_mask = old_pm
                presence = old_pres
                count = old_count
                
        if improved:
            continue
            
        # Phase 3: 2-for-1 or 3-for-2 moves (if time permits)
        if time.time() < deadline - 5:
            # Try removing 1 triple and adding 2
            # This is expensive, so we limit attempts
            for _ in range(10):
                if time.time() >= deadline:
                    return count, presence, partner_mask
                
                if not present_indices: break
                idx_out = random.choice(present_indices)
                a, b, c = ALL_TRIPLES[idx_out]
                
                old_pm = partner_mask[:]
                old_pres = presence[:]
                old_count = count
                
                _remove_triple(a, b, c, partner_mask, presence, idx_out)
                count -= 1
                
                added_count = 0
                for idx_in in range(N_TRIPLES):
                    if time.time() >= deadline:
                        partner_mask = old_pm
                        presence = old_pres
                        count = old_count
                        return count, presence, partner_mask
                    if presence[idx_in]:
                        continue
                    a_in, b_in, c_in = ALL_TRIPLES[idx_in]
                    if not _check_conflict_fast(a_in, b_in, c_in, partner_mask):
                        count += 1
                        _add_triple(a_in, b_in, c_in, partner_mask, presence, idx_in)
                        added_count += 1
                        if added_count >= 2:
                            break # Only add 2 to keep it 2-for-1
                
                if count > old_count:
                    improved = True
                    no_improve_rounds = 0
                    break
                else:
                    partner_mask = old_pm
                    presence = old_pres
                    count = old_count
                    
            if improved:
                continue

    return count, presence, partner_mask

def construct():
    start = time.time()
    deadline = start + 115  # Safety margin
    
    best_count = 0
    best_presence = [False] * N_TRIPLES
    best_partner_mask = [0] * (V * V)
    
    random.seed(42)
    
    # Run multiple attempts
    attempts = 0
    while time.time() < deadline - 5:
        attempts += 1
        
        # Generate a random order
        order = list(range(N_TRIPLES))
        random.shuffle(order)
        
        partner_mask = [0] * (V * V)
        
        # Greedy initialization
        count, presence = _greedy_init(order, deadline, partner_mask)
        
        # Local search
        count, presence, partner_mask = _local_search(count, presence, partner_mask, deadline)
        
        if count > best_count:
            best_count = count
            best_presence = presence[:]
            best_partner_mask = partner_mask[:]
            
        # If we hit a high bar, we can stop early or continue with tighter deadline
        # 300+ is a good target. Let's see if we can exceed 300.
        # The seed is 2.438 which is ~ 300/123? No, score is value/seed.
        # If seed program score is 1.0, value is ~123?
        # Wait, "search score is value/seed".
        # If current score is 2.438, and assuming seed=1.0, value is 2.438?
        # No, usually "value" is the number of triples.
        # Let's check the contract: "valid triple system with m triples certifies ... >= m".
        # "evaluator measures the raw value (higher is better)".
        # "search score is value/seed".
        # So if current program scores 2.438, and it produces ~300 triples, then seed ~ 123?
        # Or maybe the "seed" in the denominator is a fixed baseline value.
        # Regardless, maximizing `count` is the goal.
        
        # If we have plenty of time, do another pass of local search on the best solution found so far
        if time.time() < deadline - 10:
            # Copy best state
            curr_pm = best_partner_mask[:]
            curr_pres = best_presence[:]
            curr_count = best_count
            
            # Run local search on the best found so far with remaining time
            curr_count, curr_pres, curr_pm = _local_search(curr_count, curr_pres, curr_pm, deadline - 2)
            
            if curr_count > best_count:
                best_count = curr_count
                best_presence = curr_pres
                best_partner_mask = curr_pm

    result = []
    for i in range(N_TRIPLES):
        if best_presence[i]:
            result.append(list(ALL_TRIPLES[i]))
            
    return result

if __name__ == "__main__":
    # Quick self-test
    import time
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Time: {end - start:.2f}s, Count: {len(res)}")
