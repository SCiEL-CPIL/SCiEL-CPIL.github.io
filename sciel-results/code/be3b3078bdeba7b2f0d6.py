import time
import random
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
N_TRIPLES = len(ALL_TRIPLES)  # 560

# Precompute triple index lookup
_triple_idx = {}
for _i, _t in enumerate(ALL_TRIPLES):
    _triple_idx[_t] = _i

def _check_conflict_fast(a, b, c, partner_mask):
    """
    Check if adding triple (a,b,c) creates a K(2,2,2).
    Uses precomputed partner_mask[u][v] which is a bitmask of vertices w
    such that (u,v,w) is present.
    """
    # We look for a', b', c' distinct from a,b,c and each other
    # such that all 8 transversal triples exist.
    # (a,b,c) is the new one.
    
    # a' must be in partner_mask[b][c] (since (a',b,c) must exist)
    mask_a_candidates = partner_mask[b*V + c]
    mask_a_candidates &= ~(1 << a)
    mask_a_candidates &= ~(1 << b)
    mask_a_candidates &= ~(1 << c)
    
    # b' must be in partner_mask[a][c] (since (a,b',c) must exist)
    mask_b_base = partner_mask[a*V + c]
    mask_b_base &= ~(1 << a)
    mask_b_base &= ~(1 << b)
    mask_b_base &= ~(1 << c)
    
    # c' must be in partner_mask[a][b] (since (a,b,c') must exist)
    mask_c_base = partner_mask[a*V + b]
    mask_c_base &= ~(1 << a)
    mask_c_base &= ~(1 << b)
    mask_c_base &= ~(1 << c)
    
    while mask_a_candidates:
        lsb = mask_a_candidates & -mask_a_candidates
        a_p = lsb.bit_length() - 1
        mask_a_candidates ^= lsb
        
        # b' must also be in partner_mask[a'][c] (since (a',b',c) must exist)
        mask_b_valid = mask_b_base & partner_mask[a_p*V + c]
        mask_b_valid &= ~(1 << a_p)
        
        if mask_b_valid == 0:
            continue
            
        # c' must also be in partner_mask[a'][b] (since (a',b,c') must exist)
        mask_c_valid = mask_c_base & partner_mask[a_p*V + b]
        mask_c_valid &= ~(1 << a_p)
        
        if mask_c_valid == 0:
            continue
        
        temp_b = mask_b_valid
        while temp_b:
            lsb_b = temp_b & -temp_b
            b_p = lsb_b.bit_length() - 1
            temp_b ^= lsb_b
            
            # c' must be in partner_mask[a][b'] (since (a,b',c') must exist)
            # and in partner_mask[a'][b'] (since (a',b',c') must exist)
            mask_c_final = mask_c_valid
            mask_c_final &= partner_mask[a*V + b_p]
            mask_c_final &= partner_mask[a_p*V + b_p]
            mask_c_final &= ~(1 << b_p)
            
            if mask_c_final:
                return True
                
    return False

def _add_triple(a, b, c, partner_mask, presence, idx):
    presence[idx] = True
    partner_mask[a*V + b] |= (1 << c)
    partner_mask[b*V + a] |= (1 << c)
    partner_mask[a*V + c] |= (1 << b)
    partner_mask[c*V + a] |= (1 << b)
    partner_mask[b*V + c] |= (1 << a)
    partner_mask[c*V + b] |= (1 << a)

def _remove_triple(a, b, c, partner_mask, presence, idx):
    presence[idx] = False
    partner_mask[a*V + b] &= ~(1 << c)
    partner_mask[b*V + a] &= ~(1 << c)
    partner_mask[a*V + c] &= ~(1 << b)
    partner_mask[c*V + a] &= ~(1 << b)
    partner_mask[b*V + c] &= ~(1 << a)
    partner_mask[c*V + b] &= ~(1 << a)

def _greedy_with_presence(order, deadline, partner_mask):
    """Greedy: add triples in order, skipping conflicts. Returns (count, presence)."""
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
    Local search to improve the current solution.
    Tries to add missing triples, and if stuck, tries swaps.
    """
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try to add any missing triple
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
                # Restart scan from beginning as adding one might unblock others?
                # Actually, adding a triple only increases constraints, so it won't unblock others.
                # So we can just continue. But wait, if we add a triple, it might block others.
                # So simple forward scan is fine.
                pass
        
        if improved:
            continue
            
        # 2. If no adds possible, try swaps
        # Pick a random present triple to remove, then try to add as many as possible
        # If we can add more than 1, we net gain.
        # To be safe and fast, we try a few random swaps.
        
        present_indices = [i for i in range(N_TRIPLES) if presence[i]]
        if not present_indices:
            break
            
        # Limit number of swap attempts per "improvement cycle" to avoid wasting time
        swaps_tried = 0
        max_swaps = 50
        
        while swaps_tried < max_swaps and time.time() < deadline:
            swaps_tried += 1
            idx_out = random.choice(present_indices)
            a, b, c = ALL_TRIPLES[idx_out]
            
            # Save state
            old_partner_mask = partner_mask[:]
            old_presence = presence[:]
            old_count = count
            
            # Remove idx_out
            _remove_triple(a, b, c, partner_mask, presence, idx_out)
            count -= 1
            
            # Try to add any missing triples
            added_any = False
            for idx_in in range(N_TRIPLES):
                if time.time() >= deadline:
                    # Revert and break out
                    partner_mask = old_partner_mask
                    presence = old_presence
                    count = old_count
                    return count, presence, partner_mask
                if presence[idx_in]:
                    continue
                a_in, b_in, c_in = ALL_TRIPLES[idx_in]
                if not _check_conflict_fast(a_in, b_in, c_in, partner_mask):
                    count += 1
                    _add_triple(a_in, b_in, c_in, partner_mask, presence, idx_in)
                    added_any = True
            
            if count > old_count:
                # Accept improvement
                improved = True
                # Update present_indices list is expensive, so we just break and restart outer loop
                # which will re-scan for adds (which will find none if we just added all possible)
                # and then try new swaps.
                break
            else:
                # Revert
                partner_mask = old_partner_mask
                presence = old_presence
                count = old_count
                
    return count, presence, partner_mask

def construct():
    start = time.time()
    deadline = start + 115  # safety margin
    
    best_count = 0
    best_presence = [False] * N_TRIPLES
    best_partner_mask = [0] * (V * V)
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        order = list(range(N_TRIPLES))
        random.shuffle(order)
        
        # Reset partner_mask
        partner_mask = [0] * (V * V)
        
        count, presence = _greedy_with_presence(order, deadline, partner_mask)
        
        # Local Search
        count, presence, partner_mask = _local_search(count, presence, partner_mask, deadline)
        
        if count > best_count:
            best_count = count
            best_presence = presence[:]
            best_partner_mask = partner_mask[:]
            
    result = []
    for i in range(N_TRIPLES):
        if best_presence[i]:
            result.append(list(ALL_TRIPLES[i]))
    return result

random.seed(42)
