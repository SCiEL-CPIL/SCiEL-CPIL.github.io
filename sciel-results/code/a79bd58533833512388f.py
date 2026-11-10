# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    # Compute forbidden differences F
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Symmetrize: D = F U -F
    # Since -1 is a square mod 5 and mod 41, -1 is a square mod 205.
    # So F = -F. Thus D = F.
    # But let's be safe and compute D explicitly.
    forbidden_sym = set()
    for d in forbidden:
        forbidden_sym.add(d)
        forbidden_sym.add((-d) % m)
    forbidden_sym.discard(0)
    
    # Precompute conflict masks
    # conflict_masks[x] is a bitmask where bit s is set if (x-s) % m is in forbidden_sym
    conflict_masks = [0] * m
    for x in range(m):
        mask = 0
        for d in forbidden_sym:
            s = (x - d) % m
            mask |= (1 << s)
        conflict_masks[x] = mask
        
    def is_valid_addition(M_S, x):
        return (M_S & conflict_masks[x]) == 0

    def greedy_construct(order):
        S = []
        M_S = 0
        for x in order:
            if is_valid_addition(M_S, x):
                S.append(x)
                M_S |= (1 << x)
        return S, M_S

    def local_search(S, M_S):
        improved = True
        while improved:
            improved = False
            # Phase 1: Try to add any element
            for x in range(m):
                if not (M_S >> x) & 1:
                    if is_valid_addition(M_S, x):
                        S.append(x)
                        M_S |= (1 << x)
                        improved = True
                        break
            
            if improved:
                continue
            
            # Phase 2: Try 1-1 swaps to escape local optimum
            # We perform a systematic search for a valid swap.
            # To avoid getting stuck in a cycle, we rely on the fact that
            # valid swaps change the set. If we keep swapping, we might cycle.
            # However, we only swap if it's valid.
            # We will try a few random valid swaps if systematic is too slow, 
            # but m=205 is small, so systematic is fine.
            
            swap_found = False
            # Shuffle S indices to randomize which u we try first
            u_indices = list(range(len(S)))
            random.shuffle(u_indices)
            
            for ui in u_indices:
                u = S[ui]
                # Try all v not in S
                # To speed up, we can iterate v in random order
                v_candidates = [x for x in range(m) if not (M_S >> x) & 1]
                random.shuffle(v_candidates)
                
                for v in v_candidates:
                    # Check if S - {u} + {v} is valid
                    # M_S without u
                    M_temp = M_S ^ (1 << u)
                    if (M_temp & conflict_masks[v]) == 0:
                        # Valid swap
                        S[ui] = v
                        M_S = M_temp | (1 << v)
                        swap_found = True
                        break
                if swap_found:
                    break
            
            if not swap_found:
                # No valid 1-1 swap exists?
                # This means S is an isolated node in the 1-swap graph.
                # We might need 2-2 swaps or a restart.
                # For now, break and let the outer loop restart.
                break
                
        return S, M_S

    best_set = []
    best_M_S = 0
    best_size = 0
    
    # Deadline guard
    deadline = time.time() + 610.0
    
    for i in range(10000):
        if time.time() > deadline:
            break
        
        # Random order for greedy
        order = list(range(m))
        random.shuffle(order)
        
        S, M_S = greedy_construct(order)
        S, M_S = local_search(S, M_S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            best_M_S = M_S
            
            # If we reach a very high size, we might stop?
            # Theoretical max is unknown, but 12 is current record.
            # If we find 15, it's a big jump.
            
    if not best_set:
        best_set = [0]
        
    return np.array(best_set, dtype=np.int64)

# EVOLVE-BLOCK-END
