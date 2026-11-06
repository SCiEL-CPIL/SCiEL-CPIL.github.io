"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

# Global precomputation
ALL_MASKS = []
N_MASKS = 0
CONFLICTS = None
FULL_MASK = (1 << 16) - 1
MASK_TO_IDX = {}

def _precompute():
    global ALL_MASKS, N_MASKS, CONFLICTS, MASK_TO_IDX
    
    if ALL_MASKS:
        return
    
    # Generate all 4-subsets of 16 elements as bitmasks
    for combo in combinations(range(16), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        ALL_MASKS.append(mask)
    
    N_MASKS = len(ALL_MASKS)
    for i, m in enumerate(ALL_MASKS):
        MASK_TO_IDX[m] = i
    
    # Precompute conflict bitmasks
    # CONFLICTS[i][j] is a Python int bitmask where bit k is set iff {i,j,k} is a sunflower
    CONFLICTS = [[0] * N_MASKS for _ in range(N_MASKS)]
    
    conflict_cache = {}
    
    for i in range(N_MASKS):
        A = ALL_MASKS[i]
        for j in range(i + 1, N_MASKS):
            B = ALL_MASKS[j]
            
            I = A & B
            S = A | B
            I_size = bin(I).count('1')
            P_size = 4 - I_size
            
            if P_size < 0:
                continue
            
            U = FULL_MASK ^ S  # elements not in A or B
            
            key = (I, U, P_size)
            if key in conflict_cache:
                conflict_mask = conflict_cache[key]
            else:
                # Find all P_size-subsets of U
                bits_in_U = []
                temp_U = U
                while temp_U:
                    lsb = temp_U & -temp_U
                    bits_in_U.append(lsb)
                    temp_U ^= lsb
                
                if len(bits_in_U) < P_size:
                    conflict_mask = 0
                else:
                    conflict_mask = 0
                    for P_combo in combinations(bits_in_U, P_size):
                        P_mask = 0
                        for b in P_combo:
                            P_mask |= b
                        C_mask = I | P_mask
                        if C_mask in MASK_TO_IDX:
                            idx_C = MASK_TO_IDX[C_mask]
                            conflict_mask |= (1 << idx_C)
                
                conflict_cache[key] = conflict_mask
            
            CONFLICTS[i][j] = conflict_mask
            CONFLICTS[j][i] = conflict_mask

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    _precompute()
    
    best_size = 0
    best_set = []
    
    indices = list(range(N_MASKS))
    
    # Strategy 1: Deterministic greedy (lexicographic order)
    selected_mask = 0
    blocked_mask = 0
    selected_indices = []
    
    for u in indices:
        if time.time() > deadline:
            break
        bit_u = 1 << u
        if (selected_mask & bit_u) or (blocked_mask & bit_u):
            continue
        
        selected_mask |= bit_u
        selected_indices.append(u)
        
        # Update blocked mask using precomputed conflicts
        m = selected_mask & ~bit_u
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            blocked_mask |= CONFLICTS[u][v]
            m ^= lsb
    
    current_size = len(selected_indices)
    if current_size > best_size:
        best_size = current_size
        best_set = selected_indices[:]
    
    # Strategy 2: Randomized greedy restarts
    # With precomputed conflicts, each restart is much faster
    # We can afford many more restarts
    
    restart_count = 0
    while time.time() < deadline - 2.0:
        restart_count += 1
        random.seed(42 + restart_count)
        random.shuffle(indices)
        
        selected_mask = 0
        blocked_mask = 0
        selected_indices = []
        
        for u in indices:
            if time.time() > deadline - 2.0:
                break
            bit_u = 1 << u
            if (selected_mask & bit_u) or (blocked_mask & bit_u):
                continue
            
            selected_mask |= bit_u
            selected_indices.append(u)
            
            # Update blocked mask using precomputed conflicts
            m = selected_mask & ~bit_u
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                blocked_mask |= CONFLICTS[u][v]
                m ^= lsb
        
        current_size = len(selected_indices)
        if current_size > best_size:
            best_size = current_size
            best_set = selected_indices[:]
            
            if best_size >= 250:
                break
    
    # Convert indices back to sorted lists of elements
    result = []
    for idx in best_set:
        mask = ALL_MASKS[idx]
        elems = []
        for i in range(16):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
    
    return result
# EVOLVE-BLOCK-END
