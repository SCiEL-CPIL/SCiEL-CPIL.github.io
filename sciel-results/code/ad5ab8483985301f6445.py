import time
import random
import numpy as np
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
N_TRIPLES = len(ALL_TRIPLES)  # 560

# Precompute indices for NumPy broadcasting
# A_prime, B_prime, C_prime are 3D arrays of indices
_A_idx = np.arange(V)[:, None, None]
_B_idx = np.arange(V)[None, :, None]
_C_idx = np.arange(V)[None, None, :]

# Precompute valid mask for distinct i,j,k
# We will generate this once and reuse, or generate per check if it's cheap.
# Generating 16x16x16 mask is cheap.
def _get_valid_mask(a, b, c):
    """Returns a 3D boolean array where True indicates (i,j,k) are distinct and not in {a,b,c}."""
    # i, j, k ranges
    # Exclude a,b,c
    mask = np.ones((V, V, V), dtype=bool)
    mask[a, :, :] = False
    mask[:, b, :] = False
    mask[:, :, c] = False
    # Ensure distinct i, j, k
    # i != j
    i_range = np.arange(V)[:, None]
    j_range = np.arange(V)[None, :]
    distinct_ik = (i_range != j_range)
    # Broadcast to 3D
    distinct_ik = distinct_ik[:, :, None]
    mask = mask & distinct_ik
    
    # j != k
    j_range_2 = np.arange(V)[:, None]
    k_range = np.arange(V)[None, :]
    distinct_jk = (j_range_2 != k_range)
    distinct_jk = distinct_jk[None, :, :]
    mask = mask & distinct_jk
    
    # i != k
    i_range_2 = np.arange(V)[:, None]
    k_range_2 = np.arange(V)[None, :]
    distinct_ik_2 = (i_range_2 != k_range_2)
    distinct_ik_2 = distinct_ik_2[:, None, :]
    mask = mask & distinct_ik_2
    
    return mask

def _check_conflict_np(state, a, b, c):
    """
    Check if adding triple (a,b,c) to state creates a K(2,2,2).
    state is a 16x16x16 boolean numpy array.
    """
    # Get valid (a', b', c') candidates
    mask = _get_valid_mask(a, b, c)
    
    # Construct the 7 terms
    # 1. {a', b, c} -> state[i, b, c]
    t1 = state[_A_idx, b, c]
    # 2. {a, b', c} -> state[a, j, c]
    t2 = state[a, _B_idx, c]
    # 3. {a, b, c'} -> state[a, b, k]
    t3 = state[a, b, _C_idx]
    # 4. {a', b', c} -> state[i, j, c]
    t4 = state[_A_idx, _B_idx, c]
    # 5. {a', b, c'} -> state[i, b, k]
    t5 = state[_A_idx, b, _C_idx]
    # 6. {a, b', c'} -> state[a, j, k]
    t6 = state[a, _B_idx, _C_idx]
    # 7. {a', b', c'} -> state[i, j, k]
    t7 = state[_A_idx, _B_idx, _C_idx]
    
    # Combine
    conflict = mask & t1 & t2 & t3 & t4 & t5 & t6 & t7
    
    return np.any(conflict)

def _set_triple(state, t, val):
    """Set triple t to val in state (symmetric)."""
    a, b, c = t
    state[a, b, c] = val
    state[a, c, b] = val
    state[b, a, c] = val
    state[b, c, a] = val
    state[c, a, b] = val
    state[c, b, a] = val

def construct():
    start = time.time()
    deadline = start + 115  # Safety margin
    
    # Initial state: empty
    state = np.zeros((V, V, V), dtype=bool)
    count = 0
    
    best_state = state.copy()
    best_count = 0
    
    # List of all triples for random selection
    triples = ALL_TRIPLES
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.999
    
    # Pre-shuffle triples for faster random access?
    # We can just pick random index
    
    while time.time() < deadline:
        # Choose move type
        # 50% Add, 20% Remove, 30% Swap
        r = random.random()
        
        if r < 0.5:
            # Add
            idx = random.randint(0, N_TRIPLES - 1)
            t = triples[idx]
            a, b, c = t
            # Check if already present
            if state[a, b, c]:
                continue
            
            if not _check_conflict_np(state, a, b, c):
                _set_triple(state, t, True)
                count += 1
                if count > best_count:
                    best_count = count
                    best_state = state.copy()
            else:
                # Reject
                pass
        elif r < 0.7:
            # Remove
            # Pick a random present triple
            present_indices = [i for i in range(N_TRIPLES) if state[ALL_TRIPLES[i][0], ALL_TRIPLES[i][1], ALL_TRIPLES[i][2]]]
            if not present_indices:
                continue
            idx = random.choice(present_indices)
            t = triples[idx]
            _set_triple(state, t, False)
            count -= 1
            # Always accept removal in SA (it's a "bad" move for objective, but good for exploration)
            # Actually, in SA, we accept with prob exp(-Delta/T). Delta = +1 (cost).
            # Here we just accept to allow exploration.
            
        else:
            # Swap
            # Pick a random present and a random absent
            present_indices = [i for i in range(N_TRIPLES) if state[ALL_TRIPLES[i][0], ALL_TRIPLES[i][1], ALL_TRIPLES[i][2]]]
            absent_indices = [i for i in range(N_TRIPLES) if not state[ALL_TRIPLES[i][0], ALL_TRIPLES[i][1], ALL_TRIPLES[i][2]]]
            
            if not present_indices or not absent_indices:
                continue
                
            idx_out = random.choice(present_indices)
            idx_in = random.choice(absent_indices)
            
            t_out = triples[idx_out]
            t_in = triples[idx_in]
            
            # Temporarily apply swap
            _set_triple(state, t_out, False)
            _set_triple(state, t_in, True)
            
            # Check if t_in creates conflict in this new state
            a, b, c = t_in
            if not _check_conflict_np(state, a, b, c):
                # Accept
                count += 0  # Count unchanged
                if count > best_count:
                    best_count = count
                    best_state = state.copy()
            else:
                # Reject: revert
                _set_triple(state, t_in, False)
                _set_triple(state, t_out, True)
        
        T *= cooling
        if T < 0.1:
            T = 0.1

    # Extract result from best_state
    result = []
    for t in ALL_TRIPLES:
        a, b, c = t
        if state[a, b, c]: # Wait, should use best_state
            pass
            
    # Use best_state
    for t in ALL_TRIPLES:
        a, b, c = t
        if best_state[a, b, c]:
            result.append(list(t))
            
    return result

random.seed(42)
