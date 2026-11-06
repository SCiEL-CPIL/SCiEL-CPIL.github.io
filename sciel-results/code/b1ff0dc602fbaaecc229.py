"""Vectorized Monte Carlo spin-system optimization for low AP discrepancy."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_N = 256
_DEADLINE_FRAC = 0.85  # safety margin: stop at 85% of 60s

def _build_ap_data():
    """Precompute for each position i (0-indexed): list of (ap_id, sign_in_ap)
    Actually we need: for each AP, the set of indices it covers.
    We'll build:
      - ap_indices: list of arrays, ap_indices[k] = np.array of 0-based positions in AP k
      - pos_ap_count: for each position, how many APs contain it (for energy update)
      - For O(1) flip: energy = sum_k S_k^2 where S_k = sum_{i in AP_k} x_i
        Flipping x_j: delta_energy = -2 * x_j * (sum of S_k over APs containing j) + 2 * (count of APs containing j)
        Let T_j = sum of S_k over all APs k that contain position j.
        Then delta = -2*x_j*T_j + 2*C_j where C_j = number of APs containing j.
        We maintain T_j dynamically.
    """
    aps = []
    seen = set()
    for start in range(_N):
        for step in range(1, _N - start):
            length = (_N - 1 - start) // step + 1
            if length < 2:
                continue
            indices = tuple(start + k * step for k in range(length))
            if indices not in seen:
                seen.add(indices)
                aps.append(indices)
    # Include singletons (length 1) - they don't affect max discrepancy meaningfully 
    # since |x_i| = 1 always, but the task says "including one". 
    # Singletons always have sum ±1, so max is at least 1. We can ignore them 
    # for optimization since they're constant, but we must account for them 
    # in the final answer. Actually, the max over ALL APs includes singletons 
    # which are always 1, so the minimum possible max is at least 1.
    # For the energy, singletons contribute 1^2 = 1 each, constant. Skip them.
    
    ap_arrays = [np.array(a, dtype=np.int64) for a in aps]
    n_aps = len(ap_arrays)
    
    # C_j: number of APs containing position j
    C = np.zeros(_N, dtype=np.int64)
    for a in ap_arrays:
        C[a] += 1
    
    # Initial S_k for a given sign vector
    # We'll compute S = ap_arrays.T @ signs  (but ap_arrays is ragged)
    # Instead, build incidence matrix M of shape (n_aps, _N) with 0/1
    # This might be large: n_aps * 256. Let's estimate n_aps.
    # For start=0: steps 1..255 -> 255 APs
    # For start=1: steps 1..254 -> 254 APs
    # ... total ~ sum_{s=0}^{255} (255-s) = 255*256/2 = 32640 APs
    # 32640 * 256 = ~8.4M entries, fine for int8.
    
    M = np.zeros((n_aps, _N), dtype=np.int8)
    for k, a in enumerate(ap_arrays):
        M[k, a] = 1
    
    return M, C, ap_arrays

def _compute_max_abs_ap_sum(signs, ap_arrays):
    """Compute max |sum| over all APs. signs: array of shape (256,) with values ±1."""
    # sums = M @ signs
    # But M is int8, signs is int8 -> int16 result, fine
    # We'll compute via dot
    sums = np.zeros(len(ap_arrays), dtype=np.int32)
    for k, a in enumerate(ap_arrays):
        sums[k] = np.sum(signs[a])
    return int(np.max(np.abs(sums)))

def _compute_max_abs_ap_sum_fast(signs, M):
    """Use precomputed M for speed."""
    sums = M.dot(signs)
    return int(np.max(np.abs(sums)))

def construct():
    t0 = time.time()
    deadline = t0 + 60.0 * _DEADLINE_FRAC
    
    M, C, ap_arrays = _build_ap_data()
    n_aps = len(ap_arrays)
    
    # Initialize: random signs
    rng = np.random.RandomState(42)
    signs = rng.choice([-1, 1], size=_N).astype(np.int8)
    
    # Compute initial S = M @ signs
    S = M.dot(signs).astype(np.int32)  # shape (n_aps,)
    
    # T_j = sum of S_k over APs k containing position j = M.T @ S
    T = M.T.dot(S).astype(np.int32)  # shape (_N,)
    
    # Energy = sum S_k^2 (plus constant for singletons, which we ignore)
    energy = int(np.sum(S.astype(np.int64) ** 2))
    
    best_signs = signs.copy()
    best_max = _compute_max_abs_ap_sum_fast(signs, M)
    best_energy = energy
    
    # Simulated Annealing / Metropolis
    T_start = 50.0
    T_end = 0.1
    n_aps_float = float(n_aps)
    
    # We'll do sweeps. Each sweep: visit each position once in random order.
    # Per flip: 
    #   delta_S = -2 * signs[j] for each AP containing j
    #   new_S_k = S_k - 2*signs[j] for k in APs(j)
    #   delta_energy = sum_{k in APs(j)} (new_S_k^2 - S_k^2)
    #                = sum_{k in APs(j)} ((S_k - 2x)^2 - S_k^2) where x=signs[j]
    #                = sum_{k in APs(j)} (-4*x*S_k + 4)
    #                = -4*x * (sum_{k in APs(j)} S_k) + 4 * C_j
    #                = -4*x*T_j + 4*C_j
    #   This is O(1) if we maintain T_j!
    
    # But we also need to update S and T after a flip.
    # If we flip position j:
    #   For each AP k containing j: S_k -= 2*signs[j]
    #   For each position l: T_l changes if l is in any AP k that contains j.
    #   T_l = sum_{k: l in AP_k} S_k
    #   delta_T_l = sum_{k: l in AP_k and j in AP_k} (-2*signs[j])
    #   This is not O(1) in general. 
    #   
    # Alternative: don't maintain T. Instead, for each flip, compute delta_energy
    # by looking at the APs containing j. But that's O(C_j) per flip.
    # C_j can be up to ~255 for j=0. That's 255 ops per flip.
    # With 256 positions per sweep, that's 256*255 = 65280 ops per sweep.
    # In NumPy vectorized form, we can do better.
    
    # Let's use a different approach: vectorized batch flips.
    # Actually, let's just do pure Python with the O(1) formula but maintain T.
    # Updating T: when S_k changes by d_k, T_l changes by d_k for each l in AP_k.
    # So flipping j: for each k in APs(j), S_k -= 2x, and for each l in AP_k, T_l -= 2x.
    # That's sum_{k in APs(j)} len(AP_k) operations. Could be large.
    
    # Let's just compute delta_energy directly each time by summing over APs containing j.
    # Precompute: for each j, list of AP indices containing j.
    aps_of_pos = [list() for _ in range(_N)]
    for k, a in enumerate(ap_arrays):
        for pos in a:
            aps_of_pos[pos].append(k)
    
    best_max = _compute_max_abs_ap_sum_fast(signs, M)
    
    # Annealing schedule
    n_sweeps = 2000
    for sweep in range(n_sweeps):
        if time.time() > deadline:
            break
        T = T_start * (T_end / T_start) ** (sweep / max(n_sweeps, 1))
        
        order = rng.permutation(_N)
        for j in order:
            x = signs[j]
            # Compute delta_energy = -4*x*T_j + 4*C_j where T_j = sum_{k in APs(j)} S_k
            ap_list = aps_of_pos[j]
            T_j = int(np.sum(S[ap_list]))
            delta_e = -4 * x * T_j + 4 * C[j]
            
            if delta_e <= 0 or rng.random() < np.exp(-delta_e / (T + 1e-10)):
                # Accept flip
                signs[j] = -x
                for k in ap_list:
                    S[k] -= 2 * x
                energy += delta_e
                
    # Final evaluation
    final_max = _compute_max_abs_ap_sum_fast(signs, M)
    if final_max < best_max:
        best_signs = signs.copy()
        best_max = final_max
    
    # Also try a few random restarts if time permits
    while time.time() < deadline - 5.0:
        signs2 = rng.choice([-1, 1], size=_N).astype(np.int8)
        S2 = M.dot(signs2).astype(np.int32)
        max2 = int(np.max(np.abs(S2)))
        if max2 < best_max:
            best_signs = signs2.copy()
            best_max = max2
    
    return list(best_signs.astype(int))

# EVOLVE-BLOCK-END
