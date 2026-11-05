"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product
from collections import defaultdict

def _count_nonzero_square_fast(coeffs):
    """Count nonzero terms in the exact square using NumPy convolution."""
    if len(coeffs) == 0:
        return 0
    c = np.array(coeffs, dtype=np.int64)
    sq = np.convolve(c, c)
    return int(np.count_nonzero(sq))

def _validate(coeffs):
    """Check all constraints on the coefficient list."""
    if len(coeffs) > 41:
        return False
    if len(coeffs) == 0 or coeffs[-1] == 0:
        return False
    for c in coeffs:
        if not isinstance(c, int) or c < -64 or c > 64:
            return False
    nonzero = sum(1 for c in coeffs if c != 0)
    if nonzero != 12:
        return False
    return True

def _sparse_to_dense(sparse, max_deg=40):
    """Convert sparse list of (exponent, coeff) to dense coefficient list."""
    coeffs = [0] * (max_deg + 1)
    for exp, c in sparse:
        if 0 <= exp <= max_deg:
            coeffs[exp] = c
    while len(coeffs) > 1 and coeffs[-1] == 0:
        coeffs.pop()
    return coeffs

def _dense_to_sparse(coeffs):
    """Convert dense coefficient list to sparse representation."""
    return [(i, c) for i, c in enumerate(coeffs) if c != 0]

def _square_nonzero_count(sparse):
    """Count nonzero terms in square from sparse representation directly."""
    dense = _sparse_to_dense(sparse)
    return _count_nonzero_square_fast(dense)

def _fast_square_count_from_signs(signs, exponents, base_coeffs):
    """
    Fast square nonzero count given signs applied to base coefficients at fixed exponents.
    Uses the fact that square coefficients are sums of products, so we can compute
    the square array once per sign assignment efficiently.
    """
    n = len(signs)
    # Build dense array
    max_exp = max(exponents)
    dense = np.zeros(max_exp + 1, dtype=np.int64)
    for i in range(n):
        dense[exponents[i]] = base_coeffs[i] * signs[i]
    sq = np.convolve(dense, dense)
    return int(np.count_nonzero(sq))

def _brute_force_sign_search_consecutive(deadline, start=0, length=12):
    """
    Exhaustive sign search on consecutive exponents [start, start+length-1].
    For 12 terms with ±1 coefficients, there are 2^12 = 4096 sign assignments.
    We also try small coefficient magnitudes.
    """
    exponents = list(range(start, start + length))
    if exponents[-1] > 40:
        return None, float('inf')
    
    best_sq = float('inf')
    best_signs = None
    
    # First: pure ±1 signs
    for signs in product([1, -1], repeat=length):
        if time.time() > deadline:
            break
        dense = [0] * (exponents[-1] + 1)
        for i in range(length):
            dense[exponents[i]] = signs[i]
        sq = _count_nonzero_square_fast(dense)
        if sq < best_sq:
            best_sq = sq
            best_signs = signs
    
    # Then: try small coefficient magnitudes {±1, ±2}
    coeff_mags = [1, 2]
    # For each magnitude pattern, try all sign combinations
    # This is 2^12 * 2^12 = too many, so we'll do it smarter
    # Try: all coefficients = ±1 (done above), then try flipping one to ±2
    for flip_idx in range(length):
        for mag in [2, 3]:
            for sign in [1, -1]:
                if time.time() > deadline:
                    break
                # Start from best ±1 signs, modify one
                if best_signs is None:
                    continue
                dense = [0] * (exponents[-1] + 1)
                for i in range(length):
                    if i == flip_idx:
                        dense[exponents[i]] = sign * mag
                    else:
                        dense[exponents[i]] = best_signs[i]
                sq = _count_nonzero_square_fast(dense)
                if sq < best_sq:
                    best_sq = sq
                    best_signs = tuple(dense[exponents[i]] for i in range(length))
    
    if best_signs is None:
        return None, float('inf')
    
    # Convert to sparse
    sparse = [(exponents[i], best_signs[i]) for i in range(length)]
    return sparse, best_sq

def _coeffs_only_optimize(exponents, deadline, coeff_choices=None, seed=42):
    """
    Optimize ONLY the coefficients on a fixed set of exponents using
    simulated annealing.
    """
    if coeff_choices is None:
        coeff_choices = [-4, -3, -2, -1, 1, 2, 3, 4]
    
    n = len(exponents)
    random.seed(seed)
    
    # Initialize with random coefficients
    current_coeffs = [random.choice(coeff_choices) for _ in range(n)]
    current_dense = _sparse_to_dense([(exponents[i], current_coeffs[i]) for i in range(n)])
    current_sq = _count_nonzero_square_fast(current_dense)
    
    best_coeffs = list(current_coeffs)
    best_sq = current_sq
    
    # Simulated annealing parameters
    T_start = 10.0
    T_end = 0.1
    n_iter = max(1, int((deadline - time.time()) * 800))
    stall_count = 0
    max_stall = 5000
    
    for it in range(n_iter):
        if time.time() >= deadline - 2.0:
            break
        
        progress = it / max(1, n_iter - 1)
        T = T_start * (T_end / T_start) ** progress
        
        move_type = random.choices(
            ['single_coeff', 'flip_sign', 'pair_coeff', 'swap_coeff'],
            weights=[40, 25, 20, 15]
        )[0]
        
        trial_coeffs = list(current_coeffs)
        valid = True
        
        if move_type == 'single_coeff':
            idx = random.randrange(n)
            new_c = random.choice(coeff_choices)
            if new_c != trial_coeffs[idx]:
                trial_coeffs[idx] = new_c
            else:
                valid = False
                
        elif move_type == 'flip_sign':
            idx = random.randrange(n)
            trial_coeffs[idx] = -trial_coeffs[idx]
            
        elif move_type == 'pair_coeff':
            if n >= 2:
                i, j = random.sample(range(n), 2)
                new_ci = random.choice(coeff_choices)
                new_cj = random.choice(coeff_choices)
                if new_ci != trial_coeffs[i] or new_cj != trial_coeffs[j]:
                    trial_coeffs[i] = new_ci
                    trial_coeffs[j] = new_cj
                else:
                    valid = False
            else:
                valid = False
                
        elif move_type == 'swap_coeff':
            if n >= 2:
                i, j = random.sample(range(n), 2)
                if trial_coeffs[i] != trial_coeffs[j]:
                    trial_coeffs[i], trial_coeffs[j] = trial_coeffs[j], trial_coeffs[i]
                else:
                    valid = False
            else:
                valid = False
        
        if not valid:
            continue
        
        trial_sparse = [(exponents[i], trial_coeffs[i]) for i in range(n)]
        trial_dense = _sparse_to_dense(trial_sparse)
        
        if len(trial_dense) > 41 or trial_dense[-1] == 0:
            continue
        if sum(1 for c in trial_dense if c != 0) != 12:
            continue
        if any(c < -64 or c > 64 for c in trial_dense):
            continue
        
        trial_sq = _count_nonzero_square_fast(trial_dense)
        
        delta = trial_sq - current_sq
        if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-6)):
            current_coeffs = trial_coeffs
            current_sq = trial_sq
            stall_count = 0
            
            if trial_sq < best_sq:
                best_sq = trial_sq
                best_coeffs = list(trial_coeffs)
        else:
            stall_count += 1
    
    return best_coeffs, best_sq

def _general_hill_climb(sparse_init, deadline, coeff_choices=None):
    """
    General hill-climbing that can modify both exponents and coefficients.
    """
    if coeff_choices is None:
        coeff_choices = [-4, -3, -2, -1, 1, 2, 3, 4]
    
    current_sparse = list(sparse_init)
    current_dense = _sparse_to_dense(current_sparse)
    current_sq = _count_nonzero_square_fast(current_dense)
    
    best_sparse = list(current_sparse)
    best_sq = current_sq
    
    random.seed(123)
    max_exp = 40
    stall_count = 0
    max_stall = 3000
    
    while time.time() < deadline - 3.0 and stall_count < max_stall:
        move_type = random.choices(
            ['flip_sign', 'shift_exp', 'change_coeff', 'shift_pair', 'coeff_pair', 'swap_exps'],
            weights=[20, 25, 20, 15, 10, 10]
        )[0]
        
        trial_sparse = list(current_sparse)
        valid = True
        
        if move_type == 'flip_sign':
            idx = random.randrange(len(trial_sparse))
            e, c = trial_sparse[idx]
            trial_sparse[idx] = (e, -c)
            
        elif move_type == 'shift_exp':
            idx = random.randrange(len(trial_sparse))
            e, c = trial_sparse[idx]
            shift = random.choice([-3, -2, -1, 1, 2, 3])
            new_e = e + shift
            if 0 <= new_e <= max_exp:
                existing_exps = set(x[0] for x in trial_sparse if x[0] != e)
                if new_e not in existing_exps:
                    trial_sparse[idx] = (new_e, c)
                else:
                    valid = False
            else:
                valid = False
                
        elif move_type == 'change_coeff':
            idx = random.randrange(len(trial_sparse))
            e, c = trial_sparse[idx]
            new_c = random.choice(coeff_choices)
            if new_c != c:
                trial_sparse[idx] = (e, new_c)
            else:
                valid = False
                
        elif move_type == 'shift_pair':
            if len(trial_sparse) >= 2:
                i, j = random.sample(range(len(trial_sparse)), 2)
                ei, ci = trial_sparse[i]
                ej, cj = trial_sparse[j]
                shift_i = random.choice([-2, -1, 1, 2])
                shift_j = random.choice([-2, -1, 1, 2])
                new_ei = ei + shift_i
                new_ej = ej + shift_j
                if 0 <= new_ei <= max_exp and 0 <= new_ej <= max_exp:
                    other_exps = set(x[0] for k, x in enumerate(trial_sparse) if k not in (i, j))
                    if new_ei not in other_exps and new_ej not in other_exps and new_ei != new_ej:
                        trial_sparse[i] = (new_ei, ci)
                        trial_sparse[j] = (new_ej, cj)
                    else:
                        valid = False
                else:
                    valid = False
            else:
                valid = False
                
        elif move_type == 'coeff_pair':
            if len(trial_sparse) >= 2:
                i, j = random.sample(range(len(trial_sparse)), 2)
                ei, ci = trial_sparse[i]
                ej, cj = trial_sparse[j]
                new_ci = random.choice(coeff_choices)
                new_cj = random.choice(coeff_choices)
                if new_ci != ci or new_cj != cj:
                    trial_sparse[i] = (ei, new_ci)
                    trial_sparse[j] = (ej, new_cj)
                else:
                    valid = False
            else:
                valid = False
                
        elif move_type == 'swap_exps':
            if len(trial_sparse) >= 2:
                i, j = random.sample(range(len(trial_sparse)), 2)
                ei, ci = trial_sparse[i]
                ej, cj = trial_sparse[j]
                trial_sparse[i] = (ej, ci)
                trial_sparse[j] = (ei, cj)
        
        if not valid:
            continue
        
        if len(trial_sparse) != 12:
            continue
        exps = [x[0] for x in trial_sparse]
        if len(exps) != len(set(exps)):
            continue
        
        trial_dense = _sparse_to_dense(trial_sparse)
        if not _validate(trial_dense):
            continue
        
        trial_sq = _count_nonzero_square_fast(trial_dense)
        
        if trial_sq < current_sq:
            current_sparse = trial_sparse
            current_dense = trial_dense
            current_sq = trial_sq
            stall_count = 0
            if trial_sq < best_sq:
                best_sq = trial_sq
                best_sparse = list(trial_sparse)
        else:
            stall_count += 1
            if random.random() < 0.03:
                current_sparse = trial_sparse
                current_dense = trial_dense
                current_sq = trial_sq
    
    return best_sparse, best_sq

def _try_ap_construction(start, k, deadline, coeff_choices=None, seed=0):
    """
    Try an arithmetic progression with given start and step k.
    """
    if start < 0 or start + 11 * k > 40:
        return None, float('inf')
    
    exponents = [start + i * k for i in range(12)]
    best_coeffs, best_sq = _coeffs_only_optimize(exponents, deadline, coeff_choices, seed)
    sparse = [(exponents[i], best_coeffs[i]) for i in range(12)]
    return sparse, best_sq

def _structured_search(deadline):
    """
    Try specific algebraic constructions to find good starting points.
    """
    best_sparse = None
    best_sq = float('inf')
    
    random.seed(99)
    
    # Construction 1: Arithmetic progressions with various steps
    for k in range(1, 4):
        for start in range(0, 41 - 11 * k + 1):
            if time.time() > deadline - 5.0:
                return best_sparse, best_sq
            sparse, sq = _try_ap_construction(start, k, deadline - 5.0, seed=start * 100 + k)
            if sq < best_sq:
                best_sq = sq
                best_sparse = list(sparse)
    
    # Construction 2: Random with pairwise cancellation targeting
    for _ in range(200):
        if time.time() > deadline - 5.0:
            break
        exps = sorted(random.sample(range(0, 41), 12))
        sparse = [(e, random.choice([-2, -1, 1, 2])) for e in exps]
        dense = _sparse_to_dense(sparse)
        if not _validate(dense):
            continue
        sq = _count_nonzero_square_fast(dense)
        if sq < best_sq:
            best_sq = sq
            best_sparse = list(sparse)
    
    return best_sparse, best_sq

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin for 90s limit
    
    best_sparse = None
    best_sq = float('inf')
    
    # Phase 1: Exhaustive sign search on consecutive exponents 0..11
    # This is the key insight from the inspiration: densest structure + sign search
    if time.time() < deadline - 60.0:
        try:
            sign_sparse, sign_sq = _brute_force_sign_search_consecutive(deadline - 55.0, start=0, length=12)
            if sign_sparse is not None:
                best_sparse = sign_sparse
                best_sq = sign_sq
        except Exception:
            pass
    
    # Phase 2: Also try consecutive exponents starting at other positions
    for start in [1, 2, 3]:
        if time.time() > deadline - 50.0:
            break
        try:
            sign_sparse, sign_sq = _brute_force_sign_search_consecutive(deadline - 45.0, start=start, length=12)
            if sign_sparse is not None and sign_sq < best_sq:
                best_sq = sign_sq
                best_sparse = list(sign_sparse)
        except Exception:
            pass
    
    # Phase 3: Intensive coefficient optimization on the best exponents found
    if best_sparse is not None and time.time() < deadline - 20.0:
        try:
            exps = [x[0] for x in best_sparse]
            coeffs, sq = _coeffs_only_optimize(
                exps, deadline - 15.0,
                coeff_choices=[-4, -3, -2, -1, 1, 2, 3, 4],
                seed=42
            )
            sparse = [(exps[i], coeffs[i]) for i in range(12)]
            dense = _sparse_to_dense(sparse)
            if _validate(dense) and sq < best_sq:
                best_sq = sq
                best_sparse = list(sparse)
        except Exception:
            pass
    
    # Phase 4: Try other AP structures with coefficient optimization
    if time.time() < deadline - 15.0:
        ap_configs = []
        for k in range(1, 4):
            for start in range(0, 41 - 11 * k + 1):
                ap_configs.append((start, k))
        
        ap_configs.sort(key=lambda x: x[1])
        
        remaining = deadline - time.time() - 10.0
        n_configs = min(10, len(ap_configs))
        time_per_config = max(1.0, remaining / n_configs)
        
        for i, (start, k) in enumerate(ap_configs[:n_configs]):
            if time.time() >= deadline - 10.0:
                break
            
            exponents = [start + j * k for j in range(12)]
            if exponents[-1] > 40:
                continue
            
            coeffs, sq = _coeffs_only_optimize(
                exponents, time.time() + time_per_config,
                coeff_choices=[-4, -3, -2, -1, 1, 2, 3, 4],
                seed=start * 1000 + k
            )
            
            sparse = [(exponents[j], coeffs[j]) for j in range(12)]
            dense = _sparse_to_dense(sparse)
            if not _validate(dense):
                continue
            
            if sq < best_sq:
                best_sq = sq
                best_sparse = list(sparse)
    
    # Phase 5: General hill-climbing from best found so far
    if best_sparse is not None and time.time() < deadline - 5.0:
        try:
            found_sparse, found_sq = _general_hill_climb(best_sparse, deadline)
            if found_sq < best_sq:
                best_sq = found_sq
                best_sparse = list(found_sparse)
        except Exception:
            pass
    
    # Final: One more intensive coefficient optimization from best
    if best_sparse is not None and time.time() < deadline - 3.0:
        try:
            exps = [x[0] for x in best_sparse]
            coeffs, sq = _coeffs_only_optimize(
                exps, deadline,
                coeff_choices=[-4, -3, -2, -1, 1, 2, 3, 4],
                seed=999
            )
            sparse = [(exps[i], coeffs[i]) for i in range(12)]
            dense = _sparse_to_dense(sparse)
            if _validate(dense) and sq < best_sq:
                best_sq = sq
                best_sparse = list(sparse)
        except Exception:
            pass
    
    # Convert to dense and validate
    if best_sparse is None:
        best_coeffs = [1] * 12
    else:
        best_coeffs = _sparse_to_dense(best_sparse)
        if not _validate(best_coeffs):
            best_coeffs = [1] * 12
    
    return best_coeffs
# EVOLVE-BLOCK-END
