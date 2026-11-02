"""Matrix multiplication 2x3x4 rank-minimal exact decomposition."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product, combinations

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # 2, 3, 4
    schoolbook_rank = n * m * p  # 24

    # Build schoolbook solution (rank 24)
    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank

    # Try ranks from 23 down to 12
    for target_rank in range(23, 11, -1):
        if time.time() > deadline - 5:
            break
        result = _search_rank(n, m, p, target_rank, deadline)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break

    return best_result

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using Fraction."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                u_row = U[i * m + j]
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                for r in range(rank):
                    s += Fraction(u_row[r]) * Fraction(v_row[r]) * Fraction(w_row[r])
                if s != Fraction(1):
                    return False
    return True

def _snap_to_rational(U, V, W, n, m, p, max_denom=32):
    """Snap to rationals with small denominators and verify exactly."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for denom in [1, 2, 4, 8, 16, 32]:
        U_scaled = U_arr * denom
        V_scaled = V_arr * denom
        W_scaled = W_arr * denom
        
        if np.allclose(U_scaled, np.round(U_scaled), atol=1e-6) and \
           np.allclose(V_scaled, np.round(V_scaled), atol=1e-6) and \
           np.allclose(W_scaled, np.round(W_scaled), atol=1e-6):
            
            U_int = np.round(U_scaled).astype(int)
            V_int = np.round(V_scaled).astype(int)
            W_int = np.round(W_scaled).astype(int)
            
            U_frac = [[Fraction(a, denom) for a in row] for row in U_int]
            V_frac = [[Fraction(a, denom) for a in row] for row in V_int]
            W_frac = [[Fraction(a, denom) for a in row] for row in W_int]
            
            if _verify_exact(U_frac, V_frac, W_frac, n, m, p):
                return U_frac, V_frac, W_frac
    return None

def _compute_residuals(U, V, W, n, m, p):
    """Compute all residuals efficiently."""
    um, vm, wm = n * m, m * p, p * n
    residuals = np.zeros(n * m * p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * (m * p) + j * p + k
                u_row = U[i * m + j]
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                residuals[idx] = np.dot(u_row, v_row * w_row) - 1.0
    return residuals

def _search_rank(n, m, p, target_rank, deadline):
    """
    Search for a decomposition of given rank using multiple strategies:
    1. Coordinate descent with rational snapping
    2. Integer/half-integer grid search with structured patterns
    3. Perturbation-based local search
    """
    um, vm, wm = n * m, m * p, p * n
    
    # Strategy 1: Coordinate descent from multiple initializations
    np.random.seed(42 + target_rank)
    
    max_attempts = 50
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if attempt < 15:
            # Sparse binary initialization with random positions
            U = np.zeros((um, target_rank))
            V = np.zeros((vm, target_rank))
            W = np.zeros((wm, target_rank))
            for r in range(target_rank):
                U[np.random.randint(um), r] = 1.0
                V[np.random.randint(vm), r] = 1.0
                W[np.random.randint(wm), r] = 1.0
        elif attempt < 30:
            # Half-integer initialization
            choices = np.array([0, 1, -1, 0.5, -0.5])
            U = np.random.choice(choices, size=(um, target_rank))
            V = np.random.choice(choices, size=(vm, target_rank))
            W = np.random.choice(choices, size=(wm, target_rank))
        elif attempt < 40:
            # Small random values
            U = np.random.uniform(-1, 1, size=(um, target_rank))
            V = np.random.uniform(-1, 1, size=(vm, target_rank))
            W = np.random.uniform(-1, 1, size=(wm, target_rank))
        else:
            # Structured: try to use schoolbook-like patterns
            U = np.zeros((um, target_rank))
            V = np.zeros((vm, target_rank))
            W = np.zeros((wm, target_rank))
            # Assign random permutations of schoolbook columns
            perm = np.random.permutation(n * m * p)
            for r in range(target_rank):
                idx = perm[r]
                i = idx // (m * p)
                j = (idx // p) % m
                k = idx % p
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
        
        residuals = _compute_residuals(U, V, W, n, m, p)
        max_res = np.max(np.abs(residuals))
        
        if max_res < 1e-8:
            snapped = _snap_to_rational(U, V, W, n, m, p)
            if snapped:
                return {"U": snapped[0], "V": snapped[1], "W": snapped[2]}
        
        # Coordinate descent
        steps = 500
        for step in range(steps):
            if time.time() > deadline:
                break
            
            worst_idx = np.argmax(np.abs(residuals))
            wi = worst_idx // (m * p)
            wj = (worst_idx // p) % m
            wk = worst_idx % p
            
            current_worst_val = residuals[worst_idx]
            if abs(current_worst_val) < 1e-10:
                break
            
            improved = False
            
            # Try U updates
            for r in range(target_rank):
                v_val = V[wj * p + wk, r]
                w_val = W[wk * n + wi, r]
                if abs(v_val) < 1e-12 or abs(w_val) < 1e-12:
                    continue
                delta_optimal = -current_worst_val / (v_val * w_val)
                for delta in [delta_optimal, delta_optimal * 0.5, delta_optimal * 0.25, 0.5, -0.5, 1.0, -1.0]:
                    old_val = U[wi * m + wj, r]
                    U[wi * m + wj, r] = old_val + delta
                    new_res = current_worst_val + delta * v_val * w_val
                    if abs(new_res) < abs(current_worst_val):
                        residuals[worst_idx] = new_res
                        improved = True
                        break
                    else:
                        U[wi * m + wj, r] = old_val
                if improved:
                    break
            
            if improved:
                continue

            # Try V updates
            for r in range(target_rank):
                u_val = U[wi * m + wj, r]
                w_val = W[wk * n + wi, r]
                if abs(u_val) < 1e-12 or abs(w_val) < 1e-12:
                    continue
                delta_optimal = -current_worst_val / (u_val * w_val)
                for delta in [delta_optimal, delta_optimal * 0.5, delta_optimal * 0.25, 0.5, -0.5, 1.0, -1.0]:
                    old_val = V[wj * p + wk, r]
                    V[wj * p + wk, r] = old_val + delta
                    new_res = current_worst_val + delta * u_val * w_val
                    if abs(new_res) < abs(current_worst_val):
                        residuals[worst_idx] = new_res
                        improved = True
                        break
                    else:
                        V[wj * p + wk, r] = old_val
                if improved:
                    break
            
            if improved:
                continue

            # Try W updates
            for r in range(target_rank):
                u_val = U[wi * m + wj, r]
                v_val = V[wj * p + wk, r]
                if abs(u_val) < 1e-12 or abs(v_val) < 1e-12:
                    continue
                delta_optimal = -current_worst_val / (u_val * v_val)
                for delta in [delta_optimal, delta_optimal * 0.5, delta_optimal * 0.25, 0.5, -0.5, 1.0, -1.0]:
                    old_val = W[wk * n + wi, r]
                    W[wk * n + wi, r] = old_val + delta
                    new_res = current_worst_val + delta * u_val * v_val
                    if abs(new_res) < abs(current_worst_val):
                        residuals[worst_idx] = new_res
                        improved = True
                        break
                    else:
                        W[wk * n + wi, r] = old_val
                if improved:
                    break
            
            if not improved:
                # Random perturbation to escape local minima
                if step % 10 == 0:
                    idx_u = np.random.randint(um)
                    idx_v = np.random.randint(vm)
                    idx_w = np.random.randint(wm)
                    r = np.random.randint(target_rank)
                    U[idx_u, r] += np.random.choice([-1, -0.5, 0.5, 1])
                    V[idx_v, r] += np.random.choice([-1, -0.5, 0.5, 1])
                    W[idx_w, r] += np.random.choice([-1, -0.5, 0.5, 1])
                    residuals = _compute_residuals(U, V, W, n, m, p)
                    max_res = np.max(np.abs(residuals))
                    if max_res < 1e-8:
                        snapped = _snap_to_rational(U, V, W, n, m, p)
                        if snapped:
                            return {"U": snapped[0], "V": snapped[1], "W": snapped[2]}
                else:
                    break
        
        # Final check
        residuals = _compute_residuals(U, V, W, n, m, p)
        max_res = np.max(np.abs(residuals))
        if max_res < 1e-8:
            snapped = _snap_to_rational(U, V, W, n, m, p)
            if snapped:
                return {"U": snapped[0], "V": snapped[1], "W": snapped[2]}
    
    # Strategy 2: Direct search over small integer/half-integer grids for lower ranks
    if target_rank <= 20:
        result = _grid_search(n, m, p, target_rank, deadline)
        if result is not None:
            return result
    
    return None

def _grid_search(n, m, p, target_rank, deadline):
    """
    Direct search over structured patterns with small integer/half-integer values.
    This is a heuristic that tries to find sparse decompositions.
    """
    um, vm, wm = n * m, m * p, p * n
    np.random.seed(1000 + target_rank)
    
    # Try to find a decomposition where each rank-1 term has a structured pattern
    max_attempts = 100
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Initialize with random sparse patterns
        U = np.zeros((um, target_rank))
        V = np.zeros((vm, target_rank))
        W = np.zeros((wm, target_rank))
        
        for r in range(target_rank):
            # Each rank-1 term picks random rows
            u_row = np.random.randint(um)
            v_row = np.random.randint(vm)
            w_row = np.random.randint(wm)
            
            U[u_row, r] = np.random.choice([1, -1, 0.5, -0.5])
            V[v_row, r] = np.random.choice([1, -1, 0.5, -0.5])
            W[w_row, r] = np.random.choice([1, -1, 0.5, -0.5])
        
        residuals = _compute_residuals(U, V, W, n, m, p)
        max_res = np.max(np.abs(residuals))
        
        if max_res < 1e-8:
            snapped = _snap_to_rational(U, V, W, n, m, p)
            if snapped:
                return {"U": snapped[0], "V": snapped[1], "W": snapped[2]}
        
        # Local refinement
        for step in range(200):
            if time.time() > deadline:
                break
            
            worst_idx = np.argmax(np.abs(residuals))
            wi = worst_idx // (m * p)
            wj = (worst_idx // p) % m
            wk = worst_idx % p
            
            current_worst_val = residuals[worst_idx]
            if abs(current_worst_val) < 1e-10:
                break
            
            improved = False
            
            # Try adjusting the value at the worst position
            for r in range(target_rank):
                v_val = V[wj * p + wk, r]
                w_val = W[wk * n + wi, r]
                if abs(v_val) < 1e-12 or abs(w_val) < 1e-12:
                    continue
                
                # Try snapping U[wi*m+wj, r] to a nearby rational
                current_u = U[wi * m + wj, r]
                for new_u in [current_u, current_u + 0.5, current_u - 0.5, 
                              np.round(current_u), np.round(current_u * 2) / 2]:
                    old_u = U[wi * m + wj, r]
                    U[wi * m + wj, r] = new_u
                    new_res = current_worst_val + (new_u - old_u) * v_val * w_val
                    if abs(new_res) < abs(current_worst_val):
                        residuals[worst_idx] = new_res
                        improved = True
                        break
                    else:
                        U[wi * m + wj, r] = old_u
                
                if improved:
                    break
            
            if improved:
                continue
            
            # Similar for V and W
            for r in range(target_rank):
                u_val = U[wi * m + wj, r]
                w_val = W[wk * n + wi, r]
                if abs(u_val) < 1e-12 or abs(w_val) < 1e-12:
                    continue
                
                current_v = V[wj * p + wk, r]
                for new_v in [current_v, current_v + 0.5, current_v - 0.5,
                              np.round(current_v), np.round(current_v * 2) / 2]:
                    old_v = V[wj * p + wk, r]
                    V[wj * p + wk, r] = new_v
                    new_res = current_worst_val + (new_v - old_v) * u_val * w_val
                    if abs(new_res) < abs(current_worst_val):
                        residuals[worst_idx] = new_res
                        improved = True
                        break
                    else:
                        V[wj * p + wk, r] = old_v
                
                if improved:
                    break
            
            if improved:
                continue
            
            for r in range(target_rank):
                u_val = U[wi * m + wj, r]
                v_val = V[wj * p + wk, r]
                if abs(u_val) < 1e-12 or abs(v_val) < 1e-12:
                    continue
                
                current_w = W[wk * n + wi, r]
                for new_w in [current_w, current_w + 0.5, current_w - 0.5,
                              np.round(current_w), np.round(current_w * 2) / 2]:
                    old_w = W[wk * n + wi, r]
                    W[wk * n + wi, r] = new_w
                    new_res = current_worst_val + (new_w - old_w) * u_val * v_val
                    if abs(new_res) < abs(current_worst_val):
                        residuals[worst_idx] = new_res
                        improved = True
                        break
                    else:
                        W[wk * n + wi, r] = old_w
                
                if improved:
                    break
            
            if not improved:
                break
        
        residuals = _compute_residuals(U, V, W, n, m, p)
        max_res = np.max(np.abs(residuals))
        if max_res < 1e-8:
            snapped = _snap_to_rational(U, V, W, n, m, p)
            if snapped:
                return {"U": snapped[0], "V": snapped[1], "W": snapped[2]}
    
    return None
# EVOLVE-BLOCK-END
