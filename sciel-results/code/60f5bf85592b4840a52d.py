"""Product-formula coefficient optimizer for 6-site XXZ Heisenberg chain."""
import numpy as np
import time
import sys

_DEADLINE = 140.0
_T0 = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _T0)

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site open XXZ chain."""
    N = 6
    delta = 0.8
    field = 0.3
    
    # Spin-1/2 operators
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    
    I2 = np.eye(2, dtype=complex)
    
    def site_op(op, site):
        """Create operator acting on a specific site."""
        ops = [I2] * N
        ops[site] = op
        result = np.array([[1.0+0j]])
        for op in ops:
            result = np.kron(result, op)
        return result
    
    def bond_op(op1, op2, site1, site2):
        """Create operator acting on two sites."""
        ops = [I2] * N
        ops[site1] = op1
        ops[site2] = op2
        result = np.array([[1.0+0j]])
        for op in ops:
            result = np.kron(result, op)
        return result
    
    # Build A: odd bonds (0-1, 2-3, 4-5) + half field
    A = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(0, N, 2):
        A += bond_op(sx, sx, i, i+1) + bond_op(sy, sy, i, i+1) + delta * bond_op(sz, sz, i, i+1)
    # Half field
    for i in range(N):
        A += 0.5 * field * site_op(sz, i)
    
    # Build B: even bonds (1-2, 3-4) + half field
    B = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(1, N-1, 2):
        B += bond_op(sx, sx, i, i+1) + bond_op(sy, sy, i, i+1) + delta * bond_op(sz, sz, i, i+1)
    # Half field
    for i in range(N):
        B += 0.5 * field * site_op(sz, i)
    
    return A, B

def _precompute_eig(A, B):
    """Precompute eigen-decompositions for A, B, and A+B."""
    w_A, v_A = np.linalg.eigh(A)
    v_A_dag = v_A.conj().T
    
    w_B, v_B = np.linalg.eigh(B)
    v_B_dag = v_B.conj().T
    
    H_total = A + B
    w_AB, v_AB = np.linalg.eigh(H_total)
    v_AB_dag = v_AB.conj().T
    
    return w_A, v_A, v_A_dag, w_B, v_B, v_B_dag, w_AB, v_AB, v_AB_dag

def _expm_herm(w, v, v_dag, coeff):
    """Compute exp(-i * coeff * H) given eigen-decomp of H."""
    phases = np.exp(-1j * coeff * w)
    v_scaled = v * phases
    return v_scaled @ v_dag

def _product_formula(a, b, t, eig_data):
    """Compute product formula U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)."""
    w_A, v_A, v_A_dag, w_B, v_B, v_B_dag, _, _, _ = eig_data
    n = len(a)
    U = np.eye(2**6, dtype=complex)
    for k in range(n):
        U_B = _expm_herm(w_B, v_B, v_B_dag, b[k] * t)
        U = U_B @ U
        U_A = _expm_herm(w_A, v_A, v_A_dag, a[k] * t)
        U = U_A @ U
    return U

def _exact_propagator(t, eig_data):
    """Compute exact propagator exp(-i t (A+B))."""
    _, _, _, _, _, _, w_AB, v_AB, v_AB_dag = eig_data
    return _expm_herm(w_AB, v_AB, v_AB_dag, t)

def _spectral_norm_error(U, U_exact):
    """Compute spectral norm of U - U_exact."""
    return np.linalg.norm(U - U_exact, 2)

def _score(a, b, eig_data):
    """Compute mean spectral norm error at t = 0.25, 0.5, 1.0."""
    times = [0.25, 0.5, 1.0]
    errors = []
    for t in times:
        U = _product_formula(a, b, t, eig_data)
        U_exact = _exact_propagator(t, eig_data)
        errors.append(_spectral_norm_error(U, U_exact))
    return np.mean(errors)

def _normalize_sum(arr):
    """Normalize array to sum to 1, clipping to [-50, 50]."""
    arr = np.array(arr, dtype=float)
    arr = np.clip(arr, -50, 50)
    s = np.sum(arr)
    if abs(s) > 1e-15:
        arr = arr / s
    else:
        arr = np.ones_like(arr) / len(arr)
    return arr

def _random_symmetric_coefficients(rng):
    """Generate random symmetric coefficients for 12 stages."""
    # 6 parameters for A, 6 for B
    a_half = rng.uniform(-1, 1, 6)
    b_half = rng.uniform(-1, 1, 6)
    
    a_full = np.concatenate([a_half, a_half[::-1]])
    b_full = np.concatenate([b_half, b_half[::-1]])
    
    a_full = _normalize_sum(a_full)
    b_full = _normalize_sum(b_full)
    
    return a_full, b_full

def coefficients(stages):
    """Optimize product formula coefficients using Random Search + Local Refinement."""
    if stages != 12:
        # Fallback for non-12 stage counts
        substeps = stages // 2
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b
    
    A, B = _build_hamiltonians()
    eig_data = _precompute_eig(A, B)
    
    # Baseline: Uniform coefficients
    a_uniform = np.ones(12) / 12.0
    b_uniform = np.ones(12) / 12.0
    best_a = a_uniform
    best_b = b_uniform
    best_score = _score(a_uniform, b_uniform, eig_data)
    
    rng = np.random.default_rng(42)
    
    # Random Search
    max_evals = 2000
    eval_count = 0
    for _ in range(max_evals):
        if _remaining() < 10.0:
            break
            
        a_rand, b_rand = _random_symmetric_coefficients(rng)
        score = _score(a_rand, b_rand, eig_data)
        eval_count += 1
        
        if score < best_score:
            best_score = score
            best_a = a_rand
            best_b = b_rand
    
    # Optional: Local refinement of the best found
    if _remaining() > 20.0:
        # Simple local perturbation refinement
        current_a = best_a.copy()
        current_b = best_b.copy()
        current_score = best_score
        step_size = 0.1
        for _ in range(50):
            if _remaining() < 15.0:
                break
            improved = False
            for i in range(6):
                # Perturb a_half[i]
                orig_val = current_a[i]
                current_a[i] = orig_val + step_size * rng.uniform(-1, 1)
                current_a[11-i] = current_a[i] # maintain symmetry
                current_a = _normalize_sum(current_a)
                
                s = _score(current_a, current_b, eig_data)
                if s < current_score:
                    current_score = s
                    best_a = current_a.copy()
                    best_b = current_b.copy()
                    improved = True
                else:
                    current_a[i] = orig_val
                    current_a[11-i] = orig_val
                    current_a = _normalize_sum(current_a)
                    
            # Perturb b_half[i]
            for i in range(6):
                orig_val = current_b[i]
                current_b[i] = orig_val + step_size * rng.uniform(-1, 1)
                current_b[11-i] = current_b[i]
                current_b = _normalize_sum(current_b)
                
                s = _score(current_a, current_b, eig_data)
                if s < current_score:
                    current_score = s
                    best_a = current_a.copy()
                    best_b = current_b.copy()
                    improved = True
                else:
                    current_b[i] = orig_val
                    current_b[11-i] = orig_val
                    current_b = _normalize_sum(current_b)
            
            if not improved:
                step_size *= 0.5
                if step_size < 1e-4:
                    break
    
    return best_a.tolist(), best_b.tolist()
