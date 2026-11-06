"""Sparse squares of twelve-term integer polynomials - DFT spectral optimization."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start = time.time()
    deadline = start + 82.0  # safety margin before 90s kill

    best_poly = None
    best_nz = float('inf')

    # Precompute DFT matrix for 8192 points (2^13)
    N_DFT = 8192
    # We'll use a smaller DFT for speed: 4096 points
    N_DFT = 4096
    k_vals = np.arange(N_DFT, dtype=np.float64)
    # omega = exp(-2*pi*i/N_DFT)
    # For polynomial evaluation at roots of unity: P(w^k) = sum c_j * w^{j*k}
    # We need to be careful: the polynomial has degree up to 40, so we need N_DFT > 40*2 = 80
    # Actually for the square, degree goes up to 80, so N_DFT > 80 is needed for no aliasing
    # But we're using it as a surrogate, not for exact computation

    def dft_matrix(deg_max=40):
        """Create DFT evaluation matrix for polynomials of degree up to deg_max."""
        # P(x) = sum_{j=0}^{deg_max} c_j x^j
        # Evaluate at x = exp(2*pi*i*k/N_DFT) for k=0..N_DFT-1
        # M[k, j] = exp(2*pi*i*k*j/N_DFT)
        j = np.arange(deg_max + 1, dtype=np.float64)[:, None]
        k = np.arange(N_DFT, dtype=np.float64)[None, :]
        M = np.exp(2j * np.pi * k * j / N_DFT)
        return M

    # For speed, we'll precompute a smaller version
    # Actually, let's use a more efficient approach:
    # Instead of full DFT, use the fact that we only need L1 norm of |P(w^k)|^2
    # |P(w^k)|^2 = P(w^k) * conjugate(P(w^k))
    # The L1 norm of the spectrum relates to sparsity

    def square_nonzero_fast(exponents, coeffs):
        """Fast exact square nonzero count."""
        c = np.asarray(coeffs, dtype=np.int64)
        e = np.asarray(exponents, dtype=np.int64)
        degs = e[:, None] + e[None, :]
        vals = c[:, None] * c[None, :]
        result = np.bincount(degs.ravel(), weights=vals.ravel(), minlength=82)
        return int(np.count_nonzero(result))

    def spectral_surrogate(exponents, coeffs):
        """
        Compute a spectral surrogate for the number of nonzero terms in P^2.
        Uses evaluation at roots of unity.
        If P^2 is sparse, its DFT has small L1 norm relative to its max value.
        Actually, we want to MINIMIZE the number of nonzero terms.
        
        Better surrogate: The L1 norm of the DFT of P^2 coefficients.
        A sparse polynomial has a DFT that is 'flat' in some sense.
        
        Actually, let's think differently. The number of nonzero terms in P^2
        is related to how much cancellation occurs. 
        
        Let's use: sum of |P(w^k)|^4 over k as a proxy.
        If P^2 is sparse, this should be small relative to (sum |P(w^k)|^2)^2.
        
        Actually, the cleanest surrogate: 
        The L1 norm of the circular convolution of P with P (using DFT).
        But circular convolution wraps around, so we need N_DFT > 2*max_degree.
        """
        c = np.asarray(coeffs, dtype=np.float64)
        e = np.asarray(exponents, dtype=np.float64)
        
        # Build coefficient array
        max_deg = int(max(e))
        coeff_arr = np.zeros(max_deg + 1, dtype=np.float64)
        for ex, co in zip(e, c):
            coeff_arr[int(ex)] = co
        
        # Pad to length N_DFT for circular convolution
        padded = np.zeros(N_DFT, dtype=np.float64)
        padded[:max_deg+1] = coeff_arr
        
        # FFT
        F = np.fft.fft(padded)
        # P^2 in frequency domain
        F2 = F * F
        # Inverse FFT to get P^2 coefficients (circular)
        p2_coeffs = np.fft.ifft(F2).real
        # Only consider up to degree 2*max_deg
        p2_coeffs = p2_coeffs[:2*max_deg+1]
        
        # Count nonzeros (with tolerance for floating point)
        nz = np.count_nonzero(np.abs(p2_coeffs) > 1e-9)
        return float(nz)

    def spectral_l1_surrogate(exponents, coeffs):
        """
        L1 norm of the DFT of P^2 as a smooth surrogate.
        Lower L1 norm of the spectrum suggests more structure/cancellation.
        """
        c = np.asarray(coeffs, dtype=np.float64)
        e = np.asarray(exponents, dtype=np.float64)
        
        max_deg = int(max(e))
        coeff_arr = np.zeros(max_deg + 1, dtype=np.float64)
        for ex, co in zip(e, c):
            coeff_arr[int(ex)] = co
        
        padded = np.zeros(N_DFT, dtype=np.float64)
        padded[:max_deg+1] = coeff_arr
        
        F = np.fft.fft(padded)
        F2 = F * F
        # L1 norm of the spectrum of P^2
        l1 = np.sum(np.abs(F2))
        return float(l1)

    def validate(poly_coeff_list):
        if not poly_coeff_list or len(poly_coeff_list) > 41:
            return False
        if poly_coeff_list[-1] == 0:
            return False
        for c in poly_coeff_list:
            if not isinstance(c, int) or c < -64 or c > 64:
                return False
        if sum(1 for c in poly_coeff_list if c != 0) != 12:
            return False
        return True

    def try_candidate(exponents, coeffs):
        nonlocal best_poly, best_nz
        if len(exponents) != 12 or len(set(exponents)) != 12:
            return
        if any(e < 0 or e >= 41 for e in exponents):
            return
        if any(c == 0 or abs(c) > 64 for c in coeffs):
            return
        nz = square_nonzero_fast(exponents, coeffs)
        if nz < best_nz:
            best_nz = nz
            max_deg = max(exponents)
            poly = [0] * (max_deg + 1)
            for e, c in zip(exponents, coeffs):
                poly[e] = c
            best_poly = poly

    # Phase 1: Generate structured candidates
    candidates = []
    
    # AP with structured coefficients
    for d in range(1, 9):
        max_a = max(0, 40 - 11 * d)
        for a in range(0, max_a + 1):
            exps = [a + i * d for i in range(12)]
            if exps[-1] >= 41:
                continue
            for block_size in [1, 2, 3, 4, 6]:
                for sign in [1, -1]:
                    coefs = []
                    for i in range(12):
                        if (i // block_size) % 2 == 0:
                            coefs.append(sign)
                        else:
                            coefs.append(-sign)
                    candidates.append((exps, coefs))
                    for mag in [1, 2]:
                        coefs_mag = [c * mag for c in coefs]
                        if all(abs(c) <= 64 for c in coefs_mag):
                            candidates.append((exps, coefs_mag))
    
    # Two-AP structure
    for d1 in range(1, 5):
        for d2 in range(1, 5):
            for a1 in range(0, 12):
                for gap in range(3, 20):
                    a2 = a1 + gap
                    exp1 = [a1 + i * d1 for i in range(6)]
                    exp2 = [a2 + i * d2 for i in range(6)]
                    all_exps = sorted(exp1 + exp2)
                    if len(set(all_exps)) != 12 or any(e >= 41 for e in all_exps):
                        continue
                    for s1 in [1, -1]:
                        for s2 in [1, -1]:
                            coefs = [s1] * 6 + [s2] * 6
                            candidates.append((all_exps, coefs))
                            coefs_alt = [s1 * (1 if i%2==0 else -1) for i in range(6)] + \
                                       [s2 * (1 if i%2==0 else -1) for i in range(6)]
                            candidates.append((all_exps, coefs_alt))
    
    # Symmetric structures
    for center in range(6, 35):
        for offset_set in [
            [1,2,3,4,5,6], [1,2,3,4,5,7], [1,2,3,4,6,8],
            [1,2,3,5,7,9], [1,3,5,7,9,11], [2,4,6,8,10,12],
            [1,2,4,5,7,8], [1,3,4,6,7,9], [1,2,4,6,8,10],
            [2,3,4,5,6,7], [1,2,3,4,6,7], [1,2,3,5,6,8],
        ]:
            exps = sorted([center - o for o in offset_set] + [center + o for o in offset_set])
            if any(e < 0 or e >= 41 for e in exps) or len(set(exps)) != 12:
                continue
            for sign_pat in [[1]*6 + [1]*6, [1]*6 + [-1]*6, [1,-1]*3 + [1,-1]*3, [1,1,-1,-1,1,-1]*2]:
                coefs = list(sign_pat)
                candidates.append((exps, coefs))

    # Evaluate all candidates
    for exps, coefs in candidates:
        if time.time() > deadline - 45:
            break
        try_candidate(exps, coefs)
    
    # Phase 2: DFT-guided gradient descent on coefficient values
    # Fix exponents, optimize coefficients using spectral surrogate
    random.seed(42)
    
    def dft_gradient_descent(fixed_exps, initial_coefs, max_iters=5000):
        """
        Optimize coefficients for fixed exponents using DFT spectral surrogate.
        The surrogate is the L1 norm of the DFT of P^2, which is a smooth
        approximation to the count of nonzero terms.
        """
        nonlocal best_poly, best_nz
        
        cur_coefs = np.array(initial_coefs, dtype=np.float64)
        e = np.array(fixed_exps, dtype=np.float64)
        max_deg = int(max(e))
        
        # Build the linear map from coefficients to P^2 coefficients
        # P^2[j] = sum_{i: e_a + e_b = j} c_a * c_b
        # This is a quadratic form in c
        # P^2 = Q(c) where Q is a quadratic map
        
        # For gradient computation, we can use the DFT approach:
        # P(w^k) = sum_j c_j w^{j*k} (where j ranges over exponents)
        # |P(w^k)|^2 is the periodogram
        # The L1 norm of the DFT of P^2 relates to sum_k |P(w^k)|^4
        
        # Actually, let's compute the surrogate directly:
        # s(c) = sum_k |P(w^k)|^4 / (N_DFT^2)
        # This is proportional to the L2 norm squared of P^2 coefficients
        # But we want L1-like behavior...
        
        # Let's use: s(c) = sum_k |P(w^k)|^4
        # Gradient: ds/dc_a = sum_k 4 |P(w^k)|^2 * Re(conj(P(w^k)) * w^{e_a * k})
        
        k_vals = np.arange(N_DFT, dtype=np.float64)
        # Precompute w^{e_i * k} for each exponent
        # W[i, k] = exp(2*pi*i*e_i*k/N_DFT)
        W = np.exp(2j * np.pi * np.outer(e, k_vals) / N_DFT)
        
        cur_nz = square_nonzero_fast(fixed_exps, [int(round(c)) for c in cur_coefs])
        if cur_nz < best_nz:
            best_nz = cur_nz
            max_d = max(fixed_exps)
            poly = [0] * (max_d + 1)
            for ex, co in zip(fixed_exps, [int(round(c)) for c in cur_coefs]):
                poly[ex] = co
            best_poly = poly
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Compute P(w^k) = W @ c
            P_vals = W @ cur_coefs
            P_abs2 = np.abs(P_vals) ** 2
            
            # Surrogate: sum |P(w^k)|^4
            surrogate = np.sum(P_abs2 ** 2)
            
            # Gradient: d(surrogate)/dc_a = sum_k 4 * |P(w^k)|^2 * Re(conj(P(w^k)) * W[a,k])
            # = 4 * Re(conj(P_vals) * P_abs2) @ W.conj() ... let me recompute
            # s = sum_k |P_k|^4 = sum_k P_k * conj(P_k) * P_k * conj(P_k)
            # ds/dc_a = sum_k 4 * |P_k|^2 * conj(P_k) * W[a,k]
            # Taking real part for real c_a:
            # ds/dc_a = Re(sum_k 4 * |P_k|^2 * conj(P_k) * W[a,k])
            
            grad_complex = 4.0 * P_abs2 * np.conj(P_vals)
            grad = np.real(W.T @ grad_complex)
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                grad = grad / grad_norm
            
            # Adaptive step size
            step_size = 0.5 / (1.0 + it * 0.01)
            
            # Update
            new_coefs = cur_coefs - step_size * grad
            
            # Project to valid range [-64, 64], avoid zeros
            new_coefs = np.clip(new_coefs, -64, 64)
            # Avoid zeros: push away from 0
            for i in range(12):
                if abs(new_coefs[i]) < 0.5:
                    new_coefs[i] = 1.0 if new_coefs[i] >= 0 else -1.0
            
            # Evaluate
            rounded = [int(round(c)) for c in new_coefs]
            if any(c == 0 or abs(c) > 64 for c in rounded):
                continue
            
            nz = square_nonzero_fast(fixed_exps, rounded)
            if nz < cur_nz:
                cur_nz = nz
                cur_coefs = new_coefs
                if nz < best_nz:
                    best_nz = nz
                    max_d = max(fixed_exps)
                    poly = [0] * (max_d + 1)
                    for ex, co in zip(fixed_exps, rounded):
                        poly[ex] = co
                    best_poly = poly
            else:
                # Simulated annealing: accept with some probability
                if random.random() < 0.1:
                    cur_coefs = new_coefs
        
        return cur_coefs

    # Phase 3: Local search with DFT guidance
    def local_search_dft(initial_exps, initial_coefs, max_iters=3000):
        nonlocal best_poly, best_nz
        cur_exps = sorted(initial_exps)
        cur_coefs = list(initial_coefs)
        cur_nz = square_nonzero_fast(cur_exps, cur_coefs)
        
        if cur_nz < best_nz:
            best_nz = cur_nz
            max_deg = max(cur_exps)
            poly = [0] * (max_deg + 1)
            for e, c in zip(cur_exps, cur_coefs):
                poly[e] = c
            best_poly = poly
        
        no_improve = 0
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            new_exps = list(cur_exps)
            new_coefs = list(cur_coefs)
            
            r = random.random()
            if r < 0.35:
                idx = random.randint(0, 11)
                new_val = random.choice([-8,-6,-4,-3,-2,-1,1,2,3,4,5,6,8,12])
                if new_val != new_coefs[idx]:
                    new_coefs[idx] = new_val
            elif r < 0.55:
                idx = random.randint(0, 11)
                shift = random.randint(-6, 6)
                new_e = new_exps[idx] + shift
                if 0 <= new_e < 41 and new_e not in new_exps:
                    new_exps[idx] = new_e
                    new_exps.sort()
                else:
                    no_improve += 1
                    continue
            elif r < 0.65:
                i, j = random.sample(range(12), 2)
                new_coefs[i], new_coefs[j] = new_coefs[j], new_coefs[i]
            elif r < 0.75:
                i, j = random.sample(range(12), 2)
                new_coefs[i] = random.choice([-4,-3,-2,-1,1,2,3,4,5])
                new_coefs[j] = random.choice([-4,-3,-2,-1,1,2,3,4,5])
            elif r < 0.85:
                idx = random.randint(0, 11)
                shift = random.choice([-8,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,8])
                new_e = new_exps[idx] + shift
                if 0 <= new_e < 41 and new_e not in new_exps:
                    new_exps[idx] = new_e
                    new_exps.sort()
                else:
                    no_improve += 1
                    continue
            else:
                i, j = random.sample(range(12), 2)
                si = random.choice([-3,-2,-1,1,2,3])
                sj = random.choice([-3,-2,-1,1,2,3])
                ne_i = new_exps[i] + si
                ne_j = new_exps[j] + sj
                if (0 <= ne_i < 41 and 0 <= ne_j < 41 and 
                    ne_i not in new_exps and ne_j not in new_exps and ne_i != ne_j):
                    new_exps[i] = ne_i
                    new_exps[j] = ne_j
                    new_exps.sort()
                else:
                    no_improve += 1
                    continue
            
            if len(set(new_exps)) != 12 or any(e < 0 or e >= 41 for e in new_exps):
                continue
            if any(c == 0 or abs(c) > 64 for c in new_coefs):
                continue
            
            nz = square_nonzero_fast(new_exps, new_coefs)
            if nz < cur_nz:
                cur_nz = nz
                cur_exps = list(new_exps)
                cur_coefs = list(new_coefs)
                no_improve = 0
                if nz < best_nz:
                    best_nz = nz
                    max_deg = max(cur_exps)
                    poly = [0] * (max_deg + 1)
                    for e, c in zip(cur_exps, cur_coefs):
                        poly[e] = c
                    best_poly = poly
            else:
                no_improve += 1
            
            if no_improve > 200:
                break
        
        return cur_exps, cur_coefs, cur_nz

    # Run DFT gradient descent from best candidate
    if best_poly is not None and time.time() < deadline - 15:
        best_exp = [i for i, c in enumerate(best_poly) if c != 0]
        best_coef = [c for c in best_poly if c != 0]
        
        # DFT GD on best
        dft_gradient_descent(best_exp, best_coef, max_iters=3000)
        
        # Multiple restarts with perturbation
        for restart in range(15):
            if time.time() > deadline - 8:
                break
            
            pert_exp = list(best_exp)
            pert_coef = list(best_coef)
            
            for _ in range(random.randint(1, 4)):
                if random.random() < 0.5:
                    idx = random.randint(0, 11)
                    shift = random.randint(-4, 4)
                    new_e = pert_exp[idx] + shift
                    if 0 <= new_e < 41 and new_e not in pert_exp:
                        pert_exp[idx] = new_e
                        pert_exp.sort()
                else:
                    idx = random.randint(0, 11)
                    pert_coef[idx] = random.choice([-4,-3,-2,-1,1,2,3,4,5,6])
                    if pert_coef[idx] == 0:
                        pert_coef[idx] = 1
            
            if len(set(pert_exp)) == 12 and all(0 <= e < 41 for e in pert_exp) and \
               all(c != 0 and abs(c) <= 64 for c in pert_coef):
                local_search_dft(pert_exp, pert_coef, max_iters=1500)
        
        # Random starting points with DFT GD
        for _ in range(8):
            if time.time() > deadline - 8:
                break
            rtype = random.random()
            if rtype < 0.3:
                d = random.randint(1, 7)
                max_a = max(0, 40 - 11 * d)
                a = random.randint(0, max_a)
                exps = [a + i * d for i in range(12)]
                if exps[-1] >= 41:
                    continue
            elif rtype < 0.6:
                d1 = random.randint(1, 4)
                d2 = random.randint(1, 4)
                a1 = random.randint(0, 10)
                gap = random.randint(4, 18)
                a2 = a1 + gap
                exps = [a1 + i * d1 for i in range(6)] + [a2 + i * d2 for i in range(6)]
                if any(e >= 41 for e in exps) or len(set(exps)) != 12:
                    continue
                exps.sort()
            else:
                exps = sorted(random.sample(range(41), 12))
            
            coefs = [random.choice([-3,-2,-1,1,2,3]) for _ in range(12)]
            
            # First do DFT GD, then local search
            dft_gradient_descent(exps, coefs, max_iters=1000)
            if time.time() < deadline - 5:
                local_search_dft(exps, coefs, max_iters=800)
    
    # Phase 4: Targeted search for very sparse squares
    if time.time() < deadline - 10:
        for d in range(1, 6):
            max_a = max(0, 40 - 11 * d)
            for a in range(0, max_a + 1):
                base_exps = [a + i * d for i in range(12)]
                if base_exps[-1] >= 41:
                    continue
                
                for init_style in range(5):
                    if time.time() > deadline - 5:
                        break
                    if init_style == 0:
                        coefs = [1] * 12
                    elif init_style == 1:
                        coefs = [1 if i % 2 == 0 else -1 for i in range(12)]
                    elif init_style == 2:
                        coefs = [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1]
                    elif init_style == 3:
                        coefs = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
                    else:
                        coefs = [random.choice([-2,-1,1,2]) for _ in range(12)]
                    
                    local_search_dft(base_exps, coefs, max_iters=800)
    
    # Final fallback
    if best_poly is None:
        best_poly = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    if not validate(best_poly):
        best_poly = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    return best_poly

# EVOLVE-BLOCK-END
