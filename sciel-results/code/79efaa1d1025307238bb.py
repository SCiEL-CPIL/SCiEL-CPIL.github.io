# EVOLVE-BLOCK-START
import numpy as np
import time

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # 600s limit, keep margin

    def two_point_P(p, L=1.0):
        """P[X1+X2+X3 < 2*X4] for 2-point {0, L} with mass p on 0."""
        # X4 = 0 with prob p: need X1+X2+X3 < 0 -> impossible (all >= 0)
        # X4 = L with prob (1-p): need X1+X2+X3 < 2L
        #   Sum < 2L means at most one of X1,X2,X3 is L
        #   P[at most one L] = p^3 + 3*p^2*(1-p)
        return (1.0 - p) * (p**3 + 3.0 * p**2 * (1.0 - p))
    
    # Golden section search for optimal p in 2-point case
    a, b = 0.01, 0.99
    gr = (np.sqrt(5.0) - 1.0) / 2.0  # golden ratio conjugate ~0.618
    
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = two_point_P(c)
    fd = two_point_P(d)
    
    max_iters = 200
    for i in range(max_iters):
        if time.time() > deadline - 30.0:
            break
        if fc < fd:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = two_point_P(d)
        else:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = two_point_P(c)
    
    p_opt = (a + b) / 2.0
    P_opt = two_point_P(p_opt)
    
    best_P = P_opt
    best_support = np.array([0.0, 1.0])
    best_weights = np.array([p_opt, 1.0 - p_opt])
    
    # 3-point distribution: {0, a, 1} with masses (m0, ma, m1)
    # Use a smarter search: for fixed a, optimize over (m0, ma) using gradient-free methods
    # or a finer grid
    
    def three_point_P(a, m0, ma):
        """P[X1+X2+X3 < 2*X4] for 3-point {0, a, 1} with masses m0, ma, m1."""
        m1 = 1.0 - m0 - ma
        if m1 < -1e-12 or m0 < -1e-12 or ma < -1e-12:
            return 0.0
        m1 = max(m1, 0.0)
        m0 = max(m0, 0.0)
        ma = max(ma, 0.0)
        
        points = np.array([0.0, a, 1.0])
        weights = np.array([m0, ma, m1])
        
        # For each possible X4 value, compute P[X1+X2+X3 < 2*x4]
        total_P = 0.0
        for k in range(3):
            w4 = weights[k]
            if w4 <= 0:
                continue
            threshold = 2.0 * points[k]
            # Compute P[X1+X2+X3 < threshold]
            p_sum = 0.0
            for i in range(3):
                for j in range(3):
                    for l_idx in range(3):
                        s = points[i] + points[j] + points[l_idx]
                        if s < threshold:
                            p_sum += weights[i] * weights[j] * weights[l_idx]
            total_P += w4 * p_sum
        return total_P
    
    # Finer grid search over a, then refine
    a_vals = np.linspace(0.05, 0.95, 40)
    
    for a in a_vals:
        if time.time() > deadline - 120.0:
            break
        n_grid = 30
        for u in np.linspace(0.01, 0.99, n_grid):
            for v in np.linspace(0.01, 0.99, n_grid):
                m0 = u
                ma = v * (1.0 - u)
                m1 = (1.0 - v) * (1.0 - u)
                if m1 < 0.005 or m0 < 0.005 or ma < 0.005:
                    continue
                P = three_point_P(a, m0, ma)
                if P > best_P:
                    best_P = P
                    best_support = np.array([0.0, a, 1.0])
                    best_weights = np.array([m0, ma, m1])
    
    # Try 4-point distributions: {0, a, b, 1} with masses (m0, ma, mb, m1)
    # This is a 4-variable optimization; do a very coarse search
    def four_point_P(a, b, m0, ma, mb):
        """P[X1+X2+X3 < 2*X4] for 4-point {0, a, b, 1} with masses m0, ma, mb, m1."""
        m1 = 1.0 - m0 - ma - mb
        if m1 < -1e-12 or m0 < -1e-12 or ma < -1e-12 or mb < -1e-12:
            return 0.0
        m1 = max(m1, 0.0)
        m0 = max(m0, 0.0)
        ma = max(ma, 0.0)
        mb = max(mb, 0.0)
        
        points = np.array([0.0, a, b, 1.0])
        weights = np.array([m0, ma, mb, m1])
        
        total_P = 0.0
        for k in range(4):
            w4 = weights[k]
            if w4 <= 0:
                continue
            threshold = 2.0 * points[k]
            p_sum = 0.0
            for i in range(4):
                for j in range(4):
                    for l_idx in range(4):
                        s = points[i] + points[j] + points[l_idx]
                        if s < threshold:
                            p_sum += weights[i] * weights[j] * weights[l_idx]
            total_P += w4 * p_sum
        return total_P
    
    # Refined 4-point search
    if time.time() < deadline - 180.0:
        a_vals4 = np.linspace(0.1, 0.5, 8)
        b_vals4 = np.linspace(0.5, 0.9, 8)
        for a in a_vals4:
            for b in b_vals4:
                if b <= a:
                    continue
                for m0 in np.linspace(0.05, 0.5, 8):
                    for ma in np.linspace(0.05, 0.4, 8):
                        mb_max = 0.4
                        for mb in np.linspace(0.05, mb_max, 8):
                            m1 = 1.0 - m0 - ma - mb
                            if m1 < 0.05:
                                continue
                            P = four_point_P(a, b, m0, ma, mb)
                            if P > best_P:
                                best_P = P
                                best_support = np.array([0.0, a, b, 1.0])
                                best_weights = np.array([m0, ma, mb, m1])
    
    # Final refinement: try to polish the best solution with local search
    if time.time() < deadline - 30.0:
        # Simple local perturbation on the best solution
        support = best_support.copy()
        weights = best_weights.copy()
        n = len(support)
        
        def full_P(supp, wts):
            """Compute P[X1+X2+X3 < 2*X4] for general discrete measure."""
            total_P = 0.0
            for k in range(n):
                w4 = wts[k]
                if w4 <= 0:
                    continue
                threshold = 2.0 * supp[k]
                p_sum = 0.0
                for i in range(n):
                    for j in range(n):
                        for l_idx in range(n):
                            s = supp[i] + supp[j] + supp[l_idx]
                            if s < threshold:
                                p_sum += wts[i] * wts[j] * wts[l_idx]
                total_P += w4 * p_sum
            return total_P
        
        # Try small perturbations of weights and support
        best_local_P = best_P
        best_local_support = support.copy()
        best_local_weights = weights.copy()
        
        step_w = 0.01
        for _ in range(100):
            if time.time() > deadline - 10.0:
                break
            perturbed = best_local_weights.copy()
            idx = np.random.randint(0, n)
            perturbed[idx] += step_w * np.random.randn()
            perturbed = np.maximum(perturbed, 0.0)
            if perturbed.sum() < 1e-9:
                continue
            perturbed /= perturbed.sum()
            P = full_P(best_local_support, perturbed)
            if P > best_local_P:
                best_local_P = P
                best_local_weights = perturbed.copy()
        
        step_s = 0.01
        for _ in range(100):
            if time.time() > deadline - 10.0:
                break
            perturbed_supp = best_local_support.copy()
            idx = np.random.randint(0, n)
            perturbed_supp[idx] += step_s * np.random.randn()
            perturbed_supp = np.maximum(perturbed_supp, 0.0)
            perturbed_supp = np.sort(perturbed_supp)
            P = full_P(perturbed_supp, best_local_weights)
            if P > best_local_P:
                best_local_P = P
                best_local_support = perturbed_supp.copy()
        
        if best_local_P > best_P:
            best_P = best_local_P
            best_support = best_local_support
            best_weights = best_local_weights
    
    support = best_support.astype(np.float64)
    weights = best_weights.astype(np.float64)
    
    # Ensure validity
    assert len(support) == len(weights)
    assert np.all(support >= -1e-9)
    assert np.all(weights >= -1e-9)
    assert np.sum(weights) > 1e-9
    
    return support, weights

# EVOLVE-BLOCK-END
