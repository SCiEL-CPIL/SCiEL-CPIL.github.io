import numpy as np
import scipy.optimize as opt
import time
from math import factorial

def construct():
    start_time = time.time()
    deadline = 205.0  # safety margin before 210s kill

    # Precompute even moments: M[n] = ∫ x^{2n} e^{-x^2} dx = (2n)! sqrt(pi) / (4^n n!)
    max_moment = 30
    sqrt_pi = np.sqrt(np.pi)
    M = np.zeros(max_moment + 1)
    for n in range(max_moment + 1):
        M[n] = factorial(2 * n) * sqrt_pi / (4.0 ** n * factorial(n))

    # P(x) = sum_{k=0}^{6} a[k] x^{2k}
    # I0 = sum_{k,l} a[k] a[l] M[k+l]
    # I2 = sum_{k,l} a[k] a[l] M[k+l+1]
    # Q(x) = P'' - 2x P' + (x^2-1) P = sum_{i=0}^{7} q[i] x^{2i}
    # I_dd = sum_{i,j} q[i] q[j] M[i+j]
    # where q depends linearly on a:
    #   q0  = 2 a1 - a0
    #   q2  = 12 a2 - 5 a1 + a0
    #   q4  = 30 a3 - 13 a2 + a1
    #   q6  = 56 a4 - 25 a3 + a2
    #   q8  = 90 a5 - 41 a4 + a3
    #   q10 = 132 a6 - 61 a5 + a4
    #   q12 = a5 - a6
    #   q14 = a6

    # Build Gram matrices
    G0 = np.zeros((7, 7))
    G2 = np.zeros((7, 7))
    for k in range(7):
        for l in range(7):
            G0[k, l] = M[k + l]
            G2[k, l] = M[k + l + 1]

    # Q in terms of a: q = Qmat @ a  (8x7)
    Qmat = np.zeros((8, 7))
    # q0 = -a0 + 2a1
    Qmat[0, 0] = -1; Qmat[0, 1] = 2
    # q2 = a0 - 5a1 + 12a2
    Qmat[1, 0] = 1; Qmat[1, 1] = -5; Qmat[1, 2] = 12
    # q4 = a1 - 13a2 + 30a3
    Qmat[2, 1] = 1; Qmat[2, 2] = -13; Qmat[2, 3] = 30
    # q6 = a2 - 25a3 + 56a4
    Qmat[3, 2] = 1; Qmat[3, 3] = -25; Qmat[3, 4] = 56
    # q8 = a3 - 41a4 + 90a5
    Qmat[4, 3] = 1; Qmat[4, 4] = -41; Qmat[4, 5] = 90
    # q10 = a4 - 61a5 + 132a6
    Qmat[5, 4] = 1; Qmat[5, 5] = -61; Qmat[5, 6] = 132
    # q12 = a5 - a6
    Qmat[6, 5] = 1; Qmat[6, 6] = -1
    # q14 = a6
    Qmat[7, 6] = 1

    # I_dd = q^T Gq q where Gq[i,j] = M[i+j]
    Gq = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            Gq[i, j] = M[i + j]
    # I_dd = a^T (Qmat^T Gq Qmat) a
    Gdd = Qmat.T @ Gq @ Qmat

    # Precompute coefficient map from (c0, c4, c8) to a
    # c12 = -(c0 + 3*c4 + 105*c8) / 10395
    # a0  = c0 + 3*c4 + 105*c8 + 10395*c12 = 0  (since g(0)=P(0)=0)
    # a2  = -6*c4 - 420*c8 - 27720*c12
    # a4  = c4 + 210*c8 + 23100*c12
    # a6  = -28*c8 - 9240*c12
    # a8  = c8 + 1260*c12
    # a10 = -66*c12
    # a12 = c12
    # a0 = 0 always
    # a = Amap @ [c0, c4, c8]
    Amap = np.zeros((7, 3))
    inv10395 = 1.0 / 10395.0
    # a0 = 0
    # a2 = -6 c4 - 420 c8 + 27720*inv10395*(c0+3c4+105c8)
    #     = -6 c4 - 420 c8 + 27720*inv10395*c0 + 83160*inv10395*c4 + 2910600*inv10395*c8
    Amap[1, 0] = 27720.0 * inv10395
    Amap[1, 1] = -6.0 + 83160.0 * inv10395
    Amap[1, 2] = -420.0 + 2910600.0 * inv10395
    # a4 = c4 + 210 c8 - 23100*inv10395*(c0+3c4+105c8)
    Amap[3, 0] = -23100.0 * inv10395
    Amap[3, 1] = 1.0 - 69300.0 * inv10395
    Amap[3, 2] = 210.0 - 2425500.0 * inv10395
    # a6 = -28 c8 + 9240*inv10395*(c0+3c4+105c8)
    Amap[5, 0] = 9240.0 * inv10395
    Amap[5, 1] = 27720.0 * inv10395
    Amap[5, 2] = -28.0 + 970200.0 * inv10395
    # a8 = c8 - 1260*inv10395*(c0+3c4+105c8)
    Amap[6, 0] = -1260.0 * inv10395
    Amap[6, 1] = -3780.0 * inv10395
    Amap[6, 2] = 1.0 - 132300.0 * inv10395
    # a10 = -66*c12 = 66*inv10395*(c0+3c4+105c8)
    Amap[4, 0] = 66.0 * inv10395
    Amap[4, 1] = 198.0 * inv10395
    Amap[4, 2] = 6930.0 * inv10395
    # a12 = c12 = -inv10395*(c0+3c4+105c8)
    Amap[6, 0] += -1.0 * inv10395
    Amap[6, 1] += -3.0 * inv10395
    Amap[6, 2] += -105.0 * inv10395
    # Wait, a8 and a12 both go to index 6? No.
    # a = [a0, a2, a4, a6, a8, a10, a12] -> indices 0,1,2,3,4,5,6
    # Let me redo: index 0=a0, 1=a2, 2=a4, 3=a6, 4=a8, 5=a10, 6=a12
    Amap = np.zeros((7, 3))
    # a0 = 0
    # a2 (idx1)
    Amap[1, 0] = 27720.0 * inv10395
    Amap[1, 1] = -6.0 + 83160.0 * inv10395
    Amap[1, 2] = -420.0 + 2910600.0 * inv10395
    # a4 (idx2)
    Amap[2, 0] = -23100.0 * inv10395
    Amap[2, 1] = 1.0 - 69300.0 * inv10395
    Amap[2, 2] = 210.0 - 2425500.0 * inv10395
    # a6 (idx3)
    Amap[3, 0] = 9240.0 * inv10395
    Amap[3, 1] = 27720.0 * inv10395
    Amap[3, 2] = -28.0 + 970200.0 * inv10395
    # a8 (idx4)
    Amap[4, 0] = -1260.0 * inv10395
    Amap[4, 1] = -3780.0 * inv10395
    Amap[4, 2] = 1.0 - 132300.0 * inv10395
    # a10 (idx5)
    Amap[5, 0] = 66.0 * inv10395
    Amap[5, 1] = 198.0 * inv10395
    Amap[5, 2] = 6930.0 * inv10395
    # a12 (idx6)
    Amap[6, 0] = -1.0 * inv10395
    Amap[6, 1] = -3.0 * inv10395
    Amap[6, 2] = -105.0 * inv10395

    # Verify: for c0=c4=c8=0, all a should be 0. Check a2: 27720/10395 - 6 + 83160/10395 = (27720+83160)/10395 - 6 = 110880/10395 - 6 = 10.666... - 6 != 0
    # Hmm, that's wrong. Let me recompute.
    # a2 = -6 c4 - 420 c8 - 27720 c12, where c12 = -(c0+3c4+105c8)/10395
    # So a2 = -6 c4 - 420 c8 + 27720(c0+3c4+105c8)/10395
    # For c0=c4=c8=0: a2 = 0. Good.
    # Coeff of c0: 27720/10395
    # Coeff of c4: -6 + 83160/10395 = -6 + 8.0 = 2.0? 83160/10395 = 8.0. Yes! -6+8 = 2.
    # Coeff of c8: -420 + 2910600/10395 = -420 + 280.0 = -140.0? 2910600/10395 = 280.0. Yes! -420+280 = -140.
    # So a2 = (27720/10395) c0 + 2 c4 - 140 c8. Let me verify: 27720/10395 = 2.666...
    # Actually 27720/10395 = 27720/10395. 10395*2 = 20790, 27720-20790=6930, 6930/10395=0.6667. So 2.6667.
    # Let me just use the formula directly in code to avoid arithmetic errors.

    # Actually, let me just compute a directly from c0,c4,c8 each time - it's only 7 multiplications.
    # The precomputation of G0, G2, Gdd is the key speedup.

    def coeffs_to_a(c0, c4, c8):
        c12 = -(c0 + 3.0 * c4 + 105.0 * c8) * inv10395
        a = np.zeros(7)
        # a0 = c0 + 3c4 + 105c8 + 10395*c12 = 0
        a[0] = 0.0
        a[1] = -6.0 * c4 - 420.0 * c8 - 27720.0 * c12
        a[2] = c4 + 210.0 * c8 + 23100.0 * c12
        a[3] = -28.0 * c8 - 9240.0 * c12
        a[4] = c8 + 1260.0 * c12
        a[5] = -66.0 * c12
        a[6] = c12
        return a

    def compute_product(c0, c4, c8):
        a = coeffs_to_a(c0, c4, c8)
        I0 = a @ G0 @ a
        I2 = a @ G2 @ a
        I_dd = a @ Gdd @ a
        if I0 <= 0 or not np.isfinite(I0) or not np.isfinite(I2) or not np.isfinite(I_dd):
            return np.inf
        if I2 < 0 or I_dd < 0:
            return np.inf
        return (1.0 / (2.0 * np.pi)) * np.sqrt(I2 * I_dd / (I0 * I0))

    def objective(coeffs):
        c0, c4, c8 = coeffs
        if time.time() > deadline - 3:
            return np.inf
        try:
            val = compute_product(c0, c4, c8)
            if not np.isfinite(val):
                return np.inf
            return val
        except Exception:
            return np.inf

    # Initial guess (from seed)
    x0 = np.array([-1.13, 0.04, 1.0 / 3240.0])
    best_score = objective(x0)
    best_coeffs = x0.copy()

    rng = np.random.RandomState(42)

    # Phase 1: Broad differential evolution
    if time.time() < deadline - 80:
        try:
            bounds = [(-5, 5), (-5, 5), (-5, 5)]
            de_result = opt.differential_evolution(
                objective, bounds,
                maxiter=500, popsize=25,
                tol=1e-10, mutation=(0.5, 1.5),
                recombination=0.9, seed=42,
                polish=True, init='latinhypercube'
            )
            if np.isfinite(de_result.fun) and de_result.fun < best_score:
                best_score = de_result.fun
                best_coeffs = de_result.x.copy()
        except Exception:
            pass

    # Phase 2: Multi-start local refinement
    if time.time() < deadline - 20:
        start_points = [best_coeffs.copy()]
        for i in range(50):
            if time.time() > deadline - 20:
                break
            scale = 0.3 * (1 + i / 10.0)
            x_start = best_coeffs + rng.normal(0, scale, 3)
            start_points.append(x_start)

        for x_start in start_points:
            if time.time() > deadline - 12:
                break
            try:
                res = opt.minimize(objective, x_start, method='Nelder-Mead',
                                   options={'xatol': 1e-12, 'fatol': 1e-12,
                                            'maxiter': 1000, 'adaptive': True})
                if np.isfinite(res.fun) and res.fun < best_score:
                    best_score = res.fun
                    best_coeffs = res.x.copy()
            except Exception:
                pass

    # Phase 3: High-precision polish
    if time.time() < deadline - 5:
        try:
            res = opt.minimize(objective, best_coeffs, method='Nelder-Mead',
                               options={'xatol': 1e-14, 'fatol': 1e-14,
                                        'maxiter': 2000, 'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass

    # Phase 4: Even tighter polish
    if time.time() < deadline - 2:
        try:
            res = opt.minimize(objective, best_coeffs, method='Nelder-Mead',
                               options={'xatol': 1e-16, 'fatol': 1e-16,
                                        'maxiter': 5000, 'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass

    return np.array(best_coeffs)
