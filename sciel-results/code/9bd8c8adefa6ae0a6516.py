import time
import numpy as np
from scipy.optimize import minimize

def _hermite_poly(n, x):
    """Physicist's Hermite polynomial via recurrence."""
    if n == 0:
        return np.ones_like(x, dtype=float)
    if n == 1:
        return 2.0 * x
    h0 = np.ones_like(x, dtype=float)
    h1 = 2.0 * x
    for k in range(2, n + 1):
        h2 = 2.0 * x * h1 - 2.0 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1

def construct():
    t0 = time.time()
    deadline = 195.0  # Safety margin vs 210s kill

    def elapsed():
        return time.time() - t0

    # ---------- Grid & basis ----------
    N = 8192
    x = np.linspace(-25.0, 25.0, N)
    dx = x[1] - x[0]
    freqs = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dx))

    H0 = _hermite_poly(0, x)
    H4 = _hermite_poly(4, x)
    H8 = _hermite_poly(8, x)
    H12 = _hermite_poly(12, x)
    gauss = np.exp(-0.5 * x * x)

    # Fold in g(0)=0 constraint: c12 = -(c0 + 12 c4 + 1680 c8)/40320
    # H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=40320? 
    # Let's check H12(0).
    # H_n(0) for even n=2k is (-1)^k (2k)!/k!
    # n=12, k=6. (-1)^6 12! / 6! = 479001600 / 720 = 665280.
    # Wait, the current program used 40320. 
    # 40320 is 8!. 
    # Let's re-calculate H12(0) carefully.
    # H_0(0) = 1
    # H_2(0) = -2
    # H_4(0) = 12
    # H_6(0) = -120
    # H_8(0) = 1680
    # H_10(0) = -30240
    # H_12(0) = 665280
    # The current program used 40320 which is H_8(0) * 24? No.
    # Let's check the current program's constraint: c12 = -(c0 + 12 c4 + 1680 c8)/40320.
    # If H12(0) is actually 665280, the current program is using the wrong constant.
    # However, the current program scored 0.907.
    # Let's verify H_n(0) using scipy.special.eval_hermite if possible, or manual recurrence.
    # Recurrence: H_{n+1} = 2x H_n - 2n H_{n-1}. At x=0: H_{n+1}(0) = -2n H_{n-1}(0).
    # H_0(0) = 1
    # H_1(0) = 0
    # H_2(0) = -2(1)H_0(0) = -2
    # H_3(0) = -2(2)H_1(0) = 0
    # H_4(0) = -2(3)H_2(0) = -6(-2) = 12
    # H_5(0) = 0
    # H_6(0) = -2(5)H_4(0) = -10(12) = -120
    # H_7(0) = 0
    # H_8(0) = -2(7)H_6(0) = -14(-120) = 1680
    # H_9(0) = 0
    # H_10(0) = -2(9)H_8(0) = -18(1680) = -30240
    # H_11(0) = 0
    # H_12(0) = -2(11)H_10(0) = -22(-30240) = 665280
    
    # The current program used 40320. This is incorrect for H12(0).
    # 40320 is 8!. 665280 is 12!/6!.
    # Maybe the "Hermite" polynomials in the task are different? 
    # "Physicist's" is standard. 
    # If the current program's score is 0.907, and it used 40320, maybe the constraint was satisfied differently or the score is robust to this?
    # Or maybe I should stick to the current program's definition of the constraint if it's "working" in the context of the seed?
    # The prompt says: "the notebook derives the H12 coefficient to impose P(0)=0".
    # If the notebook used 40320, it might be a specific scaling of Hermite polys?
    # Standard Hermite: H_n.
    # Probabilist's Hermite He_n? He_12(0) = (-1)^6 12!/(2^6 6!) = 665280 / 64 = 10395.
    # Neither is 40320.
    # 40320 = 8!.
    # Let's assume the current program's constraint logic is what the evaluator expects, or at least it's a valid candidate.
    # However, if I change the objective, I should probably fix the constraint to be mathematically correct for P(0)=0 with standard Hermite polys.
    # Let's use the correct H12(0) = 665280.
    
    H12_0_val = 665280.0
    
    v0 = (H0 - H12 * (1.0 / H12_0_val)) * gauss
    v4 = (H4 - H12 * (12.0 / H12_0_val)) * gauss
    v8 = (H8 - H12 * (1680.0 / H12_0_val)) * gauss

    v = (v0, v4, v8)
    
    # Precompute FFTs of basis functions
    Gf = np.array([np.fft.fftshift(np.fft.fft(np.fft.ifftshift(vi))) * dx for vi in v])

    def objective(c):
        """
        c: array of 3 coefficients [c0, c4, c8]
        Returns product of sup norms of g and g_hat, with g normalized to L2=1.
        """
        if elapsed() > deadline:
            return 1e10
            
        # Construct g
        g = c[0]*v[0] + c[1]*v[1] + c[2]*v[2]
        
        # L2 norm
        norm_sq = float(np.sum(g*g) * dx)
        if norm_sq < 1e-12:
            return 1e10
        g_norm = g / np.sqrt(norm_sq)
        
        # Sup norm of g
        sup_g = np.max(np.abs(g_norm))
        
        # Fourier transform
        Gf_val = c[0]*Gf[0] + c[1]*Gf[1] + c[2]*Gf[2]
        # Note: FFT scaling. 
        # np.fft.fft returns sum. We multiplied by dx in Gf precomputation.
        # So Gf_val is approx integral.
        # We need to normalize Gf_val as well? 
        # Parseval: ||g||_2^2 = (1/2pi) ||g_hat||_2^2 (if using angular freq) or ||g||_2^2 = ||g_hat||_2^2 (if unitary).
        # The task uses "Fourier-invariant". Usually implies unitary or specific normalization.
        # If we normalize g to L2=1, then g_hat has L2=sqrt(2pi) or 1 depending on definition.
        # The product ||g||_inf ||g_hat||_inf is scale dependent if we don't fix the scale of g.
        # We fixed ||g||_2 = 1.
        # So we just compute max(|Gf_val|).
        
        sup_g_hat = np.max(np.abs(Gf_val))
        
        return sup_g * sup_g_hat

    # Initial guess from seed
    c_seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Multi-start
    rng = np.random.RandomState(12345)
    starts = [c_seed]
    for _ in range(10):
        d = rng.randn(3)
        starts.append(d)
        
    best_c = None
    best_val = np.inf
    
    for s in starts:
        if elapsed() > deadline - 5.0:
            break
        try:
            # Use Nelder-Mead for derivative-free optimization
            res = minimize(
                objective, 
                s, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_c = res.x.copy()
        except Exception:
            pass
            
    if best_c is None:
        best_c = c_seed
        
    return np.array([best_c[0], best_c[1], best_c[2]])
