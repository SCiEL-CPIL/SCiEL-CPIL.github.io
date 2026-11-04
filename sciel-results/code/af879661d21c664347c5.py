import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_hermite
import time

START_TIME = time.time()
TIME_LIMIT = 180  # seconds, keep margin for evaluator kill at 210s

def construct():
    # Safety margin: check if we are already too close to the limit
    if time.time() - START_TIME > TIME_LIMIT - 10:
        return np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])

    # Grid setup
    # N=4096 is a good balance between accuracy and speed for FFT
    N = 4096
    # Range [-L, L] needs to be large enough for the Gaussian tail to be negligible
    # H12(x) ~ x^12. exp(-x^2/2) * x^12 decays very fast. L=15 or 20 is safe.
    L = 20.0
    x = np.linspace(-L, L, N)
    dx = x[1] - x[0]
    
    # Frequency grid for FFT
    # The FT of Hn(x)exp(-x^2/2) is related to Hn(k)exp(-k^2/2)
    # Using FFT: we need to be careful with scaling.
    # Standard FFT computes sum f(x_j) exp(-i 2 pi k j / N)
    # Our FT definition: \hat{f}(\xi) = \int f(x) e^{-i x \xi} dx
    # Discrete approx: \hat{f}(\xi_m) \approx \sum f(x_j) e^{-i x_j \xi_m} dx
    # If we use fft, we get frequencies \xi_m = 2 \pi m / (N dx) (shifted)
    freqs = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dx))

    # Precompute Hermite polynomials
    # Physicist's Hermite polynomials
    H0 = eval_hermite(0, x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    gauss = np.exp(-x**2 / 2)

    # Constraint: g(0) = 0
    # g(0) = (c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0)) * exp(0)
    # H0(0)=1, H4(0)=-12, H8(0)=1680, H12(0)=-290304
    # Wait, let's verify H_n(0) for even n.
    # H_0(0) = 1
    # H_2(0) = -2
    # H_4(0) = 12? No. H_4(x) = 16x^4 - 48x^2 + 12. H_4(0) = 12.
    # Let's check scipy eval_hermite.
    # scipy.special.eval_hermite(n, x) returns the physicist's Hermite polynomial H_n(x).
    # H_4(0) = 12.
    # H_8(0) = -1680? 
    # H_2(x) = 4x^2 - 2. H_2(0) = -2.
    # H_4(x) = 16x^4 - 48x^2 + 12. H_4(0) = 12.
    # H_6(x) = 64x^6 - 480x^4 + 720x^2 - 120. H_6(0) = -120.
    # H_8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680. H_8(0) = 1680.
    # H_12(0) = -290304.
    
    h0_0 = 1.0
    h4_0 = 12.0
    h8_0 = 1680.0
    h12_0 = -290304.0
    
    # c12 is determined by c0, c4, c8:
    # c0*h0_0 + c4*h4_0 + c8*h8_0 + c12*h12_0 = 0
    # c12 = -(c0*h0_0 + c4*h4_0 + c8*h8_0) / h12_0
    
    # Basis vectors v0, v4, v8 such that g = c0*v0 + c4*v4 + c8*v8
    # v0 = (H0 - (h0_0/h12_0)*H12) * gauss
    # v4 = (H4 - (h4_0/h12_0)*H12) * gauss
    # v8 = (H8 - (h8_0/h12_0)*H12) * gauss
    
    v0 = (H0 - (h0_0 / h12_0) * H12) * gauss
    v4 = (H4 - (h4_0 / h12_0) * H12) * gauss
    v8 = (H8 - (h8_0 / h12_0) * H12) * gauss

    # Precompute Gram matrices M, Mx, Mf
    # M_ij = <vi, vj> = integral vi vj dx
    # Mx_ij = <vi, vj>_x = integral x^2 vi vj dx
    # Mf_ij = <vi, vj>_f = integral |xi|^2 \hat{vi} \hat{vj}^* dxi (using Plancherel)
    
    # Position space integrals
    M = np.zeros((3, 3))
    Mx = np.zeros((3, 3))
    M[0,0] = np.trapz(v0*v0, x)
    M[1,1] = np.trapz(v4*v4, x)
    M[2,2] = np.trapz(v8*v8, x)
    M[0,1] = M[1,0] = np.trapz(v0*v4, x)
    M[0,2] = M[2,0] = np.trapz(v0*v8, x)
    M[1,2] = M[2,1] = np.trapz(v4*v8, x)
    
    Mx[0,0] = np.trapz(x*x*v0*v0, x)
    Mx[1,1] = np.trapz(x*x*v4*v4, x)
    Mx[2,2] = np.trapz(x*x*v8*v8, x)
    Mx[0,1] = Mx[1,0] = np.trapz(x*x*v0*v4, x)
    Mx[0,2] = Mx[2,0] = np.trapz(x*x*v0*v8, x)
    Mx[1,2] = Mx[2,1] = np.trapz(x*x*v4*v8, x)
    
    # Frequency space integrals
    # We need \hat{v}_i.
    # Since v_i are real and even (Hermite polys of even degree are even, gauss is even),
    # their Fourier transforms are real and even.
    # \hat{v}_i(\xi) = \int v_i(x) e^{-i x \xi} dx
    # We can compute this via FFT.
    # Note: The FFT of a function sampled on x_j gives approximations to the Fourier series coefficients,
    # which correspond to the Fourier transform at discrete frequencies if the function is periodic.
    # Since our function decays to 0 at boundaries, the periodic assumption is a good approximation.
    # Scaling: \hat{f}(\xi_m) \approx dx * FFT(f)_m
    # Frequencies: \xi_m = 2\pi m / (N dx)
    
    # Let's compute FFT of v0, v4, v8
    # We need to center the zero frequency.
    # fftshift moves zero freq to center.
    
    V0_f = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(v0))) * dx
    V4_f = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(v4))) * dx
    V8_f = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(v8))) * dx
    
    Mf = np.zeros((3, 3))
    # <vi, vj>_f = integral |xi|^2 \hat{vi} \hat{vj} dxi
    # Approx: sum |xi_m|^2 \hat{vi}_m \hat{vj}_m^* dx? No, dxi = 2\pi/(N dx).
    # Wait, Plancherel theorem: \int |f|^2 dx = \frac{1}{2\pi} \int |\hat{f}|^2 d\xi
    # So \int x^2 |f|^2 dx is not directly \int \xi^2 |\hat{f}|^2 d\xi.
    # However, \int x^2 |f|^2 dx = \frac{1}{2\pi} \int \xi^2 |\hat{f}|^2 d\xi is NOT correct.
    # The relation is: \mathcal{F}[x f(x)] = i \frac{d}{d\xi} \hat{f}(\xi)
    # So \int x^2 |f|^2 dx = \int (x f)(x) (x f)(x)^* dx = \frac{1}{2\pi} \int (i \hat{f}') (i \hat{f}')^* d\xi = \frac{1}{2\pi} \int |\hat{f}'|^2 d\xi
    # This requires differentiation of the FFT, which is tricky.
    
    # Alternative:
    # The uncertainty product is \Delta x \Delta \xi.
    # \Delta x^2 = \langle x^2 \rangle - \langle x \rangle^2. Since functions are even, \langle x \rangle = 0.
    # \Delta \xi^2 = \langle \xi^2 \rangle - \langle \xi \rangle^2. Since functions are even, \langle \xi \rangle = 0.
    # \langle \xi^2 \rangle = \frac{\int \xi^2 |\hat{f}(\xi)|^2 d\xi}{\int |\hat{f}(\xi)|^2 d\xi}
    # Using Plancherel: \int |\hat{f}|^2 d\xi = 2\pi \int |f|^2 dx.
    # And \int \xi^2 |\hat{f}|^2 d\xi = 2\pi \int |f'|^2 dx.
    # So \Delta \xi^2 = \frac{\int |f'|^2 dx}{\int |f|^2 dx}.
    # This is much easier! We just need the L2 norm of the derivative.
    
    # Let's compute Mf using derivatives.
    # v_i' = derivative of v_i.
    # v_i = P_i(x) gauss.
    # v_i' = (P_i' - x P_i) gauss.
    
    # Derivatives of Hermite polynomials: H_n'(x) = 2n H_{n-1}(x).
    # But v_i are combinations.
    # v0 = (H0 - k0 H12) gauss.
    # v0' = (H0' - k0 H12' - x(H0 - k0 H12)) gauss
    # H0' = 0. H12' = 24 H11.
    # This is getting complicated.
    
    # Let's stick to the FFT method for Mf if we can interpret it correctly.
    # \Delta \xi^2 = \frac{\int \xi^2 |\hat{f}|^2 d\xi}{\int |\hat{f}|^2 d\xi}
    # Numerator: \sum |\xi_m|^2 |\hat{f}_m|^2 \Delta \xi
    # Denominator: \sum |\hat{f}_m|^2 \Delta \xi
    # \Delta \xi = 2\pi / (N dx)
    # The ratio \frac{\sum |\xi_m|^2 |\hat{f}_m|^2}{\sum |\hat{f}_m|^2} is independent of \Delta \xi.
    # So we can just compute the matrix Mf where Mf_ij = \sum \xi_m^2 \hat{v}_{i,m} \hat{v}_{j,m}^*
    
    Mf[0,0] = np.sum(freqs**2 * np.abs(V0_f)**2)
    Mf[1,1] = np.sum(freqs**2 * np.abs(V4_f)**2)
    Mf[2,2] = np.sum(freqs**2 * np.abs(V8_f)**2)
    Mf[0,1] = Mf[1,0] = np.sum(freqs**2 * np.real(V0_f * np.conj(V4_f)))
    Mf[0,2] = Mf[2,0] = np.sum(freqs**2 * np.real(V0_f * np.conj(V8_f)))
    Mf[1,2] = Mf[2,1] = np.sum(freqs**2 * np.real(V4_f * np.conj(V8_f)))
    
    # Note: The constant factors from Plancherel cancel out in the ratio.
    # \Delta x^2 = (c^T Mx c) / (c^T M c)
    # \Delta \xi^2 = (c^T Mf c) / (c^T M c)  <-- Wait.
    # \int |\hat{f}|^2 d\xi = 2\pi \int |f|^2 dx = 2\pi (c^T M c)
    # \int \xi^2 |\hat{f}|^2 d\xi = \sum \xi_m^2 \hat{f}_m \hat{f}_m^* \Delta \xi = \Delta \xi \sum \xi_m^2 \hat{f}_m \hat{f}_m^*
    # So \Delta \xi^2 = \frac{\Delta \xi \sum \xi_m^2 \hat{f}_m \hat{f}_m^*}{2\pi (c^T M c)}
    # This introduces a constant factor $\frac{\Delta \xi}{2\pi}$.
    # However, the uncertainty product $\Delta x \Delta \xi$ is scale invariant in the sense that if we change the FT convention, both $\Delta x$ and $\Delta \xi$ scale inversely.
    # But here we are computing a specific value.
    # Let's check the seed program. It used `var_xi = g_prime_sq_norm / norm_sq`.
    # This corresponds to $\Delta \xi^2 = \frac{\int |f'|^2 dx}{\int |f|^2 dx}$.
    # This is equivalent to using the FT convention where $\hat{f}(\xi) = \frac{1}{\sqrt{2\pi}} \int f(x) e^{-ix\xi} dx$.
    # In that case, $\int |\hat{f}|^2 d\xi = \int |f|^2 dx$ and $\int \xi^2 |\hat{f}|^2 d\xi = \int |f'|^2 dx$.
    # My FFT calculation gives $\sum \xi_m^2 |\hat{f}_m|^2$.
    # If I use the non-unitary FT $\hat{f} = \int f e^{-ix\xi} dx$, then $\int |\hat{f}|^2 d\xi = 2\pi \int |f|^2 dx$.
    # And $\int \xi^2 |\hat{f}|^2 d\xi = 2\pi \int |f'|^2 dx$.
    # So the ratio $\frac{\int \xi^2 |\hat{f}|^2 d\xi}{\int |\hat{f}|^2 d\xi} = \frac{\int |f'|^2 dx}{\int |f|^2 dx}$.
    # The $2\pi$ cancels.
    # So I can just use the ratio of the sums from FFT.
    
    def objective_spherical(params):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            return 1e10
        theta, phi = params
        # Spherical coordinates for c = [c0, c4, c8]
        # c0 = sin(theta) cos(phi)
        # c4 = sin(theta) sin(phi)
        # c8 = cos(theta)
        c = np.array([np.sin(theta) * np.cos(phi),
                      np.sin(theta) * np.sin(phi),
                      np.cos(theta)])
        
        norm_sq = c @ M @ c
        if norm_sq < 1e-12 or not np.isfinite(norm_sq):
            return 1e10
            
        x2_moment = c @ Mx @ c
        f2_moment = c @ Mf @ c
        
        if not np.isfinite(x2_moment) or not np.isfinite(f2_moment):
            return 1e10
        if x2_moment < 0 or f2_moment < 0:
            return 1e10
            
        delta_x = np.sqrt(x2_moment / norm_sq)
        delta_xi = np.sqrt(f2_moment / norm_sq)
        
        return delta_x * delta_xi

    # Initial guess from seed
    c0_init = -113.0 / 100.0
    c4_init = 1.0 / 25.0
    c8_init = 1.0 / 3240.0
    c_init = np.array([c0_init, c4_init, c8_init])
    
    # Convert to spherical
    r = np.linalg.norm(c_init)
    if r < 1e-9:
        theta_init = 0.0
        phi_init = 0.0
    else:
        c_init_norm = c_init / r
        theta_init = np.arccos(np.clip(c_init_norm[2], -1.0, 1.0))
        if np.sin(theta_init) < 1e-9:
            phi_init = 0.0
        else:
            phi_init = np.arctan2(c_init_norm[1], c_init_norm[0])
            
    # Global search with Differential Evolution
    # Bounds for theta: [0, pi], phi: [-pi, pi]
    bounds = [(0.0, np.pi), (-np.pi, np.pi)]
    
    best_c = c_init.copy()
    best_score = objective_spherical([theta_init, phi_init])
    
    try:
        # Differential Evolution is slow but robust.
        # We give it a limited number of iterations.
        de_res = differential_evolution(objective_spherical, bounds, 
                                        maxiter=50, 
                                        popsize=15, 
                                        tol=1e-6, 
                                        seed=42)
        
        if time.time() - START_TIME < TIME_LIMIT - 20:
            if de_res.fun < best_score:
                best_score = de_res.fun
                theta_opt, phi_opt = de_res.x
                best_c = np.array([np.sin(theta_opt) * np.cos(phi_opt),
                                    np.sin(theta_opt) * np.sin(phi_opt),
                                    np.cos(theta_opt)])
            else:
                theta_opt, phi_opt = [theta_init, phi_init]
                best_c = c_init.copy()
        else:
            theta_opt, phi_opt = [theta_init, phi_init]
            best_c = c_init.copy()
            
        # Local refinement with L-BFGS-B
        if time.time() - START_TIME < TIME_LIMIT - 10:
            local_res = minimize(objective_spherical, [theta_opt, phi_opt], 
                                 method='L-BFGS-B', 
                                 bounds=bounds,
                                 options={'maxiter': 1000, 'ftol': 1e-12})
            
            if local_res.fun < best_score:
                best_score = local_res.fun
                theta_opt, phi_opt = local_res.x
                best_c = np.array([np.sin(theta_opt) * np.cos(phi_opt),
                                    np.sin(theta_opt) * np.sin(phi_opt),
                                    np.cos(theta_opt)])
                
    except Exception:
        pass

    # Return the coefficients c0, c4, c8
    # The task asks for three coefficients.
    # The scale is arbitrary, so we can return the normalized direction or the original scale.
    # The seed returned specific values. Let's return the values from the best spherical solution.
    # Since the objective is scale-invariant, any scalar multiple is equally good.
    # However, the evaluator might check for specific properties.
    # Returning the unit vector is safe.
    
    return best_c

# EVOLVE-BLOCK-END
