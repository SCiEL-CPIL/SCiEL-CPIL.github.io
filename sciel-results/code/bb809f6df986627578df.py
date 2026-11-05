import numpy as np
import time
from scipy.optimize import minimize

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def compute_value(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        c = np.convolve(h, h)
        return abs(2.0 * N * np.max(c) / (s * s))

    def spectral_objective_and_grad(coeffs):
        """
        Objective and gradient in spectral domain.
        coeffs: array of length N. coeffs[0] is the DC component (baseline).
                coeffs[1:] are the amplitudes of the cosine basis functions.
        
        We construct h = A * dct_basis.
        To ensure zero mean perturbation, we fix coeffs[0] = 1 (baseline) 
        and let the optimizer find coeffs[1:].
        Actually, to allow scaling, we can let coeffs[0] vary, but the objective
        is homogeneous of degree 0 in h (J(ah) = J(h)). 
        So we can fix the norm of the perturbation or fix coeffs[0]=1.
        Let's fix coeffs[0] = 1.0 and optimize coeffs[1:].
        This defines a family of h with mean = coeffs[0] * (1/N) * sum(dct_basis[0])?
        
        Let's use scipy.fft.dctn and idctn.
        The basis functions are orthogonal.
        Let B be the DCT matrix. h = B @ coeffs.
        If we use orthogonal DCT (norm='ortho'), B is orthogonal.
        
        We want to minimize J(h).
        Let h = h_base + h_pert.
        h_base = mean(h) * ones(N).
        h_pert = h - h_base.
        
        In spectral domain (DCT-II):
        coeffs[0] corresponds to the mean (DC).
        coeffs[k>0] correspond to AC components.
        
        If we set coeffs[0] = 1 and optimize coeffs[1:], we are searching for
        the best shape of the perturbation given a fixed baseline amplitude.
        Since J is scale invariant, the absolute scale doesn't matter, only the
        ratio of coefficients.
        
        So, let x = coeffs[1:].
        Full coeffs = [1, x[0], x[1], ...]
        h = idct(coeffs, norm='ortho')
        
        Gradient:
        J(h) = 2 N max(c) / S^2
        Using soft-max for max(c) as in the previous approach, but now we need
        dJ/dcoeffs.
        
        dJ/dcoeffs_k = sum_m (dJ/dh_m) * (dh_m/dcoeffs_k)
        
        dh_m/dcoeffs_k = B[m, k] (element of DCT matrix)
        
        So grad_coeffs = B.T @ grad_h
        
        This is efficient: O(N^2) for the matrix-vector product, but N=400 is small.
        We can use scipy.fft.idctn for forward and dctn for backward.
        Actually, scipy.fft.dctn(y, type=2, norm='ortho') is the inverse of idctn?
        No, dctn is the forward transform (time to freq). idctn is freq to time.
        If h = idctn(coeffs), then coeffs = dctn(h).
        The Jacobian is the DCT matrix.
        grad_coeffs = dctn(grad_h, norm='ortho')
        
        Let's verify.
        h_m = sum_k B[m,k] c_k
        dh_m/dc_k = B[m,k]
        dJ/dc_k = sum_m dJ/dh_m B[m,k]
        This is exactly the DCT of the gradient vector.
        """
        # Reconstruct h
        full_coeffs = np.zeros(N)
        full_coeffs[0] = 1.0
        full_coeffs[1:] = coeffs
        
        h = scipy.fft.idctn(full_coeffs, type=2, norm='ortho')
        
        s = np.sum(h)
        if abs(s) < 1e-12:
            return 1e10, np.zeros_like(coeffs)
            
        c = np.convolve(h, h)
        
        # Soft-max temperature
        t = 10.0
        c_max = np.max(c)
        u = t * (c - c_max)
        exp_u = np.exp(u)
        Z = np.sum(exp_u)
        p = exp_u / Z
        
        C_soft = c_max + np.log(Z) / t
        
        # Gradient of C_soft w.r.t h
        grad_C = np.zeros(N)
        for L in range(-(N-1), N):
            j = L + N - 1
            w = p[j]
            if w < 1e-10:
                continue
            
            if L >= 0:
                if N > L:
                    grad_C[:N-L] += w * h[L:]
                if L < N:
                    grad_C[L:] += w * h[:N-L]
            else:
                Lp = -L
                if N > Lp:
                    grad_C[:N-Lp] += w * h[Lp:]
                if Lp < N:
                    grad_C[Lp:] += w * h[:N-Lp]
        
        # Objective J = 2*N*C_soft / S^2
        # dJ/dh_m = 2*N * ( grad_C[m] / S - 2*C_soft / S^2 )
        grad_h = (2.0 * N / s) * grad_C - (4.0 * N * C_soft / (s * s))
        
        # Transform gradient to spectral domain
        # grad_coeffs = dctn(grad_h)
        grad_full_coeffs = scipy.fft.dctn(grad_h, type=2, norm='ortho')
        
        # We only optimize coeffs[1:], so discard grad_full_coeffs[0]
        grad_coeffs = grad_full_coeffs[1:]
        
        val = 2.0 * N * C_soft / (s * s)
        
        return val, grad_coeffs

    import scipy.fft
    
    # Initial best: Uniform (all coefficients 0 except DC)
    # h = ones(N) -> coeffs[0]=1, others 0.
    # Value for ones: c is triangle. max(c)=N. S=N. J = 2N*N/N^2 = 2.
    
    h_best = np.ones(N)
    val_best = compute_value(h_best)
    
    # Try different initial spectral shapes
    # 1. All zeros (uniform)
    # 2. Low frequency dominant
    # 3. Random
    
    starts = [
        np.zeros(N-1),
        np.zeros(N-1) + 0.1 * np.sin(np.linspace(0, np.pi, N-1)),
        np.random.RandomState(42).randn(N-1) * 0.1,
        np.random.RandomState(43).randn(N-1) * 0.1,
    ]
    
    for x0 in starts:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(
                spectral_objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'maxls': 50, 'disp': False}
            )
            x_res = res.x
            full_coeffs = np.zeros(N)
            full_coeffs[0] = 1.0
            full_coeffs[1:] = x_res
            h_res = scipy.fft.idctn(full_coeffs, type=2, norm='ortho')
            
            val_res = compute_value(h_res)
            if val_res < val_best:
                val_best = val_res
                h_best = h_res.copy()
        except Exception:
            pass
            
    return h_best
