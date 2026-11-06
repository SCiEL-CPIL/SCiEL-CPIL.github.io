"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    N = 24
    deadline = time.time() + 130.0  # safety margin before 150s kill
    sqrt_N = np.sqrt(N)
    
    # Precompute: for each coefficient index k, the vector of z^k over M points
    M = 512
    theta = 2.0 * np.pi * np.arange(M) / M
    # V[m, k] = exp(1j * theta[m] * k)
    V = np.exp(1j * np.outer(theta, np.arange(N)))  # shape (M, N)
    
    def evaluate(coeffs):
        """Return max |P(z)| over the M sample points, and refined peaks."""
        c = np.array(coeffs, dtype=np.complex128)
        vals = V @ c  # shape (M,)
        mags = np.abs(vals)
        # Find top peaks and refine with golden section
        top_k = min(8, M)
        idx = np.argsort(mags)[-top_k:]
        best = mags[idx[0]]
        for i in idx:
            m0 = int(i)
            m1 = (m0 + 1) % M
            # Bilinear interpolation for refinement
            v0 = vals[m0]
            v1 = vals[m1]
            # Golden section on [0, 1] parameter
            a, b = 0.0, 1.0
            for _ in range(40):
                t1 = a + (1 - 0.618033988749895) * (b - a)
                t2 = a + 0.618033988749895 * (b - a)
                # Linear interpolation between v0 and v1
                f1 = np.abs((1 - t1) * v0 + t1 * v1)
                f2 = np.abs((1 - t2) * v0 + t2 * v1)
                if f1 < f2:
                    b = t2
                else:
                    a = t1
            tm = (a + b) / 2.0
            fm = np.abs((1 - tm) * v0 + tm * v1)
            if fm > best:
                best = fm
        return best / sqrt_N
    
    def try_flip(coeffs, i):
        """Flip coefficient i and return new score."""
        c = np.array(coeffs, dtype=np.complex128)
        c[i] = -c[i]
        vals = V @ c
        mags = np.abs(vals)
        top_k = min(4, M)
        idx = np.argsort(mags)[-top_k:]
        best = mags[idx[0]]
        for ii in idx:
            m0 = int(ii)
            m1 = (m0 + 1) % M
            v0 = vals[m0]
            v1 = vals[m1]
            a, b = 0.0, 1.0
            for _ in range(30):
                t1 = a + (1 - 0.618033988749895) * (b - a)
                t2 = a + 0.618033988749895 * (b - a)
                f1 = np.abs((1 - t1) * v0 + t1 * v1)
                f2 = np.abs((1 - t2) * v0 + t2 * v1)
                if f1 < f2:
                    b = t2
                else:
                    a = t1
            tm = (a + b) / 2.0
            fm = np.abs((1 - tm) * v0 + tm * v1)
            if fm > best:
                best = fm
        return best / sqrt_N
    
    # Start from Golay-Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    best_coeffs = p[:N]
    best_score = evaluate(best_coeffs)
    
    # Single-flip local search
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        for i in range(N):
            new_score = try_flip(best_coeffs, i)
            if new_score < best_score:
                best_coeffs[i] = -best_coeffs[i]
                best_score = new_score
                improved = True
    
    # Double-flip moves to escape local optima
    if time.time() < deadline - 10.0:
        for _ in range(200):
            if time.time() >= deadline - 10.0:
                break
            i = np.random.randint(N)
            j = np.random.randint(N)
            if i == j:
                continue
            best_coeffs[i] = -best_coeffs[i]
            best_coeffs[j] = -best_coeffs[j]
            new_score = evaluate(best_coeffs)
            if new_score < best_score:
                best_score = new_score
            else:
                best_coeffs[i] = -best_coeffs[i]
                best_coeffs[j] = -best_coeffs[j]
    
    # Final polish: single flips again
    if time.time() < deadline - 5.0:
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(N):
                new_score = try_flip(best_coeffs, i)
                if new_score < best_score:
                    best_coeffs[i] = -best_coeffs[i]
                    best_score = new_score
                    improved = True
    
    return list(best_coeffs)
# EVOLVE-BLOCK-END
