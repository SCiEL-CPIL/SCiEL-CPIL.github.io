import time
import numpy as np
from scipy.optimize import minimize

DEADLINE_MARGIN = 10.0
MAX_RUN_TIME = 200.0

def construct():
    """
    Returns a numpy array of shape (3,) with coefficients [c0, c4, c8].
    The coefficient of H12 is implicitly determined by the constraint g(0) = 0.
    """
    t0 = time.time()
    
    # Basis indices: 0, 4, 8, 12
    basis_idx = [0, 4, 8, 12]
    size = 13  # n=0..12
    
    # Initialize matrices
    X2 = np.zeros((size, size))
    P2 = np.zeros((size, size))
    
    # Build X^2 and P^2 matrices in the Hermite basis
    # Using the standard harmonic oscillator basis |n>
    # X = (a + a^dagger) / sqrt(2)
    # P = i(a^dagger - a) / sqrt(2)
    # X^2 = (a^2 + a^{dagger2} + a a^dagger + a^dagger a) / 2
    # P^2 = -(a^2 + a^{dagger2} + a a^dagger + a^dagger a) / 2  <-- Wait, let's verify signs.
    
    # Actually, let's derive carefully.
    # [X, P] = i (assuming hbar=1, m=1, omega=1)
    # X = (a + a†)/√2
    # P = i(a† - a)/√2
    # X^2 = (a^2 + a a† + a† a + a†2) / 2
    # P^2 = (a†2 - a† a - a a† + a^2) / 2  ... wait.
    # P^2 = i/√2 (a† - a) * i/√2 (a† - a) = -1/2 (a† - a)^2 = -1/2 (a†2 - a† a - a a† + a^2)
    # = 1/2 (a† a + a a† - a†2 - a^2)
    
    # Let's check diagonal terms.
    # <n| X^2 |n> = <n| (a a† + a† a + a^2 + a†2)/2 |n>
    # a a† |n> = (n+1)|n>, a† a |n> = n|n>.
    # <n| a^2 |n> = 0, <n| a†2 |n> = 0.
    # So X^2_{nn} = (n+1+n)/2 = n + 0.5. Correct.
    
    # <n| P^2 |n> = <n| (a† a + a a† - a†2 - a^2)/2 |n>
    # = (n + n + 1)/2 = n + 0.5. Correct.
    
    # Off-diagonal terms for X^2:
    # X^2_{n, n-2} comes from a†2 term: <n| a†2 |n-2> = √((n-1)n) = √(n(n-1)).
    # Coefficient is 1/2. So X^2_{n, n-2} = 0.5 * √(n(n-1)).
    # X^2_{n, n+2} comes from a^2 term: <n| a^2 |n+2> = √((n+2)(n+1)).
    # Coefficient is 1/2. So X^2_{n, n+2} = 0.5 * √((n+1)(n+2)).
    
    # Off-diagonal terms for P^2:
    # P^2_{n, n-2} comes from -a†2 term: -1/2 * <n| a†2 |n-2> = -0.5 * √(n(n-1)).
    # P^2_{n, n+2} comes from -a^2 term: -1/2 * <n| a^2 |n+2> = -0.5 * √((n+1)(n+2)).
    
    # The current program uses these signs. Let's stick with them.
    
    for n in range(size):
        X2[n, n] = n + 0.5
        P2[n, n] = n + 0.5
        
        if n >= 2:
            coeff = 0.5 * np.sqrt(n * (n - 1))
            X2[n, n-2] = coeff
            X2[n-2, n] = coeff
            
            P2[n, n-2] = -coeff
            P2[n-2, n] = -coeff
            
        if n < size - 2:
            coeff = 0.5 * np.sqrt((n + 1) * (n + 2))
            X2[n, n+2] = coeff
            X2[n+2, n] = coeff
            
            P2[n, n+2] = -coeff
            P2[n+2, n] = -coeff

    # Extract submatrices for basis {0, 4, 8, 12}
    X2_sub = np.zeros((4, 4))
    P2_sub = np.zeros((4, 4))
    for i, ni in enumerate(basis_idx):
        for j, nj in enumerate(basis_idx):
            X2_sub[i, j] = X2[ni, nj]
            P2_sub[i, j] = P2[ni, nj]
    
    # Compute psi_n(0) for n=0,4,8,12
    # psi_0(0) = pi^(-1/4)
    # psi_n(0) = -sqrt((n-1)/n) * psi_{n-2}(0) for even n
    psi_vals = np.zeros(size)
    psi_vals[0] = np.pi ** (-0.25)
    for n in range(2, size, 2):
        psi_vals[n] = -np.sqrt((n-1)/n) * psi_vals[n-2]
        
    h0_vec = np.array([psi_vals[0], psi_vals[4], psi_vals[8], psi_vals[12]])
    
    def get_c12(c0, c4, c8):
        # c0*psi0(0) + c4*psi4(0) + c8*psi8(0) + c12*psi12(0) = 0
        return -(c0 * h0_vec[0] + c4 * h0_vec[1] + c8 * h0_vec[2]) / h0_vec[3]
    
    def objective(c_free):
        c0, c4, c8 = c_free
        c12 = get_c12(c0, c4, c8)
        c = np.array([c0, c4, c8, c12])
        
        norm_sq = np.dot(c, c)
        if norm_sq < 1e-12:
            return np.inf
            
        x2_val = np.dot(c, np.dot(X2_sub, c))
        p2_val = np.dot(c, np.dot(P2_sub, c))
        
        # Ensure positive for sqrt
        if x2_val < 1e-12 or p2_val < 1e-12:
            return np.inf
            
        return np.sqrt(x2_val * p2_val) / norm_sq

    # Seed value
    c_seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_score = objective(c_seed)
    best_c = c_seed.copy()
    
    # Strategy: Nelder-Mead with multiple diverse starting points
    # The landscape is smooth and low-dimensional (3 variables).
    # DE is overkill and slow. NM is fast and effective for smooth functions.
    
    # Generate diverse starting points
    # 1. The seed
    # 2. Random points in a reasonable range
    # 3. Points along axes
    
    start_points = []
    start_points.append(c_seed)
    
    # Add random points
    rng = np.random.default_rng(42)
    for _ in range(10):
        p = rng.uniform(-2.0, 2.0, size=3)
        # Avoid zero
        if np.linalg.norm(p) > 1e-3:
            start_points.append(p)
    
    # Add axis-aligned points
    for i in range(3):
        for sign in [-1, 1]:
            p = np.zeros(3)
            p[i] = sign * 0.5
            start_points.append(p)
            p[i] = sign * 1.0
            start_points.append(p)
    
    # Run Nelder-Mead from each start point
    for sp in start_points:
        if time.time() - t0 > MAX_RUN_TIME - DEADLINE_MARGIN:
            break
        
        try:
            res = minimize(objective, sp, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_score:
                best_score = res.fun
                best_c = res.x.copy()
        except Exception:
            pass
    
    # If no improvement or error, return seed
    if best_score == objective(c_seed):
        return c_seed
    else:
        return np.array(best_c)

if __name__ == "__main__":
    print(construct())
