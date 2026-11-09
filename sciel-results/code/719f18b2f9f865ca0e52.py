<<<<<<< SEARCH
"""Combined program: numerical pre-screening + local optimization + exact sympy verification."""
import time
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 620  # Safety margin before 630s kill
    best_score = float('inf')
    best_z = None

    # --- Fast numerical evaluation (proxy for B/(2*pi)) ---
    def evaluate_numeric(z_list):
        """Fast floating-point evaluation to approximate B/(2*pi)."""
        try:
            from scipy.special import eval_genlaguerre
            from scipy.optimize import brentq

            alpha = -0.5
            z_arr = np.array(sorted(z_list), dtype=np.float64)
            if np.any(z_arr <= 0) or len(set(np.round(z_arr, 12))) != 12:
                return float('inf')

            # Build system: g(x) = sum c_k L_{2k}^{alpha}(x)
            # Conditions: g(0)=0, g'(0)=1, g(z_i)=g'(z_i)=0 for all i
            # 26 equations, 26 unknowns

            A_rows = []
            b_vec = []

            # g(0) = 0
            row = np.array([eval_genlaguerre(2*k, alpha, 0.0) for k in range(26)])
            A_rows.append(row)
            b_vec.append(0.0)

            # g'(0) = 1: L_{2k}'(x) = -L_{2k-1}^{alpha+1}(x)
            row = np.zeros(26)
            row[0] = 0.0
            for k in range(1, 26):
                row[k] = -eval_genlaguerre(2*k - 1, alpha + 1, 0.0)
            A_rows.append(row)
            b_vec.append(1.0)

            for zi in z_arr:
                # g(zi) = 0
                row = np.array([eval_genlaguerre(2*k, alpha, zi) for k in range(26)])
                A_rows.append(row)
                b_vec.append(0.0)

                # g'(zi) = 0
                row = np.zeros(26)
                row[0] = 0.0
                for k in range(1, 26):
                    row[k] = -eval_genlaguerre(2*k - 1, alpha + 1, zi)
                A_rows.append(row)
                b_vec.append(0.0)

            A_mat = np.array(A_rows)
            b_mat = np.array(b_vec)

            # Check condition number
            cond = np.linalg.cond(A_mat)
            if cond > 1e14:
                return float('inf')

            try:
                c_sol = np.linalg.solve(A_mat, b_mat)
            except:
                return float('inf')

            # Build g(x) and q(x) evaluation
            def g_func(x_val):
                return sum(c_sol[k] * eval_genlaguerre(2*k, alpha, x_val) for k in range(26))

            def denom_func(x_val):
                result = x_val
                for zi in z_arr:
                    result *= (x_val - zi) ** 2
                return result

            def q_func(x_val):
                d = denom_func(x_val)
                if abs(d) < 1e-300:
                    return 0.0
                return g_func(x_val) / d

            # Find real roots of q(x) by scanning for sign changes
            x_max = max(z_arr[-1] * 3, 50.0)
            x_min = 1e-8

            # Dense scan
            n_scan = 4000
            x_scan = np.linspace(x_min, x_max, n_scan)
            q_scan = np.array([q_func(xi) for xi in x_scan])

            # Find sign changes
            signs = np.sign(q_scan)
            sign_changes = []
            for i in range(n_scan - 1):
                if signs[i] * signs[i+1] < 0:
                    try:
                        root = brentq(q_func, x_scan[i], x_scan[i+1], xtol=1e-14, rtol=1e-12)
                        sign_changes.append(root)
                    except:
                        pass

            if len(sign_changes) == 0:
                return 0.0

            B = max(sign_changes)
            return B / (2 * np.pi)

        except Exception:
            return float('inf')
=======
"""Combined program: numerical pre-screening + local optimization + exact sympy verification."""
import time
import numpy as np
from scipy.special import eval_genlaguerre

# EVOLVE-BLOCK-START

# Precompute coefficients of L_{2k}^{(-1/2)}(x)
def get_laguerre_coeffs():
    import sympy as sp
    x = sp.Symbol('x')
    alpha = sp.Rational(-1, 2)
    coeffs_list = []
    for k in range(26):
        L = sp.laguerre(2*k, x, alpha)
        L_poly = sp.Poly(L, x)
        c = L_poly.all_coeffs()
        coeffs_list.append(np.array(c, dtype=np.float64))
    return coeffs_list

LAG_COEFFS = None

def construct():
    global LAG_COEFFS
    if LAG_COEFFS is None:
        LAG_COEFFS = get_laguerre_coeffs()

    deadline = time.time() + 620
    best_score = float('inf')
    best_z = None

    # --- Fast numerical evaluation (proxy for B/(2*pi)) ---
    def evaluate_numeric(z_list):
        """Fast floating-point evaluation using polynomial division."""
        try:
            z_arr = np.array(sorted(z_list), dtype=np.float64)
            if np.any(z_arr <= 0) or len(set(np.round(z_arr, 12))) != 12:
                return float('inf')

            A_rows = []
            b_vec = []

            # g(0) = 0
            row = np.array([eval_genlaguerre(2*k, -0.5, 0.0) for k in range(26)])
            A_rows.append(row)
            b_vec.append(0.0)

            # g'(0) = 1
            row = np.zeros(26)
            row[0] = 0.0
            for k in range(1, 26):
                row[k] = -eval_genlaguerre(2*k - 1, 0.5, 0.0)
            A_rows.append(row)
            b_vec.append(1.0)

            for zi in z_arr:
                row = np.array([eval_genlaguerre(2*k, -0.5, zi) for k in range(26)])
                A_rows.append(row)
                b_vec.append(0.0)

                row = np.zeros(26)
                row[0] = 0.0
                for k in range(1, 26):
                    row[k] = -eval_genlaguerre(2*k - 1, 0.5, zi)
                A_rows.append(row)
                b_vec.append(0.0)

            A_mat = np.array(A_rows)
            b_mat = np.array(b_vec)

            cond = np.linalg.cond(A_mat)
            if cond > 1e14:
                return float('inf')

            try:
                c_sol = np.linalg.solve(A_mat, b_mat)
            except:
                return float('inf')

            # Compute g(x) coefficients
            max_deg = 50
            g_coeffs = np.zeros(max_deg + 1)
            for k in range(26):
                c_k = c_sol[k]
                if c_k != 0:
                    coeffs = LAG_COEFFS[k]
                    # coeffs is [a_n, ..., a_0]
                    # g_coeffs is [a_50, ..., a_0]
                    # Align to the end
                    offset = max_deg - len(coeffs) + 1
                    g_coeffs[offset:] += c_k * coeffs

            # Compute D(x) = x * prod (x - z_i)^2
            D_coeffs = np.array([1.0])
            D_coeffs = np.concatenate([[0.0], D_coeffs])
            
            for zi in z_arr:
                factor = np.array([1.0, -2.0*zi, zi**2])
                D_coeffs = np.convolve(D_coeffs, factor)
            
            # D_coeffs is [a_25, ..., a_0]
            
            # Pad g_coeffs if needed? No, g is deg 50, D is deg 25.
            # polydiv expects monic or any leading coeff.
            # Ensure lengths are correct.
            if len(g_coeffs) < len(D_coeffs):
                return float('inf')
                
            q_coeffs, rem = np.polydiv(g_coeffs, D_coeffs)
            
            # Find roots of q
            roots = np.roots(q_coeffs)
            
            B = 0.0
            for r in roots:
                if np.isreal(r):
                    r_real = float(r.real)
                    if r_real > 0:
                        q_prime_coeffs = np.polyder(q_coeffs)
                        q_prime_val = np.polyval(q_prime_coeffs, r_real)
                        if abs(q_prime_val) > 1e-6:
                            if r_real > B:
                                B = r_real
            
            return B / (2 * np.pi)

        except Exception:
            return float('inf')
>>>>>>> REPLACE

I also need to update the rest of the program to use this new evaluator and potentially improve the search.
The current search uses `optimize_z` with Nelder-Mead.
I will keep that but ensure it uses the new `evaluate_numeric`.
I will also add more initial candidates.

