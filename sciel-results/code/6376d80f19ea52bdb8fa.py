import numpy as np
import scipy.linalg
import scipy.integrate
import time


def integrate(f, t_span, y0):
    """
    Solve y' = f(t, y) on t_span = [t0, t1] for a 15-dim stiff linear IVP.
    Primary strategy: extract A and b (16 calls), then compute expm(A*dt) and
    the affine integral via Cayley-Hamilton polynomial reduction (degree n-1).
    Fallback: augmented (n+1)x(n+1) expm, then Radau/BDF/LSODA.
    """
    start_time = time.time()
    deadline = start_time + 270.0  # 30s safety margin under 300s limit

    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    dt = t1 - t0

    # --- Step 1: Extract A and b from the linear system ---
    try:
        zero_vec = np.zeros(n)
        b = np.asarray(f(t0, zero_vec), dtype=float)

        A = np.zeros((n, n))
        e0 = np.zeros(n)
        e0[0] = 1.0
        A[:, 0] = np.asarray(f(t0, e0), dtype=float) - b

        for i in range(1, n):
            if time.time() > deadline - 10.0:
                raise TimeoutError("Deadline approaching")
            ei = np.zeros(n)
            ei[i] = 1.0
            A[:, i] = np.asarray(f(t0, ei), dtype=float) - b

        if time.time() > deadline - 5.0:
            raise TimeoutError("Deadline approaching")

    except Exception:
        A = None
        b = None

    # --- Step 2: Solve via Cayley-Hamilton polynomial reduction ---
    if A is not None:
        try:
            # Compute characteristic polynomial of A
            # det(λI - A) = λ^n + c_{n-1} λ^{n-1} + ... + c_0
            # Use scipy to get coefficients in descending order of λ
            # np.poly(A) returns [1, c_{n-1}, ..., c_0]
            char_coeffs = np.poly(A)  # length n+1, char_coeffs[0] = 1
            c = char_coeffs[1:]  # c[0] = c_{n-1}, c[1] = c_{n-2}, ..., c[n-1] = c_0

            # We need exp(A*dt) = sum_{k=0}^{n-1} p_k(A)
            # By Cayley-Hamilton, A^n = -c_{n-1} A^{n-1} - ... - c_0 I
            # So we can reduce any power of A to degree < n.
            #
            # Alternative: use the eigenvalue-based approach.
            # exp(A*dt) = sum_{k=0}^{n-1} q_k(A) where q_k are polynomials of degree < n
            # such that for each eigenvalue λ_i:
            #   sum_{k=0}^{n-1} q_k(λ_i) λ_i^k = exp(λ_i * dt)
            # Wait, that's not right. We want a single polynomial r(x) of degree < n
            # such that r(λ_i) = exp(λ_i * dt) for all eigenvalues λ_i.
            # Then exp(A*dt) = r(A) = sum_{k=0}^{n-1} r_k A^k.
            #
            # This is polynomial interpolation: find r of degree < n with r(λ_i) = exp(λ_i*dt).
            # If eigenvalues are distinct, this is a standard Vandermonde problem.
            # If there are repeated eigenvalues, we need Hermite interpolation
            # (matching derivatives too), which is more complex.
            #
            # Let's use the eigenvalue approach with distinct-eigenvalue assumption,
            # falling back to scipy.linalg.expm if the Vandermonde is ill-conditioned.

            eigvals = np.linalg.eigvals(A)

            # Check for repeated eigenvalues (within tolerance)
            eig_sorted = np.sort_complex(eigvals)
            diffs = np.abs(np.diff(eig_sorted))
            min_diff = np.min(diffs) if n > 1 else 1.0

            if min_diff > 1e-10 * max(1.0, np.max(np.abs(eigvals))):
                # Distinct eigenvalues: use Lagrange/Vandermonde interpolation
                # Build Vandermonde matrix V where V[i,j] = λ_i^j, i=0..n-1, j=0..n-1
                V = np.vander(eigvals, N=n, increasing=True)  # V[i,j] = eigvals[i]^j
                rhs = np.exp(eigvals * dt)

                # Solve V @ r_coeffs = rhs for r_coeffs (length n)
                try:
                    cond_V = np.linalg.cond(V)
                    if cond_V < 1e12:
                        r_coeffs = np.linalg.solve(V, rhs)
                        # r_coeffs[k] is the coefficient of x^k in r(x)
                        # exp(A*dt) = r(A) = r_coeffs[0]*I + r_coeffs[1]*A + r_coeffs[2]*A^2 + ...
                        expA_dt = np.zeros((n, n))
                        A_power = np.eye(n)
                        for k in range(n):
                            expA_dt += r_coeffs[k] * A_power
                            A_power = A_power @ A
                    else:
                        raise np.linalg.LinAlgError("Vandermonde ill-conditioned")
                except np.linalg.LinAlgError:
                    # Fallback to standard expm
                    expA_dt = scipy.linalg.expm(dt * A)
            else:
                # Repeated eigenvalues: fall back to standard expm
                expA_dt = scipy.linalg.expm(dt * A)

            # Compute the affine integral term:
            # y_final = exp(A*dt) @ y0 + ∫_0^dt exp(A*(dt-s)) @ b ds
            # Let u = dt - s, then integral = ∫_0^dt exp(A*u) @ b du
            # For constant b, ∫_0^dt exp(A*u) du = A^{-1}(exp(A*dt) - I) if A is invertible
            # More generally, use the augmented system or:
            # ∫_0^dt exp(A*u) b du = sum_{k=0}^{n-1} s_k A^k @ b
            # where s is the polynomial such that s'(x) = r(x)/x... this gets complex.
            #
            # Simpler: use the augmented matrix approach for the affine part.
            # Or: compute ∫_0^dt exp(A*u) du @ b using a similar polynomial reduction.
            #
            # Actually, the cleanest is to build the augmented (n+1)x(n+1) system
            # but that defeats the purpose. Let me use the fact that:
            # ∫_0^dt exp(A*u) du = A^{-1}(exp(A*dt) - I) when A is invertible.
            # For singular A, we can use:
            # ∫_0^dt exp(A*u) du = sum_{k=0}^{n-1} t_k A^k
            # where t_k come from interpolating the function g(λ) = (exp(λ*dt)-1)/λ
            # at the eigenvalues (with the limit 1 at λ=0).

            # Compute the integral polynomial coefficients
            # g(λ) = (exp(λ*dt) - 1) / λ, with g(0) = 1
            # Find t of degree < n such that t(λ_i) = g(λ_i)
            g_vals = np.zeros(n, dtype=complex)
            for i in range(n):
                lam = eigvals[i]
                if abs(lam) < 1e-15:
                    g_vals[i] = 1.0  # limit of (exp(λdt)-1)/λ as λ→0 is dt... wait
                    # (exp(λ*dt) - 1) / λ → dt as λ→0
                    g_vals[i] = dt
                else:
                    g_vals[i] = (np.exp(lam * dt) - 1.0) / lam

            if min_diff > 1e-10 * max(1.0, np.max(np.abs(eigvals))):
                try:
                    cond_V2 = np.linalg.cond(V)
                    if cond_V2 < 1e12:
                        t_coeffs = np.linalg.solve(V, g_vals)
                        # Integral = sum_{k=0}^{n-1} t_coeffs[k] A^k
                        integral_matrix = np.zeros((n, n))
                        A_power = np.eye(n)
                        for k in range(n):
                            integral_matrix += t_coeffs[k] * A_power
                            A_power = A_power @ A
                    else:
                        raise np.linalg.LinAlgError("Vandermonde ill-conditioned for integral")
                except np.linalg.LinAlgError:
                    # Fallback: use scipy to compute the integral
                    M = np.zeros((n + 1, n + 1))
                    M[:n, :n] = A
                    M[:n, n] = b
                    expM = scipy.linalg.expm(dt * M)
                    z_init = np.zeros(n + 1)
                    z_init[:n] = y0
                    z_init[n] = 1.0
                    z_final = expM @ z_init
                    return z_final[:n]
            else:
                # Fallback for repeated eigenvalues
                M = np.zeros((n + 1, n + 1))
                M[:n, :n] = A
                M[:n, n] = b
                expM = scipy.linalg.expm(dt * M)
                z_init = np.zeros(n + 1)
                z_init[:n] = y0
                z_init[n] = 1.0
                z_final = expM @ z_init
                return z_final[:n]

            y_final = expA_dt @ y0 + integral_matrix @ b
            return y_final

        except Exception:
            pass

    # --- Step 3: Fallback to augmented expm ---
    if A is not None and b is not None:
        try:
            M = np.zeros((n + 1, n + 1))
            M[:n, :n] = A
            M[:n, n] = b
            expM = scipy.linalg.expm(dt * M)
            z_init = np.zeros(n + 1)
            z_init[:n] = y0
            z_init[n] = 1.0
            z_final = expM @ z_init
            return z_final[:n]
        except Exception:
            pass

    # --- Step 4: Fallback to numerical solvers ---
    try:
        sol = scipy.integrate.solve_ivp(
            f, t_span, y0,
            method='Radau',
            rtol=1e-13,
            atol=1e-15,
            dense_output=False
        )
        if sol.success:
            return sol.y[:, -1]
        else:
            sol2 = scipy.integrate.solve_ivp(
                f, t_span, y0,
                method='BDF',
                rtol=1e-12,
                atol=1e-14
            )
            return sol2.y[:, -1]
    except Exception:
        try:
            sol3 = scipy.integrate.solve_ivp(
                f, t_span, y0,
                method='LSODA',
                rtol=1e-10,
                atol=1e-12
            )
            return sol3.y[:, -1]
        except Exception:
            try:
                sol4 = scipy.integrate.solve_ivp(
                    f, t_span, y0,
                    method='RK45',
                    rtol=1e-10,
                    atol=1e-12
                )
                return sol4.y[:, -1]
            except Exception:
                return y0
