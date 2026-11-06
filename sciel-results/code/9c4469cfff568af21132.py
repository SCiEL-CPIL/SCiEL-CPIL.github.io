import numpy as np
import scipy.linalg as la

def integrate(f, t_span, y0):
        y0 = np.array(y0, dtype=float)
        n = len(y0)
        t0, t1 = t_span
        T = t1 - t0
        
        # Check for Homogeneous Linear Time-Independent System
        # f(t, y) = A y
        is_linear_homogeneous = True
        
        # Check f(t, 0) == 0
        f_zero = f(t0, np.zeros(n))
        if np.linalg.norm(f_zero) > 1e-10:
            is_linear_homogeneous = False
        else:
            # Check time independence
            y_rand = np.random.randn(n)
            f_t0 = f(t0, y_rand)
            f_t1 = f(t1, y_rand)
            if np.linalg.norm(f_t0 - f_t1) > 1e-10 * max(1.0, np.linalg.norm(f_t0)):
                is_linear_homogeneous = False
            else:
                # Check linearity
                y_a = np.random.randn(n)
                y_b = np.random.randn(n)
                f_a = f(t0, y_a)
                f_b = f(t0, y_b)
                f_ab = f(t0, y_a + y_b)
                if np.linalg.norm(f_ab - (f_a + f_b)) > 1e-10 * max(1.0, np.linalg.norm(f_a)):
                    is_linear_homogeneous = False

        if is_linear_homogeneous:
            # Construct A
            A = np.zeros((n, n))
            for j in range(n):
                e_j = np.zeros(n)
                e_j[j] = 1.0
                A[:, j] = f(t0, e_j)
            
            # Compute matrix exponential
            M = la.expm(A * T)
            y_end = M.dot(y0)
            return y_end
        else:
            # Fallback: RK4
            # 100 steps is very accurate for this non-stiff problem
            steps = 100
            h = T / steps
            t = t0
            y = y0.copy()
            for _ in range(steps):
                k1 = f(t, y)
                k2 = f(t + h/2, y + h/2 * k1)
                k3 = f(t + h/2, y + h/2 * k2)
                k4 = f(t + h, y + h * k3)
                y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
                t += h
            return y
