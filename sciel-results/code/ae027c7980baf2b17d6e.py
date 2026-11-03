import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    
    H0_0 = 1
    H4_0 = -12
    H8_0 = -1680
    H12_0 = -665280
    
    def get_c12(c0, c4, c8):
        return (c0*H0_0 + c4*H4_0 + c8*H8_0) / (-H12_0)

    # Precompute moments
    # M[n, alpha] = integral exp(-alpha x^2) x^n dx
    # Only even n matter.
    # M[2k, alpha] = (2k-1)!! / (2*alpha)^k * sqrt(pi/alpha)
    
    def moment(n, alpha):
        if n % 2 != 0:
            return 0.0
        k = n // 2
        # (2k-1)!! = product of odd numbers from 1 to 2k-1
        # Can compute iteratively or using gamma
        # (2k-1)!! = (2k)! / (2^k k!)
        import math
        if k == 0:
            return np.sqrt(np.pi / alpha)
        double_factorial = 1
        for i in range(1, 2*k, 2):
            double_factorial *= i
        return double_factorial / ((2*alpha)**k) * np.sqrt(np.pi / alpha)

    def compute_integrals(c0, c4, c8):
        c12 = get_c12(c0, c4, c8)
        
        # Get coefficients of P(x) in power basis
        # P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
        # H0: 1
        # H4: x^4 - 6x^2 + 3
        # H8: x^8 - 28x^6 + 210x^4 - 420x^2 + 105
        # H12: x^12 - 84x^10 + 1680x^8 - 15120x^6 + 45360x^4 - 30240x^2 + 10395
        
        # p[j] is coeff of x^j
        p = np.zeros(13)
        
        # H0
        p[0] += c0
        p[2] += 0
        p[4] += 0
        
        # H4
        p[0] += c4 * 3
        p[2] += c4 * (-6)
        p[4] += c4 * 1
        
        # H8
        p[0] += c8 * 105
        p[2] += c8 * (-420)
        p[4] += c8 * 210
        p[6] += c8 * (-28)
        p[8] += c8 * 1
        
        # H12
        p[0] += c12 * 10395
        p[2] += c12 * (-30240)
        p[4] += c12 * 45360
        p[6] += c12 * (-15120)
        p[8] += c12 * 1680
        p[10] += c12 * (-84)
        p[12] += c12 * 1
        
        # Compute I2 = sum_{i,j} p[i]p[j] M[i+j, 2]
        I2 = 0.0
        for i in range(13):
            if p[i] == 0: continue
            for j in range(13):
                if p[j] == 0: continue
                I2 += p[i] * p[j] * moment(i+j, 2.0)
                
        # Compute I4 = sum_{i,j,k,l} p[i]p[j]p[k]p[l] M[i+j+k+l, 4]
        I4 = 0.0
        # Optimization: precompute sums
        # Or just loop. 13^4 is 28561, very fast.
        for i in range(13):
            if p[i] == 0: continue
            for j in range(13):
                if p[j] == 0: continue
                for k in range(13):
                    if p[k] == 0: continue
                    for l in range(13):
                        if p[l] == 0: continue
                        I4 += p[i]*p[j]*p[k]*p[l] * moment(i+j+k+l, 4.0)

        # Compute I_hat4
        # P(ik) coefficients q[j]
        # H0(ik) = 1
        # H4(ik) = k^4 + 6k^2 + 3
        # H8(ik) = k^8 + 28k^6 + 210k^4 + 420k^2 + 105
        # H12(ik) = k^12 + 84k^10 + 1680k^8 + 15120k^6 + 45360k^4 + 30240k^2 + 10395
        
        q = np.zeros(13)
        
        # H0
        q[0] += c0
        
        # H4
        q[0] += c4 * 3
        q[2] += c4 * 6
        q[4] += c4 * 1
        
        # H8
        q[0] += c8 * 105
        q[2] += c8 * 420
        q[4] += c8 * 210
        q[6] += c8 * 28
        q[8] += c8 * 1
        
        # H12
        q[0] += c12 * 10395
        q[2] += c12 * 30240
        q[4] += c12 * 45360
        q[6] += c12 * 15120
        q[8] += c12 * 1680
        q[10] += c12 * 84
        q[12] += c12 * 1
        
        I_hat4 = 0.0
        for i in range(13):
            if q[i] == 0: continue
            for j in range(13):
                if q[j] == 0: continue
                for k in range(13):
                    if q[k] == 0: continue
                    for l in range(13):
                        if q[l] == 0: continue
                        I_hat4 += q[i]*q[j]*q[k]*q[l] * moment(i+j+k+l, 1.0)
        
        I_hat4 *= (np.pi**4)
        
        return I2, I4, I_hat4

    def objective(c):
        c0, c4, c8 = c
        I2, I4, I_hat4 = compute_integrals(c0, c4, c8)
        if I2 <= 0:
            return 1e100
        return (I4 * I_hat4) / (I2**4)

    x0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    best_c = x0
    best_val = objective(x0)
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
        if res.fun < best_val:
            best_c = res.x
            best_val = res.fun
            
        res2 = minimize(objective, best_c, method='BFGS', options={'maxiter': 1000, 'gtol': 1e-10})
        if res2.fun < best_val:
            best_c = res2.x
            best_val = res2.fun
            
    except Exception as e:
        pass

    return np.array(best_c)
