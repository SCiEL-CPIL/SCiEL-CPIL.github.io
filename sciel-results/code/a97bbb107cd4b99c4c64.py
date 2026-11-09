import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0

    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = 12

    training_paths = np.asarray(training_paths, dtype=float)
    N, n_tp = training_paths.shape
    dt = T / n_exercise
    discount = np.exp(-r * dt)

    # Determine column mapping
    def get_col(k):
        if n_tp == 13:
            return k
        else:
            return k - 1

    # Basis function: Polynomial in log(S)
    poly_degree = 5
    
    def get_basis(log_S):
        basis = np.ones((len(log_S), poly_degree + 1))
        for i in range(1, poly_degree + 1):
            basis[:, i] = log_S ** i
        return basis

    V = [None] * (n_exercise + 1)
    M = [None] * (n_exercise + 1)
    
    # Maturity
    S_12 = training_paths[:, get_col(12)]
    V[n_exercise] = np.maximum(K - S_12, 0.0)
    M[n_exercise] = np.ones(N, dtype=bool)
    
    coeffs = [None] * (n_exercise + 1)
    
    for k in range(n_exercise - 1, 0, -1):
        if time.time() > deadline:
            break
            
        S_k = training_paths[:, get_col(k)]
        log_S_k = np.log(S_k)
        
        mask = M[k+1]
        
        if np.sum(mask) < 10:
            coeffs[k] = np.zeros(poly_degree + 1)
            V[k] = np.maximum(K - S_k, 0.0)
            M[k] = np.zeros(N, dtype=bool)
            continue
            
        Y = discount * V[k+1][mask]
        X = get_basis(log_S_k[mask])
        
        try:
            res = np.linalg.lstsq(X, Y, rcond=None)
            coeffs[k] = res[0]
        except:
            coeffs[k] = np.zeros(poly_degree + 1)
            
        C_k = get_basis(log_S_k) @ coeffs[k]
        Payoff_k = np.maximum(K - S_k, 0.0)
        
        ex_mask = Payoff_k > C_k
        
        V[k] = np.where(ex_mask, Payoff_k, C_k)
        M[k] = M[k+1] & (~ex_mask)
        
    class BermudanPutPolicy:
        def __init__(self, coeffs, poly_degree, K, n_exercise):
            self.coeffs = coeffs
            self.poly_degree = poly_degree
            self.K = K
            self.n_exercise = n_exercise

        def exercise(self, current_state, time_step):
            current_state = np.asarray(current_state, dtype=float)
            try:
                if isinstance(time_step, (int, np.integer)):
                    idx = int(time_step)
                else:
                    idx = int(np.round(time_step * 12))

                if idx < 1 or idx > self.n_exercise:
                    return np.zeros(len(current_state), dtype=bool)

                if idx == self.n_exercise:
                    return current_state < self.K

                c = self.coeffs[idx]
                if c is None:
                    return np.zeros(len(current_state), dtype=bool)
                    
                log_S = np.log(current_state)
                basis = np.ones((len(current_state), self.poly_degree + 1))
                for i in range(1, self.poly_degree + 1):
                    basis[:, i] = log_S ** i
                
                cont_val = basis @ c
                payoff = np.maximum(self.K - current_state, 0.0)
                
                return payoff >= cont_val - 1e-10

            except Exception:
                return np.zeros(len(current_state), dtype=bool)

    return BermudanPutPolicy(coeffs, poly_degree, K, n_exercise)
