import numpy as np
import scipy.optimize
import time

def kakeya_needle_2d_positions64():
                n = 64
                M = 20000
                a = (np.arange(n) + 1) / n
                y = (np.arange(M) + 0.5) / M
                w = (1 - y) / n
                
                # Keich Initialization
                k = int(np.log2(n))
                j_idx = np.arange(1, k + 1)
                weights = (1 - j_idx) * (2.0 ** -j_idx) / k
                x = np.zeros(n, dtype=np.float64)
                for i in range(n):
                    bin_str = bin(i)[2:]
                    eps = np.array(list(bin_str), dtype=int)
                    eps = np.pad(eps, (k - len(eps), 0), "constant")
                    x[i] = np.sum(eps * weights)
                
                def objective(x):
                    L = x[None, :] + a[None, :] * y[:, None]
                    sorted_L = np.sort(L, axis=1)
                    gaps = sorted_L[:, 1:] - sorted_L[:, :-1]
                    union_len = w + np.sum(np.minimum(gaps, w[:, None]), axis=1)
                    return np.mean(union_len)
                
                def gradient(x):
                    L = x[None, :] + a[None, :] * y[:, None]
                    idx = np.argsort(L, axis=1)
                    sorted_L = L[np.arange(M)[:, None], idx]
                    gaps = sorted_L[:, 1:] - sorted_L[:, :-1]
                    mask = gaps < w[:, None]
                    
                    flat_idx = np.nonzero(mask.ravel())[0]
                    if len(flat_idx) == 0:
                        return np.zeros(n)
                    
                    m_idx, j_idx_local = np.unravel_index(flat_idx, (M, n-1))
                    k_idx_local = j_idx_local + 1
                    
                    src_k = idx[m_idx, k_idx_local]
                    src_j = idx[m_idx, j_idx_local]
                    
                    grad = np.zeros(n)
                    # Use bincount for fast accumulation
                    grad += np.bincount(src_k, weights=a[src_k], minlength=n)
                    grad -= np.bincount(src_j, weights=a[src_j], minlength=n)
                    
                    return grad / M
                
                start_time = time.time()
                deadline = start_time + 600.0
                
                # Optimize
                try:
                    res = scipy.optimize.minimize(
                        objective,
                        x,
                        jac=gradient,
                        method='L-BFGS-B',
                        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    if res.fun < objective(x):
                        x = res.x
                except Exception:
                    pass
                
                # Check time
                if time.time() > deadline - 10:
                    return x
                    
                # Optional: Polish with coordinate descent if time permits
                # But L-BFGS-B should be sufficient.
                
                return x
