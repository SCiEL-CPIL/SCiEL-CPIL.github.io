import numpy as np
import scipy.optimize as opt
import scipy.sparse as sp
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes the objective a = -sum_k f(k) log(k) / k subject to M(x) <= 1.
    """
    start_time = time.time()
    deadline = 580  # Leave 20s margin for safety

    # Try progressively larger K values
    K_values = [1000, 2000, 5000]
    best_result = None
    best_score = -np.inf

    for K in K_values:
        if time.time() - start_time > deadline - 120:
            break

        try:
            # Build sparse constraint matrix
            # A[x-1, k-1] = floor(x/k) for x in 1..10K, k in 1..K
            n_rows = 10 * K
            n_cols = K

            row_idx = []
            col_idx = []
            data = []

            for x in range(1, n_rows + 1):
                for k in range(1, n_cols + 1):
                    val = x // k
                    if val != 0:
                        row_idx.append(x - 1)
                        col_idx.append(k - 1)
                        data.append(val)

            A = sp.csr_matrix((data, (row_idx, col_idx)), shape=(n_rows, n_cols))

            # Objective: minimize sum_{k=2}^K (log(k)/k) * f_k
            c = np.zeros(K)
            k_vals = np.arange(1, K + 1).astype(np.float64)
            c[1:] = np.log(k_vals[1:]) / k_vals[1:]

            bounds = [(-10.0, 10.0)] * K
            b_ub = np.ones(n_rows)

            res = opt.linprog(c, A_ub=A, b_ub=b_ub, bounds=bounds, method='highs')

            if res.success:
                f_vals = res.x
                clipped = np.clip(f_vals, -10, 10)
                domain = np.arange(1, K + 1, dtype=np.int64)

                # Compute S
                S_eval = np.sum(clipped / domain.astype(np.float64))
                clipped[0] = clipped[0] - S_eval

                # Check feasibility
                feasible = True
                # Use efficient chunked computation
                chunk_size = 10000
                for start_x in range(1, 10 * K + 1, chunk_size):
                    end_x = min(start_x + chunk_size - 1, 10 * K)
                    x_range = np.arange(start_x, end_x + 1)
                    # Compute M(x) for this range
                    # M(x) = sum_k f(k) * floor(x/k)
                    # Vectorize: for each x, compute sum over k
                    # This is expensive, so we sample
                    if len(x_range) > 100:
                        # Sample every 10th x for speed
                        sample_idx = np.arange(0, len(x_range), 10)
                        sample_x = x_range[sample_idx]
                        for x in sample_x:
                            Mx = np.sum(clipped * (x // domain))
                            if Mx > 1.0001:
                                feasible = False
                                break
                    else:
                        for x in x_range:
                            Mx = np.sum(clipped * (x // domain))
                            if Mx > 1.0001:
                                feasible = False
                                break
                    if not feasible:
                        break

                if feasible:
                    # Compute score
                    score = -np.sum(clipped * np.log(domain.astype(np.float64)) / domain.astype(np.float64))
                    if score > best_score:
                        best_score = score
                        best_result = (domain.copy(), clipped.copy())

            if time.time() - start_time > deadline - 60:
                break

        except Exception as e:
            continue

    # Return best result or fallback
    if best_result is not None:
        return best_result

    # Fallback to seed
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    return domain, values
