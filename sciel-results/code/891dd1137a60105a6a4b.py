"""Direct combinatorial search for rank-reduced tensor decomposition of 2x3x7 matmul."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill

    # Index mapping for the tensor T[i,j,k] where:
    #   i in [0,n), j in [0,m), k in [0,p)
    #   U index: i*m + j   (n*m rows)
    #   V index: j*p + k   (m*p rows)
    #   W index: k*n + i   (p*n rows)  -- note: W[k*n+i, r] per trace convention
    #
    # The tensor entry T[(i*m+j), (j*p+k), (k*n+i)] = 1 for all valid (i,j,k), else 0.
    #
    # We seek rank R such that:
    #   For all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #   For all other (a,b,c) not corresponding to valid (i,j,k): sum_r ... = 0
    #
    # Strategy: Try ranks from 39 down to 35, using a structured initialization
    # with sparsity and integer/half-integer values, then refine via local search.

    def get_all_indices():
        """Return list of all (a,b,c) triples with their tensor values."""
        valid = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    valid.add((a, b, c))
        all_triples = []
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    all_triples.append((a, b, c, 1.0 if (a, b, c) in valid else 0.0))
        return all_triples, valid

    def schoolbook():
        R = n * m * p
        U = np.zeros((n * m, R), dtype=float)
        V = np.zeros((m * p, R), dtype=float)
        W = np.zeros((p * n, R), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def compute_error(U, V, W, all_triples):
        """Compute sum of squared errors over all tensor entries."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        R = Ua.shape[1]
        err = 0.0
        for a, b, c, target in all_triples:
            val = np.dot(np.dot(Ua[a], Va[b]), Wa[c])
            err += (val - target) ** 2
        return err

    def compute_error_fast(U, V, W, all_triples, sample=None):
        """Fast error using vectorized ops; sample if needed."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        R = Ua.shape[1]
        if sample is not None and len(all_triples) > sample:
            indices = random.sample(range(len(all_triples)), sample)
            triples = [all_triples[i] for i in indices]
        else:
            triples = all_triples
        err = 0.0
        for a, b, c, target in triples:
            val = np.dot(np.dot(Ua[a], Va[b]), Wa[c])
            err += (val - target) ** 2
        return err

    def try_rank(R, all_triples, time_budget):
        """Try to find a valid decomposition of rank R."""
        start_t = time.time()

        # Candidate values for sparse entries: {-1, -0.5, 0, 0.5, 1}
        # We'll use a structured approach: initialize with schoolbook-like patterns
        # and then do coordinate descent / local search.

        # Number of valid tensor entries
        num_valid = n * m * p
        num_total = n * m * m * p * p * n

        # We need: for each valid (a,b,c): sum_r U[a,r]V[b,r]W[c,r] = 1
        #           for each invalid (a,b,c): sum_r U[a,r]V[b,r]W[c,r] = 0

        # Approach: Use a basis of structured vectors.
        # Initialize U, V, W with random sparse patterns from candidate set.

        candidate_vals = [-1.0, -0.5, 0.5, 1.0]

        best_U = None
        best_V = None
        best_W = None
        best_err = float('inf')

        # Multiple random restarts
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            if time.time() > start_t + time_budget:
                break

            # Random initialization: sparse, each row has ~2-4 nonzeros
            U = np.zeros((n * m, R), dtype=float)
            V = np.zeros((m * p, R), dtype=float)
            W = np.zeros((p * n, R), dtype=float)

            # For each rank dimension r, pick a random pattern
            for r in range(R):
                # Each factor gets a subset of rows with candidate values
                # U: n*m rows
                nnz_u = random.randint(1, min(4, n * m))
                rows_u = random.sample(range(n * m), nnz_u)
                for row in rows_u:
                    U[row, r] = random.choice(candidate_vals)

                nnz_v = random.randint(1, min(4, m * p))
                rows_v = random.sample(range(m * p), nnz_v)
                for row in rows_v:
                    V[row, r] = random.choice(candidate_vals)

                nnz_w = random.randint(1, min(4, p * n))
                rows_w = random.sample(range(p * n), nnz_w)
                for row in rows_w:
                    W[row, r] = random.choice(candidate_vals)

            # Coordinate descent: try changing individual entries to reduce error
            # Focus on valid entries first
            Ua = U.copy()
            Va = V.copy()
            Wa = W.copy()

            # Compute initial error on a sample for speed
            err = compute_error_fast(Ua, Va, Wa, all_triples, sample=5000)

            # Local search: iterate, trying to fix one entry at a time
            local_iter = 0
            max_local_iter = 200
            while local_iter < max_local_iter:
                if time.time() > deadline:
                    break
                local_iter += 1

                # Pick a random factor and entry to modify
                factor = random.choice(['U', 'V', 'W'])
                if factor == 'U':
                    arr = Ua
                    arr_name = 'U'
                    size = n * m
                elif factor == 'V':
                    arr = Va
                    arr_name = 'V'
                    size = m * p
                else:
                    arr = Wa
                    arr_name = 'W'
                    size = p * n

                row = random.randint(0, size - 1)
                col = random.randint(0, R - 1)
                old_val = arr[row, col]

                # Try candidate values
                new_val = random.choice(candidate_vals + [0.0])
                if new_val == old_val:
                    continue

                arr[row, col] = new_val
                new_err = compute_error_fast(Ua, Va, Wa, all_triples, sample=2000)

                if new_err < err:
                    err = new_err
                    # Accept
                else:
                    # Revert (simple local search)
                    arr[row, col] = old_val

                if err < 1e-10:
                    break

            if err < best_err:
                best_err = err
                best_U = Ua.copy()
                best_V = Va.copy()
                best_W = Wa.copy()

            if best_err < 1e-10:
                break

        # Final validation with full error
        if best_U is not None and best_err < 1e-10:
            final_err = compute_error_fast(best_U, best_V, best_W, all_triples, sample=None)
            if final_err < 1e-8:
                return best_U.tolist(), best_V.tolist(), best_W.tolist()

        return None

    # Get all triples
    all_triples, valid_set = get_all_indices()

    # Try ranks from 39 down to 33
    # Known lower bound for 2x3x7: at least 2*3*7 - something, but let's try practical range
    for R in range(39, 32, -1):
        if time.time() > deadline:
            break

        time_remaining = deadline - time.time()
        if time_remaining < 5:
            break

        # Allocate time budget for this rank attempt
        time_budget = min(25, time_remaining - 5)

        result = try_rank(R, all_triples, time_budget)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
