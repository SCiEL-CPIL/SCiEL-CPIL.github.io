import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin

    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def validate_fast(U, V, W):
        """
        Validate using exact Fraction arithmetic.
        U, V, W are lists of lists of numbers (int, float, or Fraction).
        """
        try:
            rank = len(U[0])
            U_mat = np.array(U, dtype=object)
            V_mat = np.array(V, dtype=object)
            W_mat = np.array(W, dtype=object)
            
            # Convert to Fractions for exactness
            # This is slow but necessary for "exact" validation
            # Optimization: only convert if floats are present or just use Fraction directly in loop
            
            U_frac = np.array([[Fraction(x) for x in row] for row in U], dtype=object)
            V_frac = np.array([[Fraction(x) for x in row] for row in V], dtype=object)
            W_frac = np.array([[Fraction(x) for x in row] for row in W], dtype=object)
            
            U_3d = U_frac.reshape(n, m, rank)
            V_3d = V_frac.reshape(m, p, rank)
            W_3d = W_frac.reshape(p, n, rank)
            
            # T_ijk = sum_r U_ijr * V_jkr * W_kir
            # We need T_ijk == 1 for all i,j,k
            
            # Iterate manually to avoid complex einsum with object dtype
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = Fraction(0)
                        for r in range(rank):
                            s += U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
                        if s != 1:
                            return False
            return True
        except Exception:
            return False

    def validate_float(U, V, W, tol=1e-9):
        """
        Fast float validation for candidate generation.
        """
        rank = len(U[0])
        U_mat = np.array(U, dtype=float)
        V_mat = np.array(V, dtype=float)
        W_mat = np.array(W, dtype=float)
        
        U_3d = U_mat.reshape(n, m, rank)
        V_3d = V_mat.reshape(m, p, rank)
        W_3d = W_mat.reshape(p, n, rank)
        
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        if not np.allclose(T_recon, 1.0, atol=tol):
            return False
        return True

    def to_half_integers(mat):
        """
        Attempt to convert a float matrix to a matrix of Fractions that are
        integers or half-integers. Returns None if fails.
        """
        result = []
        for row in mat:
            new_row = []
            for val in row:
                # Check if close to integer
                nearest_int = round(val)
                if abs(val - nearest_int) < 1e-9:
                    new_row.append(Fraction(nearest_int))
                    continue
                # Check if close to half-integer
                nearest_half_num = round(2 * val)
                nearest_half = Fraction(nearest_half_num, 2)
                if abs(val - float(nearest_half)) < 1e-9:
                    new_row.append(nearest_half)
                    continue
                return None
            result.append(new_row)
        return result

    def try_structured_search(target_rank, max_attempts=10000):
        """
        Try to find a decomposition by fixing the first r-1 columns randomly
        and solving for the r-th column.
        """
        # Pre-compute indices for speed
        indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Generate random integer matrices for the first r-1 columns
            # We use small integers {-1, 0, 1} to keep numbers small and increase chance of consistency
            # Using randint(-1, 2) gives {-1, 0, 1}
            
            if target_rank == 1:
                U_prev = np.zeros((n * m, 0))
                V_prev = np.zeros((m * p, 0))
                W_prev = np.zeros((p * n, 0))
            else:
                U_prev = np.random.randint(-1, 2, size=(n * m, target_rank - 1))
                V_prev = np.random.randint(-1, 2, size=(m * p, target_rank - 1))
                W_prev = np.random.randint(-1, 2, size=(p * n, target_rank - 1))
            
            U_last = np.random.randint(-1, 2, size=(n * m, 1))
            V_last = np.random.randint(-1, 2, size=(m * p, 1))
            
            # Compute T_prev
            if target_rank > 1:
                U_prev_3d = U_prev.reshape(n, m, target_rank - 1)
                V_prev_3d = V_prev.reshape(m, p, target_rank - 1)
                W_prev_3d = W_prev.reshape(p, n, target_rank - 1)
                T_prev = np.einsum('ijr,jkr,kir->ijk', U_prev_3d, V_prev_3d, W_prev_3d)
            else:
                T_prev = np.zeros((n, m, p))
            
            # RHS = 1 - T_prev
            RHS = 1.0 - T_prev
            
            # Solve for W_last
            # For each (i, k), W_last[k*n+i] must satisfy:
            # Sum_j U_last[i*m+j] * V_last[j*p+k] * W_last[k*n+i] = RHS[i,j,k]
            # Note: W_last[k*n+i] is constant for fixed i,k.
            # So for each i,k:
            # Let C_ijk = U_last[i*m+j] * V_last[j*p+k]
            # We need: Sum_j C_ijk * w_ik = RHS[i,j,k]  <-- WAIT.
            
            # Let's re-read the tensor definition.
            # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            # For the last term (r = target_rank-1):
            # U_last[i*m+j] * V_last[j*p+k] * W_last[k*n+i]
            # This term depends on j.
            # The equation is:
            # T_prev[i,j,k] + U_last[i*m+j] * V_last[j*p+k] * W_last[k*n+i] = 1
            # => U_last[i*m+j] * V_last[j*p+k] * W_last[k*n+i] = 1 - T_prev[i,j,k] = RHS[i,j,k]
            
            # For a fixed (i, k), W_last[k*n+i] is a single scalar w_ik.
            # The equation must hold for ALL j in 0..m-1.
            # So for all j:
            # (U_last[i*m+j] * V_last[j*p+k]) * w_ik = RHS[i,j,k]
            
            # Let A_ijk = U_last[i*m+j] * V_last[j*p+k]
            # We need A_ijk * w_ik = RHS[i,j,k] for all j.
            
            W_last_candidate = np.zeros((p * n), dtype=float)
            valid = True
            
            for i in range(n):
                for k in range(p):
                    w_val = None
                    for j in range(m):
                        c_val = U_last[i * m + j, 0] * V_last[j * p + k, 0]
                        rhs_val = RHS[i, j, k]
                        
                        if c_val == 0:
                            if abs(rhs_val) > 1e-9:
                                valid = False
                                break
                        else:
                            # w_val = rhs_val / c_val
                            ratio = rhs_val / c_val
                            # Check if ratio is close to an integer or half-integer
                            # Since we want the final matrix to have integer/half-integer entries,
                            # and U, V are integers, W must be such that the product works out.
                            # Actually, the constraint is on W entries being int/half-int.
                            # So ratio must be int or half-int.
                            
                            nearest_int = round(ratio)
                            if abs(ratio - nearest_int) < 1e-9:
                                candidate = nearest_int
                            else:
                                nearest_half = round(2 * ratio) / 2.0
                                if abs(ratio - nearest_half) < 1e-9:
                                    candidate = nearest_half
                                else:
                                    valid = False
                                    break
                            
                            if w_val is None:
                                w_val = candidate
                            else:
                                if abs(w_val - candidate) > 1e-9:
                                    valid = False
                                    break
                    if not valid:
                        break
                    if w_val is not None:
                        W_last_candidate[k * n + i] = w_val
                if not valid:
                    break
            
            if valid:
                # Construct full U, V, W as lists of floats for fast validation
                if target_rank > 1:
                    U_full = np.hstack([U_prev, U_last])
                    V_full = np.hstack([V_prev, V_last])
                    W_full = np.hstack([W_prev, W_last_candidate.reshape(-1, 1)])
                else:
                    U_full = U_last
                    V_full = V_last
                    W_full = W_last_candidate.reshape(-1, 1)
                
                # Fast float validation
                if validate_float(U_full.tolist(), V_full.tolist(), W_full.tolist()):
                    # Convert to half-integers
                    U_frac = to_half_integers(U_full.tolist())
                    V_frac = to_half_integers(V_full.tolist())
                    W_frac = to_half_integers(W_full.tolist())
                    
                    if U_frac is not None and V_frac is not None and W_frac is not None:
                        # Exact validation
                        if validate_fast(U_frac, V_frac, W_frac):
                            return {
                                "U": U_frac,
                                "V": V_frac,
                                "W": W_frac
                            }
        
        return None

    # Try ranks from 20 to 29
    # Note: Rank 30 is schoolbook. We try to beat it.
    # Lower bounds for matrix multiplication tensor rank are not tight for small dims.
    # For 2x3x5, rank is likely less than 30.
    
    for target_rank in range(20, 30):
        if time.time() > deadline:
            break
        
        result = try_structured_search(target_rank, max_attempts=2000)
        if result is not None:
            return result

    # Fallback to schoolbook
    return get_schoolbook()
# EVOLVE-BLOCK-END
