"""Search for rank-minimal exact tensor decomposition of 3x5 * 5x7 matmul."""
import random
import time
import itertools

DIMENSIONS = [3, 5, 7]
N, M, P = 3, 5, 7
DEADLINE = 115.0  # safety margin under 120s

# EVOLVE-BLOCK-START

def _verify(U, V, W, tol=1e-9):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c] == W[k*n+i,c] for all i,j,k,c."""
    n, m, p = N, M, P
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                uc = U[i*m+j]
                vc = V[j*p+k]
                wc = W[k*n+i]
                for c in range(rank):
                    s += uc[c] * vc[c]
                    # accumulate in a way that's stable
                if abs(s - 1.0) > tol:
                    return False
                # also ensure W entries are consistent (they should be 1 for the
                # standard tensor, but the convention allows any values; the
                # actual tensor entry T[i,j,k] must equal sum_c U[i,j,c]*V[j,k,c]
                # and the output factor is W[k*n+i, c]. For the matmul tensor,
                # T[i,j,k] = 1 for all i,j,k, so we need the sum to be 1.
                # The W matrix here is actually the "output factor" - in the
                # standard decomposition, the tensor is sum_c U_c outer V_c outer W_c
                # where W_c[k*n+i] should be the coefficient. Let me re-read.
                pass
    return True


def _verify2(U, V, W, tol=1e-9):
    """
    The tensor T[i,j,k] for matrix multiplication (n x m) * (m x p) is 1 for all i,j,k.
    The decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    Wait, re-reading: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    and "output factor follows the official trace convention W[k*n+i,r]"
    
    So the tensor is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    And T[i,j,k] = 1 for all i,j,k (since it's the identity tensor for matmul).
    """
    n, m, p = N, M, P
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                uc = U[i*m+j]
                vc = V[j*p+k]
                wc = W[k*n+i]
                for r in range(rank):
                    s += uc[r] * vc[r] * wc[r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _round_to_half(x):
    """Round to nearest half-integer."""
    return round(x * 2) / 2.0


def _to_int_list(flat):
    """Convert a list of half-integers to a list of ints (multiply by 2) for exact checking."""
    return [round(x * 2) for x in flat]


def _verify_exact(U, V, W):
    """Exact verification using integer arithmetic (entries are half-integers, so *2 are ints)."""
    n, m, p = N, M, P
    rank = len(U[0])
    # Convert to integer representation (multiply by 2)
    U2 = [[u * 2 for u in row] for row in U]
    V2 = [[v * 2 for v in row] for row in V]
    W2 = [[w * 2 for w in row] for row in W]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                uc = U2[i*m+j]
                vc = V2[j*p+k]
                wc = W2[k*n+i]
                for r in range(rank):
                    s += uc[r] * vc[r] * wc[r]
                # T[i,j,k] = s / 8 should equal 1, so s should equal 8
                if s != 8:
                    return False
    return True


def _make_schoolbook():
    n, m, p = N, M, P
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W


def _try_rank(rank, time_limit, rng):
    """
    Try to find a decomposition of given rank using random initialization + local search.
    Returns (U, V, W) or None.
    """
    n, m, p = N, M, P
    size_U = n * m
    size_V = m * p
    size_W = p * n
    
    best_result = None
    start = time.time()
    
    for attempt in range(50):
        if time.time() - start > time_limit:
            break
        
        # Random initialization with half-integer values
        U = [[rng.choice([-1, 0, 1, 2]) / 2 for _ in range(rank)] for _ in range(size_U)]
        V = [[rng.choice([-1, 0, 1, 2]) / 2 for _ in range(rank)] for _ in range(size_V)]
        W = [[rng.choice([-1, 0, 1, 2]) / 2 for _ in range(rank)] for _ in range(size_W)]
        
        # Local search: try to fix the tensor by adjusting entries
        # We need T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k
        
        # Compute current tensor values
        def compute_tensor(U, V, W):
            T = {}
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(rank):
                            s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                        T[(i,j,k)] = s
            return T
        
        # Gradient-based local search: for each (i,j,k) where T != 1, try to fix
        # by adjusting one entry at a time
        for step in range(200):
            if time.time() - start > time_limit:
                break
            
            T = compute_tensor(U, V, W)
            # Find the worst mismatch
            worst_key = None
            worst_err = 0.0
            for key, val in T.items():
                err = abs(val - 1.0)
                if err > worst_err:
                    worst_err = err
                    worst_key = key
            
            if worst_err < 1e-9:
                # Check exactness
                U_rounded = [[_round_to_half(x) for x in row] for row in U]
                V_rounded = [[_round_to_half(x) for x in row] for row in V]
                W_rounded = [[_round_to_half(x) for x in row] for row in W]
                if _verify_exact(U_rounded, V_rounded, W_rounded):
                    return U_rounded, V_rounded, W_rounded
                # Otherwise continue
                break
            
            i, j, k = worst_key
            # Try to adjust U[i*m+j][r], V[j*p+k][r], or W[k*n+i][r] for some r
            # The contribution of rank r to T[i,j,k] is U[i,j,r]*V[j,k,r]*W[k,i,r]
            # We want to change this by (1 - T[i,j,k])
            
            # Try adjusting W[k*n+i][r] for each r
            improved = False
            for r in range(rank):
                current_contrib = U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if U[i*m+j][r] * V[j*p+k][r] == 0:
                    continue
                # We want new W such that U*V*newW = current_contrib + (1-T)
                target = current_contrib + (1.0 - T[(i,j,k)])
                newW = target / (U[i*m+j][r] * V[j*p+k][r])
                newW_rounded = _round_to_half(newW)
                if abs(newW_rounded - W[k*n+i][r]) < 1e-9:
                    continue
                # Check if this helps overall
                old_W = W[k*n+i][r]
                W[k*n+i][r] = newW_rounded
                T2 = compute_tensor(U, V, W)
                err2 = sum(abs(v - 1.0) for v in T2.values())
                err1 = sum(abs(v - 1.0) for v in T.values())
                if err2 < err1 - 1e-12:
                    improved = True
                    break
                else:
                    W[k*n+i][r] = old_W
            
            if not improved:
                # Try a random perturbation
                mat_choice = rng.choice(['U', 'V', 'W'])
                if mat_choice == 'U':
                    row = rng.randrange(size_U)
                    col = rng.randrange(rank)
                    delta = rng.choice([-1, 1]) / 2
                    U[row][col] += delta
                elif mat_choice == 'V':
                    row = rng.randrange(size_V)
                    col = rng.randrange(rank)
                    delta = rng.choice([-1, 1]) / 2
                    V[row][col] += delta
                else:
                    row = rng.randrange(size_W)
                    col = rng.randrange(rank)
                    delta = rng.choice([-1, 1]) / 2
                    W[row][col] += delta
        
        # Final check
        U_rounded = [[_round_to_half(x) for x in row] for row in U]
        V_rounded = [[_round_to_half(x) for x in row] for row in V]
        W_rounded = [[_round_to_half(x) for x in row] for row in W]
        if _verify_exact(U_rounded, V_rounded, W_rounded):
            return U_rounded, V_rounded, W_rounded
    
    return None


def _try_structured_rank(rank, time_limit, rng):
    """
    Try structured decompositions: use rank-1 tensors with specific patterns.
    For matmul, known constructions exist. Let's try random rank-1 combinations.
    """
    n, m, p = N, M, P
    size_U = n * m
    size_V = m * p
    size_W = p * n
    
    start = time.time()
    
    for attempt in range(100):
        if time.time() - start > time_limit:
            break
        
        # Initialize with random half-integers
        U = [[rng.choice([-2, -1, 0, 1, 2]) / 2 for _ in range(rank)] for _ in range(size_U)]
        V = [[rng.choice([-2, -1, 0, 1, 2]) / 2 for _ in range(rank)] for _ in range(size_V)]
        W = [[rng.choice([-2, -1, 0, 1, 2]) / 2 for _ in range(rank)] for _ in range(size_W)]
        
        # Iteratively project: for each (i,j,k), adjust to make T[i,j,k]=1
        for iteration in range(500):
            if time.time() - start > time_limit:
                break
            
            # Compute residuals
            residuals = {}
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(rank):
                            s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                        residuals[(i,j,k)] = 1.0 - s
            
            max_res = max(abs(v) for v in residuals.values())
            if max_res < 1e-9:
                U_rounded = [[_round_to_half(x) for x in row] for row in U]
                V_rounded = [[_round_to_half(x) for x in row] for row in V]
                W_rounded = [[_round_to_half(x) for x in row] for row in W]
                if _verify_exact(U_rounded, V_rounded, W_rounded):
                    return U_rounded, V_rounded, W_rounded
                break
            
            # Pick the worst residual
            worst_key = max(residuals, key=lambda k: abs(residuals[k]))
            i, j, k = worst_key
            res = residuals[worst_key]
            
            # Try to fix by adjusting one factor
            fixed = False
            for r in range(rank):
                u_val = U[i*m+j][r]
                v_val = V[j*p+k][r]
                w_val = W[k*n+i][r]
                
                # Try adjusting W
                if u_val * v_val != 0:
                    new_w = (u_val * v_val * w_val + res) / (u_val * v_val)
                    new_w_r = _round_to_half(new_w)
                    if abs(new_w_r - w_val) > 1e-9:
                        old = W[k*n+i][r]
                        W[k*n+i][r] = new_w_r
                        # Quick check: does this reduce total error?
                        total_err = 0.0
                        for ii in range(n):
                            for jj in range(m):
                                for kk in range(p):
                                    s = 0.0
                                    for rr in range(rank):
                                        s += U[ii*m+jj][rr] * V[jj*p+kk][rr] * W[kk*n+ii][rr]
                                    total_err += abs(s - 1.0)
                        # Compare with before
                        if total_err < sum(abs(v) for v in residuals.values()) - 1e-12:
                            fixed = True
                            break
                        else:
                            W[k*n+i][r] = old
                
                # Try adjusting V
                if u_val * w_val != 0:
                    new_v = (u_val * v_val * w_val + res) / (u_val * w_val)
                    new_v_r = _round_to_half(new_v)
                    if abs(new_v_r - v_val) > 1e-9:
                        old = V[j*p+k][r]
                        V[j*p+k][r] = new_v_r
                        total_err = 0.0
                        for ii in range(n):
                            for jj in range(m):
                                for kk in range(p):
                                    s = 0.0
                                    for rr in range(rank):
                                        s += U[ii*m+jj][rr] * V[jj*p+kk][rr] * W[kk*n+ii][rr]
                                    total_err += abs(s - 1.0)
                        if total_err < sum(abs(v) for v in residuals.values()) - 1e-12:
                            fixed = True
                            break
                        else:
                            V[j*p+k][r] = old
                
                # Try adjusting U
                if v_val * w_val != 0:
                    new_u = (u_val * v_val * w_val + res) / (v_val * w_val)
                    new_u_r = _round_to_half(new_u)
                    if abs(new_u_r - u_val) > 1e-9:
                        old = U[i*m+j][r]
                        U[i*m+j][r] = new_u_r
                        total_err = 0.0
                        for ii in range(n):
                            for jj in range(m):
                                for kk in range(p):
                                    s = 0.0
                                    for rr in range(rank):
                                        s += U[ii*m+jj][rr] * V[jj*p+kk][rr] * W[kk*n+ii][rr]
                                    total_err += abs(s - 1.0)
                        if total_err < sum(abs(v) for v in residuals.values()) - 1e-12:
                            fixed = True
                            break
                        else:
                            U[i*m+j][r] = old
            
            if not fixed:
                # Random kick
                choice = rng.choice(['U', 'V', 'W'])
                if choice == 'U':
                    row = rng.randrange(size_U)
                    col = rng.randrange(rank)
                    U[row][col] += rng.choice([-1, 1]) / 2
                elif choice == 'V':
                    row = rng.randrange(size_V)
                    col = rng.randrange(rank)
                    V[row][col] += rng.choice([-1, 1]) / 2
                else:
                    row = rng.randrange(size_W)
                    col = rng.randrange(rank)
                    W[row][col] += rng.choice([-1, 1]) / 2
    
    return None


def construct():
    n, m, p = N, M, P
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Start with schoolbook as fallback
    U_fb, V_fb, W_fb = _make_schoolbook()
    best_rank = n * m * p
    best_result = (U_fb, V_fb, W_fb)
    
    rng = random.Random(42)
    
    # Try ranks in increasing order
    # Known: rank for 3x5*5x7 is at most 105 (schoolbook)
    # Lower bounds: max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    # Try from 35 upward
    
    for rank in range(35, best_rank + 1):
        if time.time() > deadline - 10:
            break
        
        time_left = deadline - time.time()
        if time_left < 5:
            break
        
        # Allocate time proportionally
        time_for_rank = min(30.0, time_left * 0.5)
        
        result = _try_structured_rank(rank, time_for_rank, rng)
        if result is not None:
            U, V, W = result
            if _verify_exact(U, V, W):
                best_rank = rank
                best_result = (U, V, W)
                # Found one, try to go lower? No, we're going up. 
                # Actually we want the minimum, so if we find rank r, we should
                # check if lower ranks work. But we're iterating up from 35.
                # If we find a valid one at rank r, that's our answer (since we
                # started from the lower bound).
                break
        
        # Also try the other method
        if time.time() < deadline - 5:
            time_for_rank2 = min(15.0, deadline - time.time() - 5)
            result2 = _try_rank(rank, time_for_rank2, rng)
            if result2 is not None:
                U, V, W = result2
                if _verify_exact(U, V, W):
                    best_rank = rank
                    best_result = (U, V, W)
                    break
    
    U, V, W = best_result
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
