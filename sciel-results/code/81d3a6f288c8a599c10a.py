import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0  # safety margin before 120s limit
    
    # First, return the known rank-11 solution as fallback
    rank11_solution = _get_rank11_solution()
    
    # Try to find a rank-10 solution
    try:
        rank10_solution = _search_rank10(start_time, deadline)
        if rank10_solution is not None:
            return rank10_solution
    except Exception:
        pass
    
    return rank11_solution


def _get_rank11_solution():
    n, m, p = DIMENSIONS
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def set_u(i, j, r, val): U[i*m+j][r] = val
    def set_v(j, k, r, val): V[j*p+k][r] = val
    def set_w(k, i, r, val): W[k*n+i][r] = val

    # 1. (A00+A11)(B00+B11) -> M1
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1)
    set_w(1, 1, 0, 1)

    # 2. (A10+A11)B00 -> M2
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1)
    set_w(1, 1, 1, -1)

    # 3. A00(B01-B11) -> M3
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1)
    set_w(1, 1, 2, 1)

    # 4. A11(B10-B00) -> M4
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1)
    set_w(0, 1, 3, 1)

    # 5. (A00+A01)B11 -> M5
    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1)
    set_w(1, 0, 4, 1)

    # 6. (A10-A00)(B00+B01) -> M6
    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1)

    # 7. (A01-A11)(B10+B11) -> M7
    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(0, 0, 6, 1)

    # 8. A00 B02 -> M8
    set_u(0, 0, 7, 1)
    set_v(0, 2, 7, 1)
    set_w(2, 0, 7, 1)

    # 9. A01 B12 -> M9
    set_u(0, 1, 8, 1)
    set_v(1, 2, 8, 1)
    set_w(2, 0, 8, 1)

    # 10. A10 B02 -> M10
    set_u(1, 0, 9, 1)
    set_v(0, 2, 9, 1)
    set_w(2, 1, 9, 1)

    # 11. A11 B12 -> M11
    set_u(1, 1, 10, 1)
    set_v(1, 2, 10, 1)
    set_w(2, 1, 10, 1)

    return {"U": U, "V": V, "W": W}


def _search_rank10(start_time, deadline):
    """Try to find a rank-10 decomposition using structured parameterization."""
    n, m, p = DIMENSIONS
    rank = 10
    
    # The multiplication tensor T[i,j,k] = 1 if i+k corresponds to proper output
    # For C = A*B where A is 2x2, B is 2x3:
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all valid (i,j,k) where i in [0,2), j in [0,2), k in [0,3)
    
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
    # where i in range(n), j in range(m), k in range(p)
    
    # This is equivalent to finding rank-10 factors
    
    # Strategy: Use the known rank-11 structure and try to merge two terms
    # The four "corner" terms (8-11) are simple A[i,j]*B[j,k] terms for k=2
    # These might be mergeable with other terms
    
    # Let's try a parameterized approach where we fix most structure and optimize
    
    # Based on the rank-11 solution, the last 4 terms handle the k=2 column of B
    # Terms 1-7 handle k=0,1 columns with bilinear combinations
    
    # Hypothesis: We can express some of the k=2 contributions using the existing
    # bilinear structure by extending the U and V factors
    
    # Let's set up a least-squares problem
    # Variables: U (4x10), V (6x10), W (6x10) = 100 variables
    # Constraints: For each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    # This is a tensor decomposition problem. Let's use a structured initialization
    # based on the rank-11 solution with one dimension removed
    
    # Initialize from rank-11 by dropping one term and adjusting
    U11 = np.array([[0]*11 for _ in range(4)], dtype=float)
    V11 = np.array([[0]*11 for _ in range(6)], dtype=float)
    W11 = np.array([[0]*11 for _ in range(6)], dtype=float)
    
    sol11 = _get_rank11_solution()
    for i in range(4):
        for r in range(11):
            U11[i,r] = sol11['U'][i][r]
    for i in range(6):
        for r in range(11):
            V11[i,r] = sol11['V'][i][r]
    for i in range(6):
        for r in range(11):
            W11[i,r] = sol11['W'][i][r]
    
    # Try different strategies to reduce from 11 to 10
    best_solution = None
    best_obj = float('inf')
    
    # Strategy 1: Try to merge the two A00*B02 and A01*B12 terms (both contribute to C02)
    # and A10*B02 and A11*B12 (both contribute to C12)
    # by absorbing them into existing bilinear terms
    
    # Let's parameterize: keep terms 1-7 as is, but allow terms 8-11 to be modified
    # and allow small perturbations to all terms
    
    for attempt in range(5):
        if time.time() - start_time > deadline - 10:
            break
        
        # Different initializations
        if attempt == 0:
            # Drop term 11 (A11*B12) and adjust
            U0 = U11[:, :10].copy()
            V0 = V11[:, :10].copy()
            W0 = W11[:, :10].copy()
            # Add contribution of term 11 distributed among other terms
            U0 += 0.1 * U11[:, 10:11]
            V0 += 0.1 * V11[:, 10:11]
            W0 += 0.1 * W11[:, 10:11]
        elif attempt == 1:
            # Drop term 10 (A10*B02)
            U0 = U11[:, :9].copy()
            V0 = V11[:, :9].copy()
            W0 = W11[:, :9].copy()
            U0 = np.column_stack([U0, U11[:, 10]])
            V0 = np.column_stack([V0, V11[:, 10]])
            W0 = np.column_stack([W0, W11[:, 10]])
            # Perturb
            U0 += 0.05 * np.random.randn(4, 10)
            V0 += 0.05 * np.random.randn(6, 10)
            W0 += 0.05 * np.random.randn(6, 10)
        elif attempt == 2:
            # Merge terms 8 and 9 (both for C02)
            U0 = U11[:, :8].copy()
            V0 = V11[:, :8].copy()
            W0 = W11[:, :8].copy()
            # New term 8: A00*B02 + A01*B12
            U0[:, 8] = [1, 0, 0, 0]  # A00
            U0[:, 9] = [0, 1, 0, 0]  # A01
            V0[:, 8] = [0, 0, 1, 0, 0, 0]  # B02
            V0[:, 9] = [0, 0, 0, 0, 1, 0]  # B12
            W0[:, 8] = [0, 0, 1, 0, 0, 0]  # C02
            W0[:, 9] = [0, 0, 1, 0, 0, 0]  # C02
            # But we still need C12 from terms 10,11
            # This won't work directly - we need 4 separate terms for the 4 k=2 entries
            # Unless we can share...
            U0 += 0.05 * np.random.randn(4, 10)
            V0 += 0.05 * np.random.randn(6, 10)
            W0 += 0.05 * np.random.randn(6, 10)
        elif attempt == 3:
            # Random perturbation of full rank-11, then project to rank-10
            U0 = U11[:, :10].copy() + 0.1 * np.random.randn(4, 10)
            V0 = V11[:, :10].copy() + 0.1 * np.random.randn(6, 10)
            W0 = W11[:, :10].copy() + 0.1 * np.random.randn(6, 10)
        else:
            # Try a completely different structure
            # Use Strassen-like approach for the 2x2 part, handle k=2 separately
            U0 = np.zeros((4, 10))
            V0 = np.zeros((6, 10))
            W0 = np.zeros((6, 10))
            
            # 7 terms for the 2x2 x 2x2 part (k=0,1)
            # 3 terms for k=2 column
            # Total: 10
            
            # Term 0: (A00+A11)(B00+B11)
            U0[0,0] = 1; U0[3,0] = 1
            V0[0,0] = 1; V0[3,0] = 1
            W0[0,0] = 1; W0[5,0] = 1
            
            # Term 1: (A10+A11)B00
            U0[2,1] = 1; U0[3,1] = 1
            V0[0,1] = 1
            W0[2,1] = 1; W0[5,1] = -1
            
            # Term 2: A00(B01-B11)
            U0[0,2] = 1
            V0[1,2] = 1; V0[3,2] = -1
            W0[1,2] = 1; W0[5,2] = 1
            
            # Term 3: A11(B10-B00)
            U0[3,3] = 1
            V0[2,3] = 1; V0[0,3] = -1
            W0[0,3] = 1; W0[2,3] = 1
            
            # Term 4: (A00+A01)B11
            U0[0,4] = 1; U0[1,4] = 1
            V0[3,4] = 1
            W0[0,4] = -1; W0[1,4] = 1
            
            # Term 5: (A10-A00)(B00+B01)
            U0[2,5] = 1; U0[0,5] = -1
            V0[0,5] = 1; V0[1,5] = 1
            W0[5,5] = 1
            
            # Term 6: (A01-A11)(B10+B11)
            U0[1,6] = 1; U0[3,6] = -1
            V0[2,6] = 1; V0[3,6] = 1
            W0[0,6] = 1
            
            # Terms 7-9: For k=2 column
            # C02 = A00*B02 + A01*B12
            # C12 = A10*B02 + A11*B12
            # We need 4 independent products but only have 3 slots
            # Try: Term 7: A00*B02, Term 8: A01*B12, Term 9: (A10+A11)*B02 or similar
            
            U0[0,7] = 1  # A00
            V0[2,7] = 1  # B02
            W0[4,7] = 1  # C02 (index 2*2+0=4)
            
            U0[1,8] = 1  # A01
            V0[5,8] = 1  # B12
            W0[4,8] = 1  # C02
            
            # For C12: need A10*B02 + A11*B12
            # Can we do it in one term? No, it's a sum of two products
            # Unless we use a bilinear trick
            U0[2,9] = 1; U0[3,9] = 1  # A10+A11
            V0[2,9] = 1  # B02
            W0[5,9] = 1  # C12 (index 2*2+1=5)
            # But this gives (A10+A11)*B02, not A10*B02+A11*B12
            # We're missing A11*B12 and have extra A10*B02... 
            # This doesn't work. Let's just optimize from here.
            
            U0 += 0.05 * np.random.randn(4, 10)
            V0 += 0.05 * np.random.randn(6, 10)
            W0 += 0.05 * np.random.randn(6, 10)
        
        # Optimize
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        def objective(x):
            Ux = x[:40].reshape(4, 10)
            Vx = x[40:100].reshape(6, 10)
            Wx = x[100:160].reshape(6, 10)
            
            # Compute tensor and compare to target
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(rank):
                            val += Ux[i*m+j, r] * Vx[j*p+k, r] * Wx[k*n+i, r]
                        err += (val - 1.0) ** 2
            return err
        
        # Try optimization
        try:
            res = minimize(objective, x0, method='BFGS', 
                          options={'maxiter': 100, 'gtol': 1e-8})
            
            if res.fun < best_obj:
                best_obj = res.fun
                if res.fun < 1e-10:
                    Ux = res.x[:40].reshape(4, 10)
                    Vx = res.x[40:100].reshape(6, 10)
                    Wx = res.x[100:160].reshape(6, 10)
                    
                    # Round to nearest half-integer
                    Ux = np.round(Ux * 2) / 2
                    Vx = np.round(Vx * 2) / 2
                    Wx = np.round(Wx * 2) / 2
                    
                    # Verify
                    if _verify_decomposition(Ux, Vx, Wx):
                        return _format_solution(Ux, Vx, Wx)
                    
                    # If rounding broke it, try without rounding
                    if _verify_decomposition(res.x[:40].reshape(4, 10), 
                                            res.x[40:100].reshape(6, 10), 
                                            res.x[100:160].reshape(6, 10)):
                        return _format_solution(res.x[:40].reshape(4, 10), 
                                              res.x[40:100].reshape(6, 10), 
                                              res.x[100:160].reshape(6, 10))
                
                best_solution = (res.x[:40].reshape(4, 10), 
                                res.x[40:100].reshape(6, 10), 
                                res.x[100:160].reshape(6, 10))
        except Exception:
            pass
    
    return None


def _verify_decomposition(U, V, W):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    n, m, p = DIMENSIONS
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(U.shape[1]):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-6:
                    return False
    return True


def _format_solution(U, V, W):
    """Format the solution as the required dictionary with list-of-lists."""
    return {
        "U": [[U[i, r] for r in range(U.shape[1])] for i in range(U.shape[0])],
        "V": [[V[i, r] for r in range(V.shape[1])] for i in range(V.shape[0])],
        "W": [[W[i, r] for r in range(W.shape[1])] for i in range(W.shape[0])]
    }
# EVOLVE-BLOCK-END
