"""Tensor flattening + sparse LP-inspired greedy for matrix multiplication rank minimization."""
import numpy as np
from scipy.optimize import linprog
from itertools import product
import random

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    total = nm * mp * pn
    
    # Target rank: start with schoolbook, try to reduce
    # Theoretical lower bound for 3x5 * 5x6 is at least max(nm, mp, pn) = 30
    # But actual tensor rank is higher. Let's aim for rank < 90.
    
    def schoolbook():
        r = n * m * p
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    def check_decomposition(U, V, W):
        """Check if U, V, W form a valid decomposition with integer/half-integer entries."""
        # Check entries are integer or half-integer
        for M in [U, V, W]:
            M2 = 2 * M
            Mround = np.round(M2)
            if not np.allclose(M2, Mround, atol=1e-8):
                return False
        # Check tensor equality
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Tensor flattening: T[(i*m+j)*(m*p) + (j*p+k)]... 
        # Actually, the tensor is indexed by (i, j, k) where i in [0,n), j in [0,m), k in [0,p)
        # Flattened: idx = (i * m + j) * p + k? No, the convention is specific.
        # Let's use the standard: T[i,j,k] = (i*m+j, j*p+k, k*n+i)
        # Actually the output factor W has index W[k*n+i, r], so the third mode is (k, i)
        # So T is a 3-tensor with modes (i*m+j), (j*p+k), (k*n+i)? 
        # No, matrix mult tensor: T[a, b, c] where a indexes (i,j), b indexes (j,k), c indexes (k,i)
        # But j must match between a and b, k must match between b and c, i must match between c and a.
        # So T is not a full tensor on [nm]x[mp]x[pn], it's sparse.
        # The decomposition requires: for all valid (i,j,k): T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
        # And for invalid tuples, the product should be 0 (but they're not part of the tensor).
        # Actually, the tensor is defined only on the "fiber" where indices are consistent.
        # Let's verify by computing sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] for each (i,j,k) and checking it equals 1.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(U.shape[1]):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if not np.isclose(val, 1.0, atol=1e-8):
                        return False
        # Also need to check that "illegal" combinations give 0? 
        # The problem says "exactly decompose the matrix multiplication tensor".
        # The tensor T has T[i,j,k] = 1 if indices are consistent, 0 otherwise.
        # But in the flattened space, entries where j doesn't match, etc., are 0.
        # We need: for all a in [nm], b in [mp], c in [pn]:
        #   sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
        # where T[a,b,c] = 1 if a=(i*m+j), b=(j'*p+k), c=(k'*n+i') with j=j', k=k', i=i', else 0.
        for a in range(nm):
            i_a = a // m
            j_a = a % m
            for b in range(mp):
                j_b = b // p
                k_b = b % p
                for c in range(pn):
                    k_c = c // n
                    i_c = c % n
                    target = 1.0 if (j_a == j_b and k_b == k_c and i_c == i_a) else 0.0
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[a, r] * V[b, r] * W[c, r]
                    if not np.isclose(val, target, atol=1e-8):
                        return False
        return True
    
    # Start with schoolbook
    U, V, W = schoolbook()
    current_rank = U.shape[1]
    
    # Try to reduce rank using greedy tensor flattening + LP
    # Tensor flattening: view T as a vector of length nm*mp*pn
    # T_flat[idx] = T[a,b,c] where idx = a*mp*pn + b*pn + c
    T_flat = np.zeros(total)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                idx = a * mp * pn + b * pn + c
                T_flat[idx] = 1.0
    
    # Try rank reduction from current_rank - 1 down
    for target_rank in range(current_rank - 1, max(nm, mp, pn) - 1, -1):
        # Use greedy: fix U, V randomly sparse, solve for W
        # Or use the LP idea: for each column, we want u_c^T (V_c ⊗ W_c) = T_flat
        # This is hard. Let's use ALS with random restarts and rational rounding.
        
        best_found = False
        for attempt in range(5):
            random.seed(attempt)
            np.random.seed(attempt)
            
            U_try = np.zeros((nm, target_rank))
            V_try = np.zeros((mp, target_rank))
            W_try = np.zeros((pn, target_rank))
            
            # Sparse initialization
            for r in range(target_rank):
                # Pick random nonzero positions
                for _ in range(random.randint(1, 3)):
                    U_try[random.randint(0, nm-1), r] = random.choice([1, -1])
                for _ in range(random.randint(1, 3)):
                    V_try[random.randint(0, mp-1), r] = random.choice([1, -1])
                for _ in range(random.randint(1, 3)):
                    W_try[random.randint(0, pn-1), r] = random.choice([1, -1])
            
            # ALS iterations
            for it in range(50):
                # Solve for U: fix V, W
                # For each a: sum_r U[a,r] * (V[:,r] ⊙ W[:,r] at (b,c)) = T[a,b,c]
                # Build linear system: for each a, variables are U[a,0..r-1]
                # A_a is mp*pn x target_rank
                solved_U = True
                for a in range(nm):
                    # Build A: for each r, A[b,c,r] = V[b,r] * W[c,r]
                    A = np.zeros((mp * pn, target_rank))
                    for b in range(mp):
                        for c in range(pn):
                            for r in range(target_rank):
                                A[b * pn + c, r] = V_try[b, r] * W_try[c, r]
                    b_vec = T_flat[a * mp * pn : (a+1) * mp * pn]
                    # Solve least squares
                    try:
                        sol, residuals, rank_A, sv = np.linalg.lstsq(A, b_vec, rcond=None)
                        U_try[a, :] = sol
                    except:
                        solved_U = False
                        break
                
                if not solved_U:
                    break
                
                # Similarly for V and W
                solved_V = True
                for b in range(mp):
                    A = np.zeros((nm * pn, target_rank))
                    for a in range(nm):
                        for c in range(pn):
                            for r in range(target_rank):
                                A[a * pn + c, r] = U_try[a, r] * W_try[c, r]
                    b_vec = T_flat.reshape(nm, mp, pn)[:, b, :].flatten()
                    try:
                        sol, residuals, rank_A, sv = np.linalg.lstsq(A, b_vec, rcond=None)
                        V_try[b, :] = sol
                    except:
                        solved_V = False
                        break
                if not solved_V:
                    break
                
                solved_W = True
                for c in range(pn):
                    A = np.zeros((nm * mp, target_rank))
                    for a in range(nm):
                        for b in range(mp):
                            for r in range(target_rank):
                                A[a * mp + b, r] = U_try[a, r] * V_try[b, r]
                    b_vec = T_flat.reshape(nm, mp, pn)[:, :, c].flatten()
                    try:
                        sol, residuals, rank_A, sv = np.linalg.lstsq(A, b_vec, rcond=None)
                        W_try[c, :] = sol
                    except:
                        solved_W = False
                        break
                if not solved_W:
                    break
            
            # Check if real-valued solution is close to integer/half-integer
            U2 = 2 * U_try
            V2 = 2 * V_try
            W2 = 2 * W_try
            if (np.allclose(U2, np.round(U2), atol=1e-6) and
                np.allclose(V2, np.round(V2), atol=1e-6) and
                np.allclose(W2, np.round(W2), atol=1e-6)):
                U_int = np.round(U2) / 2.0
                V_int = np.round(V2) / 2.0
                W_int = np.round(W2) / 2.0
                if check_decomposition(U_int, V_int, W_int):
                    U, V, W = U_int, V_int, W_int
                    current_rank = target_rank
                    best_found = True
                    break
        
        if best_found:
            break
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
