import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 2, 11]
TIME_LIMIT = 140  # Safety margin before 150s kill

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    n, m, p = DIMENSIONS
    # n*m = 4, m*p = 22, p*n = 22
    
    # Build the multiplication tensor as a set of equations.
    # The tensor T has entries T[a_idx, b_idx, c_idx] = 1 if and only if
    # the product term A[a_i, a_j] * B[a_j, b_k] contributes to C[a_i, b_k].
    # For matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j, k] contributes when the term is A[i,j]*B[j,k] -> C[i,k]
    # In the decomposition: sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] = T[i,j,k]
    
    # Enumerate all non-zero tensor entries
    # T[i,j,k] = 1 when the product A[i,j]*B[j,k] contributes to C[i,k]
    # For i in [0,1], j in [0,1], k in [0,10]:
    #   The term A[i,j]*B[j,k] contributes to C[i,k]
    # So T[i, j, i*k + 0*... ] -- wait, let me re-examine the convention.
    
    # The convention is W[k*n+i, r] where the output C has dimensions n x p.
    # C[i, k] for i in [0, n-1], k in [0, p-1].
    # W index: k*n + i (column-major in some sense: k is the outer loop, i is inner)
    
    # Let me re-derive: The tensor T is of shape (n*m, m*p, p*n) = (4, 22, 22).
    # U has shape (n*m, rank) = (4, rank), indexed by (i, j) -> index i*m + j = i*2 + j
    # V has shape (m*p, rank) = (22, rank), indexed by (j, k) -> index j*p + k = j*11 + k
    # W has shape (p*n, rank) = (22, rank), indexed by (k, i) -> index k*n + i = k*2 + i
    
    # The multiplication tensor: T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # where the term A[i,j]*B[j,k] contributes to C[i,k].
    
    # So for each (i, j, k), we have one equation:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for all other (a, b, c) combinations, the sum must be 0.
    
    # Actually, the tensor is defined as:
    # T[a, b, c] = 1 if a = i*m+j, b = j'*p+k, c = k'*n+i' where j=j', k=k', i=i'
    # More precisely: T[i*m+j, j*p+k, k*n+i] = 1
    
    # Let's build the list of non-zero tensor entries
    tensor_entries = {}  # (a, b, c) -> 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor_entries[(a, b, c)] = 1
    
    # Total number of non-zero entries: n * m * p = 2*2*11 = 44
    # Each rank-1 term U[:,r] * V[:,r] * W[:,r] can match multiple tensor entries.
    
    # For a backtracking search, we need to find rank r such that:
    # sum_{t=0}^{r-1} U[a,t]*V[b,t]*W[c,t] = T[a,b,c] for all (a,b,c)
    
    # This is a very hard problem. Let me try a different approach.
    
    # The known rank-39 construction uses Strassen for pairs of columns.
    # Can we do better? The lower bound for rank of (2,2,11) matrix multiplication...
    # For (n,m,p) matrix multiplication, the rank is at least max(nm, mp, np) / something.
    # Actually, the rank of (n,m,p) tensor is at least min(nm*p, ...) but that's not helpful.
    # Known: rank of (2,2,2) is 7 (Strassen).
    # For (2,2,p), we can use Strassen on each pair of columns: ceil(p/2)*7 + (p mod 2)*4
    # For p=11: 5*7 + 1*4 = 35 + 4 = 39. That's what we have.
    
    # Can we share factors between adjacent pairs? This is the key question.
    # If we could, we might get rank < 39.
    
    # Let me try a search approach: assume rank r, and try to find U, V, W with
    # small integer coefficients.
    
    # Strategy: For each tensor entry (a,b,c), we need:
    # sum_t U[a,t]*V[b,t]*W[c,t] = 1
    # For all other (a',b',c') not in tensor_entries, sum = 0.
    
    # This is a system of bilinear equations, which is NP-hard in general.
    # But with small rank and small coefficients, we might be able to search.
    
    # Let me try rank = 38 and see if a solution exists with coefficients in {-1, 0, 1, 2, -2}.
    
    # Fall back to the known rank-39 construction
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # 5 pairs of columns using Strassen's algorithm
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        v_e = k0
        v_f = k1
        v_g = 11 + k0
        v_h = 11 + k1
        
        w_c00 = 2 * k0
        w_c10 = 2 * k0 + 1
        w_c01 = 2 * k1
        w_c11 = 2 * k1 + 1
        
        # m1 = (a+d)(e+h)
        U[0][r] = 1; U[3][r] = 1
        V[v_e][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1; W[w_c11][r] = 1
        r += 1
        
        # m2 = (c+d)e
        U[2][r] = 1; U[3][r] = 1
        V[v_e][r] = 1
        W[w_c10][r] = 1; W[w_c11][r] = -1
        r += 1
        
        # m3 = a(f-h)
        U[0][r] = 1
        V[v_f][r] = 1; V[v_h][r] = -1
        W[w_c01][r] = 1; W[w_c11][r] = 1
        r += 1
        
        # m4 = d(g-e)
        U[3][r] = 1
        V[v_g][r] = 1; V[v_e][r] = -1
        W[w_c00][r] = 1; W[w_c10][r] = 1
        r += 1
        
        # m5 = (a+b)h
        U[0][r] = 1; U[1][r] = 1
        V[v_h][r] = 1
        W[w_c00][r] = -1; W[w_c01][r] = 1
        r += 1
        
        # m6 = (c-a)(e+f)
        U[2][r] = 1; U[0][r] = -1
        V[v_e][r] = 1; V[v_f][r] = 1
        W[w_c11][r] = 1
        r += 1
        
        # m7 = (b-d)(g+h)
        U[1][r] = 1; U[3][r] = -1
        V[v_g][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1
        r += 1

    # Single column k2 = 10 using schoolbook
    k2 = 10
    v_e2 = k2
    v_g2 = 11 + k2
    w_c02 = 2 * k2
    w_c12 = 2 * k2 + 1
    
    U[0][r] = 1
    V[v_e2][r] = 1
    W[w_c02][r] = 1
    r += 1
    
    U[1][r] = 1
    V[v_g2][r] = 1
    W[w_c02][r] = 1
    r += 1
    
    U[2][r] = 1
    V[v_e2][r] = 1
    W[w_c12][r] = 1
    r += 1
    
    U[3][r] = 1
    V[v_g2][r] = 1
    W[w_c12][r] = 1
    r += 1
    
    return {"U": U, "V": V, "W": W}
