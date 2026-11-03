import numpy as np
import time
import random

n, m, p = 3, 4, 6

def Aidx(i, j): return i * m + j
def Bidx(j, k): return j * p + k
def Cidx(i, k): return k * n + i  # W[k*n+i]

def build_decomposition(i_parts, j_parts, k_parts):
    """Build decomposition from block partitions."""
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_rank1_terms(sub_U, sub_V, sub_W):
        """Add rank-1 terms for a sub-block. sub_U is list of (i,j) positions, etc."""
        pass  # Placeholder
    
    # This approach is complex; let me use the explicit construction instead
    pass

def baseline_66():
    U = [[0]*66 for _ in range(n*m)]
    V = [[0]*66 for _ in range(m*p)]
    W = [[0]*66 for _ in range(p*n)]
    col = 0

    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A00 = Aidx(a_r, a_c); A01 = Aidx(a_r, a_c+1)
        A10 = Aidx(a_r+1, a_c); A11 = Aidx(a_r+1, a_c+1)
        B00 = Bidx(b_r, b_c); B01 = Bidx(b_r, b_c+1)
        B10 = Bidx(b_r+1, b_c); B11 = Bidx(b_r+1, b_c+1)
        C00 = Cidx(c_r, c_c); C01 = Cidx(c_r, c_c+1)
        C10 = Cidx(c_r+1, c_c); C11 = Cidx(c_r+1, c_c+1)
        # M1
        U[A00][col]+=1; U[A11][col]+=1; V[B00][col]+=1; V[B11][col]+=1; W[C00][col]+=1; W[C11][col]+=1; col+=1
        # M2
        U[A10][col]+=1; U[A11][col]+=1; V[B00][col]+=1; W[C10][col]+=1; W[C11][col]-=1; col+=1
        # M3
        U[A00][col]+=1; V[B01][col]+=1; V[B11][col]-=1; W[C01][col]+=1; W[C11][col]+=1; col+=1
        # M4
        U[A11][col]+=1; V[B10][col]+=1; V[B00][col]-=1; W[C00][col]+=1; W[C10][col]+=1; col+=1
        # M5
        U[A00][col]+=1; U[A01][col]+=1; V[B11][col]+=1; W[C00][col]-=1; W[C01][col]+=1; col+=1
        # M6
        U[A10][col]+=1; U[A00][col]-=1; V[B00][col]+=1; V[B01][col]+=1; W[C11][col]+=1; col+=1
        # M7
        U[A01][col]+=1; U[A11][col]-=1; V[B10][col]+=1; V[B11][col]+=1; W[C00][col]+=1; col+=1

    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A0 = Aidx(a_r, a_c); A1 = Aidx(a_r, a_c+1)
        B00 = Bidx(b_r, b_c); B01 = Bidx(b_r, b_c+1)
        B10 = Bidx(b_r+1, b_c); B11 = Bidx(b_r+1, b_c+1)
        C0 = Cidx(c_r, c_c); C1 = Cidx(c_r, c_c+1)
        U[A0][col]+=1; V[B00][col]+=1; W[C0][col]+=1; col+=1
        U[A1][col]+=1; V[B10][col]+=1; W[C0][col]+=1; col+=1
        U[A0][col]+=1; V[B01][col]+=1; W[C1][col]+=1; col+=1
        U[A1][col]+=1; V[B11][col]+=1; W[C1][col]+=1; col+=1

    add_1x2_2x2(0,0,0,0,0,0); add_1x2_2x2(0,2,2,0,0,0)
    add_1x2_2x2(0,0,0,2,0,2); add_1x2_2x2(0,2,2,2,0,2)
    add_1x2_2x2(0,0,0,4,0,4); add_1x2_2x2(0,2,2,4,0,4)
    add_strassen(1,0,0,0,1,0); add_strassen(1,2,2,0,1,0)
    add_strassen(1,0,0,2,1,2); add_strassen(1,2,2,2,1,2)
    add_strassen(1,0,0,4,1,4); add_strassen(1,2,2,4,1,4)
    assert col == 66
    return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)

def verify_np(U, V, W):
    r = U.shape[1]
    U_r = U.reshape(n, m, r)
    V_r = V.reshape(m, p, r)
    W_r = W.reshape(p, n, r)
    result = np.einsum('ijl,jkl,kil->ijk', U_r, V_r, W_r)
    return np.allclose(result, 1.0, atol=1e-9)

def verify_np_fast(U, V, W):
    """Faster verification using batched matrix product."""
    r = U.shape[1]
    U_r = U.reshape(n, m, r)
    V_r = V.reshape(m, p, r)
    W_r = W.reshape(p, n, r)
    # Compute sum over l of U[i,j,l]*V[j,k,l]*W[k,i,l]
    # Use einsum with optimize
    result = np.einsum('ijl,jkl,kil->ijk', U_r, V_r, W_r, optimize=True)
    return np.allclose(result, 1.0, atol=1e-9)

def try_merge(U, V, W, l1, l2):
    """Try to merge two rank-1 terms into one.
    Check if U[:,l1]+U[:,l2], V[:,l1]+V[:,l2], W[:,l1]+W[:,l2] is valid
    when we replace both with their sum as a single term."""
    # The idea: if term l1 and l2 can be combined into a single rank-1 term,
    # then removing both and adding their sum should still be valid.
    # Actually, merging two rank-1 terms into one is only valid if the sum
    # of two rank-1 tensors equals one rank-1 tensor, which is rare.
    # Better approach: check if removing both l1 and l2 and adding (U1+U2, V1+V2, W1+W2) works.
    # But that's just a different term. Let's think differently.
    # 
    # Actually, the correct "merge" is: can we find a new term (u,v,w) such that
    # u⊗v⊗w = u1⊗v1⊗w1 + u2⊗v2⊗w2? This is generally not possible.
    #
    # A more practical approach: try to see if two terms are "redundant" in the sense
    # that their sum can be represented more compactly. For now, let's just try
    # removing one term and checking if the other compensates.
    # 
    # Let's use a different strategy: try to zero out one term by adjusting another.
    pass

def try_drop(U, V, W, l):
    U_new = np.delete(U, l, axis=1)
    V_new = np.delete(V, l, axis=1)
    W_new = np.delete(W, l, axis=1)
    return verify_np_fast(U_new, V_new, W_new)

def try_drop(U, V, W, l):
    U_new = np.delete(U, l, axis=1)
    V_new = np.delete(V, l, axis=1)
    W_new = np.delete(W, l, axis=1)
    return verify_np(U_new, V_new, W_new)

def local_search(U, V, W, deadline, rng):
    r = U.shape[1]
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        indices = list(range(r))
        rng.shuffle(indices)
        for l in indices:
            if time.time() > deadline - 10:
                break
            if try_drop(U, V, W, l):
                U = np.delete(U, l, axis=1)
                V = np.delete(V, l, axis=1)
                W = np.delete(W, l, axis=1)
                r -= 1
                improved = True
                break
    return U, V, W

def local_search_v2(U, V, W, deadline, rng):
    """Enhanced local search with drop and pairwise merge attempts."""
    r = U.shape[1]
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        indices = list(range(r))
        rng.shuffle(indices)
        
        # First pass: try drops
        for l in indices:
            if time.time() > deadline - 10:
                break
            if try_drop(U, V, W, l):
                U = np.delete(U, l, axis=1)
                V = np.delete(V, l, axis=1)
                W = np.delete(W, l, axis=1)
                r -= 1
                improved = True
                break
        
        if improved:
            continue
        
        # Second pass: try merging pairs (replace two terms with their "sum" term)
        # This is valid if: the tensor sum of two rank-1 terms equals one rank-1 term
        # In practice, we check if removing both and adding a combined term works.
        # A simpler heuristic: try to see if two terms can be "absorbed" by adjusting
        # coefficients. Since our entries are integers, let's try:
        # For each pair (i,j), check if U[:,i]+U[:,j], V[:,i]+V[:,j], W[:,i]+W[:,j]
        # gives a valid decomposition when we remove i and j and add this as a new term.
        # But that increases rank by 0 (remove 2, add 1 = net -1) only if the sum is rank-1.
        # The sum of two rank-1 tensors is rank-1 only in special cases.
        #
        # Alternative: try to find if one term is a scalar multiple of another.
        for i in range(r):
            if time.time() > deadline - 10:
                break
            for j in range(i+1, r):
                if time.time() > deadline - 10:
                    break
                # Check if term j is a scalar multiple of term i
                Ui = U[:, i]
                Uj = U[:, j]
                Vi = V[:, i]
                Vj = V[:, j]
                Wi = W[:, i]
                Wj = W[:, j]
                
                # Check if Ui and Uj are proportional
                if np.any(Ui != 0):
                    scale = Uj[Ui != 0][0] / Ui[Ui != 0][0]
                    if np.allclose(Uj, scale * Ui, atol=1e-9) and \
                       np.allclose(Vj, scale * Vi, atol=1e-9) and \
                       np.allclose(Wj, scale * Wi, atol=1e-9):
                        # Term j is a multiple of term i; remove j
                        U = np.delete(U, j, axis=1)
                        V = np.delete(V, j, axis=1)
                        W = np.delete(W, j, axis=1)
                        r -= 1
                        improved = True
                        break
            if improved:
                break
    return U, V, W

def construct():
    deadline = time.time() + 140
    rng = random.Random(42)
    U, V, W = baseline_66()
    U, V, W = local_search_v2(U, V, W, deadline, rng)
    if not verify_np(U, V, W):
        U, V, W = baseline_66()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
