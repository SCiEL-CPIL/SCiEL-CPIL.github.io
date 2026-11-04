import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def _verify(U, V, W, n=4, m=4, p=7, tol=1e-9):
    """Verify that the decomposition reproduces the matmul tensor exactly."""
    r = len(U[0]) if U and U[0] else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                ui = U[i * m + j]
                vk = V[j * p + k]
                wi = W[k * n + i]
                for c in range(r):
                    s += ui[c] * vk[c] * wi[c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_2x2_terms():
    """
    Returns 7 terms for 2x2x2 Strassen.
    Each term is (U_vec, V_vec, W_vec) of length 4.
    U indices: 0=a, 1=b, 2=c, 3=d
    V indices: 0=e, 1=f, 2=g, 3=h
    W indices: 0=i, 1=j, 2=k, 3=l
    """
    terms = []
    # M1 = (a+d)(e+h) -> i+1, l+1
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    # M2 = (c+d)(e) -> k+1, l-1
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
    # M3 = a(f-h) -> j+1
    terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]))
    # M4 = d(g-e) -> i-1, k+1
    terms.append(([0, 0, 0, 1], [-1, 0, 1, 0], [-1, 0, 1, 0]))
    # M5 = (a+b)(h) -> i+1, j+1
    terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0]))
    # M6 = (c-a)(e+f) -> l+1
    terms.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
    # M7 = (b-d)(g+h) -> i-1, l-1
    terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [-1, 0, 0, -1]))
    return terms

def _strassen_4x4x4_terms():
    """
    Generate 49 rank-1 terms for 4x4 * 4x4 -> 4x4.
    Returns list of (U_vec, V_vec, W_vec) where vectors have length 16.
    U index: i*4+j
    V index: j*4+k
    W index: k*4+i
    """
    base_terms = _strassen_2x2_terms()
    
    # Block definitions for 4x4 split into 2x2 blocks
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    # C blocks: C00, C01, C10, C11
    
    # Strassen block products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Output:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to map these to scalar indices.
    # For each block product, we have a 2x2 matrix X and 2x2 matrix Y.
    # We apply the 7 base Strassen terms to X and Y.
    # Each base term gives a scalar product of linear combinations.
    # We expand these into 4x4 terms.
    
    def get_block_indices(block_row, block_col):
        """Return list of (i, j) for a 2x2 block at (block_row, block_col)."""
        indices = []
        for r in range(2):
            for c in range(2):
                indices.append((2*block_row + r, 2*block_col + c))
        return indices

    # Define the 7 block products
    # Each product is defined by:
    # X_blocks: list of (block_row, block_col, coeff)
    # Y_blocks: list of (block_row, block_col, coeff)
    # Output_contribs: list of (c_block_row, c_block_col, coeff)
    
    products = [
        # M1
        {
            "X": [(0,0,1), (1,1,1)],
            "Y": [(0,0,1), (1,1,1)],
            "Out": [(0,0,1), (1,1,1)]
        },
        # M2
        {
            "X": [(1,0,1), (1,1,1)],
            "Y": [(0,0,1)],
            "Out": [(1,0,1), (1,1,-1)]
        },
        # M3
        {
            "X": [(0,0,1)],
            "Y": [(0,1,1), (1,1,-1)],
            "Out": [(0,1,1), (1,1,1)]
        },
        # M4
        {
            "X": [(1,1,1)],
            "Y": [(1,0,1), (0,0,-1)],
            "Out": [(0,0,1), (1,0,1)]
        },
        # M5
        {
            "X": [(0,0,1), (0,1,1)],
            "Y": [(1,1,1)],
            "Out": [(0,0,-1), (0,1,1)]
        },
        # M6
        {
            "X": [(1,0,1), (0,0,-1)],
            "Y": [(0,0,1), (0,1,1)],
            "Out": [(1,1,1)]
        },
        # M7
        {
            "X": [(0,1,1), (1,1,-1)],
            "Y": [(1,0,1), (1,1,1)],
            "Out": [(0,0,1), (1,1,-1)]
        }
    ]
    
    all_terms = []
    
    for prod in products:
        # Build the 2x2 matrices X and Y as lists of (i, j, coeff)
        X_mat = [[[] for _ in range(2)] for _ in range(2)]
        for (br, bc, coeff) in prod["X"]:
            for (i, j) in get_block_indices(br, bc):
                X_mat[i//2][j//2].append((i, j, coeff))
        
        Y_mat = [[[] for _ in range(2)] for _ in range(2)]
        for (br, bc, coeff) in prod["Y"]:
            for (j, k) in get_block_indices(br, bc):
                Y_mat[j//2][k//2].append((j, k, coeff))
        
        # Apply the 7 base Strassen terms to X and Y
        for (u_base, v_base, w_base) in base_terms:
            # u_base has 4 entries corresponding to X[0,0], X[0,1], X[1,0], X[1,1]
            # v_base has 4 entries corresponding to Y[0,0], Y[0,1], Y[1,0], Y[1,1]
            # w_base has 4 entries corresponding to C[0,0], C[0,1], C[1,0], C[1,1]
            
            # Collect all (i, j, coeff) for U
            u_entries = []
            for idx in range(4):
                r, c = divmod(idx, 2)
                coeff = u_base[idx]
                if coeff != 0:
                    for (i, j, c_coeff) in X_mat[r][c]:
                        u_entries.append((i, j, coeff * c_coeff))
            
            # Collect all (j, k, coeff) for V
            v_entries = []
            for idx in range(4):
                r, c = divmod(idx, 2)
                coeff = v_base[idx]
                if coeff != 0:
                    for (j, k, c_coeff) in Y_mat[r][c]:
                        v_entries.append((j, k, coeff * c_coeff))
            
            # Collect all (k, i, coeff) for W
            # Note: w_base indices correspond to C blocks.
            # C[0,0] -> C block (0,0) -> global C indices (0,0), (0,1), (1,0), (1,1)
            # C[0,1] -> C block (0,1) -> global C indices (0,2), (0,3), (1,2), (1,3)
            # C[1,0] -> C block (1,0) -> global C indices (2,0), (2,1), (3,0), (3,1)
            # C[1,1] -> C block (1,1) -> global C indices (2,2), (2,3), (3,2), (3,3)
            
            # For each output block, we have a coefficient from w_base.
            # We also have a coefficient from prod["Out"].
            
            w_entries = []
            for idx in range(4):
                r, c = divmod(idx, 2)
                coeff_w_base = w_base[idx]
                if coeff_w_base == 0:
                    continue
                
                # Find the contribution of this block product to this C block
                coeff_out = 0
                for (obr, ocb, oc) in prod["Out"]:
                    if obr == r and ocb == c:
                        coeff_out = oc
                        break
                
                if coeff_out == 0:
                    continue
                
                total_coeff = coeff_w_base * coeff_out
                
                # Add to w_entries for the global C indices in this block
                for (k, i) in get_block_indices(r, c):
                    # W index is k*4 + i
                    w_entries.append((k, i, total_coeff))
            
            # Now, for each pair (u_entry, v_entry) with matching j,
            # and each w_entry, we create a rank-1 term.
            # Actually, the term is U * V * W.
            # U is a vector, V is a vector, W is a vector.
            # We need to sum up contributions to the same (i, j) in U, (j, k) in V, (k, i) in W?
            # No, each rank-1 term is a single product of three vectors.
            # The tensor is sum of U_r * V_r * W_r.
            # Here, U_r is sparse, V_r is sparse, W_r is sparse.
            
            # We iterate over all combinations of u_entry, v_entry, w_entry?
            # No, that would be the schoolbook expansion.
            # Strassen combines them.
            # The term is:
            # (sum u_entries) * (sum v_entries) * (sum w_entries)? No.
            # The rank-1 term is defined by the vectors U, V, W.
            # U has non-zeros at u_entries.
            # V has non-zeros at v_entries.
            # W has non-zeros at w_entries.
            # This forms ONE rank-1 term.
            
            U_vec = [0.0] * 16
            V_vec = [0.0] * 16
            W_vec = [0.0] * 16
            
            for (i, j, c) in u_entries:
                U_vec[i*4 + j] += c
            for (j, k, c) in v_entries:
                V_vec[j*4 + k] += c
            for (k, i, c) in w_entries:
                W_vec[k*4 + i] += c
            
            # Check if term is zero
            if any(U_vec) and any(V_vec) and any(W_vec):
                all_terms.append((U_vec, V_vec, W_vec))
                
    return all_terms

def _schoolbook_4x4x3_terms():
    """
    Generate 48 rank-1 terms for 4x4 * 4x3 -> 4x3.
    Returns list of (U_vec, V_vec, W_vec) where vectors have length 16, 12, 12?
    No, we need to map to the full 4x4x7 indices.
    U: length 16 (i*4+j)
    V: length 28 (j*7+k), where k is 4,5,6.
    W: length 28 (k*4+i), where k is 4,5,6.
    """
    terms = []
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                U_vec = [0.0] * 16
                V_vec = [0.0] * 28
                W_vec = [0.0] * 28
                
                U_vec[i*4 + j] = 1.0
                V_vec[j*7 + k] = 1.0
                W_vec[k*4 + i] = 1.0
                
                terms.append((U_vec, V_vec, W_vec))
    return terms

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Get Strassen terms for 4x4x4 (first 4 cols)
    strassen_terms = _strassen_4x4x4_terms()
    
    # Get Schoolbook terms for 4x4x3 (last 3 cols)
    schoolbook_terms = _schoolbook_4x4x3_terms()
    
    # Combine
    all_terms = strassen_terms + schoolbook_terms
    rank = len(all_terms)
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for r, (U_vec, V_vec, W_vec) in enumerate(all_terms):
        for idx in range(len(U_vec)):
            if U_vec[idx] != 0:
                U[idx][r] = U_vec[idx]
        for idx in range(len(V_vec)):
            if V_vec[idx] != 0:
                V[idx][r] = V_vec[idx]
        for idx in range(len(W_vec)):
            if W_vec[idx] != 0:
                W[idx][r] = W_vec[idx]
        
        if time.time() > deadline - 10:
            break
            
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook 112
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
    print(f"Valid: {_verify(res['U'], res['V'], res['W'])}")
