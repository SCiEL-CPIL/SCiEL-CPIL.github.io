import time
import sys
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

# Strassen coefficients for 2x2x2
# M1 = (A11 + A22)(B11 + B22)
# M2 = (A21 + A22)B11
# M3 = A11(B21 - B22)
# M4 = A22(B12 - B11)
# M5 = (A11 + A12)B22
# M6 = (A21 - A11)(B11 + B12)
# M7 = (A12 - A22)(B21 + B22)
# C11 = M1 + M4 - M5 + M7
# C12 = M3 + M5
# C21 = M2 + M4
# C22 = M1 - M2 + M3 + M6

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # We will construct the decomposition by summing contributions from blocks.
    # The final U, V, W will be lists of lists (or numpy arrays, but lists are fine for small sizes).
    # We use a dictionary to accumulate coefficients for each (row, col) in U, V, W.
    # U: shape (n*m, rank)
    # V: shape (m*p, rank)
    # W: shape (p*n, rank)
    
    # Since rank is not known a priori, we will build the factors as a list of "terms".
    # Each term is a tuple (u_dict, v_dict, w_dict) where u_dict maps (i,j) -> coeff, etc.
    # Finally, we will flatten these into matrices.
    
    terms = []
    
    def add_block(n_b, m_b, p_b, i_off, j_off, k_off):
        """
        Adds a block of size n_b x m_b x p_b at offsets (i_off, j_off, k_off).
        """
        if n_b == 0 or m_b == 0 or p_b == 0:
            return
        
        # If the block is 2x2x2, use Strassen
        if n_b == 2 and m_b == 2 and p_b == 2:
            # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
            # A sub-block: A[i_off+i, j_off+j]
            # B sub-block: B[j_off+j, k_off+k]
            # C sub-block: C[k_off+k, i_off+i] (Note the W index convention: k*n+i)
            
            # Define the 7 Strassen multiplications
            # Each multiplication corresponds to a "term" in the decomposition.
            # A term r has:
            # U_r: linear combination of A entries
            # V_r: linear combination of B entries
            # W_r: linear combination of C entries
            
            # M1 = (A11 + A22)(B11 + B22)
            # M2 = (A21 + A22)B11
            # M3 = A11(B21 - B22)
            # M4 = A22(B12 - B11)
            # M5 = (A11 + A12)B22
            # M6 = (A21 - A11)(B11 + B12)
            # M7 = (A12 - A22)(B21 + B22)
            
            # Note: Indices in Strassen are 1-based in formula, 0-based here.
            # A11 = A[0,0], A12 = A[0,1], A21 = A[1,0], A22 = A[1,1]
            # B11 = B[0,0], B12 = B[0,1], B21 = B[1,0], B22 = B[1,1]
            
            strassen_mults = [
                # (U_coeffs, V_coeffs)
                # U is dict: (i,j) -> coeff
                # V is dict: (j,k) -> coeff
                # We need to map local (i,j) to global (i_off+i, j_off+j)
                
                # M1
                ({(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}), # Wait, V is (j,k). B11+B22 is B[0,0]+B[1,1]. So V coeffs: (0,0):1, (1,1):1.
                # Correction: V index is (j,k). B11 is B[0,0] -> (0,0). B22 is B[1,1] -> (1,1).
                # So M1 V: {(0,0): 1, (1,1): 1}
                
                # M2: (A21+A22) B11
                ({(1,0): 1, (1,1): 1}, {(0,0): 1}),
                
                # M3: A11 (B21-B22)
                ({(0,0): 1}, {(1,0): 1, (1,1): -1}),
                
                # M4: A22 (B12-B11)
                ({(1,1): 1}, {(0,1): 1, (0,0): -1}),
                
                # M5: (A11+A12) B22
                ({(0,0): 1, (0,1): 1}, {(1,1): 1}),
                
                # M6: (A21-A11) (B11+B12)
                ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
                
                # M7: (A12-A22) (B21+B22)
                ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1})
            ]
            
            # C contributions
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # W index is (k, i).
            # C11 is at local (k=0, i=0) -> global (k_off+0, i_off+0)
            # C12 is at local (k=0, i=1) -> global (k_off+0, i_off+1)
            # C21 is at local (k=1, i=0) -> global (k_off+1, i_off+0)
            # C22 is at local (k=1, i=1) -> global (k_off+1, i_off+1)
            
            c_contribs = [
                # For each of the 7 mults, which C entries does it affect and with what sign?
                # M1: C11(+1), C22(+1)
                { (0,0): 1, (1,1): 1 },
                # M2: C21(+1), C22(-1)
                { (1,0): 1, (1,1): -1 },
                # M3: C12(+1), C22(+1)
                { (0,1): 1, (1,1): 1 },
                # M4: C11(+1), C21(+1)
                { (0,0): 1, (1,0): 1 },
                # M5: C11(-1), C12(+1)
                { (0,0): -1, (0,1): 1 },
                # M6: C22(+1)
                { (1,1): 1 },
                # M7: C11(+1)
                { (0,0): 1 }
            ]
            
            for u_local, v_local in strassen_mults:
                u_global = {}
                v_global = {}
                w_global = {}
                
                for (li, lj), coeff in u_local.items():
                    gi = i_off + li
                    gj = j_off + lj
                    u_global[(gi, gj)] = coeff
                    
                for (lj, lk), coeff in v_local.items():
                    gj = j_off + lj
                    gk = k_off + lk
                    v_global[(gj, gk)] = coeff
                    
                for (lk, li), coeff in c_contribs[strassen_mults.index((u_local, v_local))].items():
                    gk = k_off + lk
                    gi = i_off + li
                    w_global[(gk, gi)] = coeff
                    
                terms.append((u_global, v_global, w_global))
            return
        
        # Otherwise, split the block
        # Heuristic: Split the largest dimension
        if n_b >= 2:
            # Split n into 2 and n_b-2? Or 1 and n_b-1?
            # For n=3, split into 2 and 1.
            # For n=4, split into 2 and 2.
            if n_b % 2 == 0:
                n1 = n_b // 2
                n2 = n_b - n1
            else:
                n1 = n_b // 2 + 1 # e.g. 3 -> 2
                n2 = n_b - n1     # e.g. 3 -> 1
            
            add_block(n1, m_b, p_b, i_off, j_off, k_off)
            add_block(n2, m_b, p_b, i_off + n1, j_off, k_off)
        elif m_b >= 2:
            if m_b % 2 == 0:
                m1 = m_b // 2
                m2 = m_b - m1
            else:
                m1 = m_b // 2 + 1
                m2 = m_b - m1
            
            add_block(n_b, m1, p_b, i_off, j_off, k_off)
            add_block(n_b, m2, p_b, i_off, j_off + m1, k_off)
        elif p_b >= 2:
            if p_b % 2 == 0:
                p1 = p_b // 2
                p2 = p_b - p1
            else:
                p1 = p_b // 2 + 1
                p2 = p_b - p1
            
            add_block(n_b, m_b, p1, i_off, j_off, k_off)
            add_block(n_b, m_b, p2, i_off, j_off, k_off + p1)
        else:
            # 1x1x1 block: Schoolbook
            # A[i,j] * B[j,k] -> C[k,i]
            u_global = { (i_off, j_off): 1 }
            v_global = { (j_off, k_off): 1 }
            w_global = { (k_off, i_off): 1 }
            terms.append((u_global, v_global, w_global))
            
    # Start recursion
    add_block(n, m, p, 0, 0, 0)
    
    # Flatten terms into U, V, W matrices
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][r] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][r] = coeff
        for (k, i), coeff in w_dict.items():
            W[k * n + i][r] = coeff
            
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
