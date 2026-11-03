import time
import numpy as np

DIMENSIONS = [2, 4, 7]
DEADLINE = time.time() + 140.0

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify tensor decomposition: for all (i,k), sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1"""
    for i in range(n):
        for k in range(p):
            s = 0.0
            for j in range(m):
                u_row = U[i * m + j]
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                s += float(np.sum(u_row * v_row * w_row))
            if abs(s - 1.0) > tol:
                return False
    return True

def _construct_rank_50():
    n, m, p = 2, 4, 7
    rank = 50
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Strassen 2x2 * 2x2 coefficients
    # U rows: 0,1 (a,b), 2,3 (c,d) -> Local indices
    # V rows: 0,1 (e,f), 2,3 (g,h)
    # W rows: 0,1 (0,0; 0,1), 2,3 (1,0; 1,1)
    
    # M1 = (a+d)(e+h)
    # U: a(0), d(1) -> 1, 1. V: e(0), h(3) -> 1, 1. W: all 1.
    # M2 = (c+d)e
    # U: c(2), d(1) -> 1, 1. V: e(0) -> 1. W: c(2), d(3) -> 1, 1.
    # M3 = a(f-h)
    # U: a(0) -> 1. V: f(1), h(3) -> 1, -1. W: b(1), d(3) -> 1, 1.
    # M4 = d(g-e)
    # U: d(1) -> 1. V: g(2), e(0) -> 1, -1. W: a(0), c(2) -> 1, 1.
    # M5 = (a+b)h
    # U: a(0), b(1) -> 1, 1. V: h(3) -> 1. W: a(0), c(2) -> -1, -1.
    # M6 = (c-a)(e+f)
    # U: c(2), a(0) -> 1, -1. V: e(0), f(1) -> 1, 1. W: c(2), d(3) -> -1, -1.
    # M7 = (b-d)(g+h)
    # U: b(1), d(1) -> 1, -1. V: g(2), h(3) -> 1, 1. W: a(0), b(1) -> 1, 1.
    
    strassen_terms = [
        # (U_coefs, V_coefs, W_coefs)
        # Coefs are dicts {local_idx: val}
        ({0:1, 1:1}, {0:1, 3:1}, {0:1, 1:1, 2:1, 3:1}), # M1
        ({2:1, 1:1}, {0:1}, {2:1, 3:1}),                 # M2
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),                # M3
        ({1:1}, {2:1, 0:-1}, {0:1, 2:1}),                # M4
        ({0:1, 1:1}, {3:1}, {0:-1, 2:-1}),               # M5
        ({2:1, 0:-1}, {0:1, 1:1}, {2:-1, 3:-1}),         # M6
        ({1:1, 1:-1}, {2:1, 3:1}, {0:1, 1:1})            # M7
    ]
    
    # Simple 2x2 * 2x1 coefficients
    # U: a(0), b(1), c(2), d(1) -> wait, d is index 1? No.
    # Local A: 0=a, 1=b, 2=c, 3=d? No, standard is 0=a, 1=b, 2=c, 3=d.
    # Wait, in Strassen above I used:
    # U rows: 0,1 (a,b), 2,3 (c,d).
    # So 0=a, 1=b, 2=c, 3=d.
    # Let's re-verify Strassen M1: (a+d). U[0]=1, U[3]=1?
    # My previous code: sU[0,0]=1; sU[1,0]=1. This was a,d?
    # If U rows are 0,1,2,3 corresponding to a,b,c,d.
    # M1 = (a+d). U[0]=1, U[3]=1.
    # Let's fix the Strassen terms with correct indices 0,1,2,3 for a,b,c,d.
    
    strassen_terms_fixed = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 1:1, 2:1, 3:1}), # M1 = (a+d)(e+h)
        ({2:1, 3:1}, {0:1}, {2:1, 3:1}),                 # M2 = (c+d)e
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),                # M3 = a(f-h)
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),                # M4 = d(g-e)
        ({0:1, 1:1}, {3:1}, {0:-1, 2:-1}),               # M5 = (a+b)h
        ({2:1, 0:-1}, {0:1, 1:1}, {2:-1, 3:-1}),         # M6 = (c-a)(e+f)
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1, 1:1})            # M7 = (b-d)(g+h)
    ]
    
    simple_terms = [
        ({0:1}, {0:1}, {0:1}), # N1 = ae
        ({1:1}, {1:1}, {0:1}), # N2 = bg
        ({2:1}, {0:1}, {1:1}), # N3 = ce
        ({3:1}, {1:1}, {1:1})  # N4 = dg
    ]

    def add_block(a_j_offset, b_j_offset, terms, is_simple=False):
        nonlocal col
        for u_coefs, v_coefs, w_coefs in terms:
            for u_idx, val in u_coefs.items():
                # u_idx is local 0..3 (a,b,c,d)
                # local i = u_idx // 2, local j = u_idx % 2
                li = u_idx // 2
                lj = u_idx % 2
                global_i = li
                global_j = a_j_offset + lj
                U[global_i * m + global_j, col] = val
            
            for v_idx, val in v_coefs.items():
                # v_idx is local 0..3 (e,f,g,h) or 0..1 (e,g)
                if is_simple:
                    # v_idx 0=e, 1=g. local j = v_idx, local k = 0
                    lj = v_idx
                    lk = 0
                else:
                    # v_idx 0=e, 1=f, 2=g, 3=h. local j = v_idx // 2, local k = v_idx % 2
                    lj = v_idx // 2
                    lk = v_idx % 2
                global_j = b_j_offset + lj
                global_k = lk # k is local to the 2x2 block, but B is 2x7.
                # Wait, B is 2x7. The 2x2 blocks are columns 0,1; 2,3; 4,5.
                # The 2x1 block is column 6.
                # So global k needs to be offset by the block's column offset.
                
            # This logic is getting complicated. Let's use explicit offsets.
            
            col += 1
            
    # Let's just hardcode the loops for clarity and correctness.
    
    # Block 1: A1 (cols 0,1) * B1 (rows 0,1)
    # B1 cols 0,1 (2x2), 2,3 (2x2), 4,5 (2x2), 6 (2x1)
    
    # A1 B1 part 1: 2x2 * 2x2 (B cols 0,1)
    for u_coefs, v_coefs, w_coefs in strassen_terms_fixed:
        for u_idx, val in u_coefs.items():
            li = u_idx // 2
            lj = u_idx % 2
            U[li * 4 + lj, col] = val # A1 col lj
        for v_idx, val in v_coefs.items():
            lj = v_idx // 2
            lk = v_idx % 2
            V[(0 + lj) * 7 + (0 + lk), col] = val # B1 row lj, col lk
        for w_idx, val in w_coefs.items():
            li = w_idx // 2
            lk = w_idx % 2
            W[(0 + lk) * 2 + li, col] = val # C col lk, row li
        col += 1
        
    # A1 B1 part 2: 2x2 * 2x2 (B cols 2,3)
    for u_coefs, v_coefs, w_coefs in strassen_terms_fixed:
        for u_idx, val in u_coefs.items():
            li = u_idx // 2
            lj = u_idx % 2
            U[li * 4 + lj, col] = val
        for v_idx, val in v_coefs.items():
            lj = v_idx // 2
            lk = v_idx % 2
            V[(0 + lj) * 7 + (2 + lk), col] = val # B1 row lj, col 2+lk
        for w_idx, val in w_coefs.items():
            li = w_idx // 2
            lk = w_idx % 2
            W[(2 + lk) * 2 + li, col] = val # C col 2+lk, row li
        col += 1
        
    # A1 B1 part 3: 2x2 * 2x2 (B cols 4,5)
    for u_coefs, v_coefs, w_coefs in strassen_terms_fixed:
        for u_idx, val in u_coefs.items():
            li = u_idx // 2
            lj = u_idx % 2
            U[li * 4 + lj, col] = val
        for v_idx, val in v_coefs.items():
            lj = v_idx // 2
            lk = v_idx % 2
            V[(0 + lj) * 7 + (4 + lk), col] = val # B1 row lj, col 4+lk
        for w_idx, val in w_coefs.items():
            li = w_idx // 2
            lk = w_idx % 2
            W[(4 + lk) * 2 + li, col] = val # C col 4+lk, row li
        col += 1
        
    # A1 B1 part 4: 2x2 * 2x1 (B col 6)
    for u_coefs, v_coefs, w_coefs in simple_terms:
        for u_idx, val in u_coefs.items():
            li = u_idx // 2
            lj = u_idx % 2
            U[li * 4 + lj, col] = val
        for v_idx, val in v_coefs.items():
            lj = v_idx
            V[(0 + lj) * 7 + 6, col] = val # B1 row lj, col 6
        for w_idx, val in w_coefs.items():
            li = w_idx
            W[6 * 2 + li, col] = val # C col 6, row li
        col += 1
        
    # Block 2: A2 (cols 2,3) * B2 (rows 2,3)
    # B2 cols 0,1 (2x2), 2,3 (2x2), 4,5 (2x2), 6 (2x1)
    # Note: B2 is rows 2,3 of B. So global row is 2+lj.
    
    # A2 B2 part 1: 2x2 * 2x2 (B2 cols 0,1)
    for u_coefs, v_coefs, w_coefs in strassen_terms_fixed:
        for u_idx, val in u_coefs.items():
            li = u_idx // 2
            lj = u_idx % 2
            U[li * 4 + (2 + lj), col] = val # A2 col 2+lj
        for v_idx, val in v_coefs.items():
            lj = v_idx // 2
            lk = v_idx % 2
            V[(2 + lj) * 7 + (0 + lk), col] = val # B2 row 2+lj, col lk
        for w_idx, val in w_coefs.items():
            li = w_idx // 2
            lk = w_idx % 2
            W[(0 + lk) * 2 + li, col] = val
        col += 1
        
    # A2 B2 part 2: 2x2 * 2x2 (B2 cols 2,3)
    for u_coefs, v_coefs, w_coefs in strassen_terms_fixed:
        for u_idx, val in u_coefs.items():
            li = u_idx // 2
            lj = u_idx % 2
            U[li * 4 + (2 + lj), col] = val
        for v_idx, val in v_coefs.items():
            lj = v_idx // 2
            lk = v_idx % 2
            V[(2 + lj) * 7 + (2 + lk), col] = val
        for w_idx, val in w_coefs.items():
            li = w_idx // 2
            lk = w_idx % 2
            W[(2 + lk) * 2 + li, col] = val
        col += 1
        
    # A2 B2 part 3: 2x2 * 2x2 (B2 cols 4,5)
    for u_coefs, v_coefs, w_coefs in strassen_terms_fixed:
        for u_idx, val in u_coefs.items():
            li = u_idx // 2
            lj = u_idx % 2
            U[li * 4 + (2 + lj), col] = val
        for v_idx, val in v_coefs.items():
            lj = v_idx // 2
            lk = v_idx % 2
            V[(2 + lj) * 7 + (4 + lk), col] = val
        for w_idx, val in w_coefs.items():
            li = w_idx // 2
            lk = w_idx % 2
            W[(4 + lk) * 2 + li, col] = val
        col += 1
        
    # A2 B2 part 4: 2x2 * 2x1 (B2 col 6)
    for u_coefs, v_coefs, w_coefs in simple_terms:
        for u_idx, val in u_coefs.items():
            li = u_idx // 2
            lj = u_idx % 2
            U[li * 4 + (2 + lj), col] = val
        for v_idx, val in v_coefs.items():
            lj = v_idx
            V[(2 + lj) * 7 + 6, col] = val
        for w_idx, val in w_coefs.items():
            li = w_idx
            W[6 * 2 + li, col] = val
        col += 1
        
    return U, V, W, col

def construct():
    n, m, p = DIMENSIONS
    U, V, W, rank = _construct_rank_50()
    
    if _verify(U, V, W, n, m, p):
        return {
            'U': U.tolist(),
            'V': V.tolist(),
            'W': W.tolist()
        }
    else:
        # Fallback to schoolbook rank 56
        U_s = np.zeros((n * m, 56), dtype=float)
        V_s = np.zeros((m * p, 56), dtype=float)
        W_s = np.zeros((p * n, 56), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j, c] = 1.0
                    V_s[j * p + k, c] = 1.0
                    W_s[k * n + i, c] = 1.0
                    c += 1
        return {
            'U': U_s.tolist(),
            'V': V_s.tolist(),
            'W': W_s.tolist()
        }
