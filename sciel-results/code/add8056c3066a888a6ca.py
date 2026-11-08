import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=4
    # U: n*m = 6 x rank
    # V: m*p = 12 x rank
    # W: p*n = 8 x rank
    
    # Rank 16 decomposition:
    # Block 1: A[0:2,0:2] x B[0:2,0:2] via Strassen (rank 7)
    # Block 2: A[0:2,0:2] x B[0:2,2:4] via Strassen (rank 7)
    # Block 3: A[0:2,2] x B[2,0:4] via outer products (rank 2)
    
    rank = 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_indices, v_indices, w_indices, col):
        for idx, val in u_indices:
            U[idx][col] += val
        for idx, val in v_indices:
            V[idx][col] += val
        for idx, val in w_indices:
            W[idx][col] += val
    
    # A indices: A00=0, A01=1, A02=2, A10=3, A11=4, A12=5
    # B indices: B00=0, B01=1, B02=2, B03=3, B10=4, B11=5, B12=6, B13=7, B20=8, B21=9, B22=10, B23=11
    # C indices (W[k*n+i]): C00=0, C01=2, C02=4, C03=6, C10=1, C11=3, C12=5, C13=7
    
    # --- Block 1: A[0:2,0:2] x B[0:2,0:2] (Strassen, rank 7) ---
    # M1: (A00+A11)(B00+B11) -> C00+C11
    add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
    # M2: (A10+A11)B00 -> C10-C11
    add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
    # M3: A00(B01-B11) -> C01+C11
    add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
    # M4: A11(B10-B00) -> C00+C10
    add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
    # M5: (A00+A01)B11 -> -C00+C01
    add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
    # M6: (A10-A00)(B00+B01) -> C11
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
    # M7: (A01-A11)(B10+B11) -> C00
    add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)
    
    # --- Block 2: A[0:2,0:2] x B[0:2,2:4] (Strassen, rank 7) ---
    # B indices: B02=2, B03=3, B12=6, B13=7
    # M1': (A00+A11)(B02+B13) -> C02+C13
    add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
    # M2': (A10+A11)B02 -> C12-C13
    add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
    # M3': A00(B03-B13) -> C03+C13
    add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
    # M4': A11(B12-B02) -> C02+C12
    add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
    # M5': (A00+A01)B13 -> -C02+C03
    add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
    # M6': (A10-A00)(B02+B03) -> C13
    add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
    # M7': (A01-A11)(B12+B13) -> C02
    add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)
    
    # --- Block 3: A[0:2,2] x B[2,0:4] (outer products, rank 2) ---
    # A02=2, A12=5; B20=8, B21=9, B22=10, B23=11
    # A02 * B2j -> C0j; A12 * B2j -> C1j
    # Column 14: A02 * (B20 + B21 + B22 + B23) -> C00+C01+C02+C03
    add_term([(2, 1)], [(8, 1), (9, 1), (10, 1), (11, 1)], [(0, 1), (2, 1), (4, 1), (6, 1)], 14)
    # Column 15: A12 * (B20 + B21 + B22 + B23) -> C10+C11+C12+C13
    add_term([(5, 1)], [(8, 1), (9, 1), (10, 1), (11, 1)], [(1, 1), (3, 1), (5, 1), (7, 1)], 15)
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    U = np.array(res['U'], dtype=float)
    V = np.array(res['V'], dtype=float)
    W = np.array(res['W'], dtype=float)
    
    # Verify: for random A, B, check that sum_r U*A * V*B * W gives correct C
    np.random.seed(42)
    A = np.random.randn(2, 3)
    B = np.random.randn(3, 4)
    C_true = A @ B
    
    # Compute via decomposition
    # u_i = U[i,:] dot A_flat (row-major), v_i = V[i,:] dot B_flat
    A_flat = A.flatten()  # 6 elements
    B_flat = B.flatten()  # 12 elements
    C_comp = np.zeros((4, 2))  # W is p*n = 8, so output is 4x2 in column-major
    for r in range(len(res['U'][0])):
        u_val = float(U[:, r] @ A_flat)
        v_val = float(V[:, r] @ B_flat)
        C_comp += W[:, r] * (u_val * v_val)
    
    # W[k*n+i] means C_comp is indexed as [col, row], so C_comp.T should be C_true
    C_from_W = C_comp.T  # shape (2, 4)
    error = np.max(np.abs(C_from_W - C_true))
    print(f"Rank: {len(res['U'][0])}")
    print(f"Max error: {error:.2e}")
    print(f"Time: {time.time()-t0:.3f}s")
