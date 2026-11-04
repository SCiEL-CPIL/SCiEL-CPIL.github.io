import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # Total rank: 49 (Strassen for 4x4x4) + 49 (Strassen for 4x4x4) + 16 (Schoolbook for 4x4x1)
    rank = 49 + 49 + 16
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a 2x2 Strassen decomposition
    def add_strassen_2x2(u_map, v_map, w_map, offset):
        # u_map: dict (r, c) -> U index
        # v_map: dict (r, c) -> V index
        # w_map: dict (r, c) -> W index
        
        # P1 = (x00 + x11)(y00 + y11)
        U[u_map[(0,0)]][offset] += 1
        U[u_map[(1,1)]][offset] += 1
        V[v_map[(0,0)]][offset] += 1
        V[v_map[(1,1)]][offset] += 1
        W[w_map[(0,0)]][offset] += 1
        W[w_map[(1,1)]][offset] += 1
        
        # P2 = (x10 + x11)y00
        U[u_map[(1,0)]][offset] += 1
        U[u_map[(1,1)]][offset] += 1
        V[v_map[(0,0)]][offset] += 1
        W[w_map[(1,0)]][offset] += 1
        W[w_map[(1,1)]][offset] -= 1
        
        # P3 = x00(y01 - y11)
        U[u_map[(0,0)]][offset] += 1
        V[v_map[(0,1)]][offset] += 1
        V[v_map[(1,1)]][offset] -= 1
        W[w_map[(0,1)]][offset] += 1
        W[w_map[(1,1)]][offset] += 1
        
        # P4 = x11(y10 - y00)
        U[u_map[(1,1)]][offset] += 1
        V[v_map[(1,0)]][offset] += 1
        V[v_map[(0,0)]][offset] -= 1
        W[w_map[(0,0)]][offset] += 1
        W[w_map[(1,0)]][offset] += 1
        
        # P5 = (x00 + x01)y11
        U[u_map[(0,0)]][offset] += 1
        U[u_map[(0,1)]][offset] += 1
        V[v_map[(1,1)]][offset] += 1
        W[w_map[(0,0)]][offset] -= 1
        W[w_map[(0,1)]][offset] += 1
        
        # P6 = (x10 - x00)(y00 + y01)
        U[u_map[(1,0)]][offset] += 1
        U[u_map[(0,0)]][offset] -= 1
        V[v_map[(0,0)]][offset] += 1
        V[v_map[(0,1)]][offset] += 1
        W[w_map[(1,1)]][offset] += 1
        
        # P7 = (x01 - x11)(y10 + y11)
        U[u_map[(0,1)]][offset] += 1
        U[u_map[(1,1)]][offset] -= 1
        V[v_map[(1,0)]][offset] += 1
        V[v_map[(1,1)]][offset] += 1
        W[w_map[(0,0)]][offset] += 1
        W[w_map[(1,0)]][offset] += 1

    # Process first 4 columns (k=0..3)
    offset = 0
    # A is 4x4, B is 4x4 (first 4 cols)
    # A blocks: A00 (0,0), A01 (0,1), A10 (1,0), A11 (1,1)
    # B blocks: B00 (0,0), B01 (0,1), B10 (1,0), B11 (1,1)
    
    # M1 = (A00 + A11)(B00 + B11)
    u_map = {(0,0): 0*m+0, (0,1): 0*m+1, (1,0): 2*m+0, (1,1): 2*m+1} # A00
    # Wait, u_map should map to U indices.
    # U index is i*m + j.
    # A00 is i=0, j=0 -> U[0]
    # A01 is i=0, j=1 -> U[1]
    # A10 is i=2, j=0 -> U[2*4+0] = U[8]
    # A11 is i=2, j=1 -> U[2*4+1] = U[9]
    # But M1 uses A00 + A11. So x00 = A00 + A22.
    # x00 corresponds to U[0] + U[2*4+2] = U[0] + U[10]
    # x01 corresponds to U[1] + U[11]
    # x10 corresponds to U[8] + U[12]
    # x11 corresponds to U[9] + U[13]
    
    # Let's create a helper to get U index for A element (i, j)
    def u_idx_A(i, j): return i * m + j
    # Let's create a helper to get V index for B element (j, k)
    def v_idx_B(j, k): return j * p + k
    # Let's create a helper to get W index for C element (k, i)
    def w_idx_C(k, i): return k * n + i
    
    # For M1:
    # X = A00 + A11 (blocks)
    # x00 = A[0,0] + A[2,2] -> u_idx_A(0,0) + u_idx_A(2,2)
    # x01 = A[0,1] + A[2,3] -> u_idx_A(0,1) + u_idx_A(2,3)
    # x10 = A[2,0] + A[3,2] -> u_idx_A(2,0) + u_idx_A(3,2)
    # x11 = A[2,1] + A[3,3] -> u_idx_A(2,1) + u_idx_A(3,3)
    
    # Y = B00 + B11 (blocks)
    # y00 = B[0,0] + B[2,0] -> v_idx_B(0,0) + v_idx_B(2,0)
    # y01 = B[0,1] + B[2,1] -> v_idx_B(0,1) + v_idx_B(2,1)
    # y10 = B[1,0] + B[3,0] -> v_idx_B(1,0) + v_idx_B(3,0)
    # y11 = B[1,1] + B[3,1] -> v_idx_B(1,1) + v_idx_B(3,1)
    
    # Z = M1 contributes to C00, C01, C20, C21
    # z00 -> C[0,0] -> w_idx_C(0,0)
    # z01 -> C[1,0] -> w_idx_C(1,0) ? No, C is 4x4. C[0,0], C[0,1], C[2,0], C[2,1].
    # C[0,0] is k=0, i=0 -> w_idx_C(0,0)
    # C[0,1] is k=1, i=0 -> w_idx_C(1,0)
    # C[2,0] is k=0, i=2 -> w_idx_C(0,2)
    # C[2,1] is k=1, i=2 -> w_idx_C(1,2)
    
    # We need to modify add_strassen_2x2 to accept lists of indices, not just single indices.
    # Or we can just add the contributions manually.
    
    # Let's rewrite add_strassen_2x2 to accept lists of (index, coeff) pairs.
    
    def add_strassen_2x2_v2(u_list, v_list, w_list, offset):
        # u_list: list of (index, coeff) for x00, x01, x10, x11
        # v_list: list of (index, coeff) for y00, y01, y10, y11
        # w_list: list of (index, coeff) for z00, z01, z10, z11
        
        x00, x01, x10, x11 = u_list
        y00, y01, y10, y11 = v_list
        z00, z01, z10, z11 = w_list
        
        # P1 = (x00 + x11)(y00 + y11)
        for idx, c in x00: U[idx][offset] += c
        for idx, c in x11: U[idx][offset] += c
        for idx, c in y00: V[idx][offset] += c
        for idx, c in y11: V[idx][offset] += c
        for idx, c in z00: W[idx][offset] += c
        for idx, c in z11: W[idx][offset] += c
        
        # P2 = (x10 + x11)y00
        for idx, c in x10: U[idx][offset] += c
        for idx, c in x11: U[idx][offset] += c
        for idx, c in y00: V[idx][offset] += c
        for idx, c in z10: W[idx][offset] += c
        for idx, c in z11: W[idx][offset] -= c
        
        # P3 = x00(y01 - y11)
        for idx, c in x00: U[idx][offset] += c
        for idx, c in y01: V[idx][offset] += c
        for idx, c in y11: V[idx][offset] -= c
        for idx, c in z01: W[idx][offset] += c
        for idx, c in z11: W[idx][offset] += c
        
        # P4 = x11(y10 - y00)
        for idx, c in x11: U[idx][offset] += c
        for idx, c in y10: V[idx][offset] += c
        for idx, c in y00: V[idx][offset] -= c
        for idx, c in z00: W[idx][offset] += c
        for idx, c in z10: W[idx][offset] += c
        
        # P5 = (x00 + x01)y11
        for idx, c in x00: U[idx][offset] += c
        for idx, c in x01: U[idx][offset] += c
        for idx, c in y11: V[idx][offset] += c
        for idx, c in z00: W[idx][offset] -= c
        for idx, c in z01: W[idx][offset] += c
        
        # P6 = (x10 - x00)(y00 + y01)
        for idx, c in x10: U[idx][offset] += c
        for idx, c in x00: U[idx][offset] -= c
        for idx, c in y00: V[idx][offset] += c
        for idx, c in y01: V[idx][offset] += c
        for idx, c in z11: W[idx][offset] += c
        
        # P7 = (x01 - x11)(y10 + y11)
        for idx, c in x01: U[idx][offset] += c
        for idx, c in x11: U[idx][offset] -= c
        for idx, c in y10: V[idx][offset] += c
        for idx, c in y11: V[idx][offset] += c
        for idx, c in z00: W[idx][offset] += c
        for idx, c in z10: W[idx][offset] += c

    # Define the 7 Strassen products for 4x4
    # M1 = (A00 + A11)(B00 + B11)
    u_m1 = [
        [(u_idx_A(0,0), 1), (u_idx_A(2,2), 1)], # x00
        [(u_idx_A(0,1), 1), (u_idx_A(2,3), 1)], # x01
        [(u_idx_A(2,0), 1), (u_idx_A(3,2), 1)], # x10
        [(u_idx_A(2,1), 1), (u_idx_A(3,3), 1)]  # x11
    ]
    v_m1 = [
        [(v_idx_B(0,0), 1), (v_idx_B(2,0), 1)], # y00
        [(v_idx_B(0,1), 1), (v_idx_B(2,1), 1)], # y01
        [(v_idx_B(1,0), 1), (v_idx_B(3,0), 1)], # y10
        [(v_idx_B(1,1), 1), (v_idx_B(3,1), 1)]  # y11
    ]
    w_m1 = [
        [(w_idx_C(0,0), 1)], # z00
        [(w_idx_C(1,0), 1)], # z01
        [(w_idx_C(0,2), 1)], # z10
        [(w_idx_C(1,2), 1)]  # z11
    ]
    add_strassen_2x2_v2(u_m1, v_m1, w_m1, offset); offset += 7

    # M2 = (A10 + A11)B00
    u_m2 = [
        [(u_idx_A(2,0), 1), (u_idx_A(2,1), 1)], # x00 = A20+A21? No.
        # A10 is block (2,0), A11 is block (2,1).
        # A10 = [[A20, A21], [A30, A31]]
        # A11 = [[A22, A23], [A32, A33]]
        # X = A10 + A11
        # x00 = A20 + A22
        # x01 = A21 + A23
        # x10 = A30 + A32
        # x11 = A31 + A33
        [(u_idx_A(2,0), 1), (u_idx_A(2,2), 1)],
        [(u_idx_A(2,1), 1), (u_idx_A(2,3), 1)],
        [(u_idx_A(3,0), 1), (u_idx_A(3,2), 1)],
        [(u_idx_A(3,1), 1), (u_idx_A(3,3), 1)]
    ]
    v_m2 = [
        [(v_idx_B(0,0), 1)], # y00 = B00
        [(v_idx_B(0,1), 1)], # y01 = B01
        [(v_idx_B(1,0), 1)], # y10 = B10
        [(v_idx_B(1,1), 1)]  # y11 = B11
    ]
    w_m2 = [
        [(w_idx_C(0,0), 1)], # z00 -> C00
        [(w_idx_C(1,0), 1)], # z01 -> C01
        [(w_idx_C(0,2), 1)], # z10 -> C20
        [(w_idx_C(1,2), 1)]  # z11 -> C21
    ]
    add_strassen_2x2_v2(u_m2, v_m2, w_m2, offset); offset += 7

    # M3 = A00(B01 - B11)
    u_m3 = [
        [(u_idx_A(0,0), 1)], # x00 = A00
        [(u_idx_A(0,1), 1)], # x01 = A01
        [(u_idx_A(1,0), 1)], # x10 = A10
        [(u_idx_A(1,1), 1)]  # x11 = A11
    ]
    v_m3 = [
        [(v_idx_B(0,1), 1), (v_idx_B(1,1), -1)], # y00 = B01 - B11
        [(v_idx_B(0,2), 1), (v_idx_B(1,2), -1)], # y01 = B02 - B12
        [(v_idx_B(0,3), 1), (v_idx_B(1,3), -1)], # y10 = B03 - B13
        [(v_idx_B(0,4), 1), (v_idx_B(1,4), -1)]  # y11 = B04 - B14
    ]
    w_m3 = [
        [(w_idx_C(0,0), 1)], # z00 -> C00
        [(w_idx_C(1,0), 1)], # z01 -> C01
        [(w_idx_C(0,2), 1)], # z10 -> C20
        [(w_idx_C(1,2), 1)]  # z11 -> C21
    ]
    add_strassen_2x2_v2(u_m3, v_m3, w_m3, offset); offset += 7

    # M4 = A11(B10 - B00)
    u_m4 = [
        [(u_idx_A(2,2), 1)], # x00 = A22
        [(u_idx_A(2,3), 1)], # x01 = A23
        [(u_idx_A(3,2), 1)], # x10 = A32
        [(u_idx_A(3,3), 1)]  # x11 = A33
    ]
    v_m4 = [
        [(v_idx_B(1,0), 1), (v_idx_B(0,0), -1)], # y00 = B10 - B00
        [(v_idx_B(1,1), 1), (v_idx_B(0,1), -1)], # y01 = B11 - B01
        [(v_idx_B(1,2), 1), (v_idx_B(0,2), -1)], # y10 = B12 - B02
        [(v_idx_B(1,3), 1), (v_idx_B(0,3), -1)]  # y11 = B13 - B03
    ]
    w_m4 = [
        [(w_idx_C(0,0), 1)], # z00 -> C00
        [(w_idx_C(1,0), 1)], # z01 -> C01
        [(w_idx_C(0,2), 1)], # z10 -> C20
        [(w_idx_C(1,2), 1)]  # z11 -> C21
    ]
    add_strassen_2x2_v2(u_m4, v_m4, w_m4, offset); offset += 7

    # M5 = (A00 + A01)B11
    u_m5 = [
        [(u_idx_A(0,0), 1), (u_idx_A(0,1), 1)], # x00 = A00+A01
        [(u_idx_A(0,2), 1), (u_idx_A(0,3), 1)], # x01 = A02+A03
        [(u_idx_A(1,0), 1), (u_idx_A(1,1), 1)], # x10 = A10+A11
        [(u_idx_A(1,2), 1), (u_idx_A(1,3), 1)]  # x11 = A12+A13
    ]
    v_m5 = [
        [(v_idx_B(2,0), 1)], # y00 = B20
        [(v_idx_B(2,1), 1)], # y01 = B21
        [(v_idx_B(3,0), 1)], # y10 = B30
        [(v_idx_B(3,1), 1)]  # y11 = B31
    ]
    w_m5 = [
        [(w_idx_C(0,0), 1)], # z00 -> C00
        [(w_idx_C(1,0), 1)], # z01 -> C01
        [(w_idx_C(0,2), 1)], # z10 -> C20
        [(w_idx_C(1,2), 1)]  # z11 -> C21
    ]
    add_strassen_2x2_v2(u_m5, v_m5, w_m5, offset); offset += 7

    # M6 = (A10 - A00)(B00 + B01)
    u_m6 = [
        [(u_idx_A(2,0), 1), (u_idx_A(0,0), -1)], # x00 = A20-A00
        [(u_idx_A(2,1), 1), (u_idx_A(0,1), -1)], # x01 = A21-A01
        [(u_idx_A(3,0), 1), (u_idx_A(1,0), -1)], # x10 = A30-A10
        [(u_idx_A(3,1), 1), (u_idx_A(1,1), -1)]  # x11 = A31-A11
    ]
    v_m6 = [
        [(v_idx_B(0,0), 1), (v_idx_B(0,1), 1)], # y00 = B00+B01
        [(v_idx_B(0,2), 1), (v_idx_B(0,3), 1)], # y01 = B02+B03
        [(v_idx_B(1,0), 1), (v_idx_B(1,1), 1)], # y10 = B10+B11
        [(v_idx_B(1,2), 1), (v_idx_B(1,3), 1)]  # y11 = B12+B13
    ]
    w_m6 = [
        [(w_idx_C(0,0), 1)], # z00 -> C00
        [(w_idx_C(1,0), 1)], # z01 -> C01
        [(w_idx_C(0,2), 1)], # z10 -> C20
        [(w_idx_C(1,2), 1)]  # z11 -> C21
    ]
    add_strassen_2x2_v2(u_m6, v_m6, w_m6, offset); offset += 7

    # M7 = (A01 - A11)(B10 + B11)
    u_m7 = [
        [(u_idx_A(0,1), 1), (u_idx_A(2,2), -1)], # x00 = A01-A22
        [(u_idx_A(0,2), 1), (u_idx_A(2,3), -1)], # x01 = A02-A23
        [(u_idx_A(1,1), 1), (u_idx_A(3,2), -1)], # x10 = A11-A32
        [(u_idx_A(1,2), 1), (u_idx_A(3,3), -1)]  # x11 = A12-A33
    ]
    v_m7 = [
        [(v_idx_B(1,0), 1), (v_idx_B(2,0), 1)], # y00 = B10+B20
        [(v_idx_B(1,1), 1), (v_idx_B(2,1), 1)], # y01 = B11+B21
        [(v_idx_B(1,2), 1), (v_idx_B(2,2), 1)], # y10 = B12+B22
        [(v_idx_B(1,3), 1), (v_idx_B(2,3), 1)]  # y11 = B13+B23
    ]
    w_m7 = [
        [(w_idx_C(0,0), 1)], # z00 -> C00
        [(w_idx_C(1,0), 1)], # z01 -> C01
        [(w_idx_C(0,2), 1)], # z10 -> C20
        [(w_idx_C(1,2), 1)]  # z11 -> C21
    ]
    add_strassen_2x2_v2(u_m7, v_m7, w_m7, offset); offset += 7

    # Process next 4 columns (k=4..7)
    # Same logic, but B indices are shifted by 4.
    # B00 becomes B40, etc.
    # v_idx_B(j, k) -> v_idx_B(j, k+4)
    
    # M1 = (A00 + A11)(B40 + B41)
    u_m1_2 = [
        [(u_idx_A(0,0), 1), (u_idx_A(2,2), 1)],
        [(u_idx_A(0,1), 1), (u_idx_A(2,3), 1)],
        [(u_idx_A(2,0), 1), (u_idx_A(3,2), 1)],
        [(u_idx_A(2,1), 1), (u_idx_A(3,3), 1)]
    ]
    v_m1_2 = [
        [(v_idx_B(0,4), 1), (v_idx_B(2,4), 1)],
        [(v_idx_B(0,5), 1), (v_idx_B(2,5), 1)],
        [(v_idx_B(1,4), 1), (v_idx_B(3,4), 1)],
        [(v_idx_B(1,5), 1), (v_idx_B(3,5), 1)]
    ]
    w_m1_2 = [
        [(w_idx_C(4,0), 1)],
        [(w_idx_C(5,0), 1)],
        [(w_idx_C(4,2), 1)],
        [(w_idx_C(5,2), 1)]
    ]
    add_strassen_2x2_v2(u_m1_2, v_m1_2, w_m1_2, offset); offset += 7

    # M2 = (A10 + A11)B40
    u_m2_2 = [
        [(u_idx_A(2,0), 1), (u_idx_A(2,2), 1)],
        [(u_idx_A(2,1), 1), (u_idx_A(2,3), 1)],
        [(u_idx_A(3,0), 1), (u_idx_A(3,2), 1)],
        [(u_idx_A(3,1), 1), (u_idx_A(3,3), 1)]
    ]
    v_m2_2 = [
        [(v_idx_B(0,4), 1)],
        [(v_idx_B(0,5), 1)],
        [(v_idx_B(1,4), 1)],
        [(v_idx_B(1,5), 1)]
    ]
    w_m2_2 = [
        [(w_idx_C(4,0), 1)],
        [(w_idx_C(5,0), 1)],
        [(w_idx_C(4,2), 1)],
        [(w_idx_C(5,2), 1)]
    ]
    add_strassen_2x2_v2(u_m2_2, v_m2_2, w_m2_2, offset); offset += 7

    # M3 = A00(B41 - B51)
    u_m3_2 = [
        [(u_idx_A(0,0), 1)],
        [(u_idx_A(0,1), 1)],
        [(u_idx_A(1,0), 1)],
        [(u_idx_A(1,1), 1)]
    ]
    v_m3_2 = [
        [(v_idx_B(0,5), 1), (v_idx_B(1,5), -1)],
        [(v_idx_B(0,6), 1), (v_idx_B(1,6), -1)],
        [(v_idx_B(0,7), 1), (v_idx_B(1,7), -1)],
        [(v_idx_B(0,8), 1), (v_idx_B(1,8), -1)] # Wait, B is 4x9. k goes 0..8.
        # B41 is k=5. B51 is k=5? No.
        # B is 4x9. Rows 0..3, Cols 0..8.
        # B40 is row 0, col 4.
        # B41 is row 0, col 5.
        # B51 is row 1, col 5.
        # So v_idx_B(0,5) - v_idx_B(1,5).
        # My previous v_m3 had v_idx_B(0,1) - v_idx_B(1,1).
        # So for k=4..7, it's v_idx_B(0,5) - v_idx_B(1,5).
    ]
    # Let's re-verify v_m3_2
    # M3 = A00(B01 - B11) for first block.
    # For second block, M3 = A00(B41 - B51).
    # B41 is (0,5), B51 is (1,5).
    # B42 is (0,6), B52 is (1,6).
    # B43 is (0,7), B53 is (1,7).
    # B44 is (0,8), B54 is (1,8).
    v_m3_2 = [
        [(v_idx_B(0,5), 1), (v_idx_B(1,5), -1)],
        [(v_idx_B(0,6), 1), (v_idx_B(1,6), -1)],
        [(v_idx_B(0,7), 1), (v_idx_B(1,7), -1)],
        [(v_idx_B(0,8), 1), (v_idx_B(1,8), -1)]
    ]
    w_m3_2 = [
        [(w_idx_C(4,0), 1)],
        [(w_idx_C(5,0), 1)],
        [(w_idx_C(4,2), 1)],
        [(w_idx_C(5,2), 1)]
    ]
    add_strassen_2x2_v2(u_m3_2, v_m3_2, w_m3_2, offset); offset += 7

    # M4 = A11(B50 - B40)
    u_m4_2 = [
        [(u_idx_A(2,2), 1)],
        [(u_idx_A(2,3), 1)],
        [(u_idx_A(3,2), 1)],
        [(u_idx_A(3,3), 1)]
    ]
    v_m4_2 = [
        [(v_idx_B(1,4), 1), (v_idx_B(0,4), -1)],
        [(v_idx_B(1,5), 1), (v_idx_B(0,5), -1)],
        [(v_idx_B(1,6), 1), (v_idx_B(0,6), -1)],
        [(v_idx_B(1,7), 1), (v_idx_B(0,7), -1)]
    ]
    w_m4_2 = [
        [(w_idx_C(4,0), 1)],
        [(w_idx_C(5,0), 1)],
        [(w_idx_C(4,2), 1)],
        [(w_idx_C(5,2), 1)]
    ]
    add_strassen_2x2_v2(u_m4_2, v_m4_2, w_m4_2, offset); offset += 7

    # M5 = (A00 + A01)B51
    u_m5_2 = [
        [(u_idx_A(0,0), 1), (u_idx_A(0,1), 1)],
        [(u_idx_A(0,2), 1), (u_idx_A(0,3), 1)],
        [(u_idx_A(1,0), 1), (u_idx_A(1,1), 1)],
        [(u_idx_A(1,2), 1), (u_idx_A(1,3), 1)]
    ]
    v_m5_2 = [
        [(v_idx_B(2,4), 1)],
        [(v_idx_B(2,5), 1)],
        [(v_idx_B(3,4), 1)],
        [(v_idx_B(3,5), 1)]
    ]
    w_m5_2 = [
        [(w_idx_C(4,0), 1)],
        [(w_idx_C(5,0), 1)],
        [(w_idx_C(4,2), 1)],
        [(w_idx_C(5,2), 1)]
    ]
    add_strassen_2x2_v2(u_m5_2, v_m5_2, w_m5_2, offset); offset += 7

    # M6 = (A10 - A00)(B40 + B41)
    u_m6_2 = [
        [(u_idx_A(2,0), 1), (u_idx_A(0,0), -1)],
        [(u_idx_A(2,1), 1), (u_idx_A(0,1), -1)],
        [(u_idx_A(3,0), 1), (u_idx_A(1,0), -1)],
        [(u_idx_A(3,1), 1), (u_idx_A(1,1), -1)]
    ]
    v_m6_2 = [
        [(v_idx_B(0,4), 1), (v_idx_B(0,5), 1)],
        [(v_idx_B(0,6), 1), (v_idx_B(0,7), 1)],
        [(v_idx_B(1,4), 1), (v_idx_B(1,5), 1)],
        [(v_idx_B(1,6), 1), (v_idx_B(1,7), 1)]
    ]
    w_m6_2 = [
        [(w_idx_C(4,0), 1)],
        [(w_idx_C(5,0), 1)],
        [(w_idx_C(4,2), 1)],
        [(w_idx_C(5,2), 1)]
    ]
    add_strassen_2x2_v2(u_m6_2, v_m6_2, w_m6_2, offset); offset += 7

    # M7 = (A01 - A11)(B50 + B51)
    u_m7_2 = [
        [(u_idx_A(0,1), 1), (u_idx_A(2,2), -1)],
        [(u_idx_A(0,2), 1), (u_idx_A(2,3), -1)],
        [(u_idx_A(1,1), 1), (u_idx_A(3,2), -1)],
        [(u_idx_A(1,2), 1), (u_idx_A(3,3), -1)]
    ]
    v_m7_2 = [
        [(v_idx_B(1,4), 1), (v_idx_B(2,4), 1)],
        [(v_idx_B(1,5), 1), (v_idx_B(2,5), 1)],
        [(v_idx_B(1,6), 1), (v_idx_B(2,6), 1)],
        [(v_idx_B(1,7), 1), (v_idx_B(2,7), 1)]
    ]
    w_m7_2 = [
        [(w_idx_C(4,0), 1)],
        [(w_idx_C(5,0), 1)],
        [(w_idx_C(4,2), 1)],
        [(w_idx_C(5,2), 1)]
    ]
    add_strassen_2x2_v2(u_m7_2, v_m7_2, w_m7_2, offset); offset += 7

    # Process last column (k=8)
    # Schoolbook
    for i in range(4):
        for j in range(4):
            U[u_idx_A(i,j)][offset] = 1
            V[v_idx_B(j,8)][offset] = 1
            W[w_idx_C(8,i)][offset] = 1
            offset += 1

    return {"U": U, "V": V, "W": W}

