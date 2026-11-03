import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        U[u_idx][0] = u_val # Placeholder, need column index
        # Actually, I should just fill the arrays directly.

    # Let's just define the terms as tuples of (u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
    terms = []
    
    # 1. M1 = (A00 + A11)(B00 + B11)
    terms.append(([0, 4], [1, 1], [0, 4], [1, 1], [0, 3], [1, 1]))
    # 2. M2 = (A10 + A11)B00
    terms.append(([3, 4], [1, 1], [0], [1], [1, 3], [1, -1]))
    # 3. M3 = A00(B01 - B11)
    terms.append(([0], [1], [1, 4], [1, -1], [2, 3], [1, 1]))
    # 4. M4 = A11(B10 - B00)
    terms.append(([4], [1], [3, 0], [1, -1], [0, 1], [1, 1]))
    # 5. M5 = (A00 + A01)B11
    terms.append(([0, 1], [1, 1], [4], [1], [0, 2], [-1, 1]))
    # 6. M6 = (A10 - A00)(B00 + B01)
    terms.append(([3, 0], [1, -1], [0, 1], [1, 1], [3], [1]))
    # 7. M7 = (A01 - A11)(B10 + B11)
    terms.append(([1, 4], [1, -1], [3, 4], [1, 1], [0], [1]))
    
    # 8. A00 B02
    terms.append(([0], [1], [2], [1], [4], [1]))
    # 9. A01 B12
    terms.append(([1], [1], [5], [1], [4], [1]))
    # 10. A10 B02
    terms.append(([3], [1], [2], [1], [5], [1]))
    # 11. A11 B12
    terms.append(([4], [1], [5], [1], [5], [1]))
    
    # 12. A02 B20
    terms.append(([2], [1], [6], [1], [0], [1]))
    # 13. A02 B21
    terms.append(([2], [1], [7], [1], [2], [1]))
    # 14. A02 B22
    terms.append(([2], [1], [8], [1], [4], [1]))
    # 15. A12 B20
    terms.append(([5], [1], [6], [1], [1], [1]))
    # 16. A12 B21
    terms.append(([5], [1], [7], [1], [3], [1]))
    # 17. A12 B22
    terms.append(([5], [1], [8], [1], [5], [1]))
    
    for c, (u_idx, u_val, v_idx, v_val, w_idx, w_val) in enumerate(terms):
        for idx, val in zip(u_idx, u_val):
            U[idx][c] = val
        for idx, val in zip(v_idx, v_val):
            V[idx][c] = val
        for idx, val in zip(w_idx, w_val):
            W[idx][c] = val
            
    return {"U": U, "V": V, "W": W}
