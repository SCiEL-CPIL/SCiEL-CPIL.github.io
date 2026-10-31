DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add linear forms
    def add_form(f1, f2, sign=1):
        res = {}
        for k, v in f1.items():
            res[k] = res.get(k, 0) + v
        for k, v in f2.items():
            res[k] = res.get(k, 0) + sign * v
        return {k: v for k, v in res.items() if v != 0}
        
    def sub_form(f1, f2):
        return add_form(f1, f2, -1)
        
    # A and B block forms
    # A11: 0,1,4,5
    A11 = {(0,0): {0:1}, (0,1): {1:1}, (1,0): {4:1}, (1,1): {5:1}}
    A12 = {(0,0): {2:1}, (0,1): {3:1}, (1,0): {6:1}, (1,1): {7:1}}
    A21 = {(0,0): {8:1}, (0,1): {9:1}, (1,0): {12:1}, (1,1): {13:1}}
    A22 = {(0,0): {10:1}, (0,1): {11:1}, (1,0): {14:1}, (1,1): {15:1}}
    
    B11 = {(0,0): {0:1}, (0,1): {1:1}, (1,0): {4:1}, (1,1): {5:1}}
    B12 = {(0,0): {2:1}, (0,1): {3:1}, (1,0): {6:1}, (1,1): {7:1}}
    B21 = {(0,0): {8:1}, (0,1): {9:1}, (1,0): {12:1}, (1,1): {13:1}}
    B22 = {(0,0): {10:1}, (0,1): {11:1}, (1,0): {14:1}, (1,1): {15:1}}
    
    # M definitions
    M_defs = [
        (add_form(A11[(0,0)], A22[(0,0)]), add_form(B11[(0,0)], B22[(0,0)]), 
         add_form(A11[(0,1)], A22[(0,1)]), add_form(B11[(0,1)], B22[(0,1)]),
         add_form(A11[(1,0)], A22[(1,0)]), add_form(B11[(1,0)], B22[(1,0)]),
         add_form(A11[(1,1)], A22[(1,1)]), add_form(B11[(1,1)], B22[(1,1)])),
        # This is getting complicated with individual entries.
        # Let's stick to the block level logic and expand.
    ]
    
    # I will use the previous logic but ensure correct W indexing.
    
    # Correct W indexing: W_index = k * 4 + i
    # My C entry is (i, k).
    
    # I will generate the full code in the final block.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
