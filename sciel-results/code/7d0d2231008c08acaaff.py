import numpy as np

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use Strassen's algorithm for 2x2 blocks.
    # p=10 is 5 blocks of width 2.
    num_blocks = p // 2
    rank_per_block = 7
    total_rank = num_blocks * rank_per_block
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    # Define the Strassen components for a single 2x2 block.
    # Each component is a list of (index, coeff) for U, V, W.
    # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) relative to block start
    # W indices: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1) relative to block start in C
    
    # Note on V and W index mapping for a block b:
    # Local j=0 -> Global j=0, Local k=0,1 -> Global k=2b, 2b+1
    # V row = j*p + k
    # Local (j, k) -> Global (j, 2b+k)
    # V row for local (0,0) is 0*p + 2b = 2b
    # V row for local (0,1) is 0*p + 2b+1 = 2b+1
    # V row for local (1,0) is 1*p + 2b = 10+2b
    # V row for local (1,1) is 1*p + 2b+1 = 11+2b
    
    # W row = k*n + i
    # Local (k, i) -> Global (2b+k, i)
    # W row for local (0,0) is 0*2 + 0 = 0? No, Global k=2b.
    # W row for local (0,0) [k'=0, i=0] is (2b)*2 + 0 = 4b
    # W row for local (1,0) [k'=1, i=0] is (2b+1)*2 + 0 = 4b+2
    # W row for local (0,1) [k'=0, i=1] is (2b)*2 + 1 = 4b+1
    # W row for local (1,1) [k'=1, i=1] is (2b+1)*2 + 1 = 4b+3
    
    # Strassen definitions:
    # M1 = (A00+A11)(B00+B11) -> C00+=1, C11+=1
    # M2 = (A10+A11)B00       -> C10+=1, C11-=1
    # M3 = A00(B01-B11)       -> C01+=1, C11+=1
    # M4 = A11(B10-B00)       -> C00+=1, C10+=1
    # M5 = (A00+A10)B11       -> C00-=1, C01+=1
    # M6 = (A11-A00)(B00+B01) -> C11+=1
    # M7 = (A01-A11)(B10+B11) -> C00+=1
    
    # U indices: 00=0, 01=1, 10=2, 11=3
    # V local indices: 00=0, 01=1, 10=2, 11=3 (relative to block)
    # W local indices: 00=0, 10=1, 01=2, 11=3 (relative to block, note k*n+i order)
    # Wait, W local index mapping:
    # C00 (k'=0, i=0) -> local idx 0
    # C10 (k'=0, i=1) -> local idx 1
    # C01 (k'=1, i=0) -> local idx 2
    # C11 (k'=1, i=1) -> local idx 3
    
    strassen_terms = [
        # (U_entries, V_entries, W_entries)
        # Term 1: M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # Term 2: M2
        ([(2, 1), (3, 1)], [(0, 1)], [(1, 1), (3, -1)]),
        # Term 3: M3
        ([(0, 1)], [(1, 1), (3, -1)], [(2, 1), (3, 1)]),
        # Term 4: M4
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (1, 1)]),
        # Term 5: M5
        ([(0, 1), (2, 1)], [(3, 1)], [(0, -1), (2, 1)]),
        # Term 6: M6
        ([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # Term 7: M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])
    ]
    
    for b in range(num_blocks):
        base_rank = b * rank_per_block
        for t in range(rank_per_block):
            r = base_rank + t
            u_terms, v_terms, w_terms = strassen_terms[t]
            
            for idx, coeff in u_terms:
                U[idx, r] += coeff
                
            for idx, coeff in v_terms:
                # Map local V index to global V index
                # idx 0 -> j=0, k=0 -> 2b
                # idx 1 -> j=0, k=1 -> 2b+1
                # idx 2 -> j=1, k=0 -> 10+2b
                # idx 3 -> j=1, k=1 -> 11+2b
                if idx == 0:
                    v_row = 2 * b
                elif idx == 1:
                    v_row = 2 * b + 1
                elif idx == 2:
                    v_row = 10 + 2 * b
                elif idx == 3:
                    v_row = 11 + 2 * b
                V[v_row, r] += coeff
                
            for idx, coeff in w_terms:
                # Map local W index to global W index
                # idx 0 -> k=0, i=0 -> 4b
                # idx 1 -> k=0, i=1 -> 4b+1
                # idx 2 -> k=1, i=0 -> 4b+2
                # idx 3 -> k=1, i=1 -> 4b+3
                if idx == 0:
                    w_row = 4 * b
                elif idx == 1:
                    w_row = 4 * b + 1
                elif idx == 2:
                    w_row = 4 * b + 2
                elif idx == 3:
                    w_row = 4 * b + 3
                W[w_row, r] += coeff

    # Convert to lists of lists as required by the interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
