import time
import random
from fractions import Fraction
from itertools import combinations

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS
    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21
    
    # Generate irregular sub-tensors that admit rank reductions
    # Each block is a set of (i,j,k) triples that can be decomposed with fewer
    # terms than the product of their extents
    all_blocks = []
    
    # 2x2x2 Strassen blocks (rank 7 vs 8 schoolbook)
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            indices = set()
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        indices.add((i, j, k))
                            all_blocks.append({
                                'type': 'strassen_2x2x2',
                                'params': (i1, i2, j1, j2, k1, k2),
                                'indices': indices,
                                'rank': 7,
                                'schoolbook': 8
                            })
    
    # 2x3x2 blocks (rank 10 vs 12 schoolbook) - use 2 Strassen 2x2x2 sharing a plane
    # Actually, let's use a different approach: 2x3x2 can be done with 10 terms
    # by decomposing the 3 dimension into 2+1
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 2):
                for k1 in range(p - 1):
                    for k2 in range(k1 + 1, p):
                        # Use 2x2x2 on j1,j1+1 and 2x1x2 on j1+2
                        indices = set()
                        for i in (i1, i2):
                            for j in (j1, j1+1, j1+2):
                                for k in (k1, k2):
                                    indices.add((i, j, k))
                        all_blocks.append({
                            'type': 'strassen_2x3x2',
                            'params': (i1, i2, j1, k1, k2),
                            'indices': indices,
                            'rank': 7 + 4,  # 7 for 2x2x2 + 4 for 2x1x2
                            'schoolbook': 12
                        })
    
    # 3x2x2 blocks
    for j1 in range(m - 1):
        for j2 in range(j1 + 1, m):
            for k1 in range(p - 1):
                for k2 in range(k1 + 1, p):
                    for i1 in range(n - 2):
                        indices = set()
                        for i in (i1, i1+1, i1+2):
                            for j in (j1, j2):
                                for k in (k1, k2):
                                    indices.add((i, j, k))
                        all_blocks.append({
                            'type': 'strassen_3x2x2',
                            'params': (j1, j2, k1, k2, i1),
                            'indices': indices,
                            'rank': 7 + 4,
                            'schoolbook': 12
                        })
    
    # 2x2x3 blocks
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 2):
                        indices = set()
                        for i in (i1, i2):
                            for j in (j1, j2):
                                for k in (k1, k1+1, k1+2):
                                    indices.add((i, j, k))
                        all_blocks.append({
                            'type': 'strassen_2x2x3',
                            'params': (i1, i2, j1, j2, k1),
                            'indices': indices,
                            'rank': 7 + 4,
                            'schoolbook': 12
                        })
    
    # 3x3x2 blocks
    for i1 in range(n - 2):
        for j1 in range(m - 2):
            for k1 in range(p - 1):
                for k2 in range(k1 + 1, p):
                    indices = set()
                    for i in (i1, i1+1, i1+2):
                        for j in (j1, j1+1, j1+2):
                            for k in (k1, k2):
                                indices.add((i, j, k))
                    all_blocks.append({
                        'type': 'strassen_3x3x2',
                        'params': (i1, j1, k1, k2),
                        'indices': indices,
                        'rank': 7 + 7 + 4,  # two 2x2x2 + one 2x1x2
                        'schoolbook': 18
                    })
    
    # 3x2x3 blocks
    for j1 in range(m - 1):
        for j2 in range(j1 + 1, m):
            for k1 in range(p - 2):
                for i1 in range(n - 2):
                    indices = set()
                    for i in (i1, i1+1, i1+2):
                        for j in (j1, j2):
                            for k in (k1, k1+1, k1+2):
                                indices.add((i, j, k))
                    all_blocks.append({
                        'type': 'strassen_3x2x3',
                        'params': (j1, j2, k1, i1),
                        'indices': indices,
                        'rank': 7 + 4 + 4,  # 2x2x2 + two 2x1x2
                        'schoolbook': 18
                    })
    
    # 2x3x3 blocks
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 2):
                for k1 in range(p - 2):
                    indices = set()
                    for i in (i1, i2):
                        for j in (j1, j1+1, j1+2):
                            for k in (k1, k1+1, k1+2):
                                indices.add((i, j, k))
                    all_blocks.append({
                        'type': 'strassen_2x3x3',
                        'params': (i1, i2, j1, k1),
                        'indices': indices,
                        'rank': 7 + 4 + 4,
                        'schoolbook': 18
                    })
    
    # 3x3x3 blocks
    for i1 in range(n - 2):
        for j1 in range(m - 2):
            for k1 in range(p - 2):
                indices = set()
                for i in (i1, i1+1, i1+2):
                    for j in (j1, j1+1, j1+2):
                        for k in (k1, k1+1, k1+2):
                            indices.add((i, j, k))
                all_blocks.append({
                    'type': 'strassen_3x3x3',
                    'params': (i1, j1, k1),
                    'indices': indices,
                    'rank': 7 + 7 + 7 + 7 + 4 + 4 + 4,  # complex decomposition
                    'schoolbook': 27
                })
    
    # Greedy search for maximum disjoint packing with multiple seeds
    best_num_blocks = 0
    best_packing = []
    best_total_rank = 105  # schoolbook
    
    rng = random.Random(42)
    num_trials = 12000
    
    for t in range(num_trials):
        if time.time() > deadline - 20.0:
            break
            
        order = list(range(len(all_blocks)))
        rng.shuffle(order)
        
        used_indices = set()
        current_packing = []
        current_rank = 0
        
        for idx in order:
            block = all_blocks[idx]
            if not (block['indices'] & used_indices):
                current_packing.append(block)
                used_indices |= block['indices']
                current_rank += block['rank']
                
        # Calculate total rank: blocks + schoolbook for uncovered
        uncovered = 105 - len(used_indices)
        total_rank = current_rank + uncovered
        
        if total_rank < best_total_rank:
            best_total_rank = total_rank
            best_packing = current_packing
            best_num_blocks = len(current_packing)
            if best_total_rank <= 85:
                break
    
    # Construct U, V, W based on the best packing found
    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_entries, v_entries, w_entries):
        u = [0] * U_rows
        v = [0] * V_rows
        w = [0] * W_rows
        for idx, val in u_entries.items():
            u[idx] = val
        for idx, val in v_entries.items():
            v[idx] = val
        for idx, val in w_entries.items():
            w[idx] = val
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)

    def add_strassen_2x2x2(i1, i2, j1, j2, k1, k2):
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        
        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2
        
        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2
        
        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})

    def add_strassen_2x3x2(i1, i2, j1, k1, k2):
        # 2x2x2 on (i1,i2) x (j1,j1+1) x (k1,k2)
        add_strassen_2x2x2(i1, i2, j1, j1+1, k1, k2)
        # 2x1x2 on (i1,i2) x (j1+2) x (k1,k2)
        for i in (i1, i2):
            for k in (k1, k2):
                a = i * m + (j1+2)
                b = (j1+2) * p + k
                c = k * n + i
                add_term({a: 1}, {b: 1}, {c: 1})

    def add_strassen_3x2x2(j1, j2, k1, k2, i1):
        # 2x2x2 on (i1,i1+1) x (j1,j2) x (k1,k2)
        add_strassen_2x2x2(i1, i1+1, j1, j2, k1, k2)
        # 1x2x2 on (i1+2) x (j1,j2) x (k1,k2)
        for j in (j1, j2):
            for k in (k1, k2):
                a = (i1+2) * m + j
                b = j * p + k
                c = k * n + (i1+2)
                add_term({a: 1}, {b: 1}, {c: 1})

    def add_strassen_2x2x3(i1, i2, j1, j2, k1):
        # 2x2x2 on (i1,i2) x (j1,j2) x (k1,k1+1)
        add_strassen_2x2x2(i1, i2, j1, j2, k1, k1+1)
        # 2x2x1 on (i1,i2) x (j1,j2) x (k1+2)
        for i in (i1, i2):
            for j in (j1, j2):
                a = i * m + j
                b = j * p + (k1+2)
                c = (k1+2) * n + i
                add_term({a: 1}, {b: 1}, {c: 1})

    def add_strassen_3x3x2(i1, j1, k1, k2):
        # 2x2x2 on (i1,i1+1) x (j1,j1+1) x (k1,k2)
        add_strassen_2x2x2(i1, i1+1, j1, j1+1, k1, k2)
        # 2x2x2 on (i1,i1+1) x (j1+1,j1+2) x (k1,k2) - wait, this overlaps
        # Better: 2x2x2 on (i1,i1+1) x (j1,j1+1) x (k1,k2)
        #         1x2x2 on (i1+2) x (j1,j1+1) x (k1,k2)
        #         2x1x2 on (i1,i1+1) x (j1+2) x (k1,k2)
        #         1x1x2 on (i1+2) x (j1+2) x (k1,k2)
        add_strassen_2x2x2(i1, i1+1, j1, j1+1, k1, k2)
        # 1x2x2
        for j in (j1, j1+1):
            for k in (k1, k2):
                a = (i1+2) * m + j
                b = j * p + k
                c = k * n + (i1+2)
                add_term({a: 1}, {b: 1}, {c: 1})
        # 2x1x2
        for i in (i1, i1+1):
            for k in (k1, k2):
                a = i * m + (j1+2)
                b = (j1+2) * p + k
                c = k * n + i
                add_term({a: 1}, {b: 1}, {c: 1})
        # 1x1x2
        for k in (k1, k2):
            a = (i1+2) * m + (j1+2)
            b = (j1+2) * p + k
            c = k * n + (i1+2)
            add_term({a: 1}, {b: 1}, {c: 1})

    def add_strassen_3x2x3(j1, j2, k1, i1):
        # 2x2x2 on (i1,i1+1) x (j1,j2) x (k1,k1+1)
        add_strassen_2x2x2(i1, i1+1, j1, j2, k1, k1+1)
        # 1x2x2 on (i1+2) x (j1,j2) x (k1,k1+1)
        for j in (j1, j2):
            for k in (k1, k1+1):
                a = (i1+2) * m + j
                b = j * p + k
                c = k * n + (i1+2)
                add_term({a: 1}, {b: 1}, {c: 1})
        # 2x2x1 on (i1,i1+1) x (j1,j2) x (k1+2)
        for i in (i1, i1+1):
            for j in (j1, j2):
                a = i * m + j
                b = j * p + (k1+2)
                c = (k1+2) * n + i
                add_term({a: 1}, {b: 1}, {c: 1})
        # 1x2x1 on (i1+2) x (j1,j2) x (k1+2)
        for j in (j1, j2):
            a = (i1+2) * m + j
            b = j * p + (k1+2)
            c = (k1+2) * n + (i1+2)
            add_term({a: 1}, {b: 1}, {c: 1})

    def add_strassen_2x3x3(i1, i2, j1, k1):
        # 2x2x2 on (i1,i2) x (j1,j1+1) x (k1,k1+1)
        add_strassen_2x2x2(i1, i2, j1, j1+1, k1, k1+1)
        # 2x1x2 on (i1,i2) x (j1+2) x (k1,k1+1)
        for i in (i1, i2):
            for k in (k1, k1+1):
                a = i * m + (j1+2)
                b = (j1+2) * p + k
                c = k * n + i
                add_term({a: 1}, {b: 1}, {c: 1})
        # 2x2x1 on (i1,i2) x (j1,j1+1) x (k1+2)
        for i in (i1, i2):
            for j in (j1, j1+1):
                a = i * m + j
                b = j * p + (k1+2)
                c = (k1+2) * n + i
                add_term({a: 1}, {b: 1}, {c: 1})
        # 2x1x1 on (i1,i2) x (j1+2) x (k1+2)
        for i in (i1, i2):
            a = i * m + (j1+2)
            b = (j1+2) * p + (k1+2)
            c = (k1+2) * n + i
            add_term({a: 1}, {b: 1}, {c: 1})

    def add_strassen_3x3x3(i1, j1, k1):
        # Decompose into smaller pieces
        # 2x2x2 on (i1,i1+1) x (j1,j1+1) x (k1,k1+1)
        add_strassen_2x2x2(i1, i1+1, j1, j1+1, k1, k1+1)
        # 1x2x2 on (i1+2) x (j1,j1+1) x (k1,k1+1)
        for j in (j1, j1+1):
            for k in (k1, k1+1):
                a = (i1+2) * m + j
                b = j * p + k
                c = k * n + (i1+2)
                add_term({a: 1}, {b: 1}, {c: 1})
        # 2x1x2 on (i1,i1+1) x (j1+2) x (k1,k1+1)
        for i in (i1, i1+1):
            for k in (k1, k1+1):
                a = i * m + (j1+2)
                b = (j1+2) * p + k
                c = k * n + i
                add_term({a: 1}, {b: 1}, {c: 1})
        # 1x1x2 on (i1+2) x (j1+2) x (k1,k1+1)
        for k in (k1, k1+1):
            a = (i1+2) * m + (j1+2)
            b = (j1+2) * p + k
            c = k * n + (i1+2)
            add_term({a: 1}, {b: 1}, {c: 1})
        # 2x2x1 on (i1,i1+1) x (j1,j1+1) x (k1+2)
        for i in (i1, i1+1):
            for j in (j1, j1+1):
                a = i * m + j
                b = j * p + (k1+2)
                c = (k1+2) * n + i
                add_term({a: 1}, {b: 1}, {c: 1})
        # 1x2x1 on (i1+2) x (j1,j1+1) x (k1+2)
        for j in (j1, j1+1):
            a = (i1+2) * m + j
            b = j * p + (k1+2)
            c = (k1+2) * n + (i1+2)
            add_term({a: 1}, {b: 1}, {c: 1})
        # 2x1x1 on (i1,i1+1) x (j1+2) x (k1+2)
        for i in (i1, i1+1):
            a = i * m + (j1+2)
            b = (j1+2) * p + (k1+2)
            c = (k1+2) * n + i
            add_term({a: 1}, {b: 1}, {c: 1})
        # 1x1x1 on (i1+2) x (j1+2) x (k1+2)
        a = (i1+2) * m + (j1+2)
        b = (j1+2) * p + (k1+2)
        c = (k1+2) * n + (i1+2)
        add_term({a: 1}, {b: 1}, {c: 1})

    covered = set()
    for block in best_packing:
        if time.time() > deadline - 10.0:
            break
        btype = block['type']
        params = block['params']
        if btype == 'strassen_2x2x2':
            add_strassen_2x2x2(*params)
        elif btype == 'strassen_2x3x2':
            add_strassen_2x3x2(*params)
        elif btype == 'strassen_3x2x2':
            add_strassen_3x2x2(*params)
        elif btype == 'strassen_2x2x3':
            add_strassen_2x2x3(*params)
        elif btype == 'strassen_3x3x2':
            add_strassen_3x3x2(*params)
        elif btype == 'strassen_3x2x3':
            add_strassen_3x2x3(*params)
        elif btype == 'strassen_2x3x3':
            add_strassen_2x3x3(*params)
        elif btype == 'strassen_3x3x3':
            add_strassen_3x3x3(*params)
        covered |= block['indices']

    # Add schoolbook terms for uncovered (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline - 5.0:
                    break
                if (i, j, k) not in covered:
                    add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

    rank = len(U_cols)
    
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    for c in range(rank):
        for r in range(U_rows):
            U[r][c] = U_cols[c][r]
        for r in range(V_rows):
            V[r][c] = V_cols[c][r]
        for r in range(W_rows):
            W[r][c] = W_cols[c][r]

    return {"U": U, "V": V, "W": W}
