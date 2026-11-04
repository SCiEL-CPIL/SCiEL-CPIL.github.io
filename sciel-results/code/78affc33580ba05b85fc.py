import time
import random

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21
    
    # We'll build the decomposition by splitting dimensions
    # Strategy: Split n=3 into 1+2, m=5 into 2+3, p=7 into 3+4
    # This gives 4 sub-problems:
    # (1,2,3), (1,3,4), (2,2,3), (2,3,4)
    
    # For each sub-problem, we need a decomposition
    # We'll use a combination of Strassen blocks and schoolbook
    
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
    
    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        """Add a 2x2x2 Strassen block using 7 multiplications"""
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
        
        # M1 = (a11+a22)*(b11+b22)
        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
        
        # M2 = (a21+a22)*b11
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
        
        # M3 = a11*(b12-b22)
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})
        
        # M4 = a22*(b21-b11)
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})
        
        # M5 = (a11+a12)*b22
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
        
        # M6 = (a21-a11)*(b11+b12)
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
        
        # M7 = (a12-a22)*(b21+b22)
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})
    
    def add_rectangular_block(n_sub, m_sub, p_sub, i_offset, j_offset, k_offset):
        """
        Add a decomposition for an n_sub x m_sub x p_sub sub-problem
        at offsets (i_offset, j_offset, k_offset)
        """
        # For 1xmxn: schoolbook (mn terms)
        if n_sub == 1:
            for j in range(m_sub):
                for k in range(p_sub):
                    i = i_offset
                    jj = j_offset + j
                    kk = k_offset + k
                    add_term(
                        {i * m + jj: 1},
                        {jj * p + kk: 1},
                        {kk * n + i: 1}
                    )
            return
        
        # For 2x2x2: use Strassen (7 terms)
        if n_sub == 2 and m_sub == 2 and p_sub == 2:
            add_strassen_block(
                i_offset, i_offset + 1,
                j_offset, j_offset + 1,
                k_offset, k_offset + 1
            )
            return
        
        # For other sizes, use a greedy packing of Strassen blocks + schoolbook
        # Generate all possible 2x2x2 sub-blocks within this rectangular region
        all_sub_blocks = []
        for si1 in range(n_sub - 1):
            for si2 in range(si1 + 1, n_sub):
                for sj1 in range(m_sub - 1):
                    for sj2 in range(sj1 + 1, m_sub):
                        for sk1 in range(p_sub - 1):
                            for sk2 in range(sk1 + 1, p_sub):
                                indices = set()
                                for si in (si1, si2):
                                    for sj in (sj1, sj2):
                                        for sk in (sk1, sk2):
                                            indices.add((si, sj, sk))
                                all_sub_blocks.append(
                                    ((si1, si2, sj1, sj2, sk1, sk2), indices)
                                )
        
        # Greedy packing with randomization
        best_packing = []
        best_count = 0
        rng = random.Random(hash((n_sub, m_sub, p_sub)) % 10000)
        
        num_trials = 200
        for t in range(num_trials):
            if time.time() > deadline - 10.0:
                break
            order = list(range(len(all_sub_blocks)))
            rng.shuffle(order)
            used_indices = set()
            current_packing = []
            
            for idx in order:
                params, block_indices = all_sub_blocks[idx]
                if not (block_indices & used_indices):
                    current_packing.append(params)
                    used_indices |= block_indices
            
            if len(current_packing) > best_count:
                best_count = len(current_packing)
                best_packing = current_packing
        
        # Add Strassen blocks from best packing
        covered = set()
        for params in best_packing:
            si1, si2, sj1, sj2, sk1, sk2 = params
            add_strassen_block(
                i_offset + si1, i_offset + si2,
                j_offset + sj1, j_offset + sj2,
                k_offset + sk1, k_offset + sk2
            )
            for si in (si1, si2):
                for sj in (sj1, sj2):
                    for sk in (sk1, sk2):
                        covered.add((si, sj, sk))
        
        # Add schoolbook for uncovered
        for si in range(n_sub):
            for sj in range(m_sub):
                for sk in range(p_sub):
                    if (si, sj, sk) not in covered:
                        i = i_offset + si
                        jj = j_offset + sj
                        kk = k_offset + sk
                        add_term(
                            {i * m + jj: 1},
                            {jj * p + kk: 1},
                            {kk * n + i: 1}
                        )
    
    # Split the dimensions: n=3 into 1+2, m=5 into 2+3, p=7 into 3+4
    n_splits = [1, 2]
    m_splits = [2, 3]
    p_splits = [3, 4]
    
    i_offset = 0
    for n_sub in n_splits:
        j_offset = 0
        for m_sub in m_splits:
            k_offset = 0
            for p_sub in p_splits:
                add_rectangular_block(n_sub, m_sub, p_sub, i_offset, j_offset, k_offset)
                k_offset += p_sub
            j_offset += m_sub
        i_offset += n_sub
    
    rank = len(U_cols)
    
    # Convert to matrix format
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
