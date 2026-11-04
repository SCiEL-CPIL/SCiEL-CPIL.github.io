import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # We will store the terms as lists of (row, coeff) for U, V, W
    # Then convert to dense matrices at the end.
    # Rank is determined by the number of terms.
    
    terms = []
    
    def add_term(u_entries, v_entries, w_entries):
        """
        u_entries: list of (local_i, local_j, coeff)
        v_entries: list of (local_j, local_k, coeff)
        w_entries: list of (local_k, local_i, coeff)
        """
        terms.append((u_entries, v_entries, w_entries))
        
    def decompose(n_sub, m_sub, p_sub, i_off, j_off, k_off):
        if n_sub == 2 and m_sub == 2 and p_sub == 2:
            # Strassen
            # Term 1
            add_term(
                [(0,0,1), (1,1,1)],
                [(0,0,1), (1,1,1)],
                [(0,0,1), (1,1,1)]
            )
            # Term 2
            add_term(
                [(1,0,1), (1,1,1)],
                [(0,0,1)],
                [(0,1,1), (1,1,-1)]
            )
            # Term 3
            add_term(
                [(0,0,1)],
                [(0,1,1), (1,1,-1)],
                [(1,0,1), (1,1,1)]
            )
            # Term 4
            add_term(
                [(1,1,1)],
                [(1,0,1), (0,0,-1)],
                [(0,0,1), (0,1,1)]
            )
            # Term 5
            add_term(
                [(0,0,1), (0,1,1)],
                [(1,1,1)],
                [(0,0,-1), (1,0,1)]
            )
            # Term 6
            add_term(
                [(1,0,1), (0,0,-1)],
                [(0,0,1), (0,1,1)],
                [(1,1,1)]
            )
            # Term 7
            add_term(
                [(0,1,1), (1,1,-1)],
                [(1,0,1), (1,1,1)],
                [(0,0,1)]
            )
        elif m_sub > 2 and m_sub % 2 == 0:
            m_half = m_sub // 2
            decompose(n_sub, m_half, p_sub, i_off, j_off, k_off)
            decompose(n_sub, m_half, p_sub, i_off, j_off + m_half, k_off)
        elif p_sub > 2 and p_sub % 2 == 0:
            p_half = p_sub // 2
            decompose(n_sub, m_sub, p_half, i_off, j_off, k_off)
            decompose(n_sub, m_sub, p_half, i_off, j_off, k_off + p_half)
        elif n_sub > 2:
            n1 = 2
            n2 = n_sub - 2
            decompose(n1, m_sub, p_sub, i_off, j_off, k_off)
            decompose(n2, m_sub, p_sub, i_off + n1, j_off, k_off)
        else:
            # Schoolbook
            for i in range(n_sub):
                for j in range(m_sub):
                    for k in range(p_sub):
                        add_term(
                            [(i, j, 1)],
                            [(j, k, 1)],
                            [(k, i, 1)]
                        )
                        
    decompose(n, m, p, 0, 0, 0)
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_entries, v_entries, w_entries) in enumerate(terms):
        for (li, lj, c) in u_entries:
            gi = li # local i is relative to subproblem, but we passed offsets?
            # Wait, the offsets were passed to decompose, but add_term uses local coords.
            # We need to apply offsets in add_term or after.
            # It's better to apply offsets when adding to global matrices.
            # But we stored local coords in `terms`.
            # We lost the offsets in `terms`!
            # We need to store the offsets or apply them during generation.
            
            # Correction: Store global row indices directly in terms?
            # Or re-run and pass offsets to a helper that builds the matrices directly.
            
            pass # This block is invalid logic for the current storage.

    # Let's restart the accumulation part to be safe.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We need to re-generate or store offsets.
    # Let's modify `add_term` to accept offsets or store them.
    # Easiest: Re-implement the recursion to build U, V, W directly.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    rank_counter = [0]
    
    def add_term_direct(u_entries, v_entries, w_entries, i_off, j_off, k_off):
        r = rank_counter[0]
        rank_counter[0] += 1
        for (li, lj, c) in u_entries:
            gi = i_off + li
            gj = j_off + lj
            U[gi * m + gj][r] = c
        for (lj, lk, c) in v_entries:
            gj = j_off + lj
            gk = k_off + lk
            V[gj * p + gk][r] = c
        for (lk, li, c) in w_entries:
            gk = k_off + lk
            gi = i_off + li
            W[gk * n + gi][r] = c

    def decompose_direct(n_sub, m_sub, p_sub, i_off, j_off, k_off):
        if n_sub == 2 and m_sub == 2 and p_sub == 2:
            add_term_direct([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], i_off, j_off, k_off)
            add_term_direct([(1,0,1), (1,1,1)], [(0,0,1)], [(0,1,1), (1,1,-1)], i_off, j_off, k_off)
            add_term_direct([(0,0,1)], [(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], i_off, j_off, k_off)
            add_term_direct([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], i_off, j_off, k_off)
            add_term_direct([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (1,0,1)], i_off, j_off, k_off)
            add_term_direct([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)], i_off, j_off, k_off)
            add_term_direct([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)], i_off, j_off, k_off)
        elif m_sub > 2 and m_sub % 2 == 0:
            m_half = m_sub // 2
            decompose_direct(n_sub, m_half, p_sub, i_off, j_off, k_off)
            decompose_direct(n_sub, m_half, p_sub, i_off, j_off + m_half, k_off)
        elif p_sub > 2 and p_sub % 2 == 0:
            p_half = p_sub // 2
            decompose_direct(n_sub, m_sub, p_half, i_off, j_off, k_off)
            decompose_direct(n_sub, m_sub, p_half, i_off, j_off, k_off + p_half)
        elif n_sub > 2:
            n1 = 2
            n2 = n_sub - 2
            decompose_direct(n1, m_sub, p_sub, i_off, j_off, k_off)
            decompose_direct(n2, m_sub, p_sub, i_off + n1, j_off, k_off)
        else:
            for i in range(n_sub):
                for j in range(m_sub):
                    for k in range(p_sub):
                        add_term_direct([(i, j, 1)], [(j, k, 1)], [(k, i, 1)], i_off, j_off, k_off)

    decompose_direct(n, m, p, 0, 0, 0)
    
    return {"U": U, "V": V, "W": W}
