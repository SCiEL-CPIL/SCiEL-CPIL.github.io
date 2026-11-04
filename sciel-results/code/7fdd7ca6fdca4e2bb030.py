"""Matrix multiplication 3x5 by 5x6: rank-minimal tensor decomposition."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
from fractions import Fraction

_start_time = time.time()
TIME_LIMIT = 120.0

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    def get_rank_2x2x2():
        return 7  # Strassen
    
    def get_rank_2x2x1():
        return 4
    
    def get_rank_2x1x2():
        return 4
    
    def get_rank_2x1x1():
        return 2
    
    def get_rank_1x2x2():
        return 4
    
    def get_rank_1x2x1():
        return 2
    
    def get_rank_1x1x2():
        return 2
    
    def get_rank_1x1x1():
        return 1
    
    def get_rank_1x1x1_2x1x1_1x1x1():
        return 3
    
    def block_rank(n_size, m_size, p_size):
        if n_size == 2 and m_size == 2 and p_size == 2:
            return 7
        elif n_size == 2 and m_size == 2 and p_size == 1:
            return 4
        elif n_size == 2 and m_size == 1 and p_size == 2:
            return 4
        elif n_size == 2 and m_size == 1 and p_size == 1:
            return 2
        elif n_size == 1 and m_size == 2 and p_size == 2:
            return 4
        elif n_size == 1 and m_size == 2 and p_size == 1:
            return 2
        elif n_size == 1 and m_size == 1 and p_size == 2:
            return 2
        elif n_size == 1 and m_size == 1 and p_size == 1:
            return 1
        else:
            return n_size * m_size * p_size  # schoolbook
    
    def generate_partitions(dim, max_block=2):
        """Generate all ways to partition dim into blocks of size 1 or max_block."""
        if dim == 0:
            yield []
            return
        if dim == 1:
            yield [1]
            return
        if dim == 2:
            yield [2]
            yield [1, 1]
            return
        if dim == 3:
            yield [2, 1]
            yield [1, 2]
            yield [1, 1, 1]
            return
        if dim == 4:
            yield [2, 2]
            yield [2, 1, 1]
            yield [1, 2, 1]
            yield [1, 1, 2]
            yield [1, 1, 1, 1]
            return
        if dim == 5:
            yield [2, 2, 1]
            yield [2, 1, 2]
            yield [1, 2, 2]
            yield [2, 1, 1, 1]
            yield [1, 2, 1, 1]
            yield [1, 1, 2, 1]
            yield [1, 1, 1, 2]
            yield [1, 1, 1, 1, 1]
            return
        if dim == 6:
            yield [2, 2, 2]
            yield [2, 2, 1, 1]
            yield [2, 1, 2, 1]
            yield [2, 1, 1, 2]
            yield [1, 2, 2, 1]
            yield [1, 2, 1, 2]
            yield [1, 1, 2, 2]
            yield [2, 1, 1, 1, 1]
            yield [1, 2, 1, 1, 1]
            yield [1, 1, 2, 1, 1]
            yield [1, 1, 1, 2, 1]
            yield [1, 1, 1, 1, 2]
            yield [1, 1, 1, 1, 1, 1]
            return
        # Fallback
        for i in range(1, min(dim, max_block) + 1):
            for rest in generate_partitions(dim - i, max_block):
                yield [i] + rest
    
    def build_decomposition(n_parts, m_parts, p_parts):
        """Build the full decomposition from block partitions."""
        # Compute block start indices
        n_starts = [0]
        for s in n_parts[:-1]:
            n_starts.append(n_starts[-1] + s)
        m_starts = [0]
        for s in m_parts[:-1]:
            m_starts.append(m_starts[-1] + s)
        p_starts = [0]
        for s in p_parts[:-1]:
            p_starts.append(p_starts[-1] + s)
        
        terms = []
        
        for ni, ns in enumerate(n_starts):
            ne = ns + n_parts[ni]
            for mi, ms in enumerate(m_starts):
                me = ms + m_parts[mi]
                for pi, ps in enumerate(p_starts):
                    pe = ps + p_parts[pi]
                    
                    n_size = ne - ns
                    m_size = me - ms
                    p_size = pe - ps
                    
                    a_rows = list(range(ns, ne))
                    c_cols = list(range(ms, me))
                    d_cols = list(range(ps, pe))
                    
                    if n_size == 2 and m_size == 2 and p_size == 2:
                        # Strassen 7 terms
                        # M1 = (A00+A11)(B00+B11)
                        terms.append((
                            [(a_rows[0]*m+c_cols[0], 1), (a_rows[1]*m+c_cols[1], 1)],
                            [(c_cols[0]*p+d_cols[0], 1), (c_cols[1]*p+d_cols[1], 1)],
                            [(d_cols[0]*n+a_rows[0], 1), (d_cols[1]*n+a_rows[1], 1)]
                        ))
                        # M2 = (A10+A11)B00
                        terms.append((
                            [(a_rows[1]*m+c_cols[0], 1), (a_rows[1]*m+c_cols[1], 1)],
                            [(c_cols[0]*p+d_cols[0], 1)],
                            [(d_cols[0]*n+a_rows[0], 1), (d_cols[1]*n+a_rows[1], -1)]
                        ))
                        # M3 = A00(B01-B11)
                        terms.append((
                            [(a_rows[0]*m+c_cols[0], 1)],
                            [(c_cols[0]*p+d_cols[1], 1), (c_cols[1]*p+d_cols[1], -1)],
                            [(d_cols[1]*n+a_rows[0], 1), (d_cols[1]*n+a_rows[1], 1)]
                        ))
                        # M4 = A11(B10-B00)
                        terms.append((
                            [(a_rows[1]*m+c_cols[1], 1)],
                            [(c_cols[1]*p+d_cols[0], 1), (c_cols[0]*p+d_cols[0], -1)],
                            [(d_cols[0]*n+a_rows[0], 1), (d_cols[0]*n+a_rows[1], 1)]
                        ))
                        # M5 = (A00+A01)B11
                        terms.append((
                            [(a_rows[0]*m+c_cols[0], 1), (a_rows[0]*m+c_cols[1], 1)],
                            [(c_cols[1]*p+d_cols[1], 1)],
                            [(d_cols[0]*n+a_rows[0], -1), (d_cols[1]*n+a_rows[0], 1)]
                        ))
                        # M6 = (A10-A00)(B00+B01)
                        terms.append((
                            [(a_rows[1]*m+c_cols[0], 1), (a_rows[0]*m+c_cols[0], -1)],
                            [(c_cols[0]*p+d_cols[0], 1), (c_cols[0]*p+d_cols[1], 1)],
                            [(d_cols[1]*n+a_rows[1], 1)]
                        ))
                        # M7 = (A01-A11)(B10+B11)
                        terms.append((
                            [(a_rows[0]*m+c_cols[1], 1), (a_rows[1]*m+c_cols[1], -1)],
                            [(c_cols[1]*p+d_cols[0], 1), (c_cols[1]*p+d_cols[1], 1)],
                            [(d_cols[0]*n+a_rows[0], 1)]
                        ))
                    elif n_size == 2 and m_size == 2 and p_size == 1:
                        # Schoolbook 2x2x1: 4 terms
                        terms.append(([(a_rows[0]*m+c_cols[0], 1)], [(c_cols[0]*p+d_cols[0], 1)], [(d_cols[0]*n+a_rows[0], 1)]))
                        terms.append(([(a_rows[0]*m+c_cols[1], 1)], [(c_cols[1]*p+d_cols[0], 1)], [(d_cols[0]*n+a_rows[0], 1)]))
                        terms.append(([(a_rows[1]*m+c_cols[0], 1)], [(c_cols[0]*p+d_cols[0], 1)], [(d_cols[0]*n+a_rows[1], 1)]))
                        terms.append(([(a_rows[1]*m+c_cols[1], 1)], [(c_cols[1]*p+d_cols[0], 1)], [(d_cols[0]*n+a_rows[1], 1)]))
                    elif n_size == 2 and m_size == 1 and p_size == 2:
                        # Schoolbook 2x1x2: 4 terms
                        terms.append(([(a_rows[0]*m+c_cols[0], 1)], [(c_cols[0]*p+d_cols[0], 1)], [(d_cols[0]*n+a_rows[0], 1)]))
                        terms.append(([(a_rows[0]*m+c_cols[0], 1)], [(c_cols[0]*p+d_cols[1], 1)], [(d_cols[1]*n+a_rows[0], 1)]))
                        terms.append(([(a_rows[1]*m+c_cols[0], 1)], [(c_cols[0]*p+d_cols[0], 1)], [(d_cols[0]*n+a_rows[1], 1)]))
                        terms.append(([(a_rows[1]*m+c_cols[0], 1)], [(c_cols[0]*p+d_cols[1], 1)], [(d_cols[1]*n+a_rows[1], 1)]))
                    elif n_size == 2 and m_size == 1 and p_size == 1:
                        # Schoolbook 2x1x1: 2 terms
                        terms.append(([(a_rows[0]*m+c_cols[0], 1)], [(c_cols[0]*p+d_cols[0], 1)], [(d_cols[0]*n+a_rows[0], 1)]))
                        terms.append(([(a_rows[1]*m+c_cols[0], 1)], [(c_cols[0]*p+d_cols[0], 1)], [(d_cols[0]*n+a_rows[1], 1)]))
                    elif n_size == 1 and m_size == 2 and p_size == 2:
                        # Schoolbook 1x2x2: 4 terms
                        terms.append(([(a_rows[0]*m+c_cols[0], 1)], [(c_cols[0]*p+d_cols[0], 1)], [(d_cols[0]*n+a_rows[0], 1)]))
                        terms.append(([(a_rows[0]*m+c_cols[1], 1)], [(c_cols[1]*p+d_cols[0], 1)], [(d_cols[0]*n+a_rows[0], 1)]))
                        terms.append(([(a_rows[0]*m+c_cols[0], 1)], [(c_cols[0]*p+d_cols[1], 1)], [(d_cols[1]*n+a_rows[0], 1)]))
                        terms.append(([(a_rows[0]*m+c_cols[1], 1)], [(c_cols[1]*p+d_cols[1], 1)], [(d_cols[1]*n+a_rows[0], 1)]))
                    elif n_size == 1 and m_size == 2 and p_size == 1:
                        # Schoolbook 1x2x1: 2 terms
                        terms.append(([(a_rows[0]*m+c_cols[0], 1)], [(c_cols[0]*p+d_cols[0], 1)], [(d_cols[0]*n+a_rows[0], 1)]))
                        terms.append(([(a_rows[0]*m+c_cols[1], 1)], [(c_cols[1]*p+d_cols[0], 1)], [(d_cols[0]*n+a_rows[0], 1)]))
                    elif n_size == 1 and m_size == 1 and p_size == 2:
                        # Schoolbook 1x1x2: 2 terms
                        terms.append(([(a_rows[0]*m+c_cols[0], 1)], [(c_cols[0]*p+d_cols[0], 1)], [(d_cols[0]*n+a_rows[0], 1)]))
                        terms.append(([(a_rows[0]*m+c_cols[0], 1)], [(c_cols[0]*p+d_cols[1], 1)], [(d_cols[1]*n+a_rows[0], 1)]))
                    elif n_size == 1 and m_size == 1 and p_size == 1:
                        # Schoolbook 1x1x1: 1 term
                        terms.append(([(a_rows[0]*m+c_cols[0], 1)], [(c_cols[0]*p+d_cols[0], 1)], [(d_cols[0]*n+a_rows[0], 1)]))
                    else:
                        raise ValueError(f"Unhandled block: {n_size}x{m_size}x{p_size}")
        
        return terms
    
    def verify_decomposition(terms, n, m, p):
        """Verify that the decomposition is correct."""
        rank = len(terms)
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        for ci, (u_entries, v_entries, w_entries) in enumerate(terms):
            for row, coeff in u_entries:
                U[row, ci] += coeff
            for row, coeff in v_entries:
                V[row, ci] += coeff
            for row, coeff in w_entries:
                W[row, ci] += coeff
        
        # Check all valid triples
        for a in range(n):
            for c in range(m):
                for b in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[a*m+c, r] * V[c*p+b, r] * W[b*n+a, r]
                    if abs(s - 1.0) > 1e-10:
                        return False, s
        
        # Check some invalid triples
        for a in range(n):
            for c in range(m):
                for b in range(p):
                    for b2 in range(p):
                        if b2 != b:
                            s = 0
                            for r in range(rank):
                                s += U[a*m+c, r] * V[c*p+b2, r] * W[b*n+a, r]
                            if abs(s) > 1e-10:
                                return False, s
        
        return True, None
    
    # Search over all partitions
    best_rank = float('inf')
    best_terms = None
    best_parts = None
    
    n_partitions = list(generate_partitions(n, 2))
    m_partitions = list(generate_partitions(m, 2))
    p_partitions = list(generate_partitions(p, 2))
    
    for n_parts in n_partitions:
        for m_parts in m_partitions:
            for p_parts in p_partitions:
                if time.time() - _start_time > TIME_LIMIT - 30:
                    break
                
                # Estimate rank
                est_rank = 0
                n_starts = [0]
                for s in n_parts[:-1]:
                    n_starts.append(n_starts[-1] + s)
                m_starts = [0]
                for s in m_parts[:-1]:
                    m_starts.append(m_starts[-1] + s)
                p_starts = [0]
                for s in p_parts[:-1]:
                    p_starts.append(p_starts[-1] + s)
                
                for ni, ns in enumerate(n_starts):
                    ne = ns + n_parts[ni]
                    for mi, ms in enumerate(m_starts):
                        me = ms + m_parts[mi]
                        for pi, ps in enumerate(p_starts):
                            pe = ps + p_parts[pi]
                            n_size = ne - ns
                            m_size = me - ms
                            p_size = pe - ps
                            est_rank += block_rank(n_size, m_size, p_size)
                
                if est_rank < best_rank:
                    # Try to build and verify
                    try:
                        terms = build_decomposition(n_parts, m_parts, p_parts)
                        actual_rank = len(terms)
                        if actual_rank < best_rank:
                            valid, _ = verify_decomposition(terms, n, m, p)
                            if valid:
                                best_rank = actual_rank
                                best_terms = terms
                                best_parts = (n_parts, m_parts, p_parts)
                    except Exception as e:
                        pass
    
    if best_terms is None:
        # Fallback: use the original partition
        best_terms = build_decomposition([2,1], [2,2,1], [2,2,2])
        best_rank = len(best_terms)
    
    # Build final matrices
    rank = best_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for ci, (u_entries, v_entries, w_entries) in enumerate(best_terms):
        for row, coeff in u_entries:
            U[row][ci] += coeff
        for row, coeff in v_entries:
            V[row][ci] += coeff
        for row, coeff in w_entries:
            W[row][ci] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
