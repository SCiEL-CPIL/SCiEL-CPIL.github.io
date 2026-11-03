"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

_start_time = time.time()

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    deadline = _start_time + 130.0
    
    def generate_partitions(total):
        """Generate all ways to partition 'total' into parts of size 1, 2, or 3."""
        if total == 0:
            return [[]]
        results = []
        for first in [1, 2, 3]:
            if first > total:
                continue
            for rest in generate_partitions(total - first):
                results.append([first] + rest)
        return results
    
    def block_rank(n_s, m_s, p_s):
        """Return the rank needed for an n_s x m_s x p_s block."""
        if n_s == 1 and m_s == 1 and p_s == 1:
            return 1
        if n_s == 2 and m_s == 2 and p_s == 2:
            return 7  # Strassen
        # Otherwise schoolbook
        return n_s * m_s * p_s
    
    def build_decomposition(n_parts, m_parts, p_parts, n, m, p):
        """Build the full decomposition from a partitioning. Returns (U, V, W) as lists."""
        terms = []  # each: (u_entries, v_entries, w_entries)
        
        # Compute offsets
        n_offsets = [0]
        for sz in n_parts[:-1]:
            n_offsets.append(n_offsets[-1] + sz)
        m_offsets = [0]
        for sz in m_parts[:-1]:
            m_offsets.append(m_offsets[-1] + sz)
        p_offsets = [0]
        for sz in p_parts[:-1]:
            p_offsets.append(p_offsets[-1] + sz)
        
        for ni, ns in enumerate(n_parts):
            for mi, ms in enumerate(m_parts):
                for pi, ps in enumerate(p_parts):
                    a_rows = list(range(n_offsets[ni], n_offsets[ni] + ns))
                    c_cols = list(range(m_offsets[mi], m_offsets[mi] + ms))
                    d_cols = list(range(p_offsets[pi], p_offsets[pi] + ps))
                    
                    def U_idx(la, lc): return a_rows[la] * m + c_cols[lc]
                    def V_idx(lc, ld): return c_cols[lc] * p + d_cols[ld]
                    def W_idx(ld, la): return d_cols[ld] * n + a_rows[la]
                    
                    if ns == 2 and ms == 2 and ps == 2:
                        # Strassen 7 terms
                        a0, a1 = a_rows
                        c0, c1 = c_cols
                        d0, d1 = d_cols
                        
                        # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
                        # C[0,0] += M1, C[1,1] += M1
                        terms.append((
                            [(U_idx(0,0), 1), (U_idx(1,1), 1)],
                            [(V_idx(0,0), 1), (V_idx(1,1), 1)],
                            [(W_idx(0,0), 1), (W_idx(1,1), 1)]
                        ))
                        
                        # M2 = (A[1,0]+A[1,1])*B[0,0]
                        # C[1,0] += M2, C[1,1] += -M2
                        terms.append((
                            [(U_idx(1,0), 1), (U_idx(1,1), 1)],
                            [(V_idx(0,0), 1)],
                            [(W_idx(0,1), 1), (W_idx(1,1), -1)]
                        ))
                        
                        # M3 = A[0,0]*(B[0,1]-B[1,1])
                        # C[0,1] += M3, C[1,1] += M3
                        terms.append((
                            [(U_idx(0,0), 1)],
                            [(V_idx(0,1), 1), (V_idx(1,1), -1)],
                            [(W_idx(1,0), 1), (W_idx(1,1), 1)]
                        ))
                        
                        # M4 = A[1,1]*(B[1,0]-B[0,0])
                        # C[0,0] += M4, C[1,0] += M4
                        terms.append((
                            [(U_idx(1,1), 1)],
                            [(V_idx(1,0), 1), (V_idx(0,0), -1)],
                            [(W_idx(0,0), 1), (W_idx(0,1), 1)]
                        ))
                        
                        # M5 = (A[0,0]+A[0,1])*B[1,1]
                        # C[0,0] += -M5, C[0,1] += M5
                        terms.append((
                            [(U_idx(0,0), 1), (U_idx(0,1), 1)],
                            [(V_idx(1,1), 1)],
                            [(W_idx(0,0), -1), (W_idx(1,0), 1)]
                        ))
                        
                        # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
                        # C[1,1] += M6
                        terms.append((
                            [(U_idx(1,0), 1), (U_idx(0,0), -1)],
                            [(V_idx(0,0), 1), (V_idx(0,1), 1)],
                            [(W_idx(1,1), 1)]
                        ))
                        
                        # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
                        # C[0,0] += M7
                        terms.append((
                            [(U_idx(0,1), 1), (U_idx(1,1), -1)],
                            [(V_idx(1,0), 1), (V_idx(1,1), 1)],
                            [(W_idx(0,0), 1)]
                        ))
                    else:
                        # Schoolbook for this block
                        for la in range(ns):
                            for lc in range(ms):
                                for ld in range(ps):
                                    terms.append((
                                        [(U_idx(la, lc), 1)],
                                        [(V_idx(lc, ld), 1)],
                                        [(W_idx(ld, la), 1)]
                                    ))
        
        # Build matrices
        rank = len(terms)
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        for ci, (u_entries, v_entries, w_entries) in enumerate(terms):
            for row, coeff in u_entries:
                U[row][ci] += coeff
            for row, coeff in v_entries:
                V[row][ci] += coeff
            for row, coeff in w_entries:
                W[row][ci] += coeff
        
        return U, V, W, rank
    
    def verify(U_mat, V_mat, W_mat, n, m, p, tol=1e-9):
        """Check if U,V,W correctly decompose the matmul tensor."""
        r = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def verify_fast(U_mat, V_mat, W_mat, n, m, p, tol=1e-9):
        """Faster verification."""
        r = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_mat[i*m+j, :], V_mat[j*p+k, :] * W_mat[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try all possible partitionings
    n_partitions = generate_partitions(n)
    m_partitions = generate_partitions(m)
    p_partitions = generate_partitions(p)
    
    best_rank = float('inf')
    best_U = best_V = best_W = None
    best_partition = None
    
    for np_ in n_partitions:
        for mp_ in m_partitions:
            for pp_ in p_partitions:
                if time.time() > deadline:
                    break
                
                # Calculate total rank for this partitioning
                total_rank = 0
                for ns, ms, ps in product(np_, mp_, pp_):
                    total_rank += block_rank(ns, ms, ps)
                
                if total_rank >= best_rank:
                    continue
                
                U, V, W, rank = build_decomposition(np_, mp_, pp_, n, m, p)
                
                # Verify
                U_arr = np.array(U)
                V_arr = np.array(V)
                W_arr = np.array(W)
                
                if not verify_fast(U_arr, V_arr, W_arr, n, m, p):
                    continue
                
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
                best_partition = (np_, mp_, pp_)
    
    if best_U is None:
        # Fallback: pure schoolbook
        U, V, W, rank = build_decomposition([n], [m], [p], n, m, p)
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank via column elimination
    U_arr = np.array(best_U, dtype=np.float64)
    V_arr = np.array(best_V, dtype=np.float64)
    W_arr = np.array(best_W, dtype=np.float64)
    current_rank = best_rank
    
    for _ in range(3):  # Multiple rounds of elimination
        if time.time() > deadline:
            break
        found = False
        for col in range(current_rank):
            if time.time() > deadline:
                break
            if current_rank <= 1:
                break
            U_test = np.delete(U_arr, col, axis=1)
            V_test = np.delete(V_arr, col, axis=1)
            W_test = np.delete(W_arr, col, axis=1)
            if verify_fast(U_test, V_test, W_test, n, m, p):
                U_arr, V_arr, W_arr = U_test, V_test, W_test
                current_rank -= 1
                found = True
                break
        if not found:
            break
    
    # Clean up entries to be integers or half-integers
    U_final = U_arr.tolist()
    V_final = V_arr.tolist()
    W_final = W_arr.tolist()
    
    for mat in [U_final, V_final, W_final]:
        for row in mat:
            for idx in range(len(row)):
                val = row[idx]
                rounded = round(val * 2) / 2.0
                if abs(val - rounded) < 1e-9:
                    row[idx] = rounded
                else:
                    row[idx] = round(val, 10)
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
