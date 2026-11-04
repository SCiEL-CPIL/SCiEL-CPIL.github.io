import numpy as np
import time

def construct():
    n, m, p = 3, 4, 8
    
    # Target rank 16
    rank = 16
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Decomposition strategy:
    # R(3,4,8) <= R(3,4,4) + R(3,4,4)
    # R(3,4,4) <= R(3,2,4) + R(3,2,4)
    # R(3,2,4) <= R(3,2,2) + R(3,2,2)
    # R(3,2,2) = 2
    
    # We will construct the 16 terms manually.
    # Each term corresponds to a (i, j, k) combination in the sub-problems.
    
    # Sub-problem 1: p1=4, p2=4
    # Sub-problem 1.1: m1=2, m2=2
    # Sub-problem 1.1.1: p1=2, p2=2 -> Rank 2
    # Sub-problem 1.1.2: p1=2, p2=2 -> Rank 2
    # ... and so on.
    
    # Let's just iterate through the "logical" terms.
    # The decomposition is:
    # Sum over r=0..15
    # Each r corresponds to a specific (i, j, k) path.
    
    # Actually, it's easier to just use the additivity to build the matrices.
    # But we need to map the indices correctly.
    
    # Let's use the explicit construction for R(3,2,2)=2.
    # For M_{3,2,2}:
    # Term 0: j=0. U[i,0]=1, V[0,0]=1, W[k,0]=1. (Indices relative to sub-block)
    # Term 1: j=1. U[i,1]=1, V[1,1]=1, W[k,1]=1.
    
    # We need to compose these.
    
    # Let's define the mapping for the 16 terms.
    # Term index t in 0..15.
    # t = 4 * t1 + t2, where t1 in 0..3 (for p=4 split), t2 in 0..3 (for m=4 split)?
    
    # Let's just generate the 16 rank-1 tensors that sum to T.
    
    col = 0
    # Split p=8 into p1=4, p2=4
    for p_block in range(2):
        # Split m=4 into m1=2, m2=2
        for m_block in range(2):
            # Split p_block (size 4) into p_sub1=2, p_sub2=2
            for p_sub in range(2):
                # Split m_block (size 2) into m_sub1=1, m_sub2=1? No, R(3,2,2) uses j=0,1.
                # So for each (p_block, m_block, p_sub), we have 2 terms (for j=0,1 in the sub-block).
                for j_sub in range(2):
                    # This is one term.
                    # Determine global indices.
                    
                    # p index: p_block * 4 + p_sub * 2 + k_sub
                    # m index: m_block * 2 + j_sub
                    # n index: i (0..2)
                    
                    # The term is:
                    # U[i, col] = 1 for all i?
                    # V[m_idx, col] = 1 if m_idx == m_global
                    # W[p_idx, col] = 1 if p_idx == p_global
                    
                    m_global = m_block * 2 + j_sub
                    p_global_base = p_block * 4 + p_sub * 2
                    
                    for i in range(n):
                        U[i * m + m_global, col] = 1
                    V[m_global * p + p_global_base, col] = 1
                    V[m_global * p + p_global_base + 1, col] = 1
                    W[p_global_base * n + i, col] = 1
                    W[(p_global_base + 1) * n + i, col] = 1
                    
                    col += 1
    
    # Check if col == rank
    assert col == rank, f"Col {col} != Rank {rank}"
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

if __name__ == "__main__":
    res = construct()
    n, m, p = 3, 4, 8
    rank = len(res['U'][0])
    print(f"Rank: {rank}")
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                dot = np.dot(U[I], V[J] * W[K])
                if abs(dot - 1.0) > max_err:
                    max_err = abs(dot - 1.0)
    print(f"Max Error: {max_err}")
