import numpy as np
from fractions import Fraction
import time
import random

def construct():
    n, m, p = 2, 3, 7
    start_time = time.time()
    deadline = start_time + 140.0 # Leave margin for 150s limit

    # Schoolbook initialization
    rank_school = n * m * p
    U_best = [[0] * rank_school for _ in range(n * m)]
    V_best = [[0] * rank_school for _ in range(m * p)]
    W_best = [[0] * rank_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_best[a][col] = 1
                V_best[b][col] = 1
                W_best[c][col] = 1
                col += 1
                
    best_rank = rank_school

    def verify_float(U, V, W, rank, valid_indices, T_target):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # T_recon[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Check valid
        for (a, b, c) in valid_indices:
            if abs(T_recon[a, b, c] - T_target[a, b, c]) > 1e-6:
                return False
        
        # Check invalid
        mask = np.zeros(T_recon.shape, dtype=bool)
        for (a, b, c) in valid_indices:
            mask[a, b, c] = True
        invalid_vals = T_recon[~mask]
        if np.any(np.abs(invalid_vals) > 1e-6):
            return False
        return True

    def verify_exact(U, V, W, rank, valid_indices):
        Uf = [[Fraction(U[i][r]) for r in range(rank)] for i in range(len(U))]
        Vf = [[Fraction(V[i][r]) for r in range(rank)] for i in range(len(V))]
        Wf = [[Fraction(W[i][r]) for r in range(rank)] for i in range(len(W))]
        
        for (a, b, c) in valid_indices:
            s = Fraction(0)
            for r in range(rank):
                s += Uf[a][r] * Vf[b][r] * Wf[c][r]
            if s != 1:
                return False
        
        for a in range(len(U)):
            for b in range(len(V)):
                for c in range(len(W)):
                    if (a, b, c) in valid_indices:
                        continue
                    s = Fraction(0)
                    for r in range(rank):
                        s += Uf[a][r] * Vf[b][r] * Wf[c][r]
                    if s != 0:
                        return False
        return True

    def search_decomp(n_sub, m_sub, p_sub, target_rank, entries, time_limit, seed=42):
        """
        Search for a decomposition of M_{n_sub, m_sub, p_sub} with the given rank.
        Returns (U, V, W) or None.
        """
        random.seed(seed)
        start = time.time()
        
        valid_indices_sub = set()
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    a = i * m_sub + j
                    b = j * p_sub + k
                    c = k * n_sub + i
                    valid_indices_sub.add((a, b, c))
        
        T_sub = np.zeros((n_sub*m_sub, m_sub*p_sub, p_sub*n_sub), dtype=float)
        for (a, b, c) in valid_indices_sub:
            T_sub[a, b, c] = 1.0
            
        # We limit iterations to avoid excessive time, relying on time_limit
        max_iters = 20000
        for _ in range(max_iters):
            if time.time() > start + time_limit:
                break
                
            U = [[random.choice(entries) for _ in range(target_rank)] for _ in range(n_sub*m_sub)]
            V = [[random.choice(entries) for _ in range(target_rank)] for _ in range(m_sub*p_sub)]
            W = [[random.choice(entries) for _ in range(target_rank)] for _ in range(p_sub*n_sub)]
            
            # Quick float check first
            if verify_float(U, V, W, target_rank, valid_indices_sub, T_sub):
                # Exact check
                if verify_exact(U, V, W, target_rank, valid_indices_sub):
                    return U, V, W
                    
        return None

    # Strategy: Block Decomposition
    # Split B (3x7) into B1 (3x3) and B2 (3x4)
    # M_{2,3,3} + M_{2,3,4}
    
    # 1. Find decomposition for M_{2,3,3}
    # Known rank is 13. Try 13, 14, 15...
    U_sub1 = None
    V_sub1 = None
    W_sub1 = None
    rank_sub1 = None
    
    time_left = deadline - time.time()
    if time_left > 20:
        time_per_rank = max(2.0, (time_left - 10) / 10.0)
        for r_sub in range(13, 18):
            if time.time() > deadline - 30:
                break
            res = search_decomp(2, 3, 3, r_sub, [-1, 0, 1], time_per_rank, seed=100+r_sub)
            if res:
                U_sub1, V_sub1, W_sub1 = res
                rank_sub1 = r_sub
                break
    
    # 2. Find decomposition for M_{2,3,4}
    # Schoolbook is 24. 
    # Split M_{2,3,4} into two M_{2,3,2}?
    # M_{2,3,2} rank is 11. So 11+11=22 is a safe bet if we find M_{2,3,2}.
    # But we can try direct search for M_{2,3,4} with rank 20, 21, 22...
    U_sub2 = None
    V_sub2 = None
    W_sub2 = None
    rank_sub2 = None
    
    time_left = deadline - time.time()
    if time_left > 20:
        time_per_rank2 = max(2.0, (time_left - 10) / 10.0)
        # Try ranks 20 to 23
        for r_sub2 in range(20, 24):
            if time.time() > deadline - 10:
                break
            res = search_decomp(2, 3, 4, r_sub2, [-1, 0, 1], time_per_rank2, seed=200+r_sub2)
            if res:
                U_sub2, V_sub2, W_sub2 = res
                rank_sub2 = r_sub2
                break

    # Combine results
    # If we found sub-decompositions, try to combine them
    # If not, use schoolbook for that part
    
    total_rank = 0
    U_new = []
    V_new = []
    W_new = []
    
    # Initialize empty lists, we will build them
    U_new = [[0] * (rank_sub1 if U_sub1 else 18 + (rank_sub2 if U_sub2 else 24)) for _ in range(n*m)]
    V_new = [[0] * (rank_sub1 if U_sub1 else 18 + (rank_sub2 if U_sub2 else 24)) for _ in range(m*p)]
    W_new = [[0] * (rank_sub1 if U_sub1 else 18 + (rank_sub2 if U_sub2 else 24)) for _ in range(p*n)]
    
    offset = 0
    
    # Part 1: M_{2,3,3}
    if U_sub1 is not None:
        for r in range(rank_sub1):
            for i in range(2):
                for j in range(3):
                    a = i*3 + j
                    U_new[a][offset + r] = U_sub1[a][r]
                    
                    for k in range(3):
                        b_full = j*7 + k
                        b_sub = j*3 + k
                        V_new[b_full][offset + r] = V_sub1[b_sub][r]
                        
                        c_full = k*2 + i
                        c_sub = k*2 + i
                        W_new[c_full][offset + r] = W_sub1[c_sub][r]
        total_rank += rank_sub1
        offset += rank_sub1
    else:
        # Schoolbook for part 1
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    a = i*3 + j
                    b_full = j*7 + k
                    c_full = k*2 + i
                    U_new[a][offset] = 1
                    V_new[b_full][offset] = 1
                    W_new[c_full][offset] = 1
                    offset += 1
        total_rank += 18

    # Part 2: M_{2,3,4}
    if U_sub2 is not None:
        for r in range(rank_sub2):
            for i in range(2):
                for j in range(3):
                    a = i*3 + j
                    U_new[a][offset + r] = U_sub2[a][r]
                    
                    for k_prime in range(4):
                        k_full = k_prime + 3
                        b_full = j*7 + k_full
                        b_sub = j*4 + k_prime
                        V_new[b_full][offset + r] = V_sub2[b_sub][r]
                        
                        c_full = k_full*2 + i
                        c_sub = k_prime*2 + i
                        W_new[c_full][offset + r] = W_sub2[c_sub][r]
        total_rank += rank_sub2
        offset += rank_sub2
    else:
        # Schoolbook for part 2
        for i in range(2):
            for j in range(3):
                for k in range(3, 7):
                    a = i*3 + j
                    b_full = j*7 + k
                    c_full = k*2 + i
                    U_new[a][offset] = 1
                    V_new[b_full][offset] = 1
                    W_new[c_full][offset] = 1
                    offset += 1
        total_rank += 24

    # Verify the combined solution
    valid_indices_full = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_indices_full.add((a, b, c))
                
    T_full = np.zeros((n*m, m*p, p*n), dtype=float)
    for (a, b, c) in valid_indices_full:
        T_full[a, b, c] = 1.0

    if total_rank < best_rank:
        if verify_float(U_new, V_new, W_new, total_rank, valid_indices_full, T_full):
            if verify_exact(U_new, V_new, W_new, total_rank, valid_indices_full):
                best_U = U_new
                best_V = V_new
                best_W = W_new
                best_rank = total_rank

    return {
        "U": best_U,
        "V": best_V,
        "W": best_W
    }
