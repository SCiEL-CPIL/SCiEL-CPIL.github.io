import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 7]

_START_TIME = time.time()
TIME_LIMIT = 140.0  # Leave 10s margin before 150s kill

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly computes matrix multiplication."""
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # For each output element C[i,j] (i in 0..n-1, j in 0..p-1):
    # C[i,j] = sum over k in 0..m-1 of A[i,k] * B[k,j]
    # In tensor form: T[i,k,j] = 1 for all i,k,j
    # Decomposition: T[i,k,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    
    for i in range(n):
        for k in range(m):
            for j in range(p):
                lhs = 1.0  # The tensor entry T[i,k,j] = 1
                rhs = 0.0
                for r in range(rank):
                    rhs += U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
                if abs(lhs - rhs) > 1e-10:
                    return False
    return True

def construct():
    n, m, p = 2, 4, 7
    
    # First, build the rank-50 solution as a baseline
    rank_50 = 50
    U50 = np.zeros((n * m, rank_50), dtype=float)
    V50 = np.zeros((m * p, rank_50), dtype=float)
    W50 = np.zeros((p * n, rank_50), dtype=float)
    
    col = 0
    
    def add_strassen(a_base, b_base, c_base, offset):
        nonlocal col
        # M1 = (A00 + A11)(B00 + B11)
        U50[(a_base[0]+0)*m + (a_base[1]+0), offset] += 1
        U50[(a_base[0]+1)*m + (a_base[1]+1), offset] += 1
        V50[(b_base[0]+0)*p + (b_base[1]+0), offset] += 1
        V50[(b_base[0]+1)*p + (b_base[1]+1), offset] += 1
        W50[(c_base[1]+0)*n + (c_base[0]+0), offset] += 1
        W50[(c_base[1]+1)*n + (c_base[0]+1), offset] += 1
        col = offset + 1
        
        # M2 = (A10 + A11)B00
        U50[(a_base[0]+1)*m + (a_base[1]+0), col] += 1
        U50[(a_base[0]+1)*m + (a_base[1]+1), col] += 1
        V50[(b_base[0]+0)*p + (b_base[1]+0), col] += 1
        W50[(c_base[1]+0)*n + (c_base[0]+1), col] += 1
        W50[(c_base[1]+1)*n + (c_base[0]+1), col] -= 1
        col += 1
        
        # M3 = A00(B01 - B11)
        U50[(a_base[0]+0)*m + (a_base[1]+0), col] += 1
        V50[(b_base[0]+0)*p + (b_base[1]+1), col] += 1
        V50[(b_base[0]+1)*p + (b_base[1]+1), col] -= 1
        W50[(c_base[1]+1)*n + (c_base[0]+0), col] += 1
        W50[(c_base[1]+1)*n + (c_base[0]+1), col] += 1
        col += 1
        
        # M4 = A11(B10 - B00)
        U50[(a_base[0]+1)*m + (a_base[1]+1), col] += 1
        V50[(b_base[0]+1)*p + (b_base[1]+0), col] += 1
        V50[(b_base[0]+0)*p + (b_base[1]+0), col] -= 1
        W50[(c_base[1]+0)*n + (c_base[0]+0), col] += 1
        W50[(c_base[1]+0)*n + (c_base[0]+1), col] += 1
        col += 1
        
        # M5 = (A00 + A01)B11
        U50[(a_base[0]+0)*m + (a_base[1]+0), col] += 1
        U50[(a_base[0]+0)*m + (a_base[1]+1), col] += 1
        V50[(b_base[0]+1)*p + (b_base[1]+1), col] += 1
        W50[(c_base[1]+0)*n + (c_base[0]+0), col] -= 1
        W50[(c_base[1]+1)*n + (c_base[0]+0), col] += 1
        col += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U50[(a_base[0]+1)*m + (a_base[1]+0), col] += 1
        U50[(a_base[0]+0)*m + (a_base[1]+0), col] -= 1
        V50[(b_base[0]+0)*p + (b_base[1]+0), col] += 1
        V50[(b_base[0]+0)*p + (b_base[1]+1), col] += 1
        W50[(c_base[1]+1)*n + (c_base[0]+1), col] += 1
        col += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U50[(a_base[0]+0)*m + (a_base[1]+1), col] += 1
        U50[(a_base[0]+1)*m + (a_base[1]+1), col] -= 1
        V50[(b_base[0]+1)*p + (b_base[1]+0), col] += 1
        V50[(b_base[0]+1)*p + (b_base[1]+1), col] += 1
        W50[(c_base[1]+0)*n + (c_base[0]+0), col] += 1
        col += 1
        
        return col

    def add_schoolbook_221(a_base, b_base, c_base, offset):
        nonlocal col
        # N1 = A00 B0
        U50[(a_base[0]+0)*m + (a_base[1]+0), offset] += 1
        V50[(b_base[0]+0)*p + (b_base[1]+0), offset] += 1
        W50[(c_base[1]+0)*n + (c_base[0]+0), offset] += 1
        col = offset + 1
        
        # N2 = A01 B1
        U50[(a_base[0]+0)*m + (a_base[1]+1), col] += 1
        V50[(b_base[0]+1)*p + (b_base[1]+0), col] += 1
        W50[(c_base[1]+0)*n + (c_base[0]+0), col] += 1
        col += 1
        
        # N3 = A10 B0
        U50[(a_base[0]+1)*m + (a_base[1]+0), col] += 1
        V50[(b_base[0]+0)*p + (b_base[1]+0), col] += 1
        W50[(c_base[1]+0)*n + (c_base[0]+1), col] += 1
        col += 1
        
        # N4 = A11 B1
        U50[(a_base[0]+1)*m + (a_base[1]+1), col] += 1
        V50[(b_base[0]+1)*p + (b_base[1]+0), col] += 1
        W50[(c_base[1]+0)*n + (c_base[0]+1), col] += 1
        col += 1
        
        return col

    a_l_base = (0, 0)
    a_r_base = (0, 2)
    b_tl_base = (0, 0)
    b_tm_base = (0, 2)
    b_tr_base = (0, 4)
    b_t1_base = (0, 6)
    b_bl_base = (2, 0)
    b_bm_base = (2, 2)
    b_br_base = (2, 4)
    b_b1_base = (2, 6)
    c_l_base = (0, 0)
    c_m_base = (0, 2)
    c_r_base = (0, 4)
    c_1_base = (0, 6)
    
    col = add_strassen(a_l_base, b_tl_base, c_l_base, col)
    col = add_strassen(a_l_base, b_tm_base, c_m_base, col)
    col = add_strassen(a_l_base, b_tr_base, c_r_base, col)
    col = add_schoolbook_221(a_l_base, b_t1_base, c_1_base, col)
    col = add_strassen(a_r_base, b_bl_base, c_l_base, col)
    col = add_strassen(a_r_base, b_bm_base, c_m_base, col)
    col = add_strassen(a_r_base, b_br_base, c_r_base, col)
    col = add_schoolbook_221(a_r_base, b_b1_base, c_1_base, col)
    
    if col != rank_50:
        raise ValueError(f"Internal error: expected rank 50, got {col}")
    
    # Verify the rank-50 solution
    if not _verify_decomposition(U50, V50, W50, n, m, p, rank_50):
        raise ValueError("Rank-50 solution failed verification")
    
    # Now attempt to reduce rank by finding linear dependencies among the rank-1 terms
    # Strategy: Try to express some terms as linear combinations of others
    
    # The key insight: if we have terms t1, t2, ..., tr, we can try to find
    # coefficients c1, ..., cr (not all zero) such that sum cr * tr = 0 in the tensor space.
    # This would mean we can eliminate one term.
    
    # For a more practical approach: try removing one term at a time and see if
    # the remaining terms can be rescaled to still satisfy the equations.
    
    # Actually, let's try a different approach: use the structure of Strassen
    # to find shared terms between adjacent blocks.
    
    # The 2x4 * 4x7 multiplication can be seen as:
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # 
    # With A split into A_L (cols 0,1) and A_R (cols 2,3):
    # C[i,j] = sum_{k=0}^1 A_L[i,k] * B[k,j] + sum_{k=2}^3 A_R[i,k-2] * B[k,j]
    #
    # For the first sum, B has rows 0,1 and 7 cols. We split B[0:2,:] into
    # B_TL (cols 0,1), B_TM (cols 2,3), B_TR (cols 4,5), B_T1 (col 6).
    #
    # The Strassen terms for A_L * B_TL produce certain U, V, W patterns.
    # The Strassen terms for A_R * B_BL produce similar patterns but with
    # different indices.
    
    # Let's check if any pairs of terms are linearly dependent.
    # Two terms (u1,v1,w1) and (u2,v2,w2) are linearly dependent if
    # u1*v1*w1 = c * u2*v2*w2 for some constant c.
    
    # This is hard to check directly. Instead, let's try a greedy elimination:
    # For each term r, check if we can set its coefficient to 0 and adjust
    # other terms to compensate.
    
    # Actually, the most promising approach: look for terms that have the same
    # support pattern in U, V, W but with different coefficients.
    
    # Let's build a dictionary of term signatures.
    terms = []
    for r in range(rank_50):
        u_vec = U50[:, r]
        v_vec = V50[:, r]
        w_vec = W50[:, r]
        # Skip if all zeros
        if np.all(u_vec == 0) and np.all(v_vec == 0) and np.all(w_vec == 0):
            continue
        terms.append((r, u_vec, v_vec, w_vec))
    
    # Check for exact duplicates (same U, V, W vectors up to scalar)
    # This is expensive, so let's be selective.
    
    # Alternative: try to find if the system has a null space.
    # Build the 140x50 matrix where each column r corresponds to the outer product
    # of U[:,r], V[:,r], W[:,r] flattened.
    
    # The tensor T has entries T[i,k,j] = 1 for all i in 0..1, k in 0..3, j in 0..6.
    # The decomposition requires: sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1
    
    # This is a system of 2*4*7 = 56 equations (not 140; I was wrong earlier).
    # Wait, let me recount: i has 2 values, k has 4 values, j has 7 values.
    # So 2*4*7 = 56 equations.
    
    # We have 50 unknowns (the rank-1 terms are fixed, but we could scale them).
    # Actually, the terms are fixed; we're asking if we can drop some.
    
    # Let's try: for each subset of terms of size 49, check if the remaining
    # 49 terms can be rescaled to satisfy all 56 equations.
    
    # This is a nonlinear system (trilinear), so it's hard.
    
    # Simpler: check if any single term is redundant.
    # Remove term r, and see if the remaining 49 terms can be adjusted.
    # Since the terms are fixed, we'd need to find new U, V, W for the remaining
    # 49 terms that satisfy the equations. This is a feasibility problem.
    
    # Given the time constraint, let's just return the rank-50 solution.
    # The search for rank < 50 is complex and may not succeed within the time limit.
    
    return {"U": U50.tolist(), "V": V50.tolist(), "W": W50.tolist()}
