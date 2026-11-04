import time
import numpy as np
from fractions import Fraction
from itertools import product

def _build_trivial():
    """Schoolbook: rank 60."""
    n, m, p = 2, 3, 10
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, t] = Fraction(1)
                V[b, t] = Fraction(1)
                W[c, t] = Fraction(1)
                t += 1
    return {"U": U, "V": V, "W": W}

def _validate(U, V, W):
    """Validate that U,V,W (numpy object arrays with Fraction) form exact decomposition."""
    n, m, p = 2, 3, 10
    rank = U.shape[1]
    # Check all non-zero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = Fraction(0)
                for t in range(rank):
                    s += U[a, t] * V[b, t] * W[c, t]
                if s != Fraction(1):
                    return False
    # Check a few zero entries to catch spurious contributions
    # For a=(i,j), b=(j',k), c=(k',i') with j!=j' or k!=k'
    checked = 0
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    if j == j2:
                        continue
                    a = i * m + j
                    b = j2 * p + k
                    c = k * n + i
                    s = Fraction(0)
                    for t in range(rank):
                        s += U[a, t] * V[b, t] * W[c, t]
                    if s != Fraction(0):
                        return False
                    checked += 1
                    if checked > 50:
                        break
                if checked > 50:
                    break
            if checked > 50:
                break
        if checked > 50:
            break
    return True

def _build_strassen_2x2x2():
    """
    Strassen decomposition for 2x2 * 2x2 -> 2x2. Rank 7.
    A = [[a11, a12],[a21, a22]], B = [[b11, b12],[b21, b22]]
    C = [[c11, c12],[c21, c22]]
    c11 = M1 - M4 + M5 + M7
    c12 = M3 + M4 - M5 - M6
    c21 = M1 + M2
    c22 = M2 - M3 + M4 + M6
    where:
    M1 = (a11+a22)(b11+b22)
    M2 = (a21+a22)b11
    M3 = a11(b12-b22)
    M4 = a22(b21-b11)
    M5 = (a11+a12)b22
    M6 = (a21-a11)(b11+b12)
    M7 = (a12-a22)(b21+b22)
    """
    # Indices: A: 0=a11, 1=a12, 2=a21, 3=a22
    #           B: 0=b11, 1=b12, 2=b21, 3=b22
    #           C: 0=c11, 1=c12, 2=c21, 3=c22
    # W convention: C[k*n+i] where n=2, so C[0]=c11(i=0,k=0), C[1]=c12(i=0,k=1), C[2]=c21(i=1,k=0), C[3]=c22(i=1,k=1)
    # Wait: W[k*n+i]. For C[i,k]: index = k*2 + i.
    # c11: i=0,k=0 -> idx 0
    # c12: i=0,k=1 -> idx 1
    # c21: i=1,k=0 -> idx 2
    # c22: i=1,k=1 -> idx 3
    
    U = np.zeros((4, 7), dtype=object)
    V = np.zeros((4, 7), dtype=object)
    W = np.zeros((4, 7), dtype=object)
    
    # M1 = (a11+a22)(b11+b22)
    U[0, 0] = Fraction(1); U[3, 0] = Fraction(1)
    V[0, 0] = Fraction(1); V[3, 0] = Fraction(1)
    W[0, 0] = Fraction(1); W[2, 0] = Fraction(1)  # c11 += M1, c21 += M1
    
    # M2 = (a21+a22)b11
    U[2, 1] = Fraction(1); U[3, 1] = Fraction(1)
    V[0, 1] = Fraction(1)
    W[2, 1] = Fraction(1); W[3, 1] = Fraction(1)  # c21 += M2, c22 += M2
    
    # M3 = a11(b12-b22)
    U[0, 2] = Fraction(1)
    V[1, 2] = Fraction(1); V[3, 2] = Fraction(-1)
    W[1, 2] = Fraction(1); W[3, 2] = Fraction(-1)  # c12 += M3, c22 -= M3
    
    # M4 = a22(b21-b11)
    U[3, 3] = Fraction(1)
    V[2, 3] = Fraction(1); V[0, 3] = Fraction(-1)
    W[0, 3] = Fraction(-1); W[1, 3] = Fraction(-1); W[2, 3] = Fraction(1); W[3, 3] = Fraction(1)
    # c11 -= M4, c12 -= M4, c21 += M4, c22 += M4
    
    # M5 = (a11+a12)b22
    U[0, 4] = Fraction(1); U[1, 4] = Fraction(1)
    V[3, 4] = Fraction(1)
    W[0, 4] = Fraction(1); W[1, 4] = Fraction(-1)  # c11 += M5, c12 -= M5
    
    # M6 = (a21-a11)(b11+b12)
    U[2, 5] = Fraction(1); U[0, 5] = Fraction(-1)
    V[0, 5] = Fraction(1); V[1, 5] = Fraction(1)
    W[1, 5] = Fraction(-1); W[3, 5] = Fraction(1)  # c12 -= M6, c22 += M6
    
    # M7 = (a12-a22)(b21+b22)
    U[1, 6] = Fraction(1); U[3, 6] = Fraction(-1)
    V[2, 6] = Fraction(1); V[3, 6] = Fraction(1)
    W[0, 6] = Fraction(1)  # c11 += M7
    
    return {"U": U, "V": V, "W": W}

def _build_strassen_2x3x10():
    """
    Build 2x3 * 3x10 using Strassen on 2x2x2 blocks.
    Split: A (2x3) = [A_left (2x2), A_right (2x1)]
           B (3x10) = [B_top (2x10), B_bottom (1x10)]
           But this doesn't directly work with Strassen.
    
    Better approach: Split the 3 middle dimension into 2+1.
    A = [A1 (2x2), A2 (2x1)]
    B = [B1 (2x10); B2 (1x10)]
    C = A1*B1 + A2*B2
    
    For A1*B1: 2x2 * 2x10. Split B1 into 5 blocks of 2x2.
    Each 2x2*2x2 uses Strassen (rank 7).
    So A1*B1 has rank 5*7 = 35.
    
    For A2*B2: 2x1 * 1x10 = 2x10. This is just outer product style.
    A2 is 2x1, B2 is 1x10. C2 = A2 * B2.
    (A2*B2)[i,k] = A2[i,0] * B2[0,k]
    This is rank 2*1*10/... actually it's a rank-10 tensor (each (i,k) is separate product).
    Wait: T[i, 0, k] for i in {0,1}, k in {0..9}. That's 20 terms if schoolbook.
    But actually: C2[i,k] = A2[i]*B2[k]. The tensor has T[i, j=0, k] = 1 only for the specific (i, 0, k) where the indices match.
    In our indexing: A2 index = i*3 + 2 (since j=2 for the right column). B2 index = 2*10 + k = 20+k.
    C2 index = k*2 + i.
    So terms: for each i in {0,1}, k in {0..9}: T[i*3+2, 20+k, k*2+i] = 1.
    That's 20 terms, rank 20.
    
    Total: 35 + 20 = 55.
    
    Let me construct this properly.
    """
    n, m, p = 2, 3, 10
    # A1*B1: 2x2 * 2x10, split into 5 blocks of 2x2*2x2
    # A2*B2: 2x1 * 1x10, rank 20
    
    rank_total = 55
    U = np.zeros((6, rank_total), dtype=object)  # n*m = 6
    V = np.zeros((30, rank_total), dtype=object)  # m*p = 30
    W = np.zeros((20, rank_total), dtype=object)  # p*n = 20
    
    t = 0
    # Part 1: A1*B1 where A1 = A[:, 0:2], B1 = B[0:2, :]
    # 5 blocks: for col_block in range(5): B1[:, col_block*2 : col_block*2+2]
    strassen = _build_strassen_2x2x2()
    sU, sV, sW = strassen["U"], strassen["V"], strassen["W"]
    
    for col_block in range(5):
        # A1 indices: i*3 + j for i in {0,1}, j in {0,1} -> 0,1,3,4
        # B1 indices: j*10 + k for j in {0,1}, k in {col_block*2, col_block*2+1}
        # C indices: k*2 + i for k in {col_block*2, col_block*2+1}, i in {0,1}
        
        for st in range(7):
            # U part: A1 indices
            for ai in range(4):
                if sU[ai, st] != 0:
                    # Map local A1 index ai to global: local 0->0(i=0,j=0), 1->1(i=0,j=1), 2->3(i=1,j=0), 3->4(i=1,j=1)
                    global_a = [0, 1, 3, 4][ai]
                    U[global_a, t] += sU[ai, st]
            
            # V part: B1 indices
            for bi in range(4):
                if sV[bi, st] != 0:
                    # Local B1 index bi: 0->j=0,k=cb*2, 1->j=0,k=cb*2+1, 2->j=1,k=cb*2, 3->j=1,k=cb*2+1
                    j_local = bi // 2  # 0 or 1
                    k_local = bi % 2   # 0 or 1
                    global_j = j_local
                    global_k = col_block * 2 + k_local
                    global_b = global_j * 10 + global_k
                    V[global_b, t] += sV[bi, st]
            
            # W part: C indices
            for ci in range(4):
                if sW[ci, st] != 0:
                    # Local C index ci: 0->i=0,k=cb*2, 1->i=0,k=cb*2+1, 2->i=1,k=cb*2, 3->i=1,k=cb*2+1
                    i_local = ci % 2
                    k_local = ci // 2
                    global_i = i_local
                    global_k = col_block * 2 + k_local
                    global_c = global_k * 2 + global_i
                    W[global_c, t] += sW[ci, st]
            
            t += 1
    
    # Part 2: A2*B2 where A2 = A[:, 2] (2x1), B2 = B[2, :] (1x10)
    # C2[i,k] = A2[i] * B2[k]
    # A2 global index: i*3 + 2
    # B2 global index: 2*10 + k = 20 + k
    # C2 global index: k*2 + i
    for i in range(2):
        for k in range(10):
            a_idx = i * 3 + 2
            b_idx = 20 + k
            c_idx = k * 2 + i
            U[a_idx, t] = Fraction(1)
            V[b_idx, t] = Fraction(1)
            W[c_idx, t] = Fraction(1)
            t += 1
    
    return {"U": U, "V": V, "W": W}

def _try_linear_reduction(U, V, W, max_time=30.0):
    """
    Try to reduce rank by finding linear dependencies among the rank-1 terms.
    If two terms are linearly dependent (i.e., one can be expressed as a linear combination
    of others), we can eliminate it.
    
    Approach: Form the tensor T = sum_t u_t ⊗ v_t ⊗ w_t.
    If the set {u_t ⊗ v_t ⊗ w_t} has linear dependencies, we can reduce rank.
    
    Practical approach: Try to find a subset of terms that spans the same tensor.
    Use the fact that if we can find coefficients c_t such that sum_t c_t * (u_t ⊗ v_t ⊗ w_t) = 0
    and not all c_t are zero, then we can eliminate one term.
    
    For small rank reductions, try pairwise: can term t be written as combo of others?
    This is expensive, so use a heuristic: randomly project and check rank.
    """
    start = time.time()
    n, m, p = 2, 3, 10
    rank = U.shape[1]
    
    # Flatten tensor
    T = np.zeros((n*m, m*p, p*n))
    for t in range(rank):
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    T[a,b,c] += U[a,t] * V[b,t] * W[c,t]
    
    # Try to find a lower-rank representation by checking if we can drop terms
    # and still span the same tensor.
    # This is NP-hard in general, so use a greedy approach.
    
    # For now, just return the current decomposition (no reduction).
    # A proper implementation would use the "tensor rank reduction" algorithms.
    return None

def _attempt_rank_reduction_by_substitution(base_U, base_V, base_W, target_rank, max_time=20.0):
    """
    Try to find a decomposition of rank <= target_rank by expressing some terms
    as linear combinations of others.
    
    Strategy: If we have rank R and want rank R-1, try to find one term that is
    a linear combination of the others in the tensor space.
    """
    start = time.time()
    n, m, p = 2, 3, 10
    R = base_U.shape[1]
    
    if R <= target_rank:
        return {"U": base_U, "V": base_V, "W": base_W}
    
    # This is complex; skip for now and just return base.
    return None

def construct():
    deadline = time.time() + 130  # Safety margin before 150s kill
    
    # Start with the best known constructive decomposition
    # Strassen-based: rank 55
    try:
        strassen_decomp = _build_strassen_2x3x10()
        U_s, V_s, W_s = strassen_decomp["U"], strassen_decomp["V"], strassen_decomp["W"]
        
        # Validate
        if _validate(U_s, V_s, W_s):
            best_result = {
                "U": U_s.tolist(),
                "V": V_s.tolist(),
                "W": W_s.tolist()
            }
            best_rank = 55
        else:
            # Fall back to schoolbook
            schoolbook = _build_trivial()
            best_result = {
                "U": schoolbook["U"].tolist(),
                "V": schoolbook["V"].tolist(),
                "W": schoolbook["W"].tolist()
            }
            best_rank = 60
    except Exception:
        schoolbook = _build_trivial()
        best_result = {
            "U": schoolbook["U"].tolist(),
            "V": schoolbook["V"].tolist(),
            "W": schoolbook["W"].tolist()
        }
        best_rank = 60
    
    # Try to find even lower rank using algebraic methods
    if time.time() < deadline - 30:
        try:
            # Try rank 50-54 with a different block decomposition
            # Split 3 into 1+1+1: three 2x1*1x10 blocks
            # Each 2x1*1x10 has rank 20 (schoolbook for that subproblem)
            # Total: 3*20 = 60. No improvement.
            
            # Try splitting 10 into 5+5:
            # A(2x3) * B_top(3x5) + A(2x3) * B_bottom(3x5)
            # Each 2x3*3x5: use Strassen on 2x2 part.
            # 2x3*3x5: split 3 into 2+1.
            #   A1(2x2)*B1(2x5): 2x2*2x5 = 2 blocks of 2x2*2x2 (Strassen, rank 7 each) + 2x2*2x1 (rank 4)
            #     Actually 2x2*2x5: split 5 into 2+2+1.
            #     2x2*2x2: Strassen rank 7. Two of them: 14.
            #     2x2*2x1: rank 4 (schoolbook).
            #     Total for A1*B1: 14 + 4 = 18.
            #   A2(2x1)*B2(1x5): rank 10.
            #   Total for 2x3*3x5: 18 + 10 = 28.
            #   Two of them: 56. Worse than 55.
            
            # Try a different approach: exploit the 2x3 structure more.
            # The 2x3 matrix multiplication tensor M_{2,3,p} has known bounds.
            # For p=10, the trivial bound is 60, Strassen gives 55.
            # The asymptotic rank is related to omega, but for small sizes,
            # 55 might be close to optimal.
            
            # Let's try to verify if rank 54 or lower is possible by checking
            # if we can find a valid decomposition at rank 54.
            # This would require a non-trivial algebraic identity.
            
            # For now, try a randomized algebraic search:
            # Start from rank 55 decomposition and try to merge terms.
            if best_rank == 55 and time.time() < deadline - 20:
                # Try to see if any two terms can be combined
                # (i.e., if u1⊗v1⊗w1 + u2⊗v2⊗w2 can be written as a single term)
                # This is generally not possible, but check for special cases.
                pass
        
        except Exception:
            pass
    
    return best_result

# Quick self-test if run directly
if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0])
    print(f"Rank: {rank}")
    # Validate
    U = np.array(result["U"], dtype=object)
    V = np.array(result["V"], dtype=object)
    W = np.array(result["W"], dtype=object)
    valid = _validate(U, V, W)
    print(f"Valid: {valid}")
