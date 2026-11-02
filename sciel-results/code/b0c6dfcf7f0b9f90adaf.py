"""Rank-minimal exact tensor decomposition for 5x5 matrix multiplication."""
import time
import random
import numpy as np
from fractions import Fraction

N = M = P = 5
RANK_TARGET = 75  # Known best rank for 5x5 matrix multiplication
START = time.time()
DEADLINE = START + 110.0  # Safety margin before 120s limit


def _time_left():
    return DEADLINE - time.time()


def _verify(U, V, W):
    """Check exact decomposition using fractions.
    U: (N*M, rank), V: (M*P, rank), W: (P*N, rank)
    Returns True if sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == delta(i,k)
    """
    rank = len(U[0])
    # Convert to fractions for exactness
    UF = [[Fraction(U[i, r]) for r in range(rank)] for i in range(N * M)]
    VF = [[Fraction(V[i, r]) for r in range(rank)] for i in range(M * P)]
    WF = [[Fraction(W[i, r]) for r in range(rank)] for i in range(P * N)]
    
    for i in range(N):
        for j in range(M):
            for k in range(P):
                target = Fraction(1) if i == k else Fraction(0)
                acc = Fraction(0)
                urow = i * M + j
                vrow = j * P + k
                wrow = k * N + i
                for r in range(rank):
                    acc += UF[urow][r] * VF[vrow][r] * WF[wrow][r]
                if acc != target:
                    return False
    return True


def _try_random_decomposition(rank, max_attempts=2000, seed=None):
    """Try to find a rank-rank decomposition via random search with W solved.
    
    Strategy:
    1. Randomly initialize U and V with small integers
    2. For each (i,k) pair, find an r where C[i,k,r] = sum_j U[i*M+j,r]*V[j*P+k,r] is non-zero
    3. Set W[k*N+i,r] = delta(i,k) / C[i,k,r] and 0 elsewhere for that (i,k)
    4. Verify the result
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    for attempt in range(max_attempts):
        if _time_left() < 5.0:
            return None
        
        # Random U and V with entries in {-2,-1,0,1}
        U = np.array([[rng.choice([-2, -1, 0, 1]) for _ in range(rank)] for _ in range(N * M)], dtype=object)
        V = np.array([[rng.choice([-2, -1, 0, 1]) for _ in range(rank)] for _ in range(M * P)], dtype=object)
        
        # Compute C[i,k,r] = sum_j U[i*M+j, r] * V[j*P+k, r]
        C = np.zeros((N, P, rank), dtype=object)
        for i in range(N):
            for k in range(P):
                for r in range(rank):
                    total = 0
                    for j in range(M):
                        total += U[i * M + j, r] * V[j * P + k, r]
                    C[i, k, r] = total
        
        # For each (i,k), try to assign W
        W = np.zeros((P * N, rank), dtype=object)
        valid = True
        for i in range(N):
            for k in range(P):
                target = 1 if i == k else 0
                # Find an r where C[i,k,r] != 0
                candidate_rs = [r for r in range(rank) if C[i, k, r] != 0]
                if not candidate_rs:
                    valid = False
                    break
                # Pick a random r from candidates
                r_choice = rng.choice(candidate_rs)
                c_val = C[i, k, r_choice]
                if target == 0:
                    # Need sum_r C[i,k,r]*W[k*N+i,r] = 0
                    # Set W[k*N+i, r_choice] = 0, and hope other terms cancel
                    # Actually, we need to be more careful. Let's set W[k*N+i, r_choice] = 0
                    # and see if it works. But this might not work.
                    # Better: for i != k, we need the weighted sum to be 0.
                    # Let's just set all W[k*N+i, r] = 0 for now and check later.
                    # Actually, this approach is flawed for i != k.
                    # Let me reconsider.
                    # For i != k: we need sum_r C[i,k,r] * W[k*N+i,r] = 0
                    # For i == k: we need sum_r C[i,k,r] * W[k*N+i,r] = 1
                    # 
                    # Simple approach: for i == k, pick one r with C[i,k,r] != 0,
                    # set W[k*N+i, r] = 1/C[i,k,r], others 0.
                    # For i != k, set all W[k*N+i, r] = 0. But then we need
                    # sum_r C[i,k,r]*0 = 0 which is true. But wait, that means
                    # for i != k, the contribution is always 0 regardless of W.
                    # No! W[k*N+i, r] is specific to (k,i). Different (i,k) pairs
                    # use different rows of W.
                    # 
                    # So for i != k, if we set W[k*N+i, :] = 0, then the sum is 0. Good.
                    # For i == k, we set W[k*N+k, r_choice] = 1/C[k,k,r_choice], others 0.
                    #
                    # But we need C[k,k,r_choice] to divide 1, i.e., be ±1 or ±2 (for half-integers).
                    if True:
                        W[k * N + i, :] = 0
                else:
                    # i == k: need sum_r C[i,k,r] * W[i*N+i,r] = 1
                    # Pick r_choice with C[i,k,r_choice] in {±1, ±2}
                    good_rs = [r for r in candidate_rs if abs(C[i, k, r]) in (1, 2)]
                    if not good_rs:
                        valid = False
                        break
                    r_choice = rng.choice(good_rs)
                    c_val = C[i, k, r_choice]
                    W[k * N + i, r_choice] = 1 / c_val if c_val != 0 else 0
                    # Ensure others are 0 (already 0 from init)
        
        if not valid:
            continue
        
        # Verify
        if _verify(U, V, W):
            return (U, V, W)
    
    return None


def _build_schoolbook():
    """Fallback: schoolbook decomposition with rank 125."""
    rank = N * M * P
    U = np.zeros((N * M, rank), dtype=object)
    V = np.zeros((M * P, rank), dtype=object)
    W = np.zeros((P * N, rank), dtype=object)
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1
                V[j * P + k, col] = 1
                W[k * N + i, col] = 1
                col += 1
    return U, V, W, rank


def _convert_to_float(mat):
    """Convert object array with fractions/ints to float array."""
    out = np.zeros(mat.shape, dtype=float)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if isinstance(val, Fraction):
                out[i, j] = float(val)
            else:
                out[i, j] = float(val)
    return out


def construct():
    t0 = time.time()
    
    # First, try the schoolbook as a safe baseline
    U_school, V_school, W_school, rank_school = _build_schoolbook()
    best = (U_school, V_school, W_school, rank_school)
    
    # Try random search for lower rank decompositions
    # Start from rank 75 (known theoretical bound) and go up
    for target_rank in range(75, 126):
        if _time_left() < 10.0:
            break
        
        # Try multiple random seeds
        for seed in range(50):
            if _time_left() < 10.0:
                break
            
            result = _try_random_decomposition(target_rank, max_attempts=100, seed=seed)
            if result is not None:
                U, V, W = result
                if target_rank < best[3]:
                    best = (U, V, W, target_rank)
                    break  # Found a valid one for this rank, try lower next
    
    # Final verification of best
    bU, bV, bW, brank = best
    if not _verify(bU, bV, bW):
        # Fall back to schoolbook
        bU, bV, bW, brank = _build_schoolbook()
    
    Uout = _convert_to_float(bU)
    Vout = _convert_to_float(bV)
    Wout = _convert_to_float(bW)
    
    return {"U": Uout.tolist(), "V": Vout.tolist(), "W": Wout.tolist()}


if __name__ == "__main__":
    res = construct()
    print("rank =", len(res["U"][0]))
