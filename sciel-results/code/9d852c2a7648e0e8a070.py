"""Matrix multiplication 2x4 by 4x8: rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 8]
DEADLINE = 120.0
SAFETY_MARGIN = 5.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (32, rank)
    # W: (p*n, rank) = (16, rank)
    # Tensor T[i, j, k] for i in [0,n), j in [0,m), k in [0,p)
    # T[i, j, k] = 1 (standard matrix mult)
    # Decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    
    nm = n * m  # 8
    mp = m * p  # 32
    pn = p * n  # 16
    
    def time_left():
        return DEADLINE - SAFETY_MARGIN - (time.time() - start_time)
    
    # Try to find a low-rank decomposition using randomized search + local optimization
    # Start with a reasonable rank estimate and try to reduce
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                    if s != 1:
                        return False
        return True
    
    def verify_decomposition_fast(U, V, W, rank):
        """Faster verification using numpy for initial check, then exact if close."""
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Strategy: Use a structured decomposition based on known small-rank constructions
    # For 2x4 * 4x8, we can try to exploit the structure.
    
    # First, try a simple approach: use rank = n*m*p / 2 or similar and search
    # Actually, let's use a more systematic approach:
    
    # The tensor T has entries T[i,j,k]=1. We need:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # Let's try random search with small ranks
    best_rank = n * m * p  # schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Try schoolbook first as fallback
    rank_school = n * m * p
    U_school = [[0]*rank_school for _ in range(nm)]
    V_school = [[0]*rank_school for _ in range(mp)]
    W_school = [[0]*rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = rank_school
    
    # Try randomized search for lower ranks
    # Use a heuristic: try ranks from a lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(8, 32, 16) = 32
    # But actually the tensor rank can be lower than these dimensions
    
    # Let's try a structured approach: assume U has a specific form
    # For each output (i,k), the sum over j of A[i,j]*B[j,k] must equal the product
    
    # Alternative: use the fact that for fixed i,k, we need sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # This is complex. Let's try random initialization and local search.
    
    def random_decomposition(rank, scale=2):
        """Generate a random decomposition with integer/half-integer entries."""
        U = [[random.choice([0, 1, -1, 0.5, -0.5, 2, -2]) for _ in range(rank)] for _ in range(nm)]
        V = [[random.choice([0, 1, -1, 0.5, -0.5, 2, -2]) for _ in range(rank)] for _ in range(mp)]
        W = [[random.choice([0, 1, -1, 0.5, -0.5, 2, -2]) for _ in range(rank)] for _ in range(pn)]
        return U, V, W
    
    def local_search(U, V, W, rank, max_iter=1000):
        """Try to improve the decomposition by local moves."""
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        
        values = [0, 0.5, -0.5, 1, -1, 2, -2, 0.25, -0.25, 1.5, -1.5]
        
        for it in range(max_iter):
            if time_left() < 2:
                break
            # Try random modifications
            modified = False
            for _ in range(20):
                if time_left() < 2:
                    break
                # Pick a random position in U, V, or W
                which = random.choice([0, 1, 2])
                if which == 0:
                    r = random.randrange(nm)
                    c = random.randrange(rank)
                    old = U[r][c]
                    new = random.choice(values)
                    if new == old:
                        continue
                    U[r][c] = new
                    if verify_decomposition_fast(U, V, W, rank):
                        modified = True
                    else:
                        U[r][c] = old
                elif which == 1:
                    r = random.randrange(mp)
                    c = random.randrange(rank)
                    old = V[r][c]
                    new = random.choice(values)
                    if new == old:
                        continue
                    V[r][c] = new
                    if verify_decomposition_fast(U, V, W, rank):
                        modified = True
                    else:
                        V[r][c] = old
                else:
                    r = random.randrange(pn)
                    c = random.randrange(rank)
                    old = W[r][c]
                    new = random.choice(values)
                    if new == old:
                        continue
                    W[r][c] = new
                    if verify_decomposition_fast(U, V, W, rank):
                        modified = True
                    else:
                        W[r][c] = old
            if not modified and it > 50:
                break
        
        return U, V, W
    
    # Try to find a valid decomposition at various ranks
    # Start from a known valid one and try to reduce rank
    
    # Approach: Start with schoolbook, then try to find linear dependencies
    # to reduce the rank. If two columns are linearly dependent in a certain way,
    # we might be able to eliminate one.
    
    # Simpler: try random search at decreasing ranks
    # For each rank, try many random initializations
    
    for rank in range(32, min(64, best_rank)):
        if time_left() < 10:
            break
        
        found = False
        attempts = 0
        max_attempts = 500 if rank <= 40 else 200
        
        while attempts < max_attempts and time_left() > 10:
            attempts += 1
            U, V, W = random_decomposition(rank)
            
            # Quick check
            if verify_decomposition_fast(U, V, W, rank):
                # Exact verification
                if verify_decomposition(U, V, W, rank):
                    best_rank = rank
                    best_U = U
                    best_V = V
                    best_W = W
                    found = True
                    break
            
            # Try local search from this initialization
            if time_left() > 15:
                U2, V2, W2 = local_search(U, V, W, rank, max_iter=200)
                if verify_decomposition_fast(U2, V2, W2, rank):
                    if verify_decomposition(U2, V2, W2, rank):
                        best_rank = rank
                        best_U = U2
                        best_V = V2
                        best_W = W2
                        found = True
                        break
        
        if found:
            # Try to reduce further
            continue
    
    # Final exact verification
    if best_rank < n * m * p:
        if not verify_decomposition(best_U, best_V, best_W, best_rank):
            # Fallback to schoolbook
            best_U = U_school
            best_V = V_school
            best_W = W_school
            best_rank = rank_school
    
    return {"U": best_U, "V": best_V, "W": best_W}
