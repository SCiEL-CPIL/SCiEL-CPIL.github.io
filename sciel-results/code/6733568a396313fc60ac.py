"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _verify_fast(U, V, W, n, m, p):
    """Fast verification using numpy. Returns True if exact decomposition."""
    rank = U.shape[1]
    # Check all 1-entries: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j], V[j*p+k])  # This is wrong - need element-wise then multiply with W
                # Actually need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                # = dot(U[i*m+j] * V[j*p+k], W[k*n+i])
                s = np.dot(U[i*m+j] * V[j*p+k], W[k*n+i])
                if abs(s - 1.0) > 1e-9:
                    return False
    
    # Check 0-entries: for all (a,b,c) not in valid set, sum should be 0
    # Valid set: (i*m+j, j*p+k, k*n+i)
    valid_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid_set.add((i*m+j, j*p+k, k*n+i))
    
    # For efficiency, check a sample of 0-entries if rank is large
    # But for small dims, check all
    total_tuples = (n*m) * (m*p) * (p*n)
    if total_tuples <= 50000:
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    if (a, b, c) in valid_set:
                        continue
                    s = np.dot(U[a] * V[b], W[c])
                    if abs(s) > 1e-9:
                        return False
    else:
        # Sample check
        rng = np.random.RandomState(42)
        for _ in range(5000):
            a = rng.randint(0, n*m)
            b = rng.randint(0, m*p)
            c = rng.randint(0, p*n)
            if (a, b, c) in valid_set:
                continue
            s = np.dot(U[a] * V[b], W[c])
            if abs(s) > 1e-9:
                return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition as numpy arrays."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    schoolbook_rank = n * m * p
    
    # Start with schoolbook
    best_U, best_V, best_W = _schoolbook(n, m, p)
    best_rank = schoolbook_rank
    
    # Precompute valid index set
    valid_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid_set.add((i*m+j, j*p+k, k*n+i))
    
    # Strategy: Try to find lower rank via random search with structured initialization
    # Key insight: use small integer entries (-1, 0, 1) and try multiple restarts
    # Also try a "perturbation" approach: start from schoolbook and try to merge columns
    
    random.seed(12345)
    np.random.seed(12345)
    
    # Try ranks from schoolbook_rank-1 down to lower bound
    # Lower bound: max(nm, mp, pn) = 30
    lower_bound = max(nm, mp, pn)
    
    # Focus effort on promising ranks
    # For 3x5x6, try ranks 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30
    candidate_ranks = []
    for r in range(schoolbook_rank - 1, lower_bound - 1, -1):
        candidate_ranks.append(r)
    
    # Prioritize certain ranks (multiples of 5, or known good values)
    # Shuffle to diversify
    random.shuffle(candidate_ranks)
    
    for R in candidate_ranks:
        if time.time() > deadline:
            break
        
        # Multiple attempts per rank
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random initialization with {-1, 0, 1}
            # Bias towards sparsity: use -1, 0, 1 with P(1)=P(-1)=0.25, P(0)=0.5
            U = np.random.choice([-1, 0, 1], size=(nm, R), p=[0.25, 0.5, 0.25])
            V = np.random.choice([-1, 0, 1], size=(mp, R), p=[0.25, 0.5, 0.25])
            W = np.random.choice([-1, 0, 1], size=(pn, R), p=[0.25, 0.5, 0.25])
            
            # Quick check on 1-entries first
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i*m+j] * V[j*p+k], W[k*n+i])
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Full verification
            if _verify_fast(U, V, W, n, m, p):
                best_rank = R
                best_U, best_V, best_W = U, V, W
                break
        
        if best_rank < schoolbook_rank:
            # Found improvement, continue trying lower ranks
            continue
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
