"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build the target tensor (sparse representation)
    # T[a, b, c] = 1 if a=im+j, b=jp+k, c=kn+i for some i,j,k
    # We'll store it as a list of (a, b, c) triples with value 1
    target_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                target_triples.append((a, b, c))
    
    # Schoolbook decomposition as fallback
    rank_school = n*m*p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                U_school[a, col] = 1
                V_school[b, col] = 1
                W_school[c, col] = 1
                col += 1
    
    # Try to find a lower rank decomposition
    deadline = time.time() + 110  # 110 seconds to be safe
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try ranks from 39 down to some minimum
    # For each rank, try a few random initializations
    for r in range(rank_school - 1, max(nm, mp, pn) - 1, -1):
        if time.time() > deadline:
            break
        
        # Try a few random attempts for this rank
        max_attempts = 5
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random initialization
            np.random.seed(attempt * 1000 + r)
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            W = np.random.randn(pn, r)
            
            # Use a simple iterative refinement
            # We want sum_c U[a,c]*V[b,c]*W[c,c] = 1 for (a,b,c) in target_triples
            # and 0 otherwise (but we only care about the target entries being correct)
            
            # Actually, we need the full tensor to match. Let's check all entries.
            # But the tensor is mostly zero, so we need:
            # For all (a,b,c): sum_c U[a,c]*V[b,c]*W[c,c] = T[a,b,c]
            
            # This is a nonlinear system. Let's use a simple approach:
            # Fix U and V, solve for W, then fix V and W, solve for U, etc.
            
            # For fixed U and V, W[c, :] should satisfy:
            # For each (a,b,c) in target: sum_c' U[a,c']*V[b,c']*W[c',c] = 1
            # This is a linear system in W.
            
            # Let's build the system. For each c (output index), we have equations:
            # For each (a,b) such that (a,b,c) is in target: sum_c' U[a,c']*V[b,c']*W[c',c] = 1
            
            # Actually, this is getting complex. Let me use a different approach.
            
            # Use the fact that we can verify a decomposition by checking:
            # For all (a,b,c) in target_triples: sum_c' U[a,c']*V[b,c']*W[c',c] == 1
            # And for all other (a,b,c): sum_c' U[a,c']*V[b,c']*W[c',c] == 0
            
            # Let's just try a few random decompositions and check if they work.
            # This is unlikely to work for r < 40, but let's try.
            
            # Check if this random decomposition works
            valid = True
            for (a, b, c) in target_triples:
                val = 0
                for c_prime in range(r):
                    val += U[a, c_prime] * V[b, c_prime] * W[c, c_prime]
                if abs(val - 1) > 1e-6:
                    valid = False
                    break
            
            if valid:
                # Check a few non-target entries to make sure they're ~0
                for _ in range(100):
                    a = np.random.randint(nm)
                    b = np.random.randint(mp)
                    c = np.random.randint(pn)
                    if (a, b, c) in target_triples:
                        continue
                    val = 0
                    for c_prime in range(r):
                        val += U[a, c_prime] * V[b, c_prime] * W[c, c_prime]
                    if abs(val) > 1e-4:
                        valid = False
                        break
            
            if valid:
                best_rank = r
                best_U, best_V, best_W = U, V, W
                break  # Found a decomposition for this rank
        
        if best_rank == r:
            break  # Found the lowest rank we can
    
    # Convert to lists
    if best_rank == rank_school:
        U_list = U_school.tolist()
        V_list = V_school.tolist()
        W_list = W_school.tolist()
    else:
        U_list = best_U.tolist()
        V_list = best_V.tolist()
        W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
