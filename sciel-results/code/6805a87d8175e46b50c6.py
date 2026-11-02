import numpy as np
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    n, m, p = 2, 3, 6
    # Dimensions for the tensor T_{a,b,c,d,i,k}
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    # Indices:
    # a in 0..n-1, b in 0..m-1 -> U index u = a*m + b
    # c in 0..m-1, d in 0..p-1 -> V index v = c*p + d
    # k in 0..p-1, i in 0..n-1 -> W index w = k*n + i
    
    # The tensor T is defined by T_{u,v,w} = 1 if (a==i and c==k and b==d) else 0
    # Note: u=a*m+b, v=c*p+d, w=k*n+i
    # Condition: a==i, c==k, b==d
    # From u=a*m+b: a=u//m, b=u%m
    # From v=c*p+d: c=v//p, d=v%p
    # From w=k*n+i: k=w//n, i=w%n
    # So T_{u,v,w} = 1 if (u//m == w%n) and (v//p == w//n) and (u%m == v%p)

    u_dim = n * m  # 6
    v_dim = m * p  # 18
    w_dim = p * n  # 12

    # We want to find rank r decomposition.
    # The equations are: sum_{l=1}^r U_{u,l} V_{v,l} W_{w,l} = T_{u,v,w}
    # This is a multilinear system.
    # To use LLL, we can fix two matrices and solve for the third?
    # Or use a "structured" ansatz.
    
    # Alternative Strategy: 
    # The problem is equivalent to finding a set of r rank-1 tensors that sum to T.
    # We can try to find a basis for the space of solutions.
    
    # Since r is expected to be small (around 33-34), and entries are integers,
    # we can formulate this as an integer programming problem or use lattice reduction.
    
    # Let's try a different structural approach first: 
    # We know M_{2,3,6} = M_{2,3,3} \oplus M_{2,3,3} has rank 34.
    # Can we do better?
    # Theoretical lower bound for M_{2,3,6} is not well known, but likely < 34.
    # Let's try to find a rank 33 solution.
    
    # We will use a "sparse" random search with LLL refinement.
    # 1. Generate a random integer matrix U of size (6, 33) with small entries.
    # 2. Generate random V (18, 33).
    # 3. Solve for W? No, W is also unknown.
    
    # Better: Use the fact that the tensor is sparse.
    # There are 2*3*6 = 36 non-zero entries in the "fully expanded" tensor?
    # No, the tensor T has dimension 6*18*12 = 1296.
    # Non-zero entries: 
    # For each (a,b,c,d,i,k) such that a=i, c=k, b=d.
    # a: 2, b: 3, c: 3, d: 6, i: 2, k: 6.
    # Constraints: i=a (1 choice), k=c (1 choice), d=b (1 choice).
    # Free variables: a (2), b (3), c (3).
    # Total non-zeros = 2 * 3 * 3 = 18.
    # Wait, the tensor T is 6-dimensional.
    # T_{a,b,c,d,i,k} = delta_{a,i} delta_{c,k} delta_{b,d}.
    # The number of non-zero entries is indeed 18.
    # The rest are 0.
    
    # So we have 1296 equations. 18 are "1 = sum...", 1278 are "0 = sum...".
    
    # Let's try to construct a rank 33 solution using a specific structure.
    # If we assume the solution comes from M_{2,3,2} blocks, rank is 3*r232.
    # If r232=11, rank=33.
    # If r232=10, rank=30.
    
    # Let's hardcode the known rank 34 solution first to ensure we return something valid.
    # Then, if time permits, try to find rank 33.
    
    def build_rank34():
        # From the current program's fallback
        U233_seed = np.zeros((6, 17))
        V233_seed = np.zeros((9, 17))
        W233_seed = np.zeros((6, 17))
        
        U233_seed[0, 0] = 1; U233_seed[4, 0] = 1
        V233_seed[0, 0] = 1; V233_seed[4, 0] = 1
        W233_seed[0, 0] = 1
        W233_seed[3, 0] = 1
        
        U233_seed[3, 1] = 1; U233_seed[4, 1] = 1
        V233_seed[0, 1] = 1
        W233_seed[1, 1] = 1
        W233_seed[3, 1] = -1
        
        U233_seed[0, 2] = 1
        V233_seed[1, 2] = 1; V233_seed[4, 2] = -1
        W233_seed[2, 2] = 1
        W233_seed[3, 2] = 1
        
        U233_seed[4, 3] = 1
        V233_seed[3, 3] = 1; V233_seed[0, 3] = -1
        W233_seed[0, 3] = 1
        W233_seed[1, 3] = 1
        
        U233_seed[0, 4] = 1; U233_seed[1, 4] = 1
        V233_seed[4, 4] = 1
        W233_seed[0, 4] = -1
        W233_seed[2, 4] = 1
        
        U233_seed[3, 5] = 1; U233_seed[0, 5] = -1
        V233_seed[0, 5] = 1; V233_seed[1, 5] = 1
        W233_seed[3, 5] = 1
        
        U233_seed[1, 6] = 1; U233_seed[4, 6] = -1
        V233_seed[3, 6] = 1; V233_seed[4, 6] = 1
        W233_seed[0, 6] = 1
        
        U233_seed[0, 7] = 1
        V233_seed[2, 7] = 1
        W233_seed[4, 7] = 1
        
        U233_seed[1, 8] = 1
        V233_seed[5, 8] = 1
        W233_seed[4, 8] = 1
        
        U233_seed[3, 9] = 1
        V233_seed[2, 9] = 1
        W233_seed[5, 9] = 1
        
        U233_seed[4, 10] = 1
        V233_seed[5, 10] = 1
        W233_seed[5, 10] = 1
        
        U233_seed[2, 11] = 1
        V233_seed[6, 11] = 1
        W233_seed[0, 11] = 1
        
        U233_seed[2, 12] = 1
        V233_seed[7, 12] = 1
        W233_seed[2, 12] = 1
        
        U233_seed[2, 13] = 1
        V233_seed[8, 13] = 1
        W233_seed[4, 13] = 1
        
        U233_seed[5, 14] = 1
        V233_seed[6, 14] = 1
        W233_seed[1, 14] = 1
        
        U233_seed[5, 15] = 1
        V233_seed[7, 15] = 1
        W233_seed[3, 15] = 1
        
        U233_seed[5, 16] = 1
        V233_seed[8, 16] = 1
        W233_seed[5, 16] = 1
        
        U_comb = np.zeros((6, 34))
        V_comb = np.zeros((18, 34))
        W_comb = np.zeros((12, 34))
        
        U_comb[:, :17] = U233_seed
        U_comb[:, 17:] = U233_seed
        
        for j in range(3):
            for k in range(3):
                V_comb[j * 6 + k, :17] = V233_seed[j * 3 + k]
                V_comb[j * 6 + 3 + k, 17:] = V233_seed[j * 3 + k]
                
        for k in range(3):
            for i in range(2):
                W_comb[k * 2 + i, :17] = W233_seed[k * 2 + i]
                W_comb[(k + 3) * 2 + i, 17:] = W233_seed[k * 2 + i]
                
        return U_comb, V_comb, W_comb

    # Fallback solution
    U_fallback, V_fallback, W_fallback = build_rank34()
    
    # Try to find rank 33 using LLL-inspired search
    # We will try to find a rank 11 decomposition for M_{2,3,2}
    # If found, we can construct M_{2,3,6} with rank 33.
    
    def find_m232_rank11():
        # Search for rank 11 decomposition of M_{2,3,2}
        # n=2, m=3, p=2
        n2, m2, p2 = 2, 3, 2
        u_dim2 = n2 * m2  # 6
        v_dim2 = m2 * p2  # 6
        w_dim2 = p2 * n2  # 4
        
        # Tensor T2 has non-zeros at (a,b,c,d,i,k) where a=i, c=k, b=d.
        # Non-zeros: 2*3*2 = 12? No.
        # a:2, b:3, c:3, d:2, i:2, k:2.
        # i=a, k=c, d=b.
        # Free: a,b,c. 2*3*3 = 18 non-zeros.
        
        # We want U(6,11), V(6,11), W(4,11) such that sum_l U_{u,l}V_{v,l}W_{w,l} = T_{u,v,w}
        
        # Strategy: Random initialization with small integers, then local search / LLL refinement.
        # Since we are short on time, we will do a limited random search.
        
        best_err = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        # We will try a few random restarts
        for restart in range(20):
            if time.time() > deadline - 10:
                break
                
            # Initialize with random integers in {-1, 0, 1}
            U = np.random.choice([-1, 0, 1], size=(u_dim2, 11))
            V = np.random.choice([-1, 0, 1], size=(v_dim2, 11))
            W = np.random.choice([-1, 0, 1], size=(w_dim2, 11))
            
            # Calculate error
            err = 0
            for a in range(n2):
                for b in range(m2):
                    for c in range(m2):
                        for d in range(p2):
                            for i in range(n2):
                                for k in range(p2):
                                    target = 1.0 if (a == i and c == k and b == d) else 0.0
                                    u = a * m2 + b
                                    v = c * p2 + d
                                    w = k * n2 + i
                                    s = np.dot(U[u, :], np.multiply(V[v, :], W[w, :]))
                                    err += (s - target) ** 2
                                    
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
                if err == 0:
                    return best_U, best_V, best_W
                    
            # Local search: perturb one entry
            for step in range(500):
                if time.time() > deadline - 10:
                    return None if best_err > 0 else (best_U, best_V, best_W)
                    
                mat_idx = random.randint(0, 2)
                if mat_idx == 0:
                    mat = U
                    dim = u_dim2
                elif mat_idx == 1:
                    mat = V
                    dim = v_dim2
                else:
                    mat = W
                    dim = w_dim2
                    
                row = random.randint(0, dim - 1)
                col = random.randint(0, 10)
                
                old_val = mat[row, col]
                new_val = old_val + random.choice([-1, 1])
                mat[row, col] = new_val
                
                # Recalculate error (slow, but ok for small dimensions)
                new_err = 0
                for a in range(n2):
                    for b in range(m2):
                        for c in range(m2):
                            for d in range(p2):
                                for i in range(n2):
                                    for k in range(p2):
                                        target = 1.0 if (a == i and c == k and b == d) else 0.0
                                        u = a * m2 + b
                                        v = c * p2 + d
                                        w = k * n2 + i
                                        s = np.dot(U[u, :], np.multiply(V[v, :], W[w, :]))
                                        new_err += (s - target) ** 2
                                        
                if new_err < err:
                    err = new_err
                    if err == 0:
                        return U, V, W
                else:
                    mat[row, col] = old_val
                    
        return None if best_err > 0 else (best_U, best_V, best_W)

    # Attempt to find rank 33
    if time.time() < deadline - 20:
        m232_sol = find_m232_rank11()
        if m232_sol is not None:
            U232, V232, W232 = m232_sol
            # Construct M_{2,3,6} from M_{2,3,2} blocks
            U_comb = np.zeros((6, 33))
            V_comb = np.zeros((18, 33))
            W_comb = np.zeros((12, 33))
            
            # Block 1 (d=0,1 -> k=0,1)
            U_comb[:, 0:11] = U232
            for j in range(3):
                for k in range(2):
                    V_comb[j * 6 + k, 0:11] = V232[j * 2 + k]
            for k in range(2):
                for i in range(2):
                    W_comb[k * 2 + i, 0:11] = W232[k * 2 + i]
                    
            # Block 2 (d=2,3 -> k=2,3)
            U_comb[:, 11:22] = U232
            for j in range(3):
                for k in range(2):
                    V_comb[j * 6 + (2 + k), 11:22] = V232[j * 2 + k]
            for k in range(2):
                for i in range(2):
                    W_comb[(2 + k) * 2 + i, 11:22] = W232[k * 2 + i]
                    
            # Block 3 (d=4,5 -> k=4,5)
            U_comb[:, 22:33] = U232
            for j in range(3):
                for k in range(2):
                    V_comb[j * 6 + (4 + k), 22:33] = V232[j * 2 + k]
            for k in range(2):
                for i in range(2):
                    W_comb[(4 + k) * 2 + i, 22:33] = W232[k * 2 + i]
                    
            # Verify
            if verify_decomposition(U_comb, V_comb, W_comb, n, m, p):
                return {
                    "U": U_comb.tolist(),
                    "V": V_comb.tolist(),
                    "W": W_comb.tolist()
                }
                
    # Return fallback
    return {
        "U": U_fallback.tolist(),
        "V": V_fallback.tolist(),
        "W": W_fallback.tolist()
    }

def verify_decomposition(U, V, W, n, m, p):
    """
    Verifies if U, V, W exactly decompose the matrix multiplication tensor.
    """
    r = U.shape[1]
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for i in range(n):
                        for k in range(p):
                            target = 1.0 if (a == i and c == k and b == d) else 0.0
                            u = a * m + b
                            v = c * p + d
                            w = k * n + i
                            s = np.dot(U[u, :], np.multiply(V[v, :], W[w, :]))
                            if not np.isclose(s, target, atol=1e-9):
                                return False
    return True

if __name__ == "__main__":
    construct()
