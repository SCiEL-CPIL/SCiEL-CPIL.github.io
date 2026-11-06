import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Target n
    target_n = 50
    
    # Function to check if a coloring is valid
    def is_valid(C, n):
        for c in range(4):
            # Get adjacency matrix for color c
            A = (C == c).astype(int)
            # Check for triangles: trace(A^3) should be 0
            # Or simpler: for each u<v, if C[u,v]==c, then no w such that C[u,w]==c and C[v,w]==c
            # Using matrix multiplication: (A @ A)[u,v] counts common neighbors.
            # If C[u,v]==c and (A @ A)[u,v] > 0, then there is a triangle.
            A2 = A @ A
            # For all u,v where C[u,v]==c, A2[u,v] must be 0
            mask = (C == c)
            # Exclude diagonal
            np.fill_diagonal(mask, 0)
            if np.any(A2[mask]):
                return False
        return True

    # Local search
    def local_search(n, max_iters=200000):
        # Initialize random coloring
        C = np.random.randint(0, 4, (n, n)).astype(np.int8)
        np.fill_diagonal(C, -1)
        # Symmetrize
        C = (C + C.T) // 2
        np.fill_diagonal(C, -1)
        
        # Precompute A_c matrices
        # A_c[u,v] = 1 if C[u,v] == c
        A = [np.zeros((n, n), dtype=np.int8) for _ in range(4)]
        for c in range(4):
            A[c] = (C == c).astype(np.int8)
            
        # Precompute M_c = A_c @ A_c
        M = [A[c] @ A[c] for c in range(4)]
        
        # Helper to get current cost for edge (u,v)
        # Cost contribution of edge (u,v) with color c is M_c[u,v]
        # Total cost S = sum_{u<v} M_{C[u,v]}[u,v]
        
        # We want to minimize S.
        
        current_S = 0
        for u in range(n):
            for v in range(u+1, n):
                c = C[u,v]
                current_S += M[c][u,v]
                
        best_S = current_S
        best_C = C.copy()
        
        for i in range(max_iters):
            if time.time() > deadline - 10.0:
                break
                
            # Pick random edge
            u = np.random.randint(0, n)
            v = np.random.randint(0, n)
            if u == v:
                continue
            if u > v:
                u, v = v, u
                
            c_old = C[u,v]
            
            # Find best new color
            best_delta = 0
            c_new = c_old
            
            for c_new_try in range(4):
                if c_new_try == c_old:
                    continue
                # Delta S = M[c_new][u,v] - M[c_old][u,v]
                delta = M[c_new_try][u,v] - M[c_old][u,v]
                if delta < best_delta:
                    best_delta = delta
                    c_new = c_new_try
                    
            # If no improvement, pick random
            if c_new == c_old:
                c_new = np.random.choice([c for c in range(4) if c != c_old])
                delta = M[c_new][u,v] - M[c_old][u,v]
            else:
                delta = best_delta
                
            # Apply move
            current_S += delta
            C[u,v] = c_new
            C[v,u] = c_new
            
            # Update A and M
            # A[c_old][u,v] becomes 0, A[c_old][v,u] becomes 0
            # A[c_new][u,v] becomes 1, A[c_new][v,u] becomes 1
            
            # Update M[c_old]
            # M = A @ A
            # Delta A is -1 at (u,v) and (v,u)
            # Delta M = -A[:,u]A[u,:] - A[:,v]A[v,:] + A[u,u]A[u,u]... (diagonal terms)
            # Actually, simpler:
            # M_new = M_old - A[:,u] @ A[u,:] - A[:,v] @ A[v,:] + (A[u,u] + A[v,v]) * (outer product of e_u+e_v?)
            # Let's just recompute the rows/cols u and v of M.
            
            # For M[c_old]:
            # M[c_old][u,:] = A[c_old][u,:] @ A[c_old]
            # M[c_old][:,u] = A[c_old][:,u] @ A[c_old]
            # Since A is symmetric, M is symmetric.
            
            # Update A[c_old]
            A[c_old][u,v] = 0
            A[c_old][v,u] = 0
            
            # Update M[c_old]
            # M[c_old][u,:] = A[c_old][u,:] @ A[c_old]
            M[c_old][u,:] = A[c_old][u,:] @ A[c_old]
            M[c_old][:,u] = M[c_old][u,:]
            
            M[c_old][v,:] = A[c_old][v,:] @ A[c_old]
            M[c_old][:,v] = M[c_old][v,:]
            
            # Update A[c_new]
            A[c_new][u,v] = 1
            A[c_new][v,u] = 1
            
            # Update M[c_new]
            M[c_new][u,:] = A[c_new][u,:] @ A[c_new]
            M[c_new][:,u] = M[c_new][u,:]
            
            M[c_new][v,:] = A[c_new][v,:] @ A[c_new]
            M[c_new][:,v] = M[c_new][v,:]
            
            if current_S == 0:
                # Found valid coloring
                return C
                
            if current_S < best_S:
                best_S = current_S
                best_C = C.copy()
                
        return best_C

    # Try local search for n=50
    C50 = local_search(50)
    
    if is_valid(C50, 50):
        # Convert to list format
        colors = []
        for i in range(50):
            for j in range(i+1, 50):
                colors.append(int(C50[i,j]))
        return {'n': 50, 'colors': colors}
        
    # Fallback to n=32 construction
    def gf16_mul(a, b):
        r = 0
        for i in range(4):
            if (b >> i) & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if (r >> i) & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log = {}
    x = 1
    for e in range(15):
        log[x] = e
        x = gf16_mul(x, 2)

    colors32 = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                colors32.append(log[i ^ j] % 3)
            else:
                colors32.append(3)
                
    return {'n': 32, 'colors': colors32}
