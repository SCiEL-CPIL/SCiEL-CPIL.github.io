"""Program for matrix multiplication 2x4 * 4x5 with rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Build target tensor
    # T[a,b,c] where a in 0..7 (A index), b in 0..19 (B index), c in 0..9 (C index)
    # T[a,b,c] = 1 if a=i*m+k, b=k*p+j, c=j*n+i for valid (i,k,j)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i*m + k
                b = k*p + j
                c = j*n + i
                T[a,b,c] = 1.0
    
    # Start with rank-36 solution
    rank = 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    def add_strassen_2x2(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        nonlocal col
        u00 = a_row_offset * m + a_col_offset
        u01 = a_row_offset * m + a_col_offset + 1
        u10 = (a_row_offset + 1) * m + a_col_offset
        u11 = (a_row_offset + 1) * m + a_col_offset + 1
        v00 = b_row_offset * p + b_col_offset
        v01 = b_row_offset * p + b_col_offset + 1
        v10 = (b_row_offset + 1) * p + b_col_offset
        v11 = (b_row_offset + 1) * p + b_col_offset + 1
        w00 = c_col_offset * n + c_row_offset
        w01 = (c_col_offset + 1) * n + c_row_offset
        w10 = c_col_offset * n + c_row_offset + 1
        w11 = (c_col_offset + 1) * n + c_row_offset + 1
        
        U[u00,col] += 1; U[u11,col] += 1
        V[v00,col] += 1; V[v11,col] += 1
        W[w00,col] += 1; W[w11,col] += 1
        col += 1
        
        U[u10,col] += 1; U[u11,col] += 1
        V[v00,col] += 1
        W[w10,col] += 1; W[w11,col] -= 1
        col += 1
        
        U[u00,col] += 1
        V[v01,col] += 1; V[v11,col] -= 1
        W[w01,col] += 1; W[w11,col] += 1
        col += 1
        
        U[u11,col] += 1
        V[v10,col] += 1; V[v00,col] -= 1
        W[w00,col] += 1; W[w10,col] += 1
        col += 1
        
        U[u00,col] += 1; U[u01,col] += 1
        V[v11,col] += 1
        W[w00,col] -= 1; W[w01,col] += 1
        col += 1
        
        U[u10,col] += 1; U[u00,col] -= 1
        V[v00,col] += 1; V[v01,col] += 1
        W[w11,col] += 1
        col += 1
        
        U[u01,col] += 1; U[u11,col] -= 1
        V[v10,col] += 1; V[v11,col] += 1
        W[w00,col] += 1
        col += 1
    
    def add_schoolbook_2x2_2x1(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        nonlocal col
        u00 = a_row_offset * m + a_col_offset
        u01 = a_row_offset * m + a_col_offset + 1
        u10 = (a_row_offset + 1) * m + a_col_offset
        u11 = (a_row_offset + 1) * m + a_col_offset + 1
        v0 = b_row_offset * p + b_col_offset
        v1 = (b_row_offset + 1) * p + b_col_offset
        w0 = c_col_offset * n + c_row_offset
        w1 = c_col_offset * n + c_row_offset + 1
        
        U[u00,col] = 1; V[v0,col] = 1; W[w0,col] = 1
        col += 1
        U[u01,col] = 1; V[v1,col] = 1; W[w0,col] = 1
        col += 1
        U[u10,col] = 1; V[v0,col] = 1; W[w1,col] = 1
        col += 1
        U[u11,col] = 1; V[v1,col] = 1; W[w1,col] = 1
        col += 1
    
    add_strassen_2x2(0, 0, 0, 0, 0, 0)
    add_strassen_2x2(0, 2, 2, 0, 0, 0)
    add_strassen_2x2(0, 0, 0, 2, 0, 2)
    add_strassen_2x2(0, 2, 2, 2, 0, 2)
    add_schoolbook_2x2_2x1(0, 0, 0, 4, 0, 4)
    add_schoolbook_2x2_2x1(0, 2, 2, 4, 0, 4)
    
    # Verify rank-36 solution
    def verify(U, V, W, T, tol=1e-9):
        for a in range(U.shape[0]):
            for b in range(V.shape[0]):
                for c in range(W.shape[0]):
                    val = sum(U[a,t]*V[b,t]*W[c,t] for t in range(U.shape[1]))
                    if abs(val - T[a,b,c]) > tol:
                        return False
        return True
    
    # Try to reduce to rank 35 by eliminating one column
    # For each column t, try to express it as a linear combination of others
    best_rank = 36
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    if time.time() < deadline:
        # Try eliminating each of the 36 columns
        for elim in range(36):
            if time.time() > deadline - 10:
                break
            # Remove column elim and try to solve for it
            U_rem = np.delete(U, elim, axis=1)
            V_rem = np.delete(V, elim, axis=1)
            W_rem = np.delete(W, elim, axis=1)
            
            # The removed column contributes: U[:,elim] ⊗ V[:,elim] ⊗ W[:,elim]
            # We need: sum_{t!=elim} U[a,t]V[b,t]W[c,t] + U[a,elim]V[b,elim]W[c,elim] = T[a,b,c]
            # So: U[a,elim]V[b,elim]W[c,elim] = T[a,b,c] - sum_{t!=elim} U[a,t]V[b,t]W[c,t]
            
            # This is a rank-1 tensor equation: find u,v,w such that u⊗v⊗w = R
            # where R[a,b,c] = T[a,b,c] - sum_{t!=elim} U[a,t]V[b,t]W[c,t]
            
            R = T.copy()
            for t in range(35):
                R += np.outer(np.outer(U_rem[:,t], V_rem[:,t]), W_rem[:,t]).reshape(n*m, m*p, p*n)
            
            # Check if R is rank-1: R[a,b,c] = u[a]*v[b]*w[c]
            # For rank-1, all 2x2 minors of any matrix slice must be zero
            # Check: R is rank-1 iff there exist u,v,w such that R[a,b,c] = u[a]v[b]w[c]
            
            # Try to find u,v,w by: pick non-zero entries to determine the vectors
            # Find a non-zero entry
            nonzero = np.unravel_index(np.argmax(np.abs(R)), R.shape)
            if abs(R[nonzero]) < 1e-12:
                continue
            
            a0, b0, c0 = nonzero
            # R[a0,b0,c0] = u[a0]*v[b0]*w[c0]
            # We have 3 unknowns for this one equation. Need more equations.
            
            # For fixed a0: R[a0,b,c] = u[a0]*v[b]*w[c] for all b,c
            # This means the matrix R[a0,:,:] has rank 1
            slice_ac = R[a0,:,:]  # shape (m*p, p*n)
            # Check rank 1
            if slice_ac.size > 0:
                # Compute rank via SVD
                try:
                    sv = np.linalg.svd(slice_ac, compute_uv=False)
                    if len(sv) > 1 and sv[1] > 1e-8 * sv[0]:
                        continue
                except:
                    continue
            
            # Similarly check for fixed b0 and c0
            slice_bc = R[:,b0,:]  # shape (n*m, p*n)
            try:
                sv = np.linalg.svd(slice_bc, compute_uv=False)
                if len(sv) > 1 and sv[1] > 1e-8 * sv[0]:
                    continue
            except:
                continue
            
            slice_ab = R[:,:,c0]  # shape (n*m, m*p)
            try:
                sv = np.linalg.svd(slice_ab, compute_uv=False)
                if len(sv) > 1 and sv[1] > 1e-8 * sv[0]:
                    continue
            except:
                continue
            
            # R appears to be rank-1. Extract u,v,w
            # u[a] = R[a,b0,c0] / (v[b0]*w[c0])
            # v[b] = R[a0,b,c0] / (u[a0]*w[c0])
            # w[c] = R[a0,b0,c] / (u[a0]*v[b0])
            
            # Initialize
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            
            # Set u[a0] = 1 (scaling freedom)
            u[a0] = 1.0
            v[b0] = R[a0,b0,c0] / (u[a0] * w[c0]) if w[c0] != 0 else R[a0,b0,c0]
            # This is circular. Let's use:
            # R[a0,b0,c0] = u[a0]*v[b0]*w[c0]
            # Set u[a0]=1, then v[b0]*w[c0] = R[a0,b0,c0]
            # For another entry R[a0,b0,c1] = u[a0]*v[b0]*w[c1] = v[b0]*w[c1]
            # So w[c1]/w[c0] = R[a0,b0,c1]/R[a0,b0,c0]
            
            # Find another non-zero in the c-direction
            w[c0] = 1.0
            v[b0] = R[a0,b0,c0]
            for c in range(p*n):
                if c != c0 and abs(R[a0,b0,c]) > 1e-12:
                    w[c] = R[a0,b0,c] / v[b0]
            for b in range(m*p):
                if b != b0 and abs(R[a0,b,c0]) > 1e-12:
                    v[b] = R[a0,b,c0] / w[c0]
            for a in range(n*m):
                if a != a0 and abs(R[a,b0,c0]) > 1e-12:
                    u[a] = R[a,b0,c0] / (v[b0]*w[c0])
            
            # Verify
            R_recon = np.einsum('a,b,c->abc', u, v, w)
            if np.allclose(R, R_recon, atol=1e-8):
                # We found a rank-35 solution!
                best_rank = 35
                best_U = U_rem
                best_V = V_rem
                best_W = W_rem
                break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
