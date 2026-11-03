import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 120.0  # 120 seconds
    
    n, m, p = 3, 3, 4
    dim_u = n * m  # 9
    dim_v = m * p  # 12
    dim_w = p * n  # 12
    
    # First, let's try to construct a rank 35 solution using Strassen on a 2x2x2 sub-block
    # and schoolbook for the rest.
    
    # Let's verify if Strassen works for the sub-block A[0:2,0:2] x B[0:2,0:2] -> C[0:2,0:2]
    # Then add schoolbook terms for everything else.
    
    # Strassen rank-7 vectors for 2x2x2 (u,v,w each in R^4)
    # Indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # Standard Strassen:
    # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(g-h), M4 = d(f-e)
    # M5 = (a+b)h, M6 = (c-a)(e+f), M7 = (b-d)(g+h)
    # where A=[[a,b],[c,d]], B=[[e,f],[g,h]]
    
    # In our indexing:
    # A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d
    # B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h
    
    # C[0,0] = M1 + M4 - M5 - M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M7
    
    # Let me verify C[1,0] = M2 + M4 = (c+d)e + d(f-e) = ce + de + df - de = ce + df
    # But C[1,0] should be c*e + d*g.
    # So we need f = g, which means B[0,1] = B[1,0]. That's not generally true.
    
    # I think I have the B indexing wrong. Let me try:
    # B[0,0]=e, B[1,0]=f, B[0,1]=g, B[1,1]=h (column-major?)
    # No, the task says B is m x p = 3 x 4, and v-index is j*p+k.
    # So B[j,k] has index j*4+k. For the 2x2 sub-block (j in {0,1}, k in {0,1}):
    # B[0,0]=0, B[0,1]=1, B[1,0]=4, B[1,1]=5.
    
    # In the 2x2 sub-block, if I reindex locally:
    # Local B: [B[0,0], B[0,1], B[1,0], B[1,1]] = indices [0, 1, 4, 5] in global V.
    # So local index 0 -> global 0, local 1 -> global 1, local 2 -> global 4, local 3 -> global 5.
    
    # Let me just define the Strassen vectors in terms of local indices and map to global.
    
    # For the 2x2x2 block with local indices:
    # u_local: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1) -> global u: 0,1,3,4 (since i*3+j: (0,0)=0,(0,1)=1,(1,0)=3,(1,1)=4)
    # v_local: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1) -> global v: 0,1,4,5 (since j*4+k: (0,0)=0,(0,1)=1,(1,0)=4,(1,1)=5)
    # w_local: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1) -> global w: 0,3,1,4 (since k*3+i: (0,0)=0,(0,1)=3,(1,0)=1,(1,1)=4)
    
    # Wait, w-index is k*n+i = k*3+i.
    # (k,i): (0,0)=0, (0,1)=1, (1,0)=3, (1,1)=4.
    # So w_local 0=(k=0,i=0)=0, w_local 1=(k=0,i=1)=1, w_local 2=(k=1,i=0)=3, w_local 3=(k=1,i=1)=4.
    
    # Hmm, this is getting complicated. Let me just build the full tensor and try to find a decomposition.
    
    # Actually, let me just return the schoolbook solution to be safe, since I'm not confident about the Strassen vectors.
    
    # Schoolbook rank 36
    R = 36
    U = np.zeros((dim_u, R))
    V = np.zeros((dim_v, R))
    W = np.zeros((dim_w, R))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
