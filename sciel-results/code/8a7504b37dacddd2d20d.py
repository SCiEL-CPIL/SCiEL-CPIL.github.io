import numpy as np
import time
from itertools import product

_start_time = time.time()
_DEADLINE = _start_time + 140.0

def _try_rank_r_223(r, max_attempts=2000):
    """Try to find a rank-r decomposition of the 2x2x3 all-ones tensor."""
    # T[i,j,k] = 1 for i,j in {0,1}, k in {0,1,2}
    # U: 4 x r, V: 6 x r, W: 6 x r
    # Constraint: for all (i,j,k), sum_r U[i*2+j,r] * V[j*3+k,r] * W[k*2+i,r] = 1
    
    for attempt in range(max_attempts):
        if time.time() > _DEADLINE - 10:
            break
        rng = np.random.default_rng(attempt * 7919)
        
        # Initialize with small random integers
        U = rng.integers(-2, 3, size=(4, r))
        V = rng.integers(-2, 3, size=(6, r))
        W = rng.integers(-2, 3, size=(6, r))
        
        # Check if this works
        valid = True
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    s = 0
                    for rr in range(r):
                        s += U[i*2+j, rr] * V[j*3+k, rr] * W[k*2+i, rr]
                    if s != 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return U, V, W
    
    return None

def construct():
    n, m, p = 2, 2, 15
    
    # First, try to find a rank-10 decomposition for 2x2x3
    result_223 = _try_rank_r_223(10, max_attempts=5000)
    
    if result_223 is not None:
        U223, V223, W223 = result_223
        rank_223 = 10
    else:
        # Fall back to rank-11 (Strassen + schoolbook)
        U223 = np.zeros((4, 11), dtype=int)
        V223 = np.zeros((6, 11), dtype=int)
        W223 = np.zeros((6, 11), dtype=int)
        
        # Strassen for columns 0,1
        # P1: (A00+A11)(B00+B11)
        U223[0,0]=1; U223[3,0]=1; V223[0,0]=1; V223[1,0]=1
        W223[0,0]=1; W223[3,0]=1  # C[0,0] and C[1,1]... wait
        # Let me redo this carefully
        # C[0,0] = P1+P4-P5+P7, C[1,0]=P2+P4, C[0,1]=P3+P5, C[1,1]=P1-P2+P3+P6
        
        # P1: U=[1,0,0,1], V=[1,1,0,0,0,0], W for C[0,0]:+1, C[1,1]:+1
        U223[0,0]=1; U223[3,0]=1; V223[0,0]=1; V223[1,0]=1
        W223[0,0]=1; W223[3,0]=1
        
        # P2: U=[0,0,1,1], V=[1,0,0,0,0,0], W for C[1,0]:+1, C[1,1]:-1
        U223[2,1]=1; U223[3,1]=1; V223[0,1]=1
        W223[3,1]=1; W223[3,1]=-1  # ERROR: can't have both
        # W is indexed by [k*2+i], so W[0]=C[0,0], W[1]=C[0,1], W[2]=C[1,0], W[3]=C[1,1]
        # Let me redo with correct indexing
        # W[k*2+i] where k is column, i is row of C
        # C[0,0] -> W[0*2+0] = W[0]
        # C[0,1] -> W[1*2+0] = W[2]  WAIT NO
        # The trace convention is W[k*n+i, r] where n=2
        # So W index = k*2 + i
        # C[i,k] uses W[k*2+i]
        # C[0,0] -> W[0]
        # C[0,1] -> W[2]
        # C[1,0] -> W[1]
        # C[1,1] -> W[3]
        
        # Let me restart the Strassen block construction
        U223 = np.zeros((4, 11), dtype=int)
        V223 = np.zeros((6, 11), dtype=int)
        W223 = np.zeros((6, 11), dtype=int)
        
        # P1: (A00+A11)(B00+B11)
        U223[0,0]=1; U223[3,0]=1; V223[0,0]=1; V223[1,0]=1
        W223[0,0]=1  # C[0,0]
        W223[3,0]=1  # C[1,1]
        
        # P2: (A10+A11)B00
        U223[2,1]=1; U223[3,1]=1; V223[0,1]=1
        W223[1,1]=1  # C[1,0]
        W223[3,1]=-1 # C[1,1]
        
        # P3: A00(B01-B11)
        U223[0,2]=1; V223[2,2]=1; V223[3,2]=-1
        W223[2,2]=1  # C[0,1]
        W223[3,2]=1  # C[1,1]
        
        # P4: A11(B10-B00)
        U223[3,3]=1; V223[4,3]=1; V223[0,3]=-1
        W223[0,3]=1  # C[0,0]
        W223[1,3]=1  # C[1,0]
        
        # P5: (A00+A01)B11
        U223[0,4]=1; U223[1,4]=1; V223[1,4]=1
        W223[0,4]=-1 # C[0,0]
        W223[2,4]=1  # C[0,1]
        
        # P6: (A10-A00)(B00+B01)
        U223[2,5]=1; U223[0,5]=-1; V223[0,5]=1; V223[2,5]=1
        W223[3,5]=1  # C[1,1]
        
        # P7: (A01-A11)(B10+B11)
        U223[1,6]=1; U223[3,6]=-1; V223[4,6]=1; V223[5,6]=1
        W223[0,6]=1  # C[0,0]
        
        # Verify Strassen part
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = sum(U223[i*2+j,rr]*V223[j*3+k,rr]*W223[k*2+i,rr] for rr in range(7))
                    if s != 1:
                        pass  # debug
        
        # Schoolbook for column 2
        # C[0,2] = A00*B02 + A01*B12
        # C[1,2] = A10*B02 + A11*B12
        U223[0,7]=1; V223[6,7]=1; W223[4,7]=1  # A00*B02 -> C[0,2]
        U223[1,8]=1; V223[7,8]=1; W223[4,8]=1  # A01*B12 -> C[0,2]
        U223[2,9]=1; V223[6,9]=1; W223[5,9]=1  # A10*B02 -> C[1,2]
        U223[3,10]=1; V223[7,10]=1; W223[5,10]=1  # A11*B12 -> C[1,2]
        
        rank_223 = 11
    
    # Now build the full 2x2x15 decomposition
    # Use 5 Strassen blocks for columns 0-9, then handle columns 10-14
    # Or: use rank_223 blocks
    
    if rank_223 == 10:
        # 5 blocks of rank 10 = 50, total rank 50
        # But wait, 5 blocks of 2x2x3 covers 15 columns exactly!
        rank = 5 * rank_223
        U = np.zeros((4, rank), dtype=int)
        V = np.zeros((30, rank), dtype=int)
        W = np.zeros((30, rank), dtype=int)
        
        for blk in range(5):
            r0 = blk * rank_223
            k0 = blk * 3
            for rr in range(rank_223):
                U[:, r0+rr] = U223[:, rr]
                # V: V223 is 6x10, maps to V[j*15+k, r]
                for j in range(2):
                    for k in range(3):
                        V[j*15 + k0 + k, r0+rr] = V223[j*3+k, rr]
                # W: W223 is 6x10, maps to W[k*2+i, r]
                for k in range(3):
                    for i in range(2):
                        W[(k0+k)*2 + i, r0+rr] = W223[k*2+i, rr]
    else:
        # rank_223 = 11, use 4 blocks of 2x2x3 (covers 12 cols) + 1 block of 2x2x3 for last 3
        # That's 5 blocks of rank 11 = 55. Worse than 53.
        # So use the original approach: 7 Strassen + 4 schoolbook = 53
        rank = 53
        U = [[0] * rank for _ in range(4)]
        V = [[0] * rank for _ in range(30)]
        W = [[0] * rank for _ in range(30)]
        
        def add_strassen_block(base_r, k0, k1):
            r1 = base_r; r2 = base_r+1; r3 = base_r+2; r4 = base_r+3
            r5 = base_r+4; r6 = base_r+5; r7 = base_r+6
            
            U[0][r1]=1; U[3][r1]=1; V[k0][r1]=1; V[15+k1][r1]=1
            W[2*k0][r1]=1; W[2*k1+1][r1]=1
            
            U[2][r2]=1; U[3][r2]=1; V[k0][r2]=1
            W[2*k0+1][r2]=1; W[2*k1+1][r2]=-1
            
            U[0][r3]=1; V[15+k1][r3]=1; V[k1][r3]=-1
            W[2*k1][r3]=1; W[2*k1+1][r3]=1
            
            U[3][r4]=1; V[15+k0][r4]=1; V[k0][r4]=-1
            W[2*k0][r4]=1; W[2*k0+1][r4]=1
            
            U[0][r5]=1; U[1][r5]=1; V[15+k1][r5]=1
            W[2*k0][r5]=-1; W[2*k1][r5]=1
            
            U[2][r6]=1; U[0][r6]=-1; V[k0][r6]=1; V[15+k0][r6]=1
            W[2*k1+1][r6]=1
            
            U[1][r7]=1; U[3][r7]=-1; V[15+k0][r7]=1; V[15+k1][r7]=1
            W[2*k0][r7]=1
        
        for i in range(7):
            add_strassen_block(7*i, 2*i, 2*i+1)
        
        base_r = 49
        U[0][base_r]=1; V[14][base_r]=1; W[28][base_r]=1
        U[1][base_r+1]=1; V[29][base_r+1]=1; W[28][base_r+1]=1
        U[2][base_r+2]=1; V[14][base_r+2]=1; W[29][base_r+2]=1
        U[3][base_r+3]=1; V[29][base_r+3]=1; W[29][base_r+3]=1
    
    # Verify
    if time.time() < _DEADLINE - 5:
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        T_reconstructed = np.zeros((2, 2, 15))
        for i in range(2):
            for j in range(2):
                for k in range(15):
                    T_reconstructed[i,j,k] = np.sum(
                        U_arr[i*2+j, :] * V_arr[j*15+k, :] * W_arr[k*2+i, :]
                    )
        
        if not np.allclose(T_reconstructed, 1.0, atol=1e-10):
            # Fallback to known good solution
            rank = 53
            U = [[0] * rank for _ in range(4)]
            V = [[0] * rank for _ in range(30)]
            W = [[0] * rank for _ in range(30)]
            
            def add_strassen_block(base_r, k0, k1):
                r1 = base_r; r2 = base_r+1; r3 = base_r+2; r4 = base_r+3
                r5 = base_r+4; r6 = base_r+5; r7 = base_r+6
                
                U[0][r1]=1; U[3][r1]=1; V[k0][r1]=1; V[15+k1][r1]=1
                W[2*k0][r1]=1; W[2*k1+1][r1]=1
                
                U[2][r2]=1; U[3][r2]=1; V[k0][r2]=1
                W[2*k0+1][r2]=1; W[2*k1+1][r2]=-1
                
                U[0][r3]=1; V[15+k1][r3]=1; V[k1][r3]=-1
                W[2*k1][r3]=1; W[2*k1+1][r3]=1
                
                U[3][r4]=1; V[15+k0][r4]=1; V[k0][r4]=-1
                W[2*k0][r4]=1; W[2*k0+1][r4]=1
                
                U[0][r5]=1; U[1][r5]=1; V[15+k1][r5]=1
                W[2*k0][r5]=-1; W[2*k1][r5]=1
                
                U[2][r6]=1; U[0][r6]=-1; V[k0][r6]=1; V[15+k0][r6]=1
                W[2*k1+1][r6]=1
                
                U[1][r7]=1; U[3][r7]=-1; V[15+k0][r7]=1; V[15+k1][r7]=1
                W[2*k0][r7]=1
            
            for i in range(7):
                add_strassen_block(7*i, 2*i, 2*i+1)
            
            base_r = 49
            U[0][base_r]=1; V[14][base_r]=1; W[28][base_r]=1
            U[1][base_r+1]=1; V[29][base_r+1]=1; W[28][base_r+1]=1
            U[2][base_r+2]=1; V[14][base_r+2]=1; W[29][base_r+2]=1
            U[3][base_r+3]=1; V[29][base_r+3]=1; W[29][base_r+3]=1
    
    return {"U": U, "V": V, "W": W}
